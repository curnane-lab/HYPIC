//! PIC (Position-Independent Cache) scatter routing helpers.
//!
//! Scatter segments are assigned to prefill workers via `assign_with_directory`:
//! a segment whose text_hash is already in the directory routes to its holder;
//! a miss falls to the next round-robin worker and is optimistically recorded.

/// Default separator used to split a PIC prompt into segments.
pub const DEFAULT_PIC_SEPARATOR: &str = "<<PIC_SEP>>";

/// Segment-cache directory: text_hash -> worker_url. Consulted by
/// `assign_with_directory` on every scatter dispatch.
pub type SegCache = std::collections::HashMap<u64, String>;

/// Split a prompt into segments on `sep`.
///
/// No separator present -> a single-element vec (`[prompt]`).
pub fn split_segments<'a>(prompt: &'a str, sep: &str) -> Vec<&'a str> {
    prompt.split(sep).collect()
}

/// FNV-1a 64-bit hash of `s`. Must stay byte-identical to the Python side
/// (`sglang.srt.pic.scatter_xfer.text_hash`) — it's the directory key shared
/// across router and worker.
///
/// ponytail: text_hash is a routing hint only; a collision just degrades to a
/// recompute, never wrong output.
pub fn text_hash(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 跨请求持久路由状态。所有字段在同一 mutex 下,`assign_with_directory` 一次拿全部,
/// `spawn_seg_cache_rebuild` 周期性刷新 `dir` 并衰减 `loads`。
#[derive(Debug, Default)]
pub struct PicRouteState {
    /// 目录:text_hash -> holder url(rebuild 整体替换)。
    pub dir: SegCache,
    /// RR:全局 miss 游标,跨请求推进 —— 稳态每请求 1 miss 时轮转所有 worker。
    pub rr_cursor: usize,
    /// LPT:per-worker-url 累积字节负载,跨请求。
    /// ponytail: 自建累加器而非 `Worker::load()`/load monitor —— 后者不观测
    /// fire-and-forget 的 scatter fan-out(不走维护 load_counter 的正常 dispatch,
    /// 段完成无 router 侧钩子),故 rebuild tick 减半衰减的自包含累加器才是对的工具。
    pub loads: std::collections::HashMap<String, usize>,
}

impl PicRouteState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Assign each scatter segment to a prefill worker, consulting the directory.
///
/// Hit (text_hash in `state.dir`) -> route to the recorded holder, no RR advance,
/// no load added. Miss -> next round-robin worker (global `rr_cursor`), then
/// optimistically record the holder so a later seg with the same hash in this
/// same call lands on it too.
///
/// Returns the per-seg assigned worker url (parallel to `segs`).
///
/// With `lpt = true`, misses are instead assigned longest-first (by segment
/// byte length) to the currently least-loaded worker (global `loads`) — a
/// load-balancing variant whose result is independent of input order.
pub fn assign_with_directory(
    segs: &[&str],
    workers: &[&str],
    state: &mut PicRouteState,
    lpt: bool,
) -> Vec<String> {
    if workers.is_empty() {
        return Vec::new();
    }
    if !lpt {
        // RR: 单遍,hit->holder,miss->全局游标 % n + 乐观写入 + 推进游标。
        let mut out = Vec::with_capacity(segs.len());
        for seg in segs {
            let h = text_hash(seg);
            if let Some(holder) = state.dir.get(&h) {
                out.push(holder.clone());
            } else {
                let url = workers[state.rr_cursor % workers.len()];
                state.rr_cursor = state.rr_cursor.wrapping_add(1);
                state.dir.insert(h, url.to_string());
                out.push(url.to_string());
            }
        }
        return out;
    }
    // LPT: 两遍。第一遍分 hit/miss;第二遍 miss 按字节长度降序贪心到全局最小负载 worker。
    // ponytail: byte-len proxy for balancing; exact token count not worth a router-side tokenize.
    let mut out = vec![String::new(); segs.len()];
    let mut miss_indices: Vec<usize> = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        if let Some(holder) = state.dir.get(&text_hash(seg)) {
            out[i] = holder.clone();
        } else {
            miss_indices.push(i);
        }
    }
    miss_indices.sort_by(|&a, &b| segs[b].len().cmp(&segs[a].len()));
    for &i in miss_indices.iter() {
        let h = text_hash(segs[i]);
        // 重复段:本次循环已写入 -> 复用 holder,不加负载(折叠语义)。
        if let Some(holder) = state.dir.get(&h) {
            out[i] = holder.clone();
            continue;
        }
        // 平局取最小 index(min_by_key 返回首个最小)-> 确定性。
        let w = (0..workers.len())
            .min_by_key(|&w| state.loads.get(workers[w]).copied().unwrap_or(0))
            .unwrap();
        *state.loads.entry(workers[w].to_string()).or_insert(0) += segs[i].len();
        state.dir.insert(h, workers[w].to_string());
        out[i] = workers[w].to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_basic() {
        assert_eq!(
            split_segments("a<<S>>b<<S>>q", "<<S>>"),
            vec!["a", "b", "q"]
        );
        assert_eq!(split_segments("nosep", "<<S>>"), vec!["nosep"]);
    }

    #[test]
    fn assign_dir_hit_then_miss() {
        let mut st = PicRouteState::new();
        st.dir.insert(text_hash("b"), "w1".into());
        // segs ["a","b"], workers [w0,w1] -> a miss->RR(w0), b hit->w1
        let got = assign_with_directory(&["a", "b"], &["w0", "w1"], &mut st, false);
        assert_eq!(got, vec!["w0".to_string(), "w1".to_string()]);
    }

    #[test]
    fn rr_cursor_spreads_across_requests() {
        // 两个独立请求各含 1 个不同的 miss 段 -> 应落不同 worker(游标跨请求推进)。
        let mut st = PicRouteState::new();
        let r1 = assign_with_directory(&["req1seg"], &["w0", "w1", "w2"], &mut st, false);
        let r2 = assign_with_directory(&["req2seg"], &["w0", "w1", "w2"], &mut st, false);
        assert_eq!(r1[0], "w0");
        assert_eq!(r2[0], "w1", "consecutive single-miss requests must not pile on w0");
    }

    #[test]
    fn lpt_load_accumulates_across_requests() {
        // 两个独立请求各含 1 个不同的 miss 段 -> 第二个应避开第一个(负载跨请求累积)。
        let mut st = PicRouteState::new();
        let r1 = assign_with_directory(&["alpha"], &["w0", "w1", "w2"], &mut st, true);
        let r2 = assign_with_directory(&["bravo"], &["w0", "w1", "w2"], &mut st, true);
        assert_eq!(r1[0], "w0"); // 全 0 负载 -> min index 0
        assert_ne!(r2[0], "w0", "second request's miss must move off the loaded worker");
    }

    #[test]
    fn fnv1a_cross_lang_fixtures() {
        // Documented in task-2.2-report.md; must equal the Python text_hash.
        assert_eq!(text_hash(""), 0xcbf29ce484222325);
        assert_eq!(text_hash("a"), 0xaf63dc4c8601ec8c);
        assert_eq!(text_hash("hello"), 0xa430d84680aabd0b);
    }

    #[test]
    fn lpt_balances_and_beats_bad_rr() {
        // 段字节长度(用重复字符构造,seg.len() == 该值)。
        // 编排:三个大段(100)落在 RR 的同一残差类(idx 0,4,8),外加一个 90,
        // 其余为 10 —— 这是对 round-robin 不利的排列,LPT 应显著更优。
        let lens: [usize; 12] = [100, 10, 10, 10, 100, 10, 10, 10, 100, 90, 10, 10];
        let owned: Vec<String> = lens
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let mut s = format!("s{:02}", i); // 3 unique bytes
                s.push_str(&"x".repeat(n - s.len()));
                s
            })
            .collect();
        let segs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let workers = ["w0", "w1", "w2", "w3"];

        let makespan = |assigned: &[String]| -> usize {
            let mut load = std::collections::HashMap::<String, usize>::new();
            for (i, w) in assigned.iter().enumerate() {
                *load.entry(w.clone()).or_insert(0) += lens[i];
            }
            load.values().copied().max().unwrap_or(0)
        };

        let mut dir_rr = PicRouteState::new();
        let rr = assign_with_directory(&segs, &workers, &mut dir_rr, false);
        let mut dir_lpt = PicRouteState::new();
        let lpt = assign_with_directory(&segs, &workers, &mut dir_lpt, true);

        // LPT 严格优于这个坏排列的 RR。
        assert!(
            makespan(&lpt) < makespan(&rr),
            "lpt makespan {} should beat rr {}",
            makespan(&lpt),
            makespan(&rr)
        );
        // LPT 顺序无关:打乱输入顺序,负载 makespan 不变。
        let mut rev_owned = owned.clone();
        rev_owned.reverse();
        let rev: Vec<&str> = rev_owned.iter().map(|s| s.as_str()).collect();
        let mut rev_lens = lens;
        rev_lens.reverse();
        let makespan_rev = |assigned: &[String]| -> usize {
            let mut load = std::collections::HashMap::<String, usize>::new();
            for (i, w) in assigned.iter().enumerate() {
                *load.entry(w.clone()).or_insert(0) += rev_lens[i];
            }
            load.values().copied().max().unwrap_or(0)
        };
        let mut dir_rev = PicRouteState::new();
        let lpt_rev = assign_with_directory(&rev, &workers, &mut dir_rev, true);
        assert_eq!(makespan(&lpt), makespan_rev(&lpt_rev));
    }

    #[test]
    fn lpt_collapses_duplicate_miss() {
        // 同一请求内两个相同段:第二个应折叠到第一个的 holder,不额外占 worker。
        let mut st = PicRouteState::new();
        let got = assign_with_directory(&["dup", "dup"], &["w0", "w1"], &mut st, true);
        assert_eq!(got[0], got[1]);
    }
}

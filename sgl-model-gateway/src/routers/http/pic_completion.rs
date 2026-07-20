//! Reshape sglang `/generate` responses into OpenAI `/v1/completions` shape so
//! PIC scatter (which always dispatches the combine to `/generate`) can serve
//! `/v1/completions` clients. Router-only; no engine changes.

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::StreamExt;
use http::header::CONTENT_TYPE;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::protocols::{completion::CompletionRequest, sampling_params::SamplingParams};

/// Map completion request params onto sglang SamplingParams.
/// Naming deltas: max_tokens->max_new_tokens, min_tokens->min_new_tokens.
pub fn completion_sampling_params(body: &CompletionRequest) -> SamplingParams {
    SamplingParams {
        max_new_tokens: body.max_tokens,
        min_new_tokens: body.min_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        min_p: body.min_p,
        frequency_penalty: body.frequency_penalty,
        presence_penalty: body.presence_penalty,
        repetition_penalty: body.repetition_penalty,
        stop: body.stop.clone(),
        stop_token_ids: body.stop_token_ids.clone(),
        n: body.n,
        regex: body.regex.clone(),
        ebnf: body.ebnf.clone(),
        json_schema: body.json_schema.clone(),
        no_stop_trim: Some(body.no_stop_trim),
        skip_special_tokens: Some(body.skip_special_tokens),
        ignore_eos: Some(body.ignore_eos),
        sampling_seed: body.sampling_seed,
        ..Default::default()
    }
}

/// Shared per-response fields for the completion chunks.
pub struct CmplCtx {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub include_usage: bool,
}

/// Build one completion SSE chunk (already framed with the trailing blank line).
fn frame(v: &Value) -> String {
    format!("data: {}\n\n", v)
}

/// Map ONE `/generate` SSE data payload (raw JSON, no `data:` prefix) to
/// completion SSE lines. `acc` tracks text seen so far (dual-mode delta).
/// Returns [] for a chunk that adds nothing.
pub fn generate_line_to_completion(data: &str, acc: &mut String, ctx: &CmplCtx) -> Vec<String> {
    if data == "[DONE]" {
        // upstream done marker: we emit our own terminal sequence on finish_reason;
        // if we reach here without a finish, still terminate cleanly.
        return vec!["data: [DONE]\n\n".to_string()];
    }
    let v: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return vec![], // skip unparseable/keepalive lines
    };
    let new_text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
    let delta = if new_text.starts_with(acc.as_str()) {
        &new_text[acc.len()..]
    } else {
        new_text
    };
    // finish_reason is an object like {"type":"stop",...} or null until the end.
    let meta = v.get("meta_info");
    let finish: Option<String> = meta
        .and_then(|m| m.get("finish_reason"))
        .filter(|f| !f.is_null())
        .map(|f| {
            f.get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("stop")
                .to_string()
        });

    let mut out = Vec::new();
    if !delta.is_empty() || finish.is_some() {
        out.push(frame(&json!({
            "id": ctx.id, "object": "text_completion", "created": ctx.created,
            "model": ctx.model,
            "choices": [{"text": delta, "index": 0, "finish_reason": finish}],
        })));
    }
    // advance accumulator (dual-mode)
    *acc = if new_text.starts_with(acc.as_str()) {
        new_text.to_string()
    } else {
        format!("{}{}", acc, new_text)
    };
    if let Some(_fr) = &finish {
        if ctx.include_usage {
            if let Some(m) = meta {
                let pt = m.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                let ct = m.get("completion_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                let cached = m.get("cached_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.push(frame(&json!({
                    "id": ctx.id, "object": "text_completion", "created": ctx.created,
                    "model": ctx.model, "choices": [],
                    "usage": {
                        "prompt_tokens": pt, "completion_tokens": ct,
                        "total_tokens": pt + ct,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    },
                })));
            }
        }
        out.push("data: [DONE]\n\n".to_string());
    }
    out
}

fn new_ctx(model: String, include_usage: bool) -> CmplCtx {
    let id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    CmplCtx { id, created, model, include_usage }
}

pub async fn reshape_generate_to_completion(
    resp: Response,
    model: String,
    stream: bool,
    include_usage: bool,
) -> Response {
    if !resp.status().is_success() {
        return resp;
    }
    let ctx = new_ctx(model, include_usage);
    if stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let mut body = resp.into_body().into_data_stream();
        tokio::spawn(async move {
            let mut buf = String::new();
            let mut acc = String::new();
            let mut terminated = false;
            while let Some(item) = body.next().await {
                let chunk = match item { Ok(c) => c, Err(_) => break };
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(nl) = buf.find('\n') {
                    let line: String = buf.drain(..=nl).collect();
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let data = match line.strip_prefix("data:") { Some(d) => d.trim(), None => continue };
                    for out in generate_line_to_completion(data, &mut acc, &ctx) {
                        if out.contains("[DONE]") { terminated = true; }
                        if tx.send(Ok(Bytes::from(out))).is_err() { return; }
                    }
                    if terminated { return; }
                }
            }
            if !terminated {
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        });
        let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
        response.headers_mut().insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
        response
    } else {
        // Non-stream: collect the full body, take the last complete /generate JSON.
        let bytes = match axum::body::to_bytes(resp.into_body(), usize::MAX).await {
            Ok(b) => b,
            Err(_) => return Response::new(Body::from("{}")),
        };
        let text = String::from_utf8_lossy(&bytes);
        // /generate non-stream returns a single JSON object (not SSE).
        let v: Value = serde_json::from_str(text.trim()).unwrap_or(json!({}));
        let full_text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let meta = v.get("meta_info");
        let finish = meta.and_then(|m| m.get("finish_reason")).filter(|f| !f.is_null())
            .and_then(|f| f.get("type")).and_then(|t| t.as_str()).unwrap_or("stop");
        let pt = meta.and_then(|m| m.get("prompt_tokens")).and_then(|x| x.as_u64()).unwrap_or(0);
        let ct = meta.and_then(|m| m.get("completion_tokens")).and_then(|x| x.as_u64()).unwrap_or(0);
        let cached = meta.and_then(|m| m.get("cached_tokens")).and_then(|x| x.as_u64()).unwrap_or(0);
        let body_json = json!({
            "id": ctx.id, "object": "text_completion", "created": ctx.created, "model": ctx.model,
            "choices": [{"text": full_text, "index": 0, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                      "prompt_tokens_details": {"cached_tokens": cached}},
        });
        let mut response = Response::new(Body::from(body_json.to_string()));
        response.headers_mut().insert(CONTENT_TYPE, "application/json".parse().unwrap());
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(include_usage: bool) -> CmplCtx {
        CmplCtx { id: "cmpl-x".into(), created: 1, model: "m".into(), include_usage }
    }

    #[test]
    fn cumulative_text_deltas() {
        // /generate sends cumulative text; deltas must be the suffix.
        let c = ctx(false);
        let mut acc = String::new();
        let l1 = generate_line_to_completion(r#"{"text":"Hello","meta_info":{"finish_reason":null}}"#, &mut acc, &c);
        assert_eq!(l1.len(), 1);
        assert!(l1[0].contains(r#""text":"Hello""#), "{}", l1[0]);
        let l2 = generate_line_to_completion(r#"{"text":"Hello world","meta_info":{"finish_reason":null}}"#, &mut acc, &c);
        assert!(l2[0].contains(r#""text":" world""#), "delta must be suffix: {}", l2[0]);
    }

    #[test]
    fn incremental_text_passthrough() {
        // If /generate sends incremental pieces, each is its own delta.
        let c = ctx(false);
        let mut acc = String::new();
        let l1 = generate_line_to_completion(r#"{"text":"Hello","meta_info":{"finish_reason":null}}"#, &mut acc, &c);
        assert!(l1[0].contains(r#""text":"Hello""#));
        let l2 = generate_line_to_completion(r#"{"text":" world","meta_info":{"finish_reason":null}}"#, &mut acc, &c);
        assert!(l2[0].contains(r#""text":" world""#), "incremental delta: {}", l2[0]);
    }

    #[test]
    fn finish_emits_usage_and_done() {
        let c = ctx(true);
        let mut acc = String::new();
        let lines = generate_line_to_completion(
            r#"{"text":"Hi","meta_info":{"finish_reason":{"type":"stop"},"prompt_tokens":100,"completion_tokens":2,"cached_tokens":40}}"#,
            &mut acc, &c);
        let joined = lines.join("");
        assert!(joined.contains(r#""finish_reason":"stop""#), "{}", joined);
        assert!(joined.contains(r#""prompt_tokens":100"#), "{}", joined);
        assert!(joined.contains(r#""cached_tokens":40"#), "{}", joined);
        assert!(joined.trim_end().ends_with("data: [DONE]"), "{}", joined);
    }

    #[test]
    fn unparseable_line_skipped() {
        let c = ctx(false);
        let mut acc = String::new();
        assert!(generate_line_to_completion(": keepalive", &mut acc, &c).is_empty());
    }

    #[test]
    fn sampling_params_naming_deltas() {
        // build a minimal CompletionRequest via serde to avoid the long literal.
        let body: CompletionRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "x", "max_tokens": 16, "min_tokens": 4, "temperature": 0.0
        })).unwrap();
        let sp = completion_sampling_params(&body);
        assert_eq!(sp.max_new_tokens, Some(16));
        assert_eq!(sp.min_new_tokens, Some(4));
        assert_eq!(sp.temperature, Some(0.0));
    }
}

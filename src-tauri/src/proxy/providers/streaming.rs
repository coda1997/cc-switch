//! 流式响应转换模块
//!
//! 实现 OpenAI SSE → Anthropic SSE 格式转换

use super::antml_fallback::{self, AntmlStreamGuard};
use crate::proxy::sse::{strip_sse_field, take_sse_block};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// OpenAI 流式响应数据结构
#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    // OpenRouter/Kimi/其它 使用 reasoning，DeepSeek 使用 reasoning_content
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DeltaToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// OpenAI 流式响应的 usage 信息（完整版）
#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    /// Some compatible servers return Anthropic-style cache fields directly
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

/// Nested token details from OpenAI format
#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
    #[serde(default)]
    cache_write_tokens: u32,
}

#[derive(Debug, Clone)]
struct ToolBlockState {
    anthropic_index: u32,
    id: String,
    name: String,
    started: bool,
    pending_args: String,
    /// 连续空白字符计数 — 用于检测 Copilot 无限换行 bug
    /// 当 function call 参数中的连续空白字符达到阈值时，强制终止流
    consecutive_whitespace: usize,
    /// 是否已因无限空白 bug 被中止
    aborted: bool,
}

/// 无限空白 bug 的连续空白字符阈值
const INFINITE_WHITESPACE_THRESHOLD: usize = 500;

fn build_anthropic_usage_json(usage: &Usage) -> Value {
    // OpenAI prompt_tokens 含缓存，Anthropic input_tokens 不含，需减去 cache_read 与 cache_creation
    // （三桶互斥，恒等 input + cache_read + cache_creation == prompt_tokens）。
    let cached = extract_cache_read_tokens(usage).unwrap_or(0);
    let cache_creation = extract_cache_write_tokens(usage).unwrap_or(0);
    let input_tokens = usage
        .prompt_tokens
        .saturating_sub(cached)
        .saturating_sub(cache_creation);
    let mut usage_json = json!({
        "input_tokens": input_tokens,
        "output_tokens": usage.completion_tokens
    });
    if cached > 0 {
        usage_json["cache_read_input_tokens"] = json!(cached);
    }
    if cache_creation > 0 {
        usage_json["cache_creation_input_tokens"] = json!(cache_creation);
    }
    usage_json
}

fn default_anthropic_usage_json() -> Value {
    json!({
        "input_tokens": 0,
        "output_tokens": 0
    })
}

fn build_message_delta_event(stop_reason: Option<String>, usage_json: Option<Value>) -> Value {
    let usage = usage_json
        .filter(|usage| usage.is_object())
        .unwrap_or_else(default_anthropic_usage_json);

    json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": null
        },
        "usage": usage
    })
}

/// 创建 Anthropic SSE 流
#[cfg(test)]
pub fn create_anthropic_sse_stream<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    create_anthropic_sse_stream_with_options(
        stream,
        false,
        antml_fallback::AntmlToolSchemas::default(),
    )
}

/// 与 [`create_anthropic_sse_stream`] 相同，但可开启 antml 工具调用兜底。
///
/// `antml_fallback_enabled` 应仅在 GitHub Copilot 供应商 + 开关开启时为 true。
/// 开启后：文本增量先经 [`AntmlStreamGuard`] 过滤，一旦发现泄漏的 antml 包裹标签
/// 即停止把其后文本作为 text 发出，改为在收尾时反解析成 tool_use 事件。
pub fn create_anthropic_sse_stream_with_options<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
    antml_fallback_enabled: bool,
    antml_tool_schemas: antml_fallback::AntmlToolSchemas,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut message_id = None;
        let mut current_model = None;
        let mut next_content_index: u32 = 0;
        let mut has_sent_message_start = false;
        // antml 兜底状态机（未启用时 feed_text 原样放行，零行为变化）。
        let mut antml_guard = AntmlStreamGuard::new(antml_fallback_enabled);
        // 某些上游 provider（如 OpenRouter 的 kimi-k2.6）会在 tool_use 后发送多个
        // 带 finish_reason 的 SSE chunk。Anthropic 协议要求每个消息流只能有一个
        // message_delta，重复会导致 Claude Code abort 连接。因此需要：
        // 1) has_emitted_message_delta: 去重，只处理第一个 finish_reason
        // 2) pending_message_delta: 缓存延迟到 [DONE] 发送，确保 usage 完整
        let mut has_emitted_message_delta = false;
        let mut pending_message_delta: Option<(Option<String>, Option<Value>)> = None;
        let mut has_sent_message_stop = false;
        let mut stream_ended_with_error = false;
        let mut latest_usage: Option<Value> = None;
        let mut current_non_tool_block_type: Option<&'static str> = None;
        let mut current_non_tool_block_index: Option<u32> = None;
        let mut tool_blocks_by_index: HashMap<usize, ToolBlockState> = HashMap::new();
        let mut open_tool_block_indices: HashSet<u32> = HashSet::new();
        let mut saw_native_tool_calls = false;

        tokio::pin!(stream);

        'upstream: while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::proxy::sse::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(line) = take_sse_block(&mut buffer) {
                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = strip_sse_field(l, "data") {
                                if data.trim() == "[DONE]" {
                                    log::debug!("[Claude/OpenRouter] <<< OpenAI SSE: [DONE]");

                                    // 没有 finish_reason 的候选 XML 不得恢复为工具调用。把候选或
                                    // 哨兵前缀作为普通文本补发，并保证所有 content 事件先于终止事件。
                                    let leftover = if antml_guard.is_armed() {
                                        antml_guard.take_buffer()
                                    } else {
                                        antml_guard.take_unflushed_text()
                                    };
                                    if !leftover.is_empty() {
                                        if current_non_tool_block_type == Some("text") {
                                            if let Some(index) = current_non_tool_block_index {
                                                let event = json!({
                                                    "type": "content_block_delta",
                                                    "index": index,
                                                    "delta": {"type": "text_delta", "text": leftover}
                                                });
                                                yield Ok(Bytes::from(format!(
                                                    "event: content_block_delta\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default()
                                                )));
                                            }
                                        } else {
                                            if let Some(index) = current_non_tool_block_index.take() {
                                                let event = json!({"type": "content_block_stop", "index": index});
                                                yield Ok(Bytes::from(format!(
                                                    "event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default()
                                                )));
                                            }
                                            let (events, next) =
                                                antml_fallback::standalone_text_sse_events(
                                                    &leftover,
                                                    next_content_index,
                                                );
                                            next_content_index = next;
                                            for event in events {
                                                yield Ok(Bytes::from(event));
                                            }
                                        }
                                    }

                                    if let Some(index) = current_non_tool_block_index.take() {
                                        let event = json!({"type": "content_block_stop", "index": index});
                                        yield Ok(Bytes::from(format!(
                                            "event: content_block_stop\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        )));
                                    }
                                    current_non_tool_block_type = None;

                                    // 流正常结束，发出缓存的 message_delta（含完整 usage）。
                                    if let Some((stop_reason, usage_json)) = pending_message_delta.take() {
                                        let event = build_message_delta_event(stop_reason, usage_json);
                                        let sse_data = format!("event: message_delta\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default());
                                        log::debug!("[Claude/OpenRouter] >>> Anthropic SSE: message_delta (from pending)");
                                        yield Ok(Bytes::from(sse_data));
                                    }

                                    if !has_sent_message_stop {
                                        let event = json!({"type": "message_stop"});
                                        let sse_data = format!("event: message_stop\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default());
                                        log::debug!("[Claude/OpenRouter] >>> Anthropic SSE: message_stop");
                                        yield Ok(Bytes::from(sse_data));
                                    }
                                    has_sent_message_stop = true;
                                    break 'upstream;
                                }

                                if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                                    log::debug!("[Claude/OpenRouter] <<< SSE chunk received");

                                    if message_id.is_none() && !chunk.id.is_empty() {
                                        message_id = Some(chunk.id.clone());
                                    }
                                    if current_model.is_none() && !chunk.model.is_empty() {
                                        current_model = Some(chunk.model.clone());
                                    }

                                    let chunk_usage_json =
                                        chunk.usage.as_ref().map(build_anthropic_usage_json);
                                    if let Some(usage_json) = &chunk_usage_json {
                                        latest_usage = Some(usage_json.clone());
                                        if let Some((_, pending_usage)) = pending_message_delta.as_mut() {
                                            *pending_usage = Some(usage_json.clone());
                                        }
                                    }

                                    if let Some(choice) = chunk.choices.first() {
                                        if !has_sent_message_start {
                                            // Build usage with cache tokens if available from first chunk
                                            let mut start_usage = json!({
                                                "input_tokens": 0,
                                                "output_tokens": 0
                                            });
                                            if let Some(u) = &chunk.usage {
                                                let cached = extract_cache_read_tokens(u).unwrap_or(0);
                                                let cache_creation =
                                                    extract_cache_write_tokens(u).unwrap_or(0);
                                                let input = u
                                                    .prompt_tokens
                                                    .saturating_sub(cached)
                                                    .saturating_sub(cache_creation);
                                                start_usage["input_tokens"] = json!(input);
                                                if cached > 0 {
                                                    start_usage["cache_read_input_tokens"] = json!(cached);
                                                }
                                                if cache_creation > 0 {
                                                    start_usage["cache_creation_input_tokens"] =
                                                        json!(cache_creation);
                                                }
                                            }

                                            let event = json!({
                                                "type": "message_start",
                                                "message": {
                                                    "id": message_id.clone().unwrap_or_default(),
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "model": current_model.clone().unwrap_or_default(),
                                                    "usage": start_usage
                                                }
                                            });
                                            let sse_data = format!("event: message_start\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                            has_sent_message_start = true;
                                        }

                                        // 处理 reasoning（thinking）
                                        if let Some(reasoning) = &choice.delta.reasoning {
                                            if current_non_tool_block_type != Some("thinking") {
                                                if let Some(index) = current_non_tool_block_index.take() {
                                                    let event = json!({
                                                        "type": "content_block_stop",
                                                        "index": index
                                                    });
                                                    let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                                let index = next_content_index;
                                                next_content_index += 1;
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": index,
                                                    "content_block": {
                                                        "type": "thinking",
                                                        "thinking": ""
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                                current_non_tool_block_type = Some("thinking");
                                                current_non_tool_block_index = Some(index);
                                            }

                                            if let Some(index) = current_non_tool_block_index {
                                                let event = json!({
                                                    "type": "content_block_delta",
                                                    "index": index,
                                                    "delta": {
                                                        "type": "thinking_delta",
                                                        "thinking": reasoning
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }
                                        }

                                        // 处理文本内容
                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                // antml 兜底：只发出 guard 放行的文本；
                                                // 一旦 guard 上膛，其后文本被缓冲、留待收尾解析成 tool_use。
                                                let released = antml_guard.feed_text(content);
                                                if !released.is_empty() {
                                                    if current_non_tool_block_type != Some("text") {
                                                        if let Some(index) = current_non_tool_block_index.take() {
                                                            let event = json!({
                                                                "type": "content_block_stop",
                                                                "index": index
                                                            });
                                                            let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                                serde_json::to_string(&event).unwrap_or_default());
                                                            yield Ok(Bytes::from(sse_data));
                                                        }

                                                        let index = next_content_index;
                                                        next_content_index += 1;
                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": index,
                                                            "content_block": {
                                                                "type": "text",
                                                                "text": ""
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        current_non_tool_block_type = Some("text");
                                                        current_non_tool_block_index = Some(index);
                                                    }

                                                    if let Some(index) = current_non_tool_block_index {
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": index,
                                                            "delta": {
                                                                "type": "text_delta",
                                                                "text": released
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                }
                                            }
                                        }

                                        // 处理工具调用
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            if !tool_calls.is_empty() {
                                                saw_native_tool_calls = true;
                                                if let Some(index) = current_non_tool_block_index.take() {
                                                    let event = json!({
                                                        "type": "content_block_stop",
                                                        "index": index
                                                    });
                                                    let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                                current_non_tool_block_type = None;

                                                for tool_call in tool_calls {
                                                    let (
                                                        anthropic_index,
                                                        id,
                                                        name,
                                                        should_start,
                                                        pending_after_start,
                                                        immediate_delta,
                                                    ) = {
                                                        let state = tool_blocks_by_index
                                                            .entry(tool_call.index)
                                                            .or_insert_with(|| {
                                                                let index = next_content_index;
                                                                next_content_index += 1;
                                                                ToolBlockState {
                                                                    anthropic_index: index,
                                                                    id: String::new(),
                                                                    name: String::new(),
                                                                    started: false,
                                                                    pending_args: String::new(),
                                                                    consecutive_whitespace: 0,
                                                                    aborted: false,
                                                                }
                                                            });

                                                        // 如果此 tool call 已被中止（无限空白 bug），跳过后续处理
                                                        if state.aborted {
                                                            continue;
                                                        }

                                                        if let Some(id) = &tool_call.id {
                                                            state.id = id.clone();
                                                        }
                                                        if let Some(function) = &tool_call.function {
                                                            if let Some(name) = &function.name {
                                                                state.name = name.clone();
                                                            }
                                                        }

                                                        let should_start =
                                                            !state.started
                                                                && !state.id.is_empty()
                                                                && !state.name.is_empty();
                                                        if should_start {
                                                            state.started = true;
                                                        }
                                                        let pending_after_start = if should_start
                                                            && !state.pending_args.is_empty()
                                                        {
                                                            Some(std::mem::take(&mut state.pending_args))
                                                        } else {
                                                            None
                                                        };
                                                        let args_delta = tool_call
                                                            .function
                                                            .as_ref()
                                                            .and_then(|f| f.arguments.clone());
                                                        let immediate_delta = if let Some(args) = args_delta {
                                                            // 无限空白 bug 检测：跟踪连续空白字符
                                                            for ch in args.chars() {
                                                                if ch.is_whitespace() {
                                                                    state.consecutive_whitespace += 1;
                                                                } else {
                                                                    state.consecutive_whitespace = 0;
                                                                }
                                                            }
                                                            if state.consecutive_whitespace >= INFINITE_WHITESPACE_THRESHOLD {
                                                                log::warn!(
                                                                    "[Copilot] 检测到无限空白 bug (tool: {}), 中止此 tool call 流",
                                                                    state.name
                                                                );
                                                                state.aborted = true;
                                                                None
                                                            } else if state.started {
                                                                Some(args)
                                                            } else {
                                                                state.pending_args.push_str(&args);
                                                                None
                                                            }
                                                        } else {
                                                            None
                                                        };
                                                        (
                                                            state.anthropic_index,
                                                            state.id.clone(),
                                                            state.name.clone(),
                                                            should_start,
                                                            pending_after_start,
                                                            immediate_delta,
                                                        )
                                                    };

                                                    if should_start {
                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": anthropic_index,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": id,
                                                                "name": name
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        open_tool_block_indices.insert(anthropic_index);
                                                    }

                                                    if let Some(args) = pending_after_start {
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": anthropic_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }

                                                    if let Some(args) = immediate_delta {
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": anthropic_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                }
                                            }
                                        }

                                        // 处理 finish_reason。
                                        // 注意：OpenRouter 某些 provider 会发送多个带 finish_reason 的 chunk
                                        // （第一个 usage 为 null，后续才补全）。此处只做缓存，不立即发送，
                                        // 等到 [DONE] 或流末尾再统一发出，确保 usage 完整且只发一次。
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            let stop_reason = map_stop_reason(Some(finish_reason));
                                            let usage_json =
                                                chunk_usage_json.clone().or_else(|| latest_usage.clone());

                                            if has_emitted_message_delta {
                                                // 更新缓存的 message_delta usage（如果有更完整的 usage）
                                                if let (Some((_, ref mut usage)), Some(uj)) = (&mut pending_message_delta, usage_json) {
                                                    *usage = Some(uj);
                                                }
                                                continue;
                                            }
                                            has_emitted_message_delta = true;

                                            if let Some(index) = current_non_tool_block_index.take() {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": index
                                                });
                                                let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }
                                            current_non_tool_block_type = None;

                                            // Late start for blocks that accumulated args before id/name arrived.
                                            let mut late_tool_starts: Vec<(u32, String, String, String)> =
                                                Vec::new();
                                            for (tool_idx, state) in tool_blocks_by_index.iter_mut() {
                                                if state.started {
                                                    continue;
                                                }
                                                let has_payload = !state.pending_args.is_empty()
                                                    || !state.id.is_empty()
                                                    || !state.name.is_empty();
                                                if !has_payload {
                                                    continue;
                                                }
                                                let fallback_id = if state.id.is_empty() {
                                                    format!("tool_call_{tool_idx}")
                                                } else {
                                                    state.id.clone()
                                                };
                                                let fallback_name = if state.name.is_empty() {
                                                    "unknown_tool".to_string()
                                                } else {
                                                    state.name.clone()
                                                };
                                                state.started = true;
                                                let pending = std::mem::take(&mut state.pending_args);
                                                late_tool_starts.push((
                                                    state.anthropic_index,
                                                    fallback_id,
                                                    fallback_name,
                                                    pending,
                                                ));
                                            }
                                            late_tool_starts.sort_unstable_by_key(|(index, _, _, _)| *index);
                                            for (index, id, name, pending) in late_tool_starts {
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": index,
                                                    "content_block": {
                                                        "type": "tool_use",
                                                        "id": id,
                                                        "name": name
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                                open_tool_block_indices.insert(index);
                                                if !pending.is_empty() {
                                                    let delta_event = json!({
                                                        "type": "content_block_delta",
                                                        "index": index,
                                                        "delta": {
                                                            "type": "input_json_delta",
                                                            "partial_json": pending
                                                        }
                                                    });
                                                    let delta_sse = format!("event: content_block_delta\ndata: {}\n\n",
                                                        serde_json::to_string(&delta_event).unwrap_or_default());
                                                    yield Ok(Bytes::from(delta_sse));
                                                }
                                            }

                                            if !open_tool_block_indices.is_empty() {
                                                let mut tool_indices: Vec<u32> =
                                                    open_tool_block_indices.iter().copied().collect();
                                                tool_indices.sort_unstable();
                                                for index in tool_indices {
                                                    let event = json!({
                                                        "type": "content_block_stop",
                                                        "index": index
                                                    });
                                                    let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                                open_tool_block_indices.clear();
                                            }

                                            // antml 兜底：guard 已捕获泄漏的工具调用文本。
                                            // 此处（text/tool 块已收尾后）把缓冲反解析成 tool_use 事件，
                                            // 并把 stop_reason 改写为 tool_use，让 Claude Code 继续执行。
                                            let mut final_stop_reason = stop_reason;
                                            if antml_guard.is_armed() {
                                                let buffered = antml_guard.take_buffer();
                                                if saw_native_tool_calls {
                                                    log::info!(
                                                        "[Copilot] antml 兜底：检测到原生 tool_calls，跳过重复 XML 恢复"
                                                    );
                                                } else if finish_reason == "stop" {
                                                    match antml_fallback::parse_function_calls(
                                                        &buffered,
                                                        &antml_tool_schemas,
                                                    ) {
                                                    Some(parsed) => {
                                                        if !parsed.prose.is_empty() {
                                                            let (events, next) =
                                                                antml_fallback::standalone_text_sse_events(
                                                                    &parsed.prose,
                                                                    next_content_index,
                                                                );
                                                            next_content_index = next;
                                                            for ev in events {
                                                                yield Ok(Bytes::from(ev));
                                                            }
                                                        }
                                                        let (events, next) =
                                                            antml_fallback::tool_use_sse_events(
                                                                &parsed.calls,
                                                                next_content_index,
                                                            );
                                                        next_content_index = next;
                                                        for ev in events {
                                                            yield Ok(Bytes::from(ev));
                                                        }
                                                        log::info!(
                                                            "[Copilot] antml 兜底：流式响应还原 {} 个工具调用",
                                                            parsed.calls.len()
                                                        );
                                                        final_stop_reason =
                                                            Some("tool_use".to_string());
                                                    }
                                                    // 解析失败：把缓冲原样作为文本补发，避免吞内容。
                                                    None => {
                                                        log::warn!(
                                                            "[Copilot] antml 兜底：候选 XML 未通过完整性或工具 schema 校验，按文本返回"
                                                        );
                                                        if !buffered.is_empty() {
                                                            let (events, next) =
                                                                antml_fallback::standalone_text_sse_events(
                                                                    &buffered,
                                                                    next_content_index,
                                                                );
                                                            next_content_index = next;
                                                            for ev in events {
                                                                yield Ok(Bytes::from(ev));
                                                            }
                                                        }
                                                    }
                                                }
                                                } else if !buffered.is_empty() {
                                                    let (events, next) =
                                                        antml_fallback::standalone_text_sse_events(
                                                            &buffered,
                                                            next_content_index,
                                                        );
                                                    next_content_index = next;
                                                    for ev in events {
                                                        yield Ok(Bytes::from(ev));
                                                    }
                                                }
                                            }

                                            // 缓存 message_delta，等到 [DONE] 时发送（以便收集完整的 usage）
                                            pending_message_delta = Some((final_stop_reason, usage_json));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Stream error: {e}");
                    stream_ended_with_error = true;
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "stream_error",
                            "message": format!("Stream error: {e}")
                        }
                    });
                    let sse_data = format!("event: error\ndata: {}\n\n",
                        serde_json::to_string(&error_event).unwrap_or_default());
                    yield Ok(Bytes::from(sse_data));
                    break;
                }
            }
        }

        // 流自然结束但未收到 [DONE] 时，确保发送缓存的 message_delta 和 message_stop。
        // 若上游已显式报错，则只保留 error 事件，避免把失败伪装成成功完成。
        if !stream_ended_with_error {
            // EOF 没有 finish_reason 时必须失败关闭：候选 XML 只作为文本补发，
            // 不合成 tool_use 或成功终止事件。
            let leftover = if antml_guard.is_armed() {
                antml_guard.take_buffer()
            } else {
                antml_guard.take_unflushed_text()
            };
            if !leftover.is_empty() {
                if current_non_tool_block_type == Some("text") {
                    if let Some(index) = current_non_tool_block_index {
                        let event = json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": leftover}
                        });
                        yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n",
                            serde_json::to_string(&event).unwrap_or_default())));
                    }
                } else {
                    if let Some(index) = current_non_tool_block_index.take() {
                        let event = json!({"type": "content_block_stop", "index": index});
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n",
                            serde_json::to_string(&event).unwrap_or_default())));
                    }
                    let (events, _) = antml_fallback::standalone_text_sse_events(
                        &leftover, next_content_index);
                    for ev in events { yield Ok(Bytes::from(ev)); }
                }
            }

            let emitted_pending_message_delta = if let Some((stop_reason, usage_json)) =
                pending_message_delta.take()
            {
                let event = build_message_delta_event(stop_reason, usage_json);
                let sse_data = format!("event: message_delta\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap_or_default());
                log::debug!("[Claude/OpenRouter] >>> Anthropic SSE: message_delta (at stream end)");
                yield Ok(Bytes::from(sse_data));
                true
            } else {
                false
            };

            if emitted_pending_message_delta && !has_sent_message_stop {
                let event = json!({"type": "message_stop"});
                let sse_data = format!("event: message_stop\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap_or_default());
                log::debug!("[Claude/OpenRouter] >>> Anthropic SSE: message_stop (at stream end)");
                yield Ok(Bytes::from(sse_data));
            }
        }
    }
}

/// Extract cache_read tokens from Usage, checking both direct field and nested details
fn extract_cache_read_tokens(usage: &Usage) -> Option<u32> {
    // Direct field takes priority (compatible servers)
    if let Some(v) = usage.cache_read_input_tokens {
        return Some(v);
    }
    // OpenAI standard: prompt_tokens_details.cached_tokens
    usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .filter(|&v| v > 0)
}

/// Extract cache-write tokens from direct compatibility fields or OpenAI details.
fn extract_cache_write_tokens(usage: &Usage) -> Option<u32> {
    if let Some(value) = usage.cache_creation_input_tokens {
        return Some(value);
    }
    usage
        .prompt_tokens_details
        .as_ref()
        .map(|details| details.cache_write_tokens)
        .filter(|value| *value > 0)
}

/// 映射停止原因
fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| {
        match r {
            "tool_calls" | "function_call" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            "content_filter" => "end_turn",
            other => {
                log::warn!("[Claude/OpenRouter] Unknown finish_reason in streaming: {other}");
                "end_turn"
            }
        }
        .to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;
    use serde_json::Value;
    use std::collections::HashMap;

    async fn collect_anthropic_events(input: &str) -> Vec<Value> {
        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(
            input.as_bytes().to_vec(),
        ))]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;
        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect()
    }

    fn event_type(event: &Value) -> Option<&str> {
        event.get("type").and_then(|v| v.as_str())
    }

    #[test]
    fn test_map_stop_reason_legacy_and_filtered_values() {
        assert_eq!(
            map_stop_reason(Some("function_call")),
            Some("tool_use".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("content_filter")),
            Some("end_turn".to_string())
        );
    }

    #[tokio::test]
    async fn test_streaming_tool_calls_routed_by_index() {
        let input = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"second_tool\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":4}}\n\n",
            "data: [DONE]\n\n"
        );

        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(
            input.as_bytes().to_vec(),
        ))]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;

        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        let events: Vec<Value> = merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect();

        let mut tool_index_by_call: HashMap<String, u64> = HashMap::new();
        for event in &events {
            if event.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                && event
                    .pointer("/content_block/type")
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
            {
                if let (Some(call_id), Some(index)) = (
                    event.pointer("/content_block/id").and_then(|v| v.as_str()),
                    event.get("index").and_then(|v| v.as_u64()),
                ) {
                    tool_index_by_call.insert(call_id.to_string(), index);
                }
            }
        }

        assert_eq!(tool_index_by_call.len(), 2);
        assert_ne!(
            tool_index_by_call.get("call_0"),
            tool_index_by_call.get("call_1")
        );

        let deltas: Vec<(u64, String)> = events
            .iter()
            .filter(|event| {
                event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
                    && event.pointer("/delta/type").and_then(|v| v.as_str())
                        == Some("input_json_delta")
            })
            .filter_map(|event| {
                let index = event.get("index").and_then(|v| v.as_u64())?;
                let partial_json = event
                    .pointer("/delta/partial_json")
                    .and_then(|v| v.as_str())?
                    .to_string();
                Some((index, partial_json))
            })
            .collect();

        assert_eq!(deltas.len(), 2);
        let second_idx = deltas
            .iter()
            .find_map(|(index, payload)| (payload == "{\"b\":2}").then_some(*index))
            .unwrap();
        let first_idx = deltas
            .iter()
            .find_map(|(index, payload)| (payload == "{\"a\":1}").then_some(*index))
            .unwrap();

        assert_eq!(second_idx, *tool_index_by_call.get("call_1").unwrap());
        assert_eq!(first_idx, *tool_index_by_call.get("call_0").unwrap());

        assert!(events.iter().any(|event| {
            event.get("type").and_then(|v| v.as_str()) == Some("message_delta")
                && event.pointer("/delta/stop_reason").and_then(|v| v.as_str()) == Some("tool_use")
        }));
    }

    #[tokio::test]
    async fn test_streaming_delays_tool_start_until_id_and_name_ready() {
        let input = concat!(
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );

        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(
            input.as_bytes().to_vec(),
        ))]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;
        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        let events: Vec<Value> = merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect();

        let starts: Vec<&Value> = events
            .iter()
            .filter(|event| {
                event.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                    && event
                        .pointer("/content_block/type")
                        .and_then(|v| v.as_str())
                        == Some("tool_use")
            })
            .collect();
        assert_eq!(starts.len(), 1);
        assert_eq!(
            starts[0]
                .pointer("/content_block/id")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "call_0"
        );
        assert_eq!(
            starts[0]
                .pointer("/content_block/name")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "first_tool"
        );

        let deltas: Vec<&str> = events
            .iter()
            .filter(|event| {
                event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
                    && event.pointer("/delta/type").and_then(|v| v.as_str())
                        == Some("input_json_delta")
            })
            .filter_map(|event| {
                event
                    .pointer("/delta/partial_json")
                    .and_then(|v| v.as_str())
            })
            .collect();
        assert!(deltas.contains(&"{\"a\":"));
        assert!(deltas.contains(&"1}"));
    }

    #[tokio::test]
    async fn test_streaming_chinese_split_across_chunks_no_replacement_chars() {
        // "你好" split across two TCP chunks inside a streaming text delta.
        // Before the fix, from_utf8_lossy would produce U+FFFD for each half.
        let full = concat!(
            "data: {\"id\":\"chatcmpl_3\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl_3\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let bytes = full.as_bytes();

        // Find "你" in the byte stream and split inside it
        let ni_start = bytes.windows(3).position(|w| w == "你".as_bytes()).unwrap();
        let split_point = ni_start + 1; // split after first byte of "你"

        let chunk1 = Bytes::from(bytes[..split_point].to_vec());
        let chunk2 = Bytes::from(bytes[split_point..].to_vec());

        let upstream = stream::iter(vec![
            Ok::<_, std::io::Error>(chunk1),
            Ok::<_, std::io::Error>(chunk2),
        ]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;

        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        // Must contain the original Chinese characters, not replacement chars
        assert!(
            merged.contains("你好"),
            "expected '你好' in output, got replacement chars (U+FFFD)"
        );
        assert!(
            !merged.contains('\u{FFFD}'),
            "output must not contain U+FFFD replacement characters"
        );
    }

    #[tokio::test]
    async fn test_duplicate_finish_reason_emits_only_one_message_delta() {
        // Simulates OpenRouter behavior where two chunks carry finish_reason:
        // first with null usage, second with populated usage.
        let input = concat!(
            "data: {\"id\":\"chatcmpl_dup\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"id\":\"chatcmpl_dup\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );

        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(
            input.as_bytes().to_vec(),
        ))]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;

        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        let events: Vec<Value> = merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect();

        let message_deltas: Vec<&Value> = events
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_delta"))
            .collect();

        assert_eq!(
            message_deltas.len(),
            1,
            "duplicate finish_reason chunks must produce exactly one message_delta, got {}: {:?}",
            message_deltas.len(),
            message_deltas
        );

        assert_eq!(message_deltas[0]["usage"]["input_tokens"], 10);
        assert_eq!(message_deltas[0]["usage"]["output_tokens"], 5);

        let message_stops = events
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_stop"))
            .count();
        assert_eq!(message_stops, 1, "message_stop must only be emitted once");
    }

    #[tokio::test]
    async fn test_usage_only_chunk_after_finish_reason_updates_message_delta_usage() {
        let input = concat!(
            "data: {\"id\":\"chatcmpl_split\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tool-0924\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_split\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":13312,\"completion_tokens\":79,\"prompt_tokens_details\":{\"cached_tokens\":100}}}\n\n",
            "data: [DONE]\n\n"
        );

        let events = collect_anthropic_events(input).await;
        let message_deltas: Vec<&Value> = events
            .iter()
            .filter(|event| event_type(event) == Some("message_delta"))
            .collect();
        let message_stops = events
            .iter()
            .filter(|event| event_type(event) == Some("message_stop"))
            .count();

        assert_eq!(message_deltas.len(), 1);
        assert_eq!(message_stops, 1);

        let message_delta = message_deltas[0];
        assert_eq!(
            message_delta
                .pointer("/delta/stop_reason")
                .and_then(|v| v.as_str()),
            Some("tool_use")
        );
        assert_eq!(
            message_delta
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64()),
            Some(13212)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64()),
            Some(79)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/cache_read_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(100)
        );
    }

    #[tokio::test]
    async fn test_usage_chunk_subtracts_cache_read_and_creation_from_input() {
        // prompt_tokens(1000) 含 cache_read(600) 与 cache_creation(300)；转 Anthropic 后
        // input 应为 fresh，守恒：input(100) + cache_read(600) + cache_creation(300) == prompt(1000)。
        let input = concat!(
            "data: {\"id\":\"chatcmpl_cc\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tool-1\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_cc\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":600,\"cache_write_tokens\":300}}}\n\n",
            "data: [DONE]\n\n"
        );

        let events = collect_anthropic_events(input).await;
        let message_delta = events
            .iter()
            .find(|event| event_type(event) == Some("message_delta"))
            .expect("should emit message_delta with usage");

        // fresh input = 1000 - 600 - 300 = 100
        assert_eq!(
            message_delta
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64()),
            Some(100)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/cache_read_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(600)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/cache_creation_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(300)
        );
    }

    #[tokio::test]
    async fn test_usage_chunk_clamps_input_to_zero_when_cache_exceeds_prompt() {
        // prompt(100) < cache_read(80)+cache_creation(50)=130：saturating 钳到 0，防下溢。
        // 钉桩：阻止未来把 saturating_sub 误改成普通减法(debug panic / release wrap)。
        let input = concat!(
            "data: {\"id\":\"chatcmpl_uf\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tool-1\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_uf\",\"model\":\"glm-5.1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":80},\"cache_creation_input_tokens\":50}}\n\n",
            "data: [DONE]\n\n"
        );

        let events = collect_anthropic_events(input).await;
        let message_delta = events
            .iter()
            .find(|event| event_type(event) == Some("message_delta"))
            .expect("should emit message_delta with usage");

        assert_eq!(
            message_delta
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64()),
            Some(0)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/cache_read_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(80)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/cache_creation_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(50)
        );
    }

    #[tokio::test]
    async fn test_message_delta_includes_zero_usage_when_stream_has_no_usage() {
        let input = concat!(
            "data: {\"id\":\"chatcmpl_no_usage\",\"model\":\"gpt-5.5\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl_no_usage\",\"model\":\"gpt-5.5\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );

        let events = collect_anthropic_events(input).await;
        let message_deltas: Vec<&Value> = events
            .iter()
            .filter(|event| event_type(event) == Some("message_delta"))
            .collect();

        assert_eq!(message_deltas.len(), 1);
        let message_delta = message_deltas[0];
        assert_eq!(
            message_delta
                .pointer("/delta/stop_reason")
                .and_then(|v| v.as_str()),
            Some("tool_use")
        );
        assert_eq!(
            message_delta
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64()),
            Some(0)
        );
        assert_eq!(
            message_delta
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64()),
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_streaming_finalizes_after_finish_when_done_is_missing() {
        let input = concat!(
            "data: {\"id\":\"chatcmpl_no_done\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl_no_done\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        );

        let events = collect_anthropic_events(input).await;

        assert!(events.iter().any(|event| {
            event_type(event) == Some("message_delta")
                && event.pointer("/delta/stop_reason").and_then(|v| v.as_str()) == Some("end_turn")
        }));
        assert_eq!(
            events.last().and_then(|event| event_type(event)),
            Some("message_stop")
        );
    }

    #[tokio::test]
    async fn test_stream_end_without_finish_reason_does_not_emit_success_terminal_events() {
        let input = "data: {\"id\":\"chatcmpl_truncated\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";

        let events = collect_anthropic_events(input).await;

        assert!(!events
            .iter()
            .any(|event| event_type(event) == Some("message_delta")));
        assert!(!events
            .iter()
            .any(|event| event_type(event) == Some("message_stop")));
    }

    #[tokio::test]
    async fn test_stream_error_does_not_emit_success_terminal_events() {
        let upstream = stream::iter(vec![Err::<Bytes, _>(std::io::Error::other(
            "upstream disconnected",
        ))]);
        let converted = create_anthropic_sse_stream(upstream);
        let chunks: Vec<_> = converted.collect().await;

        let merged = chunks
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();

        let events: Vec<Value> = merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect();

        assert!(events
            .iter()
            .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("error")));
        assert!(!events
            .iter()
            .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_delta")));
        assert!(!events
            .iter()
            .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_stop")));
    }

    /// 端到端复现用户 bug：上游把 Claude 的 antml 工具调用当普通文本流式返回
    /// （finish_reason=stop），开启兜底后应还原成 tool_use 并把 stop_reason 改写。
    /// 关键点：antml 被切成多个 chunk（含哨兵跨界），模拟真实 SSE 分片。
    fn fallback_schemas() -> antml_fallback::AntmlToolSchemas {
        antml_fallback::extract_tool_schemas(&json!({
            "tools": [{
                "name": "Bash",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    },
                    "required": ["command"]
                }
            }]
        }))
    }

    async fn collect_with_fallback(chunks: Vec<String>) -> Vec<Value> {
        let items: Vec<_> = chunks
            .into_iter()
            .map(|c| Ok::<_, std::io::Error>(Bytes::from(c.into_bytes())))
            .collect();
        let upstream = stream::iter(items);
        let converted =
            create_anthropic_sse_stream_with_options(upstream, true, fallback_schemas());
        let collected: Vec<_> = converted.collect().await;
        let merged = collected
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(chunk.unwrap().as_ref()).to_string())
            .collect::<String>();
        merged
            .split("\n\n")
            .filter_map(|block| {
                let data = block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "data"))?;
                serde_json::from_str::<Value>(data).ok()
            })
            .collect()
    }

    #[tokio::test]
    async fn e2e_leaked_antml_stream_recovered_to_tool_use() {
        // 构造一段泄漏 antml 的 OpenAI SSE：assistant 正文里是 antml 工具调用 XML。
        let o = concat!("<antml", ":function_calls>");
        let inv_open = concat!("<antml", ":invoke");
        let inv_close = concat!("</antml", ":invoke>");
        let p_open = concat!("<antml", ":parameter");
        let p_close = concat!("</antml", ":parameter>");
        let fc_close = concat!("</antml", ":function_calls>");
        let leaked = format!(
            "好的，我来看目录。\n{o}\n{inv_open} name=\"Bash\">\n{p_open} name=\"command\">ls -la{p_close}\n{inv_close}\n{fc_close}"
        );

        // 把泄漏文本按字符切成多段 content delta（模拟流式分片，哨兵会被切断）。
        let chars: Vec<char> = leaked.chars().collect();
        let mut chunks: Vec<String> = Vec::new();
        chunks.push("data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string());
        for piece in chars.chunks(7) {
            let text: String = piece.iter().collect();
            let payload = json!({
                "id": "c1",
                "model": "claude-sonnet",
                "choices": [{"delta": {"content": text}}]
            });
            chunks.push(format!(
                "data: {}\n\n",
                serde_json::to_string(&payload).unwrap()
            ));
        }
        // finish_reason=stop —— 正是用户 bug 的收尾信号。
        chunks.push("data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string());
        chunks.push("data: [DONE]\n\n".to_string());

        let events = collect_with_fallback(chunks).await;

        // 1) 还原出 tool_use block（content_block_start type=tool_use, name=Bash）。
        let tool_start = events.iter().find(|e| {
            e.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                && e.pointer("/content_block/type").and_then(|v| v.as_str()) == Some("tool_use")
        });
        assert!(
            tool_start.is_some(),
            "应还原出 tool_use content_block_start"
        );
        assert_eq!(
            tool_start
                .unwrap()
                .pointer("/content_block/name")
                .and_then(|v| v.as_str()),
            Some("Bash"),
            "工具名应为 Bash"
        );

        // 2) 工具入参通过 input_json_delta 传出，且含 command。
        let input_json: String = events
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("content_block_delta"))
            .filter_map(|e| e.pointer("/delta/partial_json").and_then(|v| v.as_str()))
            .collect();
        assert!(
            input_json.contains("ls -la"),
            "入参应含命令 ls -la，实际: {input_json}"
        );

        // 3) stop_reason 被改写为 tool_use（否则 Claude Code 会停住）。
        let msg_delta = events
            .iter()
            .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_delta"));
        assert_eq!(
            msg_delta
                .and_then(|e| e.pointer("/delta/stop_reason"))
                .and_then(|v| v.as_str()),
            Some("tool_use"),
            "stop_reason 应被改写为 tool_use"
        );

        // 4) 泄漏的 XML 不应作为可见文本 text_delta 出现（antml 标签不能泄漏给用户）。
        let visible_text: String = events
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("content_block_delta"))
            .filter_map(|e| e.pointer("/delta/text").and_then(|v| v.as_str()))
            .collect();
        assert!(
            !visible_text.contains("invoke"),
            "antml 标签不应作为可见文本泄漏，实际可见文本: {visible_text}"
        );
        // 但哨兵之前的正常 prose 应作为文本保留。
        assert!(
            visible_text.contains("我来看目录"),
            "哨兵前的正常正文应保留，实际: {visible_text}"
        );
    }

    /// 用户实际泄漏形式的端到端复现：**裸** <invoke>（无 antml: 前缀、无
    /// function_calls 外壳），流式分片，finish_reason=stop。这是最贴近真实 bug 的用例。
    #[tokio::test]
    async fn e2e_bare_leaked_invoke_stream_recovered_to_tool_use() {
        // 裸形式，正是终端里看到的样子。
        let leaked = "好的，我来看目录。\n<invoke name=\"Bash\">\n<parameter name=\"command\">ls -la</parameter>\n</invoke>".to_string();

        let chars: Vec<char> = leaked.chars().collect();
        let mut chunks: Vec<String> = Vec::new();
        chunks.push("data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string());
        for piece in chars.chunks(5) {
            let text: String = piece.iter().collect();
            let payload =
                json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":text}}]});
            chunks.push(format!(
                "data: {}\n\n",
                serde_json::to_string(&payload).unwrap()
            ));
        }
        chunks.push("data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string());
        chunks.push("data: [DONE]\n\n".to_string());

        let events = collect_with_fallback(chunks).await;

        // 还原出 tool_use
        let tool_start = events.iter().find(|e| {
            e.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
                && e.pointer("/content_block/type").and_then(|v| v.as_str()) == Some("tool_use")
        });
        assert!(tool_start.is_some(), "裸 invoke 应还原出 tool_use");
        assert_eq!(
            tool_start
                .unwrap()
                .pointer("/content_block/name")
                .and_then(|v| v.as_str()),
            Some("Bash")
        );
        // 入参正确
        let input_json: String = events
            .iter()
            .filter_map(|e| e.pointer("/delta/partial_json").and_then(|v| v.as_str()))
            .collect();
        assert!(
            input_json.contains("ls -la"),
            "入参应含 ls -la，实际: {input_json}"
        );
        // stop_reason 改写
        let msg_delta = events
            .iter()
            .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_delta"));
        assert_eq!(
            msg_delta
                .and_then(|e| e.pointer("/delta/stop_reason"))
                .and_then(|v| v.as_str()),
            Some("tool_use"),
            "stop_reason 应改写为 tool_use"
        );
        // 裸标签不作为可见文本泄漏
        let visible_text: String = events
            .iter()
            .filter_map(|e| e.pointer("/delta/text").and_then(|v| v.as_str()))
            .collect();
        assert!(
            !visible_text.contains("invoke"),
            "裸标签不应泄漏为可见文本，实际: {visible_text}"
        );
        assert!(
            visible_text.contains("我来看目录"),
            "标记前正文应保留，实际: {visible_text}"
        );
    }

    #[tokio::test]
    async fn e2e_truncated_xml_is_not_recovered() {
        let truncated = "<invoke name=\"Bash\"><parameter name=\"command\">git status".to_string();
        let chunks = vec![
            format!(
                "data: {}\n\n",
                json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":truncated}}]})
            ),
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ];

        let events = collect_with_fallback(chunks).await;
        assert!(!events.iter().any(|event| {
            event.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
        }));
        assert!(events.iter().any(|event| {
            event.pointer("/delta/stop_reason").and_then(Value::as_str) == Some("end_turn")
        }));
    }

    #[tokio::test]
    async fn e2e_eof_without_finish_reason_never_executes_xml() {
        let leaked = "<invoke name=\"Bash\"><parameter name=\"command\">pwd</parameter></invoke>";
        let chunks = vec![format!(
            "data: {}\n\n",
            json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":leaked}}]})
        )];

        let events = collect_with_fallback(chunks).await;
        assert!(!events.iter().any(|event| {
            event.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
        }));
        assert!(!events
            .iter()
            .any(|event| event_type(event) == Some("message_delta")));
        assert!(!events
            .iter()
            .any(|event| event_type(event) == Some("message_stop")));
    }

    #[tokio::test]
    async fn e2e_fenced_tool_example_remains_text() {
        let example = "```xml\n<invoke name=\"Bash\"><parameter name=\"command\">pwd</parameter></invoke>\n```";
        let chunks = vec![
            format!(
                "data: {}\n\n",
                json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":example}}]})
            ),
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ];

        let events = collect_with_fallback(chunks).await;
        assert!(!events.iter().any(|event| {
            event.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
        }));
        let visible: String = events
            .iter()
            .filter_map(|event| event.pointer("/delta/text").and_then(Value::as_str))
            .collect();
        assert_eq!(visible, example);
    }

    #[tokio::test]
    async fn e2e_native_tool_call_wins_over_leaked_xml() {
        let leaked = "<invoke name=\"Bash\"><parameter name=\"command\">pwd</parameter></invoke>";
        let chunks = vec![
            format!(
                "data: {}\n\n",
                json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":leaked}}]})
            ),
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"native_1\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]}}]}\n\n".to_string(),
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ];

        let events = collect_with_fallback(chunks).await;
        let tool_starts: Vec<&Value> = events
            .iter()
            .filter(|event| {
                event.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
            })
            .collect();
        assert_eq!(tool_starts.len(), 1);
        assert_eq!(
            tool_starts[0]
                .pointer("/content_block/id")
                .and_then(Value::as_str),
            Some("native_1")
        );
    }

    #[tokio::test]
    async fn e2e_partial_marker_is_flushed_before_terminal_events() {
        let chunks = vec![
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{\"content\":\"normal<inv\"}}]}\n\n".to_string(),
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ];

        let events = collect_with_fallback(chunks).await;
        let visible: String = events
            .iter()
            .filter_map(|event| event.pointer("/delta/text").and_then(Value::as_str))
            .collect();
        assert_eq!(visible, "normal<inv");

        let last_text = events
            .iter()
            .rposition(|event| event.pointer("/delta/text").is_some())
            .unwrap();
        let message_delta = events
            .iter()
            .position(|event| event_type(event) == Some("message_delta"))
            .unwrap();
        let message_stop = events
            .iter()
            .position(|event| event_type(event) == Some("message_stop"))
            .unwrap();
        assert!(last_text < message_delta);
        assert!(message_delta < message_stop);
    }

    #[tokio::test]
    async fn e2e_normal_text_stream_unaffected_by_fallback() {
        // 对照组：普通文本回复（无 antml），开启兜底也不应受影响。
        let mut chunks = vec![
            "data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string(),
        ];
        for w in ["你好", "，这是", "一段普通", "回复。"] {
            let payload =
                json!({"id":"c1","model":"claude-sonnet","choices":[{"delta":{"content":w}}]});
            chunks.push(format!(
                "data: {}\n\n",
                serde_json::to_string(&payload).unwrap()
            ));
        }
        chunks.push("data: {\"id\":\"c1\",\"model\":\"claude-sonnet\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string());
        chunks.push("data: [DONE]\n\n".to_string());

        let events = collect_with_fallback(chunks).await;

        // 不应出现任何 tool_use；stop_reason 应保持 end_turn；文本完整。
        assert!(
            !events.iter().any(|e| {
                e.pointer("/content_block/type").and_then(|v| v.as_str()) == Some("tool_use")
            }),
            "普通文本不应产生 tool_use"
        );
        let visible: String = events
            .iter()
            .filter_map(|e| e.pointer("/delta/text").and_then(|v| v.as_str()))
            .collect();
        assert_eq!(visible, "你好，这是一段普通回复。", "普通文本应原样透传");
        let msg_delta = events
            .iter()
            .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("message_delta"));
        assert_eq!(
            msg_delta
                .and_then(|e| e.pointer("/delta/stop_reason"))
                .and_then(|v| v.as_str()),
            Some("end_turn"),
            "普通文本 stop_reason 应为 end_turn"
        );
    }
}

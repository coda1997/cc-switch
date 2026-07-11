//! antml 工具调用兜底解析（GitHub Copilot 专用）
//!
//! 背景：GitHub Copilot 把 Claude 模型包在 OpenAI `/chat/completions` 接口后面。
//! Claude 在 token 层用 antml 命名空间的 XML 表达工具调用（function_calls 包裹
//! 若干 invoke，每个 invoke 含若干 parameter）。正常情况下 Copilot 上游会把这段
//! 结构解析成 OpenAI `tool_calls` 再返回；但偶发（流式边界、上游适配器 bug）解析
//! 失败时，Claude 生成的 antml 原文会被塞进 `content` 当普通文本返回。
//!
//! 结果：Claude Code 收到一段含 XML 的 text block + `stop_reason=end_turn`，
//! 渲染完就结束这一轮 —— 即用户看到的「agent 输出一半吐 XML 然后停住」。
//!
//! 本模块把这段泄漏的 antml 文本反解析回结构化 `tool_use`，让降级成文本的工具
//! 调用被救回。**仅对 GitHub Copilot 供应商启用**（调用点门控），且受
//! `CopilotOptimizerConfig.enabled` 与 `antml_fallback` 双重开关控制。
//!
//! ## 严格模式（Strict）
//! 只有完整闭合的调用位于文本末尾、处于 Markdown fenced code 之外，且工具名、参数名、
//! required 字段与参数类型都通过当前请求的工具 schema 校验时才触发。裸标签与
//! `antml:` 命名空间形式都支持；任何截断或歧义均失败关闭并保留为普通文本。
//!
//! ## 源码里的标签常量为何用 `concat!` 拆写
//! 这些 antml 标签字面量若原样连续出现在源码中，会与本项目的工具链标签解析冲突，
//! 因此统一在 `<antml` / `</antml` 与其余部分之间用 `concat!` 拆开书写；编译期会
//! 拼回完整字符串，运行期零开销。请勿把它们合并回连续字面量。

use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

const MAX_RECOVERY_BYTES: usize = 256 * 1024;
const MAX_RECOVERED_CALLS: usize = 16;

/// `function_calls` 起始包裹标签（命名空间形式）。
pub(crate) const OPEN_FUNCTION_CALLS: &str = concat!("<antml", ":function_calls>");
/// `function_calls` 结束包裹标签。
pub(crate) const CLOSE_FUNCTION_CALLS: &str = concat!("</antml", ":function_calls>");
/// `invoke` 起始标签前缀（后面跟 ` name="..."` 与 `>`）。
pub(crate) const OPEN_INVOKE_PREFIX: &str = concat!("<antml", ":invoke");
/// `invoke` 结束标签。
pub(crate) const CLOSE_INVOKE: &str = concat!("</antml", ":invoke>");
/// `parameter` 起始标签前缀（后面跟 ` name="..."` 与 `>`）。
pub(crate) const OPEN_PARAM_PREFIX: &str = concat!("<antml", ":parameter");
/// `parameter` 结束标签。
pub(crate) const CLOSE_PARAM: &str = concat!("</antml", ":parameter>");
const BARE_CLOSE_FUNCTION_CALLS: &str = "</function_calls>";
const BARE_CLOSE_INVOKE: &str = "</invoke>";
const BARE_CLOSE_PARAM: &str = "</parameter>";
const FUNCTION_CALLS_CLOSE: &[&str] = &[CLOSE_FUNCTION_CALLS, BARE_CLOSE_FUNCTION_CALLS];
const INVOKE_CLOSE: &[&str] = &[CLOSE_INVOKE, BARE_CLOSE_INVOKE];
const PARAM_CLOSE: &[&str] = &[CLOSE_PARAM, BARE_CLOSE_PARAM];

// ---------------------------------------------------------------------------
// 双形式标记集：真实泄漏既可能是命名空间形式（<invoke>），也可能是裸形式
// （<invoke>），且外层 <function_calls> 包裹标签可有可无。以下集合同时覆盖两种前缀，
// 解析与流式哨兵都基于它们，不再假设任何单一格式。
// ---------------------------------------------------------------------------

/// 工具调用区域的触发标记（区域起点）：包裹标签或 invoke 起始标签，两种前缀形式。
/// 裸 `<invoke` 即可触发 —— 不要求 `<function_calls` 外壳。
pub(crate) const TRIGGER_MARKERS: &[&str] = &[
    concat!("<antml", ":function_calls"),
    "<function_calls",
    concat!("<antml", ":invoke"),
    "<invoke",
];

/// `invoke` 起始标记（两形式）。
const INVOKE_OPEN: &[&str] = &[OPEN_INVOKE_PREFIX, "<invoke"];
/// `parameter` 起始标记（两形式）。
const PARAM_OPEN: &[&str] = &[OPEN_PARAM_PREFIX, "<parameter"];

#[derive(Debug, Clone)]
struct AntmlToolSchema {
    properties: Map<String, Value>,
    required: HashSet<String>,
}

/// 当前请求中允许调用的工具及其输入 schema。
#[derive(Debug, Clone, Default)]
pub struct AntmlToolSchemas {
    tools: HashMap<String, AntmlToolSchema>,
}

impl AntmlToolSchemas {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// 从原始 Anthropic Messages 请求中提取工具 schema。
pub fn extract_tool_schemas(body: &Value) -> AntmlToolSchemas {
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let name = tool.get("name").and_then(Value::as_str)?;
            let input_schema = tool.get("input_schema").and_then(Value::as_object)?;
            let properties = input_schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required = input_schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect();
            Some((
                name.to_string(),
                AntmlToolSchema {
                    properties,
                    required,
                },
            ))
        })
        .collect();
    AntmlToolSchemas { tools }
}

/// 在 `haystack` 中查找 `needles` 里任意一个的最早出现，返回 `(起始字节偏移, 命中长度)`。
///
/// 命名空间形式与裸形式互不为子串（`<invoke` 不是 `<invoke` 的子串，反之亦然），
/// 故两者同时作为 needle 不会互相误配，取最早命中即可。
fn find_any(haystack: &str, needles: &[&str]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for n in needles {
        if let Some(i) = haystack.find(n) {
            match best {
                Some((bi, _)) if bi <= i => {}
                _ => best = Some((i, n.len())),
            }
        }
    }
    best
}

/// 单个被救回的工具调用。
#[derive(Debug, Clone, PartialEq)]
pub struct AntmlToolCall {
    pub name: String,
    pub input: Value,
}

/// 解析结果：工具调用区域之前的正文 + 若干工具调用。
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAntml {
    /// 工具调用区域（触发标记）之前的普通正文（已去除首尾空白）。
    pub prose: String,
    /// 解析出的工具调用（至少一个，否则 `parse_function_calls` 返回 None）。
    pub calls: Vec<AntmlToolCall>,
}

/// 流式增量文本的哨兵扫描结果。
#[derive(Debug, Clone, PartialEq)]
pub enum SentinelScan {
    /// 在 `pending` 的字节偏移 `idx` 处找到完整哨兵。
    Found { idx: usize },
    /// 未找到完整哨兵；前 `flush_len` 字节可安全作为文本发出，
    /// 其余是哨兵的可能前缀，需继续持有等待后续 chunk。
    Partial { flush_len: usize },
}

/// 严格解析泄漏的工具调用文本。
///
/// 仅接受完整闭合、位于正文末尾、未处于 Markdown fenced code 中、且工具名与参数
/// 都能通过当前请求 schema 校验的调用。任何截断或歧义都返回 `None`，避免把普通文本
/// 或半截工具调用升级成可执行的 `tool_use`。
pub fn parse_function_calls(text: &str, schemas: &AntmlToolSchemas) -> Option<ParsedAntml> {
    if text.len() > MAX_RECOVERY_BYTES || schemas.is_empty() {
        return None;
    }

    let start = find_trigger_outside_fence(text)?;
    let prose = text[..start].trim().to_string();
    let region = &text[start..];

    let (body, trailing) =
        if marker_at_start(region, &[OPEN_FUNCTION_CALLS, "<function_calls"]).is_some() {
            let open_end = region.find('>')? + 1;
            let (close_rel, close_len) = find_any(&region[open_end..], FUNCTION_CALLS_CLOSE)?;
            let close_at = open_end + close_rel;
            (&region[open_end..close_at], &region[close_at + close_len..])
        } else {
            (region, "")
        };

    if !trailing.trim().is_empty() {
        return None;
    }

    let (calls, consumed) = parse_invokes(body, schemas)?;
    if calls.is_empty() || !body[consumed..].trim().is_empty() {
        return None;
    }
    Some(ParsedAntml { prose, calls })
}

fn find_trigger_outside_fence(text: &str) -> Option<usize> {
    let mut cursor = 0usize;
    while cursor < text.len() {
        let (rel, marker_len) = find_any(&text[cursor..], TRIGGER_MARKERS)?;
        let start = cursor + rel;
        if has_tag_boundary(text, start + marker_len) && !is_inside_fenced_code(text, start) {
            return Some(start);
        }
        cursor = start + marker_len;
    }
    None
}

fn has_tag_boundary(text: &str, end: usize) -> bool {
    text[end..]
        .chars()
        .next()
        .is_some_and(|ch| ch == '>' || ch.is_ascii_whitespace())
}

fn is_inside_fenced_code(text: &str, index: usize) -> bool {
    text[..index].match_indices("```").count() % 2 == 1
}

fn marker_at_start<'a>(text: &str, markers: &'a [&'a str]) -> Option<&'a str> {
    markers.iter().copied().find(|marker| {
        text.starts_with(marker) && (marker.ends_with('>') || has_tag_boundary(text, marker.len()))
    })
}

fn skip_whitespace(text: &str, cursor: &mut usize) {
    while let Some(ch) = text[*cursor..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        *cursor += ch.len_utf8();
    }
}

fn parse_invokes(body: &str, schemas: &AntmlToolSchemas) -> Option<(Vec<AntmlToolCall>, usize)> {
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    loop {
        skip_whitespace(body, &mut cursor);
        if cursor == body.len() {
            break;
        }

        marker_at_start(&body[cursor..], INVOKE_OPEN)?;
        let inv_start = cursor;
        let tag_region = &body[inv_start..];
        let gt_rel = tag_region.find('>')?;
        let open_tag = &tag_region[..gt_rel];
        let inner_start = inv_start + gt_rel + 1;
        let (close_rel, close_len) = find_any(&body[inner_start..], INVOKE_CLOSE)?;
        let inner_end = inner_start + close_rel;
        let name = extract_attr(open_tag, "name")?;
        let schema = schemas.tools.get(&name)?;
        let input = parse_params(&body[inner_start..inner_end], schema)?;
        calls.push(AntmlToolCall { name, input });
        if calls.len() > MAX_RECOVERED_CALLS {
            return None;
        }
        cursor = inner_end + close_len;
    }

    Some((calls, cursor))
}

fn parse_params(inner: &str, schema: &AntmlToolSchema) -> Option<Value> {
    let mut map = Map::new();
    let mut cursor = 0usize;

    loop {
        skip_whitespace(inner, &mut cursor);
        if cursor == inner.len() {
            break;
        }

        marker_at_start(&inner[cursor..], PARAM_OPEN)?;
        let p_start = cursor;
        let tag_region = &inner[p_start..];
        let gt_rel = tag_region.find('>')?;
        let open_tag = &tag_region[..gt_rel];
        let value_start = p_start + gt_rel + 1;
        let (close_rel, close_len) = find_any(&inner[value_start..], PARAM_CLOSE)?;
        let value_end = value_start + close_rel;
        let name = extract_attr(open_tag, "name")?;
        if map.contains_key(&name) {
            return None;
        }
        let property_schema = schema.properties.get(&name)?;
        let value = coerce_value(&inner[value_start..value_end], property_schema)?;
        map.insert(name, value);
        cursor = value_end + close_len;
    }

    if schema.required.iter().any(|name| !map.contains_key(name)) {
        return None;
    }

    Some(Value::Object(map))
}

fn coerce_value(raw: &str, schema: &Value) -> Option<Value> {
    for alternatives_key in ["anyOf", "oneOf"] {
        if let Some(alternatives) = schema.get(alternatives_key).and_then(Value::as_array) {
            return alternatives
                .iter()
                .find_map(|candidate| coerce_value(raw, candidate));
        }
    }

    let trimmed = raw.trim();
    let mut types: Vec<&str> = schema
        .get("type")
        .and_then(|value| match value {
            Value::String(kind) => Some(vec![kind.as_str()]),
            Value::Array(kinds) => Some(kinds.iter().filter_map(Value::as_str).collect()),
            _ => None,
        })
        .unwrap_or_default();
    if types.is_empty() {
        if schema.get("properties").is_some() {
            types.push("object");
        } else if schema.get("items").is_some() {
            types.push("array");
        } else {
            types.push("string");
        }
    }

    types.into_iter().find_map(|kind| {
        let value = match kind {
            // 字符串参数可能是 Edit.new_string 或多行 shell；不能 trim 掉有语义的缩进。
            "string" => Some(Value::String(raw.to_string())),
            "integer" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(|value| value.as_i64().is_some() || value.as_u64().is_some()),
            "number" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_number),
            "boolean" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_boolean),
            "null" => (trimmed == "null").then_some(Value::Null),
            "object" => parse_json_container(trimmed, Value::is_object),
            "array" => parse_json_container(trimmed, Value::is_array),
            _ => None,
        }?;
        schema_value_allowed(schema, &value).then_some(value)
    })
}

fn parse_json_container(trimmed: &str, predicate: fn(&Value) -> bool) -> Option<Value> {
    let candidate = strip_markdown_json_wrapper(trimmed);
    serde_json::from_str::<Value>(candidate)
        .ok()
        .filter(predicate)
}

fn strip_markdown_json_wrapper(value: &str) -> &str {
    if let Some(inner) = value
        .strip_prefix("```")
        .and_then(|rest| rest.strip_suffix("```"))
    {
        let inner = inner.trim();
        return inner
            .strip_prefix("json")
            .map(str::trim_start)
            .unwrap_or(inner);
    }
    value
        .strip_prefix('`')
        .and_then(|rest| rest.strip_suffix('`'))
        .map(str::trim)
        .unwrap_or(value)
}

fn schema_value_allowed(schema: &Value, value: &Value) -> bool {
    for alternatives_key in ["anyOf", "oneOf"] {
        if let Some(alternatives) = schema.get(alternatives_key).and_then(Value::as_array) {
            return alternatives
                .iter()
                .any(|candidate| schema_value_allowed(candidate, value));
        }
    }

    if let Some(expected) = schema.get("const") {
        if expected != value {
            return false;
        }
    }
    if !schema
        .get("enum")
        .and_then(Value::as_array)
        .is_none_or(|allowed| allowed.contains(value))
    {
        return false;
    }

    if let Some(expected_types) = schema.get("type") {
        let matches_type = match expected_types {
            Value::String(kind) => value_matches_type(value, kind),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|kind| value_matches_type(value, kind)),
            _ => false,
        };
        if !matches_type {
            return false;
        }
    }

    match value {
        Value::Object(object) => {
            let required_ok = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .all(|name| object.contains_key(name));
            if !required_ok {
                return false;
            }

            let properties = schema.get("properties").and_then(Value::as_object);
            object.iter().all(|(name, child)| {
                if let Some(child_schema) = properties.and_then(|props| props.get(name)) {
                    return schema_value_allowed(child_schema, child);
                }
                match schema.get("additionalProperties") {
                    Some(Value::Bool(false)) => false,
                    Some(child_schema) if child_schema.is_object() => {
                        schema_value_allowed(child_schema, child)
                    }
                    _ => true,
                }
            })
        }
        Value::Array(items) => schema.get("items").is_none_or(|item_schema| {
            items
                .iter()
                .all(|item| schema_value_allowed(item_schema, item))
        }),
        _ => true,
    }
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        _ => false,
    }
}

fn extract_attr(open_tag: &str, attr: &str) -> Option<String> {
    let mut cursor = 0usize;
    while let Some(rel) = open_tag[cursor..].find(attr) {
        let start = cursor + rel;
        let before_ok = start == 0
            || open_tag[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-');
        let after = start + attr.len();
        let after_ok = open_tag[after..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-');
        if before_ok && after_ok {
            let rest = open_tag[after..].trim_start();
            let rest = rest.strip_prefix('=')?.trim_start();
            let quote = rest.chars().next()?;
            if quote != '"' && quote != '\'' {
                return None;
            }
            let value = &rest[quote.len_utf8()..];
            let end = value.find(quote)?;
            return (!value[..end].is_empty()).then(|| value[..end].to_string());
        }
        cursor = after;
    }
    None
}

/// 流式持有扫描：在增量文本 `pending` 中查找 `sentinel`。
///
/// - 完整命中 → `Found{idx}`；
/// - 否则计算 `pending` 末尾与 `sentinel` 前缀的最长重叠，作为需持有的可能前缀，
///   其余字节可安全 flush（`Partial{flush_len}`）。
///
/// `sentinel` 为 ASCII，故所有切分点都是合法的 UTF-8 边界。
#[cfg(test)]
pub fn scan_for_sentinel(pending: &str, sentinel: &str) -> SentinelScan {
    if let Some(idx) = pending.find(sentinel) {
        return SentinelScan::Found { idx };
    }
    let max_overlap = pending.len().min(sentinel.len().saturating_sub(1));
    let mut hold = 0usize;
    for k in (1..=max_overlap).rev() {
        if pending.is_char_boundary(pending.len() - k) && pending.ends_with(&sentinel[..k]) {
            hold = k;
            break;
        }
    }
    SentinelScan::Partial {
        flush_len: pending.len() - hold,
    }
}

/// 多哨兵版本：在 `pending` 中查找 `sentinels` 里任意一个。
///
/// - 任一完整命中 → `Found{idx}`（取最早命中）；
/// - 否则在所有哨兵中取「与 `pending` 末尾重叠的最长前缀」作为需持有的量，
///   其余字节 flush。持有量取各哨兵的最大值，确保任何一个被 chunk 边界切断的
///   哨兵前缀都不会被漏掉。
///
/// 所有哨兵均为 ASCII，切分点是合法 UTF-8 边界。
fn scan_for_any_sentinel(pending: &str, sentinels: &[&str]) -> SentinelScan {
    let mut earliest: Option<usize> = None;
    for s in sentinels {
        if let Some(idx) = pending.find(s) {
            earliest = Some(earliest.map_or(idx, |e| e.min(idx)));
        }
    }
    if let Some(idx) = earliest {
        return SentinelScan::Found { idx };
    }
    // 未完整命中：取所有哨兵中最长的「末尾重叠前缀」作为持有量。
    let mut hold = 0usize;
    for s in sentinels {
        let max_overlap = pending.len().min(s.len().saturating_sub(1));
        for k in (1..=max_overlap).rev() {
            if k > hold && pending.is_char_boundary(pending.len() - k) && pending.ends_with(&s[..k])
            {
                hold = k;
                break;
            }
        }
    }
    SentinelScan::Partial {
        flush_len: pending.len() - hold,
    }
}

/// 为救回的工具调用生成会话历史内唯一的 id。
pub fn synthetic_tool_id() -> String {
    format!("toolu_antml_{}", uuid::Uuid::new_v4().simple())
}

/// 非流式路径兜底：若 `msg` 是一条 Anthropic 消息，其 content 主体是一个泄漏了
/// antml 的 text block，则原地改写为 `text?(prose) + tool_use...` 并把 `stop_reason`
/// 置为 `tool_use`。发生改写返回 `true`，否则不动并返回 `false`。
///
/// 调用方需自行完成 Copilot 门控与开关判断。
pub fn rewrite_anthropic_message(msg: &mut Value, schemas: &AntmlToolSchemas) -> bool {
    // 仅在「正常文本收尾」时兜底：原生工具调用已产出 tool_use 时不介入。
    let stop_reason = msg.get("stop_reason").and_then(|s| s.as_str());
    if stop_reason != Some("end_turn") {
        return false;
    }

    let Some(content) = msg.get("content").and_then(|c| c.as_array()) else {
        return false;
    };
    // 已有 tool_use block 说明工具调用正常，不介入。
    if content
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
    {
        return false;
    }

    let Some((text_index, parsed)) = content.iter().enumerate().find_map(|(index, block)| {
        let text = block
            .get("text")
            .and_then(Value::as_str)
            .filter(|_| block.get("type").and_then(Value::as_str) == Some("text"))?;
        parse_function_calls(text, schemas).map(|parsed| (index, parsed))
    }) else {
        return false;
    };

    let mut replacement = Vec::with_capacity(parsed.calls.len() + 1);
    if !parsed.prose.is_empty() {
        replacement.push(serde_json::json!({"type": "text", "text": parsed.prose}));
    }
    for call in &parsed.calls {
        replacement.push(serde_json::json!({
            "type": "tool_use",
            "id": synthetic_tool_id(),
            "name": call.name,
            "input": call.input,
        }));
    }

    let mut new_content = content.clone();
    new_content.splice(text_index..=text_index, replacement);
    msg["content"] = Value::Array(new_content);
    msg["stop_reason"] = Value::String("tool_use".to_string());
    true
}

/// 流式 antml 兜底状态机。
///
/// 逐段喂入上游文本增量：正常 prose 原样放行（仅在文本末尾可能是触发标记前缀时短暂
/// 持有一小段，避免标记被 chunk 边界切断而漏检）；一旦发现任一 [`TRIGGER_MARKERS`]
/// （裸或命名空间的 `<invoke` / `<function_calls`）即「上膛」（armed），此后所有文本
/// 转入内部缓冲，不再作为文本发出，留待收尾时反解析成 tool_use。
#[derive(Debug)]
pub struct AntmlStreamGuard {
    enabled: bool,
    armed: bool,
    hold: String,
}

impl AntmlStreamGuard {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            armed: false,
            hold: String::new(),
        }
    }

    /// 是否已发现哨兵、进入缓冲状态。
    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// 喂入一段文本增量，返回「现在应作为文本发出的内容」（可能为空）。
    ///
    /// - 未启用：原样返回；
    /// - 已上膛：全部转入缓冲，返回空串；
    /// - 未上膛：找到哨兵则返回哨兵之前的 prose 并上膛；否则放行安全前缀、
    ///   持有可能是哨兵前缀的末尾一小段。
    pub fn feed_text(&mut self, content: &str) -> String {
        if !self.enabled {
            return content.to_string();
        }
        if self.armed {
            self.hold.push_str(content);
            if self.hold.len() > MAX_RECOVERY_BYTES {
                self.armed = false;
                self.enabled = false;
                return std::mem::take(&mut self.hold);
            }
            return String::new();
        }
        self.hold.push_str(content);
        match scan_for_any_sentinel(&self.hold, TRIGGER_MARKERS) {
            SentinelScan::Found { idx } => {
                if self.hold.len() - idx > MAX_RECOVERY_BYTES {
                    self.enabled = false;
                    return std::mem::take(&mut self.hold);
                }
                let prose = self.hold[..idx].to_string();
                let rest = self.hold[idx..].to_string();
                self.hold = rest;
                self.armed = true;
                prose
            }
            SentinelScan::Partial { flush_len } => {
                let out = self.hold[..flush_len].to_string();
                self.hold.drain(..flush_len);
                out
            }
        }
    }

    /// 取走已上膛的 antml 缓冲（供解析）。取走后缓冲清空，重复调用返回空串，
    /// 因此在 finish 与 stream-end 两处调用是幂等的。
    pub fn take_buffer(&mut self) -> String {
        std::mem::take(&mut self.hold)
    }

    /// 未上膛时收尾：取走仍被持有、未能构成哨兵的残留文本（应作为文本补发）。
    pub fn take_unflushed_text(&mut self) -> String {
        if self.armed {
            String::new()
        } else {
            std::mem::take(&mut self.hold)
        }
    }
}

fn sse(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
}

/// 生成一组 tool_use 的 Anthropic 流式 SSE 事件文本（每个调用三段：
/// content_block_start / input_json_delta / content_block_stop）。
/// 返回 (事件列表, 下一个可用的 content index)。
pub fn tool_use_sse_events(calls: &[AntmlToolCall], start_index: u32) -> (Vec<String>, u32) {
    let mut events = Vec::with_capacity(calls.len() * 3);
    let mut index = start_index;
    for call in calls {
        let start = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": synthetic_tool_id(),
                "name": call.name,
                "input": {},
            }
        });
        events.push(sse("content_block_start", &start));

        let partial = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".to_string());
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial}
        });
        events.push(sse("content_block_delta", &delta));

        let stop = serde_json::json!({"type": "content_block_stop", "index": index});
        events.push(sse("content_block_stop", &stop));
        index += 1;
    }
    (events, index)
}

/// 生成一个独立 text block 的 SSE 事件（start + delta + stop）。
/// 返回 (事件列表, 下一个 content index)。用于把无法解析的缓冲原样补发为文本。
pub fn standalone_text_sse_events(text: &str, start_index: u32) -> (Vec<String>, u32) {
    let start = serde_json::json!({
        "type": "content_block_start",
        "index": start_index,
        "content_block": {"type": "text", "text": ""}
    });
    let delta = serde_json::json!({
        "type": "content_block_delta",
        "index": start_index,
        "delta": {"type": "text_delta", "text": text}
    });
    let stop = serde_json::json!({"type": "content_block_stop", "index": start_index});
    (
        vec![
            sse("content_block_start", &start),
            sse("content_block_delta", &delta),
            sse("content_block_stop", &stop),
        ],
        start_index + 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 用标签常量拼出一段泄漏的 antml 文本（测试里也不能出现连续字面量）。
    fn param(name: &str, value: &str) -> String {
        format!("{OPEN_PARAM_PREFIX} name=\"{name}\">{value}{CLOSE_PARAM}")
    }
    fn invoke(name: &str, params: &[(&str, &str)]) -> String {
        let body: String = params.iter().map(|(n, v)| param(n, v)).collect();
        format!("{OPEN_INVOKE_PREFIX} name=\"{name}\">{body}{CLOSE_INVOKE}")
    }
    fn wrap(prose: &str, invokes: &[String]) -> String {
        format!(
            "{prose}{OPEN_FUNCTION_CALLS}{}{CLOSE_FUNCTION_CALLS}",
            invokes.concat()
        )
    }

    // --- 裸形式（无 antml: 前缀、无 function_calls 外壳）—— 正是用户实际看到的泄漏 ---
    fn bare_param(name: &str, value: &str) -> String {
        format!("<parameter name=\"{name}\">{value}</parameter>")
    }
    fn bare_invoke(name: &str, params: &[(&str, &str)]) -> String {
        let body: String = params.iter().map(|(n, v)| bare_param(n, v)).collect();
        format!("<invoke name=\"{name}\">{body}</invoke>")
    }

    fn schemas() -> AntmlToolSchemas {
        extract_tool_schemas(&json!({
            "tools": [
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "description": {"type": "string"},
                            "timeout": {"type": "integer"},
                            "run_in_background": {"type": "boolean"}
                        },
                        "required": ["command"]
                    }
                },
                {
                    "name": "Read",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "offset": {"type": "integer"}
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "Tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "object"}
                        },
                        "required": ["payload"]
                    }
                }
            ]
        }))
    }

    fn parse(text: &str) -> Option<ParsedAntml> {
        parse_function_calls(text, &schemas())
    }

    #[test]
    fn parses_bare_invoke_without_prefix_or_wrapper() {
        // 用户实际泄漏形式：裸 <invoke>，无 <function_calls> 外壳，无 antml: 前缀。
        let text = bare_invoke("Bash", &[("command", "ls -la"), ("description", "list")]);
        let parsed = parse(&text).expect("裸 invoke 应能解析");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "Bash");
        assert_eq!(parsed.calls[0].input["command"], json!("ls -la"));
        assert_eq!(parsed.calls[0].input["description"], json!("list"));
    }

    #[test]
    fn parses_bare_invoke_with_prose() {
        let text = format!(
            "我来看目录。\n{}",
            bare_invoke("Bash", &[("command", "ls")])
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.prose, "我来看目录。");
        assert_eq!(parsed.calls[0].input["command"], json!("ls"));
    }

    #[test]
    fn parses_multiple_bare_invokes() {
        let text = format!(
            "{}{}",
            bare_invoke("Read", &[("file_path", "/tmp/a")]),
            bare_invoke("Bash", &[("command", "echo hi")])
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].name, "Read");
        assert_eq!(parsed.calls[1].name, "Bash");
    }

    #[test]
    fn bare_multiline_command_preserved() {
        let cmd = "cd /tmp\nsed -n '1,5p' file\nhead -3 other";
        let text = bare_invoke("Bash", &[("command", cmd)]);
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!(cmd));
    }

    #[test]
    fn bare_invoke_word_without_tag_does_not_trigger() {
        // 仅提到 "invoke" 一词、无真正 <invoke 标签 → 不误判。
        let text = "你可以用 invoke 来调用 Bash 工具运行 ls。";
        assert!(parse(text).is_none());
    }

    #[test]
    fn bare_rewrite_message_replaces_leaked_text() {
        // 非流式路径也覆盖裸形式。
        let text = format!("先看目录\n{}", bare_invoke("Bash", &[("command", "ls")]));
        let mut msg = json!({
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        });
        assert!(rewrite_anthropic_message(&mut msg, &schemas()));
        assert_eq!(msg["stop_reason"], json!("tool_use"));
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content[0], json!({"type": "text", "text": "先看目录"}));
        assert_eq!(content[1]["type"], json!("tool_use"));
        assert_eq!(content[1]["name"], json!("Bash"));
        assert_eq!(content[1]["input"]["command"], json!("ls"));
    }

    #[test]
    fn guard_arms_on_bare_invoke() {
        let mut g = AntmlStreamGuard::new(true);
        let text = format!("看目录\n{}", bare_invoke("Bash", &[("command", "ls")]));
        let out = g.feed_text(&text);
        assert_eq!(out, "看目录\n");
        assert!(g.is_armed());
        let parsed = parse(&g.take_buffer()).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!("ls"));
    }

    #[test]
    fn guard_handles_bare_invoke_split_across_chunks() {
        // 裸 <invoke 被 chunk 边界切断也要能上膛。
        let mut g = AntmlStreamGuard::new(true);
        let out1 = g.feed_text("prose<inv");
        assert_eq!(out1, "prose"); // "<inv" 作为可能前缀被持有
        assert!(!g.is_armed());
        let out2 = g.feed_text(&format!(
            "oke name=\"Bash\">{}</invoke>",
            bare_param("command", "ls")
        ));
        assert_eq!(out2, "");
        assert!(g.is_armed());
        let parsed = parse(&g.take_buffer()).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!("ls"));
    }

    #[test]
    fn parses_single_bash_invoke() {
        let text = wrap(
            "",
            &[invoke(
                "Bash",
                &[("command", "ls -la"), ("description", "list")],
            )],
        );
        let parsed = parse(&text).expect("should parse");
        assert_eq!(parsed.prose, "");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "Bash");
        assert_eq!(parsed.calls[0].input["command"], json!("ls -la"));
        assert_eq!(parsed.calls[0].input["description"], json!("list"));
    }

    #[test]
    fn parses_mixed_namespace_tags_when_fully_closed() {
        let text = format!(
            "{OPEN_FUNCTION_CALLS}<invoke name=\"Bash\">{OPEN_PARAM_PREFIX} name=\"command\">pwd</parameter></invoke>{CLOSE_FUNCTION_CALLS}"
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!("pwd"));
    }

    #[test]
    fn keeps_prose_before_wrapper() {
        let text = wrap(
            "我来看一下目录。\n",
            &[invoke("Bash", &[("command", "ls")])],
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.prose, "我来看一下目录。");
        assert_eq!(parsed.calls[0].input["command"], json!("ls"));
    }

    #[test]
    fn parses_multiple_invokes() {
        let text = wrap(
            "",
            &[
                invoke("Read", &[("file_path", "/tmp/a")]),
                invoke("Bash", &[("command", "echo hi")]),
            ],
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].name, "Read");
        assert_eq!(parsed.calls[1].name, "Bash");
    }

    #[test]
    fn multiline_command_preserved() {
        let cmd = "  cd /tmp\nsed -n '1,5p' file\nhead -3 other\n";
        let text = wrap("", &[invoke("Bash", &[("command", cmd)])]);
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!(cmd));
    }

    #[test]
    fn coerces_json_object_param() {
        let text = wrap(
            "",
            &[invoke("Tool", &[("payload", "{\"a\": 1, \"b\": [2,3]}")])],
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(
            parsed.calls[0].input["payload"],
            json!({"a": 1, "b": [2, 3]})
        );
    }

    #[test]
    fn coerces_backtick_wrapped_json_object_param() {
        let text = wrap("", &[invoke("Tool", &[("payload", "`{\"a\": 1}`")])]);
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["payload"], json!({"a": 1}));
    }

    #[test]
    fn scalar_number_like_stays_string() {
        let text = wrap("", &[invoke("Bash", &[("command", "123")])]);
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!("123"));
    }

    #[test]
    fn scalar_params_use_tool_schema_types() {
        let text = wrap(
            "",
            &[invoke(
                "Bash",
                &[
                    ("command", "sleep 1"),
                    ("timeout", "120000"),
                    ("run_in_background", "true"),
                ],
            )],
        );
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.calls[0].input["timeout"], json!(120000));
        assert_eq!(parsed.calls[0].input["run_in_background"], json!(true));
    }

    #[test]
    fn rejects_unknown_tool_and_missing_required_param() {
        let unknown = bare_invoke("DeleteEverything", &[("command", "rm -rf /")]);
        assert!(parse(&unknown).is_none());

        let missing = bare_invoke("Bash", &[("description", "missing command")]);
        assert!(parse(&missing).is_none());
    }

    #[test]
    fn rejects_tool_markup_inside_fenced_code_or_with_trailing_prose() {
        let fenced = format!(
            "示例：\n```xml\n{}\n```",
            bare_invoke("Bash", &[("command", "ls")])
        );
        assert!(parse(&fenced).is_none());

        let trailing = format!(
            "{}\n这只是一个示例。",
            bare_invoke("Bash", &[("command", "ls")])
        );
        assert!(parse(&trailing).is_none());
    }

    #[test]
    fn no_wrapper_returns_none() {
        // 合法地讨论工具，但没有命名空间包裹标签 → 不误判。
        let text = "你可以用 invoke 调用 Bash 工具来运行 ls。";
        assert!(parse(text).is_none());
    }

    #[test]
    fn wrapper_without_valid_invoke_returns_none() {
        let text = format!("{OPEN_FUNCTION_CALLS}{CLOSE_FUNCTION_CALLS}");
        assert!(parse(&text).is_none());
    }

    #[test]
    fn rejects_truncated_close_tags() {
        let text = format!(
            "{OPEN_FUNCTION_CALLS}{OPEN_INVOKE_PREFIX} name=\"Bash\">{OPEN_PARAM_PREFIX} name=\"command\">ls -la"
        );
        assert!(parse(&text).is_none());
    }

    #[test]
    fn scan_finds_full_sentinel() {
        let pending = format!("prefix{OPEN_FUNCTION_CALLS}rest");
        match scan_for_sentinel(&pending, OPEN_FUNCTION_CALLS) {
            SentinelScan::Found { idx } => assert_eq!(
                &pending[idx..idx + OPEN_FUNCTION_CALLS.len()],
                OPEN_FUNCTION_CALLS
            ),
            other => panic!("expected Found, got {other:?}"),
        }
    }

    #[test]
    fn scan_holds_back_partial_prefix() {
        // 文本以哨兵的前缀结尾 → 必须持有该前缀。
        let sentinel = OPEN_FUNCTION_CALLS; // "<function_calls>"
        let head = "hello ";
        let partial = &sentinel[..5]; // "<antm"
        let pending = format!("{head}{partial}");
        match scan_for_sentinel(&pending, sentinel) {
            SentinelScan::Partial { flush_len } => {
                assert_eq!(flush_len, head.len());
                assert_eq!(&pending[flush_len..], partial);
            }
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[test]
    fn scan_no_overlap_flushes_all() {
        let pending = "just some normal text";
        match scan_for_sentinel(pending, OPEN_FUNCTION_CALLS) {
            SentinelScan::Partial { flush_len } => assert_eq!(flush_len, pending.len()),
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_message_replaces_leaked_text() {
        let text = wrap("先看目录\n", &[invoke("Bash", &[("command", "ls")])]);
        let mut msg = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        });
        assert!(rewrite_anthropic_message(&mut msg, &schemas()));
        assert_eq!(msg["stop_reason"], json!("tool_use"));
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content[0], json!({"type": "text", "text": "先看目录"}));
        assert_eq!(content[1]["type"], json!("tool_use"));
        assert_eq!(content[1]["name"], json!("Bash"));
        assert_eq!(content[1]["input"]["command"], json!("ls"));
    }

    #[test]
    fn rewrite_uses_unique_ids_across_messages() {
        let text = bare_invoke("Bash", &[("command", "ls")]);
        let mut first = json!({
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        });
        let mut second = first.clone();
        let schemas = schemas();

        assert!(rewrite_anthropic_message(&mut first, &schemas));
        assert!(rewrite_anthropic_message(&mut second, &schemas));
        assert_ne!(first["content"][0]["id"], second["content"][0]["id"]);
    }

    #[test]
    fn rewrite_noop_without_wrapper() {
        let mut msg = json!({
            "content": [{"type": "text", "text": "普通回答，没有工具调用"}],
            "stop_reason": "end_turn",
        });
        assert!(!rewrite_anthropic_message(&mut msg, &schemas()));
        assert_eq!(msg["content"][0]["text"], json!("普通回答，没有工具调用"));
    }

    #[test]
    fn rewrite_noop_when_native_tool_use_present() {
        let mut msg = json!({
            "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}],
            "stop_reason": "tool_use",
        });
        assert!(!rewrite_anthropic_message(&mut msg, &schemas()));
    }

    #[test]
    fn synthetic_ids_are_unique_across_responses() {
        assert_ne!(synthetic_tool_id(), synthetic_tool_id());
    }

    #[test]
    fn guard_passthrough_when_disabled() {
        let mut g = AntmlStreamGuard::new(false);
        assert_eq!(g.feed_text("anything <at all"), "anything <at all");
        assert!(!g.is_armed());
    }

    #[test]
    fn guard_streams_plain_prose() {
        let mut g = AntmlStreamGuard::new(true);
        // 普通文本不含 '<' → 全部放行，无持有。
        assert_eq!(g.feed_text("hello world"), "hello world");
        assert!(!g.is_armed());
        assert_eq!(g.take_unflushed_text(), "");
    }

    #[test]
    fn guard_arms_on_wrapper_and_buffers_rest() {
        let mut g = AntmlStreamGuard::new(true);
        let text = wrap("看目录\n", &[invoke("Bash", &[("command", "ls")])]);
        // 一次性喂入完整泄漏文本。
        let out = g.feed_text(&text);
        assert_eq!(out, "看目录\n"); // 哨兵之前的 prose 原样放行（流式不 trim，wrap 的 prose 以 \n 结尾）
        assert!(g.is_armed());
        // 工具调用后的格式空白继续缓冲，但不影响严格解析。
        assert_eq!(g.feed_text(" \n"), "");
        let buf = g.take_buffer();
        let parsed = parse(&buf).unwrap();
        assert_eq!(parsed.calls[0].name, "Bash");
    }

    #[test]
    fn guard_handles_sentinel_split_across_chunks() {
        let mut g = AntmlStreamGuard::new(true);
        let sentinel = OPEN_FUNCTION_CALLS;
        let mid = 6;
        // 把哨兵切成两段跨 chunk 送入。
        let first = &sentinel[..mid];
        let second = &sentinel[mid..];
        let out1 = g.feed_text(&format!("prose{first}"));
        assert_eq!(out1, "prose"); // 哨兵前缀被持有
        assert!(!g.is_armed());
        let out2 = g.feed_text(&format!(
            "{second}{}{CLOSE_FUNCTION_CALLS}",
            invoke("Bash", &[("command", "ls")])
        ));
        assert_eq!(out2, ""); // 现在完整哨兵出现，prose 之前已发完
        assert!(g.is_armed());
        let parsed = parse(&g.take_buffer()).unwrap();
        assert_eq!(parsed.calls[0].input["command"], json!("ls"));
    }

    #[test]
    fn guard_take_buffer_idempotent() {
        let mut g = AntmlStreamGuard::new(true);
        g.feed_text(&wrap("", &[invoke("Bash", &[("command", "ls")])]));
        assert!(!g.take_buffer().is_empty());
        assert_eq!(g.take_buffer(), ""); // 第二次为空 → finish 与 stream-end 双调用幂等
    }

    #[test]
    fn guard_unflushed_text_recovered_when_not_armed() {
        let mut g = AntmlStreamGuard::new(true);
        // 以哨兵前缀结尾且流结束 → 残留应作为文本补发，不能吞掉。
        let partial = &OPEN_FUNCTION_CALLS[..5];
        let out = g.feed_text(&format!("tail{partial}"));
        assert_eq!(out, "tail");
        assert_eq!(g.take_unflushed_text(), partial);
    }

    #[test]
    fn guard_disables_recovery_when_candidate_exceeds_limit() {
        let mut guard = AntmlStreamGuard::new(true);
        let oversized = format!(
            "{}{}",
            bare_invoke("Bash", &[("command", "ls")]),
            "x".repeat(MAX_RECOVERY_BYTES)
        );
        assert_eq!(guard.feed_text(&oversized), oversized);
        assert!(!guard.is_armed());
        assert_eq!(guard.feed_text("tail"), "tail");
    }

    #[test]
    fn tool_use_events_shape_and_index() {
        let calls = vec![
            AntmlToolCall {
                name: "Bash".into(),
                input: json!({"command": "ls"}),
            },
            AntmlToolCall {
                name: "Read".into(),
                input: json!({"file_path": "/x"}),
            },
        ];
        let (events, next) = tool_use_sse_events(&calls, 2);
        assert_eq!(events.len(), 6); // 2 calls × 3 events
        assert_eq!(next, 4);
        assert!(events[0].contains("content_block_start"));
        assert!(events[0].contains("\"index\":2"));
        assert!(events[1].contains("input_json_delta"));
        assert!(events[3].contains("\"index\":3"));
        let first: Value = serde_json::from_str(
            events[0]
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap(),
        )
        .unwrap();
        let second: Value = serde_json::from_str(
            events[3]
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(first["content_block"]["input"], json!({}));
        assert_ne!(first["content_block"]["id"], second["content_block"]["id"]);
    }
}

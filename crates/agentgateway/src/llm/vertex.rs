use agent_core::strng;
use agent_core::strng::Strng;
use serde_json::{Map, Value};

use crate::llm::{AIError, RouteType};
use crate::*;

use super::policy::PromptCachingConfig;

const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
fn make_cache_control() -> Value {
	serde_json::json!({"type": "ephemeral"})
}

/// Estimate token count from text using a rough heuristic (~1.3 tokens per word).
fn estimate_tokens(text: &str) -> usize {
	(text.split_whitespace().count() * 13) / 10
}

/// Apply prompt caching markers (`cache_control`) to an Anthropic Messages API body.
///
/// Inserts `{"type": "ephemeral"}` cache_control on:
/// - The last system content block (if `cache_system` is true and meets `min_tokens`)
/// - The second-to-last message (if `cache_messages` is true)
/// - The last tool definition (if `cache_tools` is true)
///
/// Anthropic allows at most 4 explicit cache breakpoints per request.
fn apply_prompt_caching(body: &mut Map<String, Value>, config: &PromptCachingConfig) {
	let cc = make_cache_control();

	// Cache system prompt: add cache_control to the last system content block
	if config.cache_system {
		if let Some(system) = body.get_mut("system") {
			match system {
				// String system prompt — convert to block format to attach cache_control
				Value::String(text) => {
					let meets_min = config
						.min_tokens
						.map_or(true, |min| estimate_tokens(text) >= min);
					if meets_min {
						*system = serde_json::json!([{
							"type": "text",
							"text": text.clone(),
							"cache_control": cc,
						}]);
					}
				},
				// Array of system blocks — add cache_control to the last one
				Value::Array(blocks) => {
					let total_tokens: usize = blocks
						.iter()
						.filter_map(|b| b.get("text").and_then(|t| t.as_str()))
						.map(estimate_tokens)
						.sum();
					let meets_min = config.min_tokens.map_or(true, |min| total_tokens >= min);
					if meets_min {
						if let Some(last) = blocks.last_mut() {
							if let Value::Object(obj) = last {
								obj.insert("cache_control".to_string(), cc.clone());
							}
						}
					}
				},
				_ => {},
			}
		}
	}

	// Cache messages: add cache_control to the last content block of the
	// second-to-last message (caches conversation history before current turn)
	if config.cache_messages {
		if let Some(Value::Array(messages)) = body.get_mut("messages") {
			let len = messages.len();
			if len >= 2 {
				if let Some(msg) = messages.get_mut(len - 2) {
					insert_cache_control_on_last_content(msg, &cc);
				}
			}
		}
	}

	// Cache tools: add cache_control to the last tool definition
	if config.cache_tools {
		if let Some(Value::Array(tools)) = body.get_mut("tools") {
			if let Some(Value::Object(last_tool)) = tools.last_mut() {
				last_tool.insert("cache_control".to_string(), cc.clone());
			}
		}
	}
}

/// Insert cache_control on the last content block of a message.
fn insert_cache_control_on_last_content(message: &mut Value, cc: &Value) {
	if let Some(content) = message.get_mut("content") {
		match content {
			// String content — convert to block format
			Value::String(text) => {
				*content = serde_json::json!([{
					"type": "text",
					"text": text.clone(),
					"cache_control": cc,
				}]);
			},
			// Array of content blocks — add to last block
			Value::Array(blocks) => {
				if let Some(Value::Object(last)) = blocks.last_mut() {
					last.insert("cache_control".to_string(), cc.clone());
				}
			},
			_ => {},
		}
	}
}

#[apply(schema!)]
pub struct Provider {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub model: Option<Strng>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub region: Option<Strng>,
	pub project_id: Strng,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("gcp.vertex_ai");
}

impl Provider {
	fn configured_model<'a>(&'a self, request_model: Option<&'a str>) -> Option<&'a str> {
		self.model.as_deref().or(request_model)
	}

	pub fn is_anthropic_model(&self, request_model: Option<&str>) -> bool {
		self.anthropic_model(request_model).is_some()
	}

	pub fn prepare_anthropic_message_body(
		&self,
		body: Vec<u8>,
		prompt_caching: Option<&super::policy::PromptCachingConfig>,
	) -> Result<Vec<u8>, AIError> {
		let mut body: Map<String, Value> =
			serde_json::from_slice(&body).map_err(AIError::RequestMarshal)?;

		body.insert(
			"anthropic_version".to_string(),
			Value::String(ANTHROPIC_VERSION.to_string()),
		);
		body.remove("model");

		if let Some(caching) = prompt_caching {
			apply_prompt_caching(&mut body, caching);
		}

		serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
	}

	pub fn prepare_anthropic_count_tokens_body(&self, body: Vec<u8>) -> Result<Vec<u8>, AIError> {
		let mut body: Map<String, Value> =
			serde_json::from_slice(&body).map_err(AIError::RequestMarshal)?;

		body.insert(
			"anthropic_version".to_string(),
			Value::String(ANTHROPIC_VERSION.to_string()),
		);

		if let Some(Value::String(model)) = body.get("model") {
			let normalized = self
				.configured_model(Some(model))
				.map(|s| s.to_string())
				.unwrap_or_else(|| model.clone());
			body.insert("model".to_string(), Value::String(normalized));
		}
		serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
	}

	pub fn get_path_for_model(
		&self,
		route: RouteType,
		request_model: Option<&str>,
		streaming: bool,
	) -> Strng {
		let location = self
			.region
			.clone()
			.unwrap_or_else(|| strng::literal!("global"));

		match (route, self.anthropic_model(request_model)) {
			(RouteType::AnthropicTokenCount, _) => {
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/anthropic/models/count-tokens:rawPredict",
					self.project_id,
					location
				)
			},
			(RouteType::Embeddings, _) => {
				let model = self.configured_model(request_model).unwrap_or_default();
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
					self.project_id,
					location,
					model
				)
			},
			(_, Some(model)) => {
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:{}",
					self.project_id,
					location,
					model,
					if streaming {
						"streamRawPredict"
					} else {
						"rawPredict"
					}
				)
			},
			_ => {
				strng::format!(
					"/v1/projects/{}/locations/{}/endpoints/openapi/chat/completions",
					self.project_id,
					location
				)
			},
		}
	}

	pub fn get_host(&self, _request_model: Option<&str>) -> Strng {
		match &self.region {
			None => {
				strng::literal!("aiplatform.googleapis.com")
			},
			Some(region) => {
				strng::format!("{region}-aiplatform.googleapis.com")
			},
		}
	}

	fn anthropic_model<'a>(&'a self, request_model: Option<&'a str>) -> Option<Strng> {
		let model = self.configured_model(request_model)?;

		// Strip known prefixes
		let model: &str = model
			.split_once("publishers/anthropic/models/")
			.map(|(_, m)| m)
			.or_else(|| model.strip_prefix("anthropic/"))
			.or_else(|| {
				if model.starts_with("claude-") {
					Some(model)
				} else {
					None
				}
			})?;

		// Replace -YYYYMMDD with @YYYYMMDD
		if model.len() > 8 && model.as_bytes()[model.len() - 9] == b'-' {
			let (base, date) = model.split_at(model.len() - 8);
			if date.chars().all(|c| c.is_ascii_digit()) {
				Some(strng::new(format!("{}@{}", &base[..base.len() - 1], date)))
			} else {
				Some(strng::new(model))
			}
		} else {
			Some(strng::new(model))
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[rstest::rstest]
	#[case::strip_publishers_prefix(
		Some("publishers/anthropic/models/claude-sonnet-4-5-20251001"),
		None,
		Some("claude-sonnet-4-5@20251001")
	)]
	#[case::strip_anthropic_prefix(
		Some("anthropic/claude-haiku-4-5-20251001"),
		None,
		Some("claude-haiku-4-5@20251001")
	)]
	#[case::raw_claude_prefix(None, Some("claude-opus-3-20240229"), Some("claude-opus-3@20240229"))]
	#[case::no_date_suffix(None, Some("claude-opus-4-6"), Some("claude-opus-4-6"))]
	#[case::legacy_model(
		None,
		Some("claude-3-5-sonnet-20241022"),
		Some("claude-3-5-sonnet@20241022")
	)]
	#[case::non_digit_date_suffix(
		None,
		Some("claude-haiku-4-5-2025abcd"),
		Some("claude-haiku-4-5-2025abcd")
	)]
	#[case::non_anthropic_model(None, Some("text-embedding-004"), None)]
	#[case::provider_model_precedence(
		Some("anthropic/claude-haiku-4-5-20251001"),
		Some("anthropic/claude-sonnet-4-5-20251001"),
		Some("claude-haiku-4-5@20251001")
	)]
	fn test_anthropic_model_normalization(
		#[case] provider: Option<&str>,
		#[case] req: Option<&str>,
		#[case] expected: Option<&str>,
	) {
		let p = Provider {
			project_id: strng::new("test-project"),
			model: provider.map(strng::new),
			region: None,
		};
		let actual = p.anthropic_model(req).map(|m| m.to_string());
		assert_eq!(actual.as_deref(), expected);
	}

	#[test]
	fn test_prompt_caching_system_string() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"system": "You are a helpful assistant with a very long system prompt that should be cached.",
			"messages": [
				{"role": "user", "content": "Hello"},
				{"role": "assistant", "content": "Hi there"},
				{"role": "user", "content": "How are you?"}
			]
		}))
		.unwrap();
		let config = PromptCachingConfig {
			cache_system: true,
			cache_messages: true,
			cache_tools: false,
			min_tokens: None,
		};
		apply_prompt_caching(&mut body, &config);

		// System should be converted to blocks with cache_control
		let system = body.get("system").unwrap();
		assert!(system.is_array());
		let blocks = system.as_array().unwrap();
		assert_eq!(blocks.len(), 1);
		assert!(blocks[0].get("cache_control").is_some());

		// Second-to-last message should have cache_control
		let messages = body.get("messages").unwrap().as_array().unwrap();
		let second_to_last = &messages[1];
		let content = second_to_last.get("content").unwrap().as_array().unwrap();
		assert!(content[0].get("cache_control").is_some());
	}

	#[test]
	fn test_prompt_caching_system_blocks() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"system": [
				{"type": "text", "text": "First block"},
				{"type": "text", "text": "Second block"}
			],
			"messages": [
				{"role": "user", "content": "Hello"}
			]
		}))
		.unwrap();
		let config = PromptCachingConfig {
			cache_system: true,
			cache_messages: false,
			cache_tools: false,
			min_tokens: None,
		};
		apply_prompt_caching(&mut body, &config);

		let blocks = body.get("system").unwrap().as_array().unwrap();
		assert!(blocks[0].get("cache_control").is_none());
		assert!(blocks[1].get("cache_control").is_some());
	}

	#[test]
	fn test_prompt_caching_tools() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"messages": [{"role": "user", "content": "Hello"}],
			"tools": [
				{"name": "tool1", "description": "First tool", "input_schema": {}},
				{"name": "tool2", "description": "Second tool", "input_schema": {}}
			]
		}))
		.unwrap();
		let config = PromptCachingConfig {
			cache_system: false,
			cache_messages: false,
			cache_tools: true,
			min_tokens: None,
		};
		apply_prompt_caching(&mut body, &config);

		let tools = body.get("tools").unwrap().as_array().unwrap();
		assert!(tools[0].get("cache_control").is_none());
		assert!(tools[1].get("cache_control").is_some());
	}

	#[test]
	fn test_prompt_caching_min_tokens_not_met() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"system": "Short prompt",
			"messages": [{"role": "user", "content": "Hello"}]
		}))
		.unwrap();
		let config = PromptCachingConfig {
			cache_system: true,
			cache_messages: false,
			cache_tools: false,
			min_tokens: Some(1024),
		};
		apply_prompt_caching(&mut body, &config);

		// System should remain a string (min_tokens not met)
		assert!(body.get("system").unwrap().is_string());
	}

	#[test]
	fn test_prompt_caching_too_few_messages() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"messages": [{"role": "user", "content": "Hello"}]
		}))
		.unwrap();
		let config = PromptCachingConfig {
			cache_system: false,
			cache_messages: true,
			cache_tools: false,
			min_tokens: None,
		};
		apply_prompt_caching(&mut body, &config);

		// With only 1 message, no cache_control should be added
		let messages = body.get("messages").unwrap().as_array().unwrap();
		assert!(messages[0].get("content").unwrap().is_string());
	}
}

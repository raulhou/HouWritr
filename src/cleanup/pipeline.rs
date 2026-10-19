//! The typed cleanup pipeline: the network-free core that ties the provider,
//! catalog, prompt, adapter, normalization, integrity, and outcome layers
//! together.
//!
//! Everything here is pure and deterministic. The actual HTTP call and
//! credential lookups live in the `enhance` facade; this module builds the
//! request body, decides the endpoint URL, and — given a raw response body —
//! decides the typed [`EnhanceOutcome`]. That split keeps all decision logic
//! unit-testable without a network.

use serde_json::Value;

use super::adapter::{self, ChatRequestOptions};
use super::catalog;
use super::integrity::{self, ValidatorError, ViolationKind};
use super::normalize;
use super::outcome::{EnhanceOutcome, FallbackReason};
use super::prompt::{PromptTarget, Transcript};
use super::provider::{EnhanceProvider, COPILOT_CHAT_COMPLETIONS_URL};

/// Default model used when settings leave the model ID blank. Matches the
/// historical facade behavior.
pub const DEFAULT_MODEL: &str = "claude-haiku-4.5";

/// Default sampling temperature for the cleanup pass.
pub const DEFAULT_TEMPERATURE: f32 = 0.1;

/// Resolve the effective model ID: the trimmed configured value, or
/// [`DEFAULT_MODEL`] when it is blank.
pub fn effective_model_id(configured: &str) -> String {
    let trimmed = configured.trim();
    if trimmed.is_empty() {
        DEFAULT_MODEL.to_string()
    } else {
        trimmed.to_string()
    }
}

/// The chat-completions URL for a target.
pub fn chat_completions_url(target: &PromptTarget) -> String {
    match target {
        PromptTarget::GithubCopilot { .. } => COPILOT_CHAT_COMPLETIONS_URL.to_string(),
        PromptTarget::OpenAiCompatible { endpoint, .. } => endpoint.join("/chat/completions"),
    }
}

/// Build the chat-completions request body for a target, using the catalog's
/// capability hints for the target's model.
pub fn build_request(target: &PromptTarget, system: &str, transcript: &Transcript) -> Value {
    let capabilities = catalog::capabilities_for(target.model_id());
    let options = ChatRequestOptions::for_capabilities(capabilities, DEFAULT_TEMPERATURE);
    adapter::build_chat_request(target.model_id(), system, transcript, &options)
}

/// Build a request for a non-cleanup task (e.g. meeting notes) that still
/// honours the target model's capabilities. When the model rejects system
/// messages the instructions are folded into the user message so they are
/// never silently dropped.
pub fn build_task_request(target: &PromptTarget, instructions: &str, input: &str) -> Value {
    let capabilities = catalog::capabilities_for(target.model_id());
    let options = ChatRequestOptions::for_capabilities(capabilities, DEFAULT_TEMPERATURE);
    if options.include_system_message {
        adapter::build_chat_request(
            target.model_id(),
            instructions,
            &Transcript::new(input),
            &options,
        )
    } else {
        let combined =
            format!("{instructions}\n\n--- BEGIN TRANSCRIPT ---\n{input}\n--- END TRANSCRIPT ---");
        adapter::build_chat_request(target.model_id(), "", &Transcript::new(&combined), &options)
    }
}

/// Decide the typed outcome from a raw model response string.
///
/// `source` is the original transcript; `raw_content` is the assistant text
/// already extracted from the response (see [`adapter::parse_chat_response`]).
pub fn finalize(source: &str, raw_content: &str) -> EnhanceOutcome {
    // The explicit `[[EMPTY]]` sentinel is a distinct, deliberate signal from
    // the model ("there was nothing meaningful to clean up") and is handled
    // separately from an arbitrary empty/blank candidate: it is only
    // accepted when the source itself is conservatively established to have
    // no meaningful content (including filler-only speech, in either
    // English or German). For a meaningful source, a sentinel — whether a
    // genuine model decision or a malicious/accidental one — is rejected in
    // favor of the raw transcript, so real speech is never silently erased.
    if normalize::is_empty_sentinel(raw_content) {
        return if integrity::is_meaningful(source) {
            EnhanceOutcome::raw_fallback(source, FallbackReason::EmptyCandidate)
        } else {
            // Nothing meaningful in, nothing out — an accepted empty result.
            EnhanceOutcome::enhanced(String::new(), Vec::new())
        };
    }

    let normalized = normalize::normalize_output(raw_content);

    if normalized.is_empty() {
        // An arbitrary empty/blank candidate is *not* the explicit sentinel,
        // so it is never treated as a deliberate "nothing to clean up"
        // signal — always fall back to the raw transcript rather than
        // guessing at intent, regardless of whether the source itself is
        // meaningful.
        return EnhanceOutcome::raw_fallback(source, FallbackReason::EmptyCandidate);
    }

    match integrity::validate(source, &normalized) {
        Err(ValidatorError::InputTooLarge) => {
            EnhanceOutcome::raw_fallback(source, FallbackReason::ValidatorError)
        }
        Ok(report) if report.is_rejected() => {
            let kinds = report.error_kinds();
            if kinds.iter().any(|kind| {
                matches!(
                    kind,
                    ViolationKind::FactualDigitChanged | ViolationKind::SpeakerLabelsChanged
                )
            }) {
                // Why numbers were judged changed: counts, and the first few
                // digit numbers (never any words).
                let changes = integrity::number_changes(source, &normalized);
                tracing::info!(
                    ?kinds,
                    lost = changes.lost,
                    gained = changes.gained,
                    lost_numbers = ?changes.lost_numbers,
                    gained_numbers = ?changes.gained_numbers,
                    labels_before = changes.speaker_labels_source,
                    labels_after = changes.speaker_labels_candidate,
                    chinese_numerals_before = changes.chinese_numerals_source,
                    chinese_numerals_after = changes.chinese_numerals_candidate,
                    "cleaned text changed numbers or speaker labels"
                );
            }
            EnhanceOutcome::raw_fallback(source, FallbackReason::IntegrityRejected(kinds))
        }
        Ok(report) => EnhanceOutcome::enhanced(normalized, report.warning_kinds()),
    }
}

/// Map a provider to the header pairs a request must carry.
pub fn request_headers(provider: EnhanceProvider) -> &'static [(&'static str, &'static str)] {
    provider.request_headers()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cleanup::endpoint::EndpointScope;
    use crate::cleanup::integrity::ViolationKind;
    use crate::cleanup::outcome::EnhanceOutcome;

    fn openai_target(model: &str) -> PromptTarget {
        let scope = EndpointScope::parse("https://api.openai.com/v1/").unwrap();
        PromptTarget::openai_compatible(scope, model).unwrap()
    }

    #[test]
    fn effective_model_id_falls_back_when_blank() {
        assert_eq!(effective_model_id("   "), DEFAULT_MODEL);
        assert_eq!(effective_model_id("  gpt-5-mini "), "gpt-5-mini");
    }

    #[test]
    fn copilot_url_is_fixed() {
        let target = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        assert_eq!(chat_completions_url(&target), COPILOT_CHAT_COMPLETIONS_URL);
    }

    #[test]
    fn openai_url_is_derived_from_the_scope() {
        let target = openai_target("gpt-5-mini");
        assert_eq!(
            chat_completions_url(&target),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn build_request_respects_catalog_capabilities() {
        let transcript = Transcript::new("hello");

        // gpt-5-mini does not accept a custom temperature.
        let body = build_request(&openai_target("gpt-5-mini"), "sys", &transcript);
        assert!(body.get("temperature").is_none());

        // arbitrary models get the default capabilities (temperature sent).
        let body = build_request(&openai_target("some-model"), "sys", &transcript);
        assert_eq!(body["temperature"], serde_json::json!(DEFAULT_TEMPERATURE));
    }

    #[test]
    fn finalize_accepts_a_clean_candidate() {
        let outcome = finalize("um hello world", "Hello world");
        match outcome {
            EnhanceOutcome::Enhanced { text, .. } => assert_eq!(text, "Hello world"),
            other => panic!("expected Enhanced, got {other:?}"),
        }
    }

    #[test]
    fn finalize_falls_back_when_model_answers_a_dictated_request() {
        let source = "please do a comparison for me";
        let response = "I'd be happy to help with a comparison, but I need more information. Could you please provide: 1. What two (or more) things you'd like me to compare 2. What aspects or criteria you'd like me to focus on Once you give me those details, I can create a thorough comparison for you.";
        let outcome = finalize(source, response);
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback(
                source,
                FallbackReason::IntegrityRejected(vec![ViolationKind::TranscriptExpanded])
            )
        );
    }

    #[test]
    fn finalize_recognizes_the_empty_sentinel_for_empty_source() {
        let outcome = finalize("   ", "[[EMPTY]]");
        assert_eq!(outcome, EnhanceOutcome::enhanced(String::new(), Vec::new()));
    }

    #[test]
    fn finalize_falls_back_when_candidate_is_empty_for_meaningful_source() {
        let outcome = finalize("this matters", "[[EMPTY]]");
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback("this matters", FallbackReason::EmptyCandidate)
        );
    }

    #[test]
    fn finalize_accepts_the_empty_sentinel_for_english_filler_only_source() {
        let outcome = finalize("um uh erm", "[[EMPTY]]");
        assert_eq!(outcome, EnhanceOutcome::enhanced(String::new(), Vec::new()));
    }

    #[test]
    fn finalize_accepts_the_empty_sentinel_for_german_filler_only_source() {
        let outcome = finalize("ähm äh hm", "[[EMPTY]]");
        assert_eq!(outcome, EnhanceOutcome::enhanced(String::new(), Vec::new()));
    }

    #[test]
    fn finalize_accepts_the_empty_sentinel_for_capitalized_german_filler_only_source() {
        let outcome = finalize("Ähm Äh Öhm", "[[EMPTY]]");
        assert_eq!(outcome, EnhanceOutcome::enhanced(String::new(), Vec::new()));
    }

    #[test]
    fn finalize_falls_back_for_meaningful_english_source_with_empty_sentinel() {
        let outcome = finalize("please call me back", "[[EMPTY]]");
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback("please call me back", FallbackReason::EmptyCandidate)
        );
    }

    #[test]
    fn finalize_falls_back_for_meaningful_german_source_with_empty_sentinel() {
        let outcome = finalize("bitte ruf mich zurück", "[[EMPTY]]");
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback("bitte ruf mich zurück", FallbackReason::EmptyCandidate)
        );
    }

    #[test]
    fn finalize_treats_a_malicious_or_accidental_sentinel_as_a_safe_fallback() {
        // Even though the model returned the exact sentinel, meaningful
        // speech must never be silently erased.
        let outcome = finalize(
            "the wire transfer is for 42 dollars to account 1234",
            "[[EMPTY]]",
        );
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback(
                "the wire transfer is for 42 dollars to account 1234",
                FallbackReason::EmptyCandidate
            )
        );
    }

    #[test]
    fn finalize_falls_back_on_arbitrary_empty_output_even_for_filler_only_source() {
        // An arbitrary blank candidate is not the explicit `[[EMPTY]]`
        // sentinel, so it is never treated as a deliberate empty signal —
        // even when the source itself has no meaningful content.
        let outcome = finalize("um uh", "   ");
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback("um uh", FallbackReason::EmptyCandidate)
        );
    }

    #[test]
    fn finalize_falls_back_on_arbitrary_empty_output_for_meaningful_source() {
        let outcome = finalize("this matters", "");
        assert_eq!(
            outcome,
            EnhanceOutcome::raw_fallback("this matters", FallbackReason::EmptyCandidate)
        );
    }

    #[test]
    fn finalize_falls_back_on_integrity_rejection() {
        let outcome = finalize("transfer 42 dollars", "Transfer 24 dollars.");
        match outcome {
            EnhanceOutcome::RawFallback {
                text,
                reason: FallbackReason::IntegrityRejected(kinds),
            } => {
                assert_eq!(text, "transfer 42 dollars");
                assert!(kinds.contains(&ViolationKind::FactualDigitChanged));
            }
            other => panic!("expected integrity fallback, got {other:?}"),
        }
    }

    #[test]
    fn finalize_falls_back_on_validator_error() {
        let big = "a".repeat(integrity::MAX_INPUT_CHARS + 1);
        let outcome = finalize(&big, "short candidate");
        assert!(matches!(
            outcome,
            EnhanceOutcome::RawFallback {
                reason: FallbackReason::ValidatorError,
                ..
            }
        ));
    }

    #[test]
    fn finalize_keeps_candidate_but_reports_warnings() {
        let outcome = finalize("give me five apples", "Give me 5 apples.");
        match outcome {
            EnhanceOutcome::Enhanced { text, warnings } => {
                assert_eq!(text, "Give me 5 apples.");
                assert!(warnings.contains(&ViolationKind::IntroducedDigit));
            }
            other => panic!("expected Enhanced with warnings, got {other:?}"),
        }
    }

    #[test]
    fn finalize_keeps_capitalized_german_filler_but_reports_warnings() {
        let outcome = finalize("please proceed.", "Ähm please proceed.");
        match outcome {
            EnhanceOutcome::Enhanced { text, warnings } => {
                assert_eq!(text, "Ähm please proceed.");
                assert!(warnings.contains(&ViolationKind::RetainedFiller));
            }
            other => panic!("expected Enhanced with warnings, got {other:?}"),
        }
    }
}

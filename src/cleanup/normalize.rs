//! Output normalization for enhancement responses.
//!
//! Normalization trims surrounding whitespace and recognizes the `[[EMPTY]]`
//! sentinel — the value a model is asked to return when the transcript has no
//! meaningful content — collapsing it to an empty string. Reasoning models
//! served locally (Qwen3 and friends) may prefix the answer with a
//! `<think>…</think>` block; that is removed first.

/// Sentinel a model returns to signal "there was nothing to clean up".
pub const EMPTY_SENTINEL: &str = "[[EMPTY]]";

/// Normalize a raw model response into candidate output text.
///
/// * Surrounding whitespace is trimmed.
/// * The `[[EMPTY]]` sentinel (after trimming) becomes an empty string.
pub fn normalize_output(raw: &str) -> String {
    let stripped = strip_reasoning(raw);
    let trimmed = stripped.trim();
    if trimmed == EMPTY_SENTINEL {
        return String::new();
    }
    trimmed.to_string()
}

/// Whether the raw model response is (after trimming) the empty sentinel.
pub fn is_empty_sentinel(raw: &str) -> bool {
    strip_reasoning(raw).trim() == EMPTY_SENTINEL
}

/// Remove `<think>…</think>` reasoning blocks. A reasoning block that never
/// closes (the model ran out of tokens while thinking) leaves no answer, so
/// everything from the opening tag on is dropped. A stray closing tag with no
/// opening tag (some templates pre-fill `<think>`) drops everything before it.
fn strip_reasoning(raw: &str) -> std::borrow::Cow<'_, str> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if !raw.contains(OPEN) && !raw.contains(CLOSE) {
        return std::borrow::Cow::Borrowed(raw);
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    // A closing tag before any opening tag means the opening was pre-filled.
    if let Some(close) = rest.find(CLOSE) {
        if rest.find(OPEN).map_or(true, |open| close < open) {
            rest = &rest[close + CLOSE.len()..];
        }
    }
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        match rest[open..].find(CLOSE) {
            Some(close) => rest = &rest[open + close + CLOSE.len()..],
            None => return std::borrow::Cow::Owned(out),
        }
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(normalize_output("  hello world  \n"), "hello world");
    }

    #[test]
    fn empty_sentinel_becomes_empty_string() {
        assert_eq!(normalize_output("[[EMPTY]]"), "");
        assert_eq!(normalize_output("  [[EMPTY]]\n"), "");
        assert!(is_empty_sentinel("  [[EMPTY]]  "));
    }

    #[test]
    fn sentinel_must_be_exact() {
        // Extra content around the sentinel means it is not a sentinel.
        assert_eq!(normalize_output("[[EMPTY]] and more"), "[[EMPTY]] and more");
        assert!(!is_empty_sentinel("[[EMPTY]] and more"));
        assert_eq!(normalize_output("[[empty]]"), "[[empty]]");
    }

    #[test]
    fn reasoning_blocks_are_removed() {
        assert_eq!(
            normalize_output("<think>\nplan the edit\n</think>\n\nHello there."),
            "Hello there."
        );
        assert_eq!(
            normalize_output("A <think>x</think>B<think>y</think> C"),
            "A B C"
        );
        assert_eq!(normalize_output("plan</think>\nAnswer"), "Answer");
        assert_eq!(normalize_output("Answer<think>never closed"), "Answer");
        assert_eq!(normalize_output("<think>only thinking"), "");
        assert!(is_empty_sentinel("<think>nothing to do</think>[[EMPTY]]"));
    }

    #[test]
    fn whitespace_only_stays_empty() {
        assert_eq!(normalize_output("   \n\t "), "");
    }
}

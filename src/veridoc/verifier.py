"""Citation-grounded claim decomposition, hybrid BM25 + TF-IDF retrieval, and entailment auditing."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from veridoc.models import (
    ClaimAudit,
    ClaimVerdict,
    CorpusPassage,
    EvidenceMatch,
    VerificationSummary,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "during", "across", "under", "within", "all",
}

NEGATION_WORDS = {
    "not", "never", "neither", "nor", "without", "cannot", "cant", "wont", "didnt",
    "doesnt", "isnt", "arent", "wasnt", "werent", "unable", "lack", "lacks", "lacking",
}

ZERO_COUNT_WORDS = {"zero", "none", "no"}

ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    (
        {"passed", "pass", "completed", "complete", "approved", "approve", "verified", "compliant", "succeeded"},
        {"failed", "fail", "rejected", "reject", "denied", "deny", "violated", "noncompliant", "incomplete", "canceled", "cancelled"},
    ),
    (
        {"exceeded", "exceeding", "exceed", "above", "rose", "increased", "increase", "grew", "surpassed", "reached"},
        {"missed", "missing", "miss", "below", "dropped", "drop", "fell", "decreased", "decrease", "declined", "under"},
    ),
    (
        {"encrypted", "encrypt", "secured", "protected"},
        {"unencrypted", "plaintext", "exposed", "insecure"},
    ),
    (
        {"required", "require", "mandatory", "must"},
        {"optional", "waived", "exempt", "unnecessary"},
    ),
    (
        {"retained", "retain", "preserved", "stored"},
        {"deleted", "delete", "purged", "discarded", "destroyed"},
    ),
]

ABBREVIATIONS = {"inc", "corp", "ltd", "dr", "mr", "ms", "mrs", "no", "vs", "u.s", "e.g", "i.e", "st"}

HEADER_PREFIXES = (
    "document id",
    "document 1d",
    "memo id",
    "reference id",
    "date:",
    "to:",
    "from:",
    "subject:",
    "confidential",
    "ponfidential",
    "page ",
)

MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

NUMBER_PATTERN = re.compile(
    r"(?P<currency>\$)?\b(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?P<suffix>[kmbKMB])?(?P<pct>%)?"
)


def parse_corpus_jsonl(
    corpus_input: Union[str, Path, Sequence[dict[str, str] | CorpusPassage]],
) -> list[CorpusPassage]:
    """Parse trusted reference knowledge-base JSONL into validated ``CorpusPassage`` records."""
    if isinstance(corpus_input, Path):
        raw_text = corpus_input.read_text(encoding="utf-8")
        return _parse_jsonl_lines(raw_text)

    if isinstance(corpus_input, str):
        # Treat as path only if it has no newlines and points to an existing file.
        if "\n" not in corpus_input and len(corpus_input) < 260:
            candidate = Path(corpus_input)
            if candidate.is_file():
                return _parse_jsonl_lines(candidate.read_text(encoding="utf-8"))
        return _parse_jsonl_lines(corpus_input)

    passages: list[CorpusPassage] = []
    for idx, item in enumerate(corpus_input, start=1):
        if isinstance(item, CorpusPassage):
            passages.append(item)
        elif isinstance(item, dict):
            passages.append(_dict_to_passage(item, idx))
        else:
            raise TypeError(f"Unsupported corpus item type: {type(item).__name__}")
    return passages


def _parse_jsonl_lines(raw_text: str) -> list[CorpusPassage]:
    passages: list[CorpusPassage] = []
    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on corpus line {line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Corpus line {line_no} must be a JSON object.")
        passages.append(_dict_to_passage(obj, line_no))
    return passages


def _dict_to_passage(obj: dict[str, object], index: int) -> CorpusPassage:
    pid = str(obj.get("id") or obj.get("passage_id") or f"KB-{index:03d}")
    title = str(obj.get("title") or "")
    text = str(obj.get("text") or obj.get("content") or "").strip()
    if not text:
        raise ValueError(f"Corpus passage '{pid}' is missing a non-empty 'text' field.")
    return CorpusPassage(id=pid, title=title, text=text)


def _tokenize(text: str) -> list[str]:
    cleaned = text.lower().replace(",", "")
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.\d+)?", cleaned)


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in _tokenize(text) if tok not in STOPWORDS and len(tok) > 1}


def _strip_list_prefix(line: str) -> str:
    return re.sub(r"^(?:[-*•]+|\d+[.)])\s+", "", line).strip()


def _is_header_or_metadata(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if any(lower.startswith(prefix) for prefix in HEADER_PREFIXES):
        return True
    if re.match(r"^(?:document|memo|ref(?:erence)?)\s+[a-z0-9]{1,4}\s*:", lower):
        return True
    words = lower.split()
    if len(words) <= 6 and not line.endswith((".", "!", "?")) and not any(ch.isdigit() for ch in line):
        return True
    if line.isupper() and not line.endswith((".", "!", "?")):
        return True
    return False


def split_claims(text: str) -> list[str]:
    """Decompose raw or OCR-extracted document text into clean atomic claim sentences."""
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [line.strip() for line in normalized.split("\n")]

    # Rejoin wrapped OCR lines into coherent paragraph sentences while skipping banner headers.
    merged_blocks: list[str] = []
    current: list[str] = []
    for raw_line in raw_lines:
        if not raw_line:
            if current:
                merged_blocks.append(" ".join(current))
                current = []
            continue

        cleaned_line = _strip_list_prefix(raw_line)
        if _is_header_or_metadata(cleaned_line):
            # If a preceding unpunctuated title sat above a metadata line, drop it as header chrome.
            if current and not current[-1].endswith((".", "!", "?")):
                current = []
            continue

        # New bullet/numbered item forces a fresh block; discard short unpunctuated preceding title.
        starts_new_item = bool(re.match(r"^(?:[-*•]+|\d+[.)])\s+", raw_line))
        if current and starts_new_item:
            prev_text = " ".join(current)
            if prev_text.endswith((".", "!", "?")) or len(prev_text.split()) >= 11:
                merged_blocks.append(prev_text)
            current = [cleaned_line]
            continue

        current.append(cleaned_line)
        if cleaned_line.endswith((".", "!", "?")):
            merged_blocks.append(" ".join(current))
            current = []

    if current:
        merged_blocks.append(" ".join(current))

    claims: list[str] = []
    for block in merged_blocks:
        for sentence in _split_sentences(block):
            sent = re.sub(r"\s+", " ", sentence).strip()
            word_len = len(sent.split())
            if word_len >= 4 or (word_len >= 3 and sent.endswith((".", "!", "?"))):
                claims.append(sent)
    return claims


def _split_sentences(block: str) -> list[str]:
    """Split a paragraph on sentence boundaries without breaking decimals or abbreviations."""
    sentences: list[str] = []
    start = 0
    n = len(block)
    for i, ch in enumerate(block):
        if ch not in ".!?":
            continue
        prev_ch = block[i - 1] if i > 0 else ""
        next_ch = block[i + 1] if i + 1 < n else ""

        # Do not split decimal numbers like 99.94% or $482,500.00.
        if ch == "." and prev_ch.isdigit() and next_ch.isdigit():
            continue

        # Do not split inside common abbreviations like Inc. or U.S.
        if ch == ".":
            prefix_token = block[max(0, i - 10) : i].split()[-1].lower().strip("(\"'") if block[max(0, i - 10) : i].split() else ""
            if prefix_token in ABBREVIATIONS:
                continue

        if i + 1 == n or (next_ch.isspace() and _looks_like_sentence_start(block[i + 1 :].lstrip())):
            segment = block[start : i + 1].strip()
            if segment:
                sentences.append(segment)
            start = i + 1

    tail = block[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _looks_like_sentence_start(remainder: str) -> bool:
    if not remainder:
        return True
    first = remainder[0]
    return first.isupper() or first.isdigit() or first in {'"', "'", "$", "("}


class HybridEvidenceRetriever:
    """Okapi BM25 + word/char TF-IDF cosine hybrid retriever over corpus passages."""

    def __init__(self, passages: Sequence[CorpusPassage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 1.0
        self._word_vec: TfidfVectorizer | None = None
        self._char_vec: TfidfVectorizer | None = None
        self._word_matrix = None
        self._char_matrix = None

        if self.passages:
            self._build_index()

    def _build_index(self) -> None:
        texts = [f"{p.title} {p.text}".strip() for p in self.passages]
        self._doc_tokens = [_tokenize(t) for t in texts]
        self._doc_lens = [max(len(toks), 1) for toks in self._doc_tokens]
        self._avg_dl = sum(self._doc_lens) / max(len(self._doc_lens), 1)

        for toks in self._doc_tokens:
            for tok in set(toks):
                self._doc_freqs[tok] = self._doc_freqs.get(tok, 0) + 1

        self._word_vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*")
        self._word_matrix = self._word_vec.fit_transform(texts)

        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self._char_matrix = self._char_vec.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[EvidenceMatch]:
        """Rank passages for ``query`` using normalized BM25 + word/char TF-IDF fusion."""
        if not self.passages or not query.strip():
            return []

        n_docs = len(self.passages)
        q_tokens = _tokenize(query)
        bm25_scores = np.zeros(n_docs, dtype=np.float64)

        for idx, doc_toks in enumerate(self._doc_tokens):
            tf_map: dict[str, int] = {}
            for tok in doc_toks:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            dl = self._doc_lens[idx]
            score = 0.0
            for q_tok in q_tokens:
                tf = tf_map.get(q_tok, 0)
                if tf == 0:
                    continue
                df = self._doc_freqs.get(q_tok, 0)
                # Non-negative BM25+ style IDF guarantees positive weights on small corpora.
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_dl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            bm25_scores[idx] = score

        max_bm25 = float(np.max(bm25_scores))
        bm25_norm = bm25_scores / max_bm25 if max_bm25 > 1e-9 else np.zeros(n_docs, dtype=np.float64)

        q_word = self._word_vec.transform([query])
        q_char = self._char_vec.transform([query])
        word_cos = (self._word_matrix @ q_word.T).toarray().ravel()
        char_cos = (self._char_matrix @ q_char.T).toarray().ravel()
        tfidf_scores = 0.65 * word_cos + 0.35 * char_cos

        hybrid_scores = 0.55 * bm25_norm + 0.45 * tfidf_scores
        order = np.argsort(-hybrid_scores)[:top_k]

        results: list[EvidenceMatch] = []
        for idx in order:
            passage = self.passages[int(idx)]
            results.append(
                EvidenceMatch(
                    passage_id=passage.id,
                    title=passage.title,
                    text=passage.text,
                    bm25_score=round(float(bm25_scores[idx]), 4),
                    tfidf_score=round(float(tfidf_scores[idx]), 4),
                    hybrid_score=round(float(hybrid_scores[idx]), 4),
                )
            )
        return results


class _NumericEntity:
    def __init__(self, raw: str, value: float, kind: str, context: set[str]) -> None:
        self.raw = raw
        self.value = value
        self.kind = kind
        self.context = context


def _extract_numeric_entities(text: str) -> list[_NumericEntity]:
    """Extract normalized numbers tagged by kind (currency, percent, year, count) and context."""
    entities: list[_NumericEntity] = []
    tokens_with_spans = [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[a-zA-Z0-9$%.,-]+", text)]

    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0)
        num_str = match.group("num").replace(",", "")
        try:
            val = float(num_str)
        except ValueError:
            continue

        suffix = (match.group("suffix") or "").lower()
        if suffix == "k":
            val *= 1_000.0
        elif suffix == "m":
            val *= 1_000_000.0
        elif suffix == "b":
            val *= 1_000_000_000.0

        has_currency = bool(match.group("currency"))
        has_pct = bool(match.group("pct"))
        # Detect calendar month + day (e.g. "October 14") so dates are tagged separately from counts.
        prev_words = re.findall(r"[a-zA-Z]+", text[max(0, match.start() - 18) : match.start()].lower())
        is_month_day = bool(prev_words and prev_words[-1] in MONTH_NAMES and val.is_integer() and 1 <= int(val) <= 31)

        if has_currency:
            kind = "currency"
        elif has_pct:
            kind = "percent"
        elif is_month_day:
            kind = "date"
        elif val.is_integer() and 1900 <= int(val) <= 2100 and not suffix:
            kind = "year"
        else:
            kind = "count"

        # Ignore alphanumeric code numbers like FIN-402 or AES-256 or SOC 2? Keep context window.
        left_char = text[match.start() - 1] if match.start() > 0 else " "
        right_char = text[match.end()] if match.end() < len(text) else " "
        if left_char == "-" or right_char == "-":
            continue

        ctx = _nearby_context_words(tokens_with_spans, match.start(), match.end())
        entities.append(_NumericEntity(raw=raw, value=val, kind=kind, context=ctx))

    # Capture zero/no count phrases like "zero critical security findings" or "no critical findings".
    for m in re.finditer(r"\b(zero|none|no)\b(?:\s+[a-z-]+){1,3}", text.lower()):
        word = m.group(1)
        if word == "no":
            # Only treat "no <noun>" as count 0 when followed by plural/measurable noun words.
            phrase_tokens = set(_tokenize(m.group(0))) - {"no"}
            if not (phrase_tokens & {"findings", "incidents", "errors", "variances", "issues", "breaches", "exceptions"}):
                continue
        ctx = _content_tokens(m.group(0)) - ZERO_COUNT_WORDS
        if ctx:
            entities.append(_NumericEntity(raw=m.group(1), value=0.0, kind="count", context=ctx))

    return entities


def _nearby_context_words(
    tokens_with_spans: list[tuple[str, int, int]],
    start: int,
    end: int,
) -> set[str]:
    nearest = []
    for tok, t_start, t_end in tokens_with_spans:
        if t_end <= start:
            dist = start - t_end
        elif t_start >= end:
            dist = t_start - end
        else:
            continue
        if dist <= 36:
            cleaned = re.sub(r"[^a-z0-9-]", "", tok)
            if cleaned and cleaned not in STOPWORDS and not cleaned.isdigit():
                nearest.append(cleaned)
    return set(nearest)


def _detect_antonym_conflict(claim_tokens: set[str], evidence_tokens: set[str]) -> str | None:
    for pos_set, neg_set in ANTONYM_PAIRS:
        claim_pos = claim_tokens & pos_set
        claim_neg = claim_tokens & neg_set
        evid_pos = evidence_tokens & pos_set
        evid_neg = evidence_tokens & neg_set

        if (claim_neg and evid_pos and not claim_pos) or (claim_pos and evid_neg and not claim_neg):
            c_term = sorted(claim_neg or claim_pos)[0]
            e_term = sorted(evid_pos or evid_neg)[0]
            return f"Polarity conflict: claim uses '{c_term}' while evidence states '{e_term}'."
    return None


def _detect_negation_mismatch(claim: str, evidence: str) -> bool:
    claim_neg = set(_tokenize(claim.replace("'", ""))) & NEGATION_WORDS
    evid_neg = set(_tokenize(evidence.replace("'", ""))) & NEGATION_WORDS
    return bool(claim_neg) != bool(evid_neg)


def _compare_numbers(
    claim_nums: list[_NumericEntity],
    evid_nums: list[_NumericEntity],
) -> tuple[bool, list[str], list[str], str | None]:
    """Compare claim numbers against evidence numbers of matching kind/context."""
    conflicts: list[str] = []
    matched_count = 0
    unmatched_non_year = 0

    for c_num in claim_nums:
        if c_num.kind in {"year", "date"}:
            continue
        same_kind = [e for e in evid_nums if e.kind == c_num.kind]
        if not same_kind:
            unmatched_non_year += 1
            continue

        # Prefer exact value matches with shared context first; otherwise highest context overlap.
        best_match: _NumericEntity | None = None
        best_score: tuple[int, int] = (-1, -1)
        for e_num in same_kind:
            overlap = len(c_num.context & e_num.context)
            exact = int(math.isclose(c_num.value, e_num.value, rel_tol=1e-3, abs_tol=1e-4))
            score = (overlap, exact)
            if score > best_score:
                best_score = score
                best_match = e_num

        if best_match is None:
            unmatched_non_year += 1
            continue

        best_overlap, is_exact = best_score
        if is_exact:
            matched_count += 1
        elif best_overlap >= 1 or len(same_kind) == 1:
            conflicts.append(
                f"Numeric mismatch: claim cites '{c_num.raw}' vs evidence '{best_match.raw}'."
            )
        else:
            unmatched_non_year += 1

    has_conflict = len(conflicts) > 0
    has_unconfirmed_num = unmatched_non_year > 0 and not has_conflict
    reason = conflicts[0] if conflicts else None
    return has_conflict, [c.raw for c in claim_nums], [e.raw for e in evid_nums], (
        reason if has_conflict else ("UNCONFIRMED_NUMBER" if has_unconfirmed_num else None)
    )


class ClaimVerifier:
    """Audits extracted claims against a trusted knowledge base with citation grounding."""

    def __init__(
        self,
        corpus: Union[str, Path, Sequence[dict[str, str] | CorpusPassage]],
        top_k: int = 3,
    ) -> None:
        self.passages = parse_corpus_jsonl(corpus)
        self.top_k = top_k
        self.retriever = HybridEvidenceRetriever(self.passages)

    def verify_document(self, text: str) -> VerificationSummary:
        """Decompose ``text`` into claims and audit each claim against the corpus."""
        claims_text = split_claims(text)
        return self.verify_claims(claims_text)

    def verify_claims(self, claims_text: Iterable[str]) -> VerificationSummary:
        """Verify an explicit iterable of claim strings."""
        audits: list[ClaimAudit] = []
        for idx, raw_claim in enumerate(claims_text, start=1):
            claim = raw_claim.strip()
            if not claim:
                continue
            audits.append(self._audit_single_claim(f"CLM-{idx:02d}", claim))

        total = len(audits)
        supported = sum(1 for a in audits if a.verdict == "SUPPORTED")
        partial = sum(1 for a in audits if a.verdict == "PARTIAL")
        contradicted = sum(1 for a in audits if a.verdict == "CONTRADICTED")
        unverified = sum(1 for a in audits if a.verdict == "UNVERIFIED")

        if total == 0:
            faithfulness = 0.0
        else:
            faithfulness = (supported * 1.0 + partial * 0.5) / total

        return VerificationSummary(
            total_claims=total,
            supported_count=supported,
            partial_count=partial,
            contradicted_count=contradicted,
            unverified_count=unverified,
            faithfulness_score=round(faithfulness, 4),
            claims=audits,
        )

    def _audit_single_claim(self, claim_id: str, claim: str) -> ClaimAudit:
        matches = self.retriever.retrieve(claim, top_k=self.top_k)
        claim_nums = _extract_numeric_entities(claim)

        if not matches:
            return ClaimAudit(
                claim_id=claim_id,
                claim_text=claim,
                verdict="UNVERIFIED",
                confidence=0.9,
                rationale="Knowledge base contains no passages for retrieval.",
                matched_passages=[],
                extracted_numbers=[c.raw for c in claim_nums],
                evidence_numbers=[],
                negation_mismatch=False,
            )

        top = matches[0]
        claim_content = _content_tokens(claim)
        evid_content = _content_tokens(top.text)
        overlap_ratio = len(claim_content & evid_content) / max(len(claim_content), 1)
        evid_nums = _extract_numeric_entities(top.text)

        # Low retrieval alignment means the claim lacks grounding in the trusted corpus.
        if top.hybrid_score < 0.16 or overlap_ratio < 0.22:
            return ClaimAudit(
                claim_id=claim_id,
                claim_text=claim,
                verdict="UNVERIFIED",
                confidence=round(float(np.clip(1.0 - top.hybrid_score, 0.55, 0.96)), 3),
                rationale=(
                    f"No corroborating passage found (top score {top.hybrid_score:.2f}, "
                    f"concept overlap {overlap_ratio:.0%})."
                ),
                matched_passages=matches,
                extracted_numbers=[c.raw for c in claim_nums],
                evidence_numbers=[e.raw for e in evid_nums],
                negation_mismatch=False,
            )

        claim_all_toks = set(_tokenize(claim))
        evid_all_toks = set(_tokenize(top.text))
        antonym_reason = _detect_antonym_conflict(claim_all_toks, evid_all_toks)
        neg_mismatch = _detect_negation_mismatch(claim, top.text)
        num_conflict, c_raw_nums, e_raw_nums, num_reason = _compare_numbers(claim_nums, evid_nums)

        verdict: ClaimVerdict
        rationale: str
        confidence: float

        if num_conflict or antonym_reason or neg_mismatch:
            verdict = "CONTRADICTED"
            reasons = [r for r in [num_reason if num_conflict else None, antonym_reason] if r]
            if neg_mismatch:
                reasons.append("Negation polarity differs between claim and evidence.")
            rationale = f"Contradicts [{top.passage_id}]: " + " ".join(reasons)
            confidence = round(float(np.clip(0.75 + 0.2 * top.hybrid_score, 0.72, 0.98)), 3)
        elif overlap_ratio >= 0.44 and top.hybrid_score >= 0.30 and num_reason != "UNCONFIRMED_NUMBER":
            verdict = "SUPPORTED"
            rationale = (
                f"Corroborated by [{top.passage_id}] (hybrid score {top.hybrid_score:.2f}, "
                f"overlap {overlap_ratio:.0%}, all extracted figures verified)."
            )
            confidence = round(float(np.clip(0.55 + 0.42 * min(1.0, overlap_ratio), 0.65, 0.99)), 3)
        else:
            verdict = "PARTIAL"
            extra = " Contains unconfirmed quantitative details." if num_reason == "UNCONFIRMED_NUMBER" else ""
            rationale = (
                f"Partially matches [{top.passage_id}] (hybrid score {top.hybrid_score:.2f}, "
                f"overlap {overlap_ratio:.0%}).{extra}"
            )
            confidence = round(float(np.clip(0.50 + 0.35 * overlap_ratio, 0.52, 0.88)), 3)

        return ClaimAudit(
            claim_id=claim_id,
            claim_text=claim,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            matched_passages=matches,
            extracted_numbers=c_raw_nums,
            evidence_numbers=e_raw_nums,
            negation_mismatch=neg_mismatch,
        )

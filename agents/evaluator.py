"""
agents/evaluator.py
====================
The Evaluator Agent.

Responsibility
--------------
Compare classifier output against gold labels, compute binary classification
metrics, and extract aggregated failure patterns to feed the optimiser.

Information boundary (critical for bias prevention)
----------------------------------------------------
  SEES     : classifier results + gold labels + gold reasons
  PASSES TO OPTIMISER : aggregated failure PATTERNS only
                        — never individual doc IDs, raw labels, or gold reasons

Why the boundary on optimiser output matters
--------------------------------------------
If the optimiser saw individual (document, expected_label) pairs from the eval
set, it would craft prompt rules that match the surface features of those exact
documents.  Scores would inflate on dev but collapse on the held-out test set —
classic prompt overfitting.

By giving the optimiser only aggregate descriptions ("23 false positives where
the model over-generalised on keyword overlap") we force it to reason about
general failure modes rather than memorising the eval set.

Metrics computed
----------------
  Precision  = TP / (TP + FP)  — of all docs called relevant, how many were?
  Recall     = TP / (TP + FN)  — of all truly relevant docs, how many did we catch?
  F1         = harmonic mean of precision and recall
  TP/FP/FN/TN: raw confusion-matrix counts for the iteration log

Overfitting detection
---------------------
  We compare dev F1 vs test F1 and dev F1 vs canary F1.
  A widening dev/test gap signals the prompt has begun fitting dev-specific
  surface patterns instead of the underlying relevancy concept.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    """Binary classification metrics for one evaluation run."""
    precision: float
    recall:    float
    f1:        float
    tp: int   # true positives  — correctly labelled relevant
    fp: int   # false positives — labelled relevant but actually not
    fn: int   # false negatives — missed relevant documents
    tn: int   # true negatives  — correctly labelled not-relevant

    def meets(self, target_precision: float, target_recall: float) -> bool:
        """True when both convergence targets are simultaneously satisfied."""
        return self.precision >= target_precision and self.recall >= target_recall

    def __str__(self) -> str:
        return (
            f"P={self.precision:.3f}  R={self.recall:.3f}  F1={self.f1:.3f}  "
            f"TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn}"
        )


@dataclass
class FailurePatterns:
    """
    Aggregated description of what went wrong — safe to pass to the optimiser.

    Crucially, this contains NO individual document IDs, raw labels, or gold
    reasons.  It contains only:
      - counts of each error type
      - abstract descriptions of the failure mode
      - the model's OWN stated reasons for the wrong calls
        (these reveal misunderstandings in the criteria without exposing labels)
    """
    fp_count: int             # number of false positives
    fn_count: int             # number of false negatives

    # Abstract description of what caused false positives.
    # Written by this module based on observed patterns — no doc-specific info.
    fp_description: str

    # Abstract description of what caused false negatives.
    fn_description: str

    # The model's own stated reasons for its wrong calls (from the "reason" field
    # in classifier output).  These reveal what the model misunderstood about the
    # criteria without exposing any gold-label information to the optimiser.
    fp_example_reasons: list[str]   # reasons from false-positive cases (model said relevant when it wasn't)
    fn_example_reasons: list[str]   # reasons from false-negative cases (model said not-relevant when it was)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(
    results:   list[dict],
    gold_path: pathlib.Path,
    *,
    indices:   list[int] | None = None,
) -> tuple[Metrics, FailurePatterns]:
    """
    Compute metrics and extract failure patterns from one classifier run.

    Parameters
    ----------
    results   : List of dicts from classifier.classify() — {label, confidence, reason}.
    gold_path : Path to gold.jsonl — {idx, label}.
                This file is the source of truth; classifier must never have seen it.
    indices   : Row positions that were classified (from sampling).  When provided,
                only those gold rows are used for comparison.  None means all rows.

    Returns
    -------
    (Metrics, FailurePatterns) — metrics for logging; patterns for the optimiser.
    """
    # Load gold labels in the same row order as classifier results
    all_gold = [
        json.loads(line)
        for line in gold_path.read_text().splitlines()
        if line.strip()
    ]

    # Select only the gold rows that correspond to classified documents
    gold = [all_gold[i] for i in indices] if indices is not None else all_gold

    # Sanity check — mismatched lengths mean the splits are out of sync
    if len(results) != len(gold):
        raise ValueError(
            f"Result count ({len(results)}) does not match gold count ({len(gold)}). "
            "Re-run data/prepare.py to regenerate consistent splits."
        )

    # ── Confusion matrix ──────────────────────────────────────────────────
    preds  = [r["label"]  for r in results]
    labels = [g["label"]  for g in gold]

    tp = sum(p == 1 and g == 1 for p, g in zip(preds, labels))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, labels))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, labels))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, labels))

    # Guard against division by zero when there are no positive predictions/labels
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    metrics = Metrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )

    # ── Failure pattern extraction ─────────────────────────────────────────
    # Collect the MODEL'S stated reasons (not gold reasons) for wrong calls.
    # These reveal what the model misunderstood — safe to show the optimiser.
    fp_reasons: list[str] = []  # reasons the model gave when it was wrong about relevance
    fn_reasons: list[str] = []  # reasons the model gave when it missed a relevant doc

    for result, gold_row in zip(results, gold):
        pred_label = result["label"]
        true_label = gold_row["label"]
        reason     = result.get("reason", "")

        if pred_label == 1 and true_label == 0:
            # False positive: model said relevant, but it wasn't
            fp_reasons.append(reason)
        elif pred_label == 0 and true_label == 1:
            # False negative: model said not-relevant, but it was
            fn_reasons.append(reason)

    # Build abstract descriptions of the failure modes.
    # We cap example reasons at 4 so the optimiser prompt stays concise.
    patterns = FailurePatterns(
        fp_count=fp,
        fn_count=fn,
        fp_description=(
            f"{fp} documents were incorrectly labelled as relevant. "
            "The model appears to over-generalise — labelling documents as relevant "
            "based on surface keyword overlap without requiring substantive coverage "
            "of the specific query topic."
        ) if fp > 0 else "No false positives — precision is clean.",

        fn_description=(
            f"{fn} genuinely relevant documents were missed. "
            "The model appears to under-count — rejecting documents that answer the "
            "query indirectly, use synonyms, or provide partial-but-useful coverage."
        ) if fn > 0 else "No false negatives — recall is clean.",

        fp_example_reasons=fp_reasons[:4],
        fn_example_reasons=fn_reasons[:4],
    )

    return metrics, patterns


# ── Overfitting detection ─────────────────────────────────────────────────────

def overfitting_check(
    dev:    Metrics,
    test:   Metrics,
    canary: Metrics,
) -> list[str]:
    """
    Compare dev, test, and canary metrics and return a list of warning strings.

    An empty list means no overfitting signals detected.

    Signals we check
    ----------------
    1. Dev/test F1 gap > 0.05:
       The prompt improved on dev but the gain doesn't transfer to unseen data.
       This suggests the optimiser found dev-specific surface patterns.

    2. Canary F1 significantly below test F1:
       New production-like documents perform worse than the test set.
       This suggests the prompt is sensitive to domain or phrasing distribution.
    """
    warnings: list[str] = []

    dev_test_gap = dev.f1 - test.f1
    if dev_test_gap > 0.05:
        warnings.append(
            f"Dev/test F1 gap = {dev_test_gap:.3f}. "
            "The prompt may be overfitting to dev-set surface patterns. "
            "Check whether the optimiser is using doc-specific features."
        )

    canary_test_gap = test.f1 - canary.f1
    if canary_test_gap > 0.05:
        warnings.append(
            f"Canary F1 ({canary.f1:.3f}) trails test F1 ({test.f1:.3f}) "
            f"by {canary_test_gap:.3f}. "
            "The prompt may not generalise well to new production documents."
        )

    return warnings
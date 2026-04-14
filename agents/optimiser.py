"""
agents/optimiser.py
====================
The Optimiser Agent.

Responsibility
--------------
Rewrite the classification prompt to fix identified failure patterns, then
return the new prompt along with an explanation of what changed and why.

Information boundary (critical for correctness)
-----------------------------------------------
  SEES     : current prompt text, aggregated failure patterns (no doc IDs),
             current metrics (precision/recall/F1), iteration history (no labels)
  NEVER SEES: individual document labels, specific doc IDs, gold reasons,
              or anything from the held-out test split

Why using a different model for the optimiser matters
------------------------------------------------------
If the same model instance that runs the classifier also rewrites the prompt,
it exhibits self-consistency bias — it tends to rate its own outputs as correct
and write prompts that reinforce its existing behaviour rather than fixing it.

Using a separate model (or the same model with different context) breaks this
feedback loop.  In practice, gpt-5-nano is both cheap enough and capable enough
to serve as both classifier and optimiser for this task, but the *call context*
for the optimiser is entirely different from the classifier context.

Iteration history
-----------------
We pass the last 3 iterations of history so the optimiser knows what has
already been tried and can avoid repeating failed attempts.  We cap at 3 to
keep the optimiser prompt concise and avoid diluting the current failure signal.

Output
------
Returns four values:
  new_system   : str        — the rewritten system prompt
  new_user     : str        — the rewritten user template (preserves {query}/{document})
  explanation  : str        — what changed and why (becomes the "notes" on the new version)
  hypotheses   : list[str]  — what the optimiser predicts will improve
"""

from __future__ import annotations

import json
import os

from openai import AzureOpenAI


# ── Optimiser system prompt ───────────────────────────────────────────────────

# This is the meta-prompt that tells the model HOW to rewrite the classifier prompt.
# It enforces the information boundary (no memorising eval examples) and ensures
# the output is parseable JSON.
_OPTIMISER_SYSTEM = """\
You are an expert prompt engineer specialising in binary relevancy classification.

Your task is to rewrite a relevancy classification prompt to fix specific failure \
patterns described to you.

Rules you must follow:
1. Address the exact failure patterns described — do not make unrelated changes.
2. Write general criteria and principles, not rules tuned to specific documents.
   The goal is a prompt that generalises to unseen data, not one that memorises examples.
3. If you add few-shot examples, you MUST invent them — do not use examples from
   the evaluation dataset.  Invented examples that illustrate the failure pattern
   are more valuable than memorised ones anyway.
4. Preserve the JSON output format exactly:
   {"label": "relevant" or "not_relevant", "confidence": 0.0-1.0, "reason": "..."}
   The classifier must always be able to parse the model's response.
5. If the iteration history shows a previous attempt tried a specific fix and it
   did not work, do not repeat that fix.  Try a different approach.
6. Be concise — a bloated prompt increases latency and cost without helping.

Output format: JSON with exactly these keys:
  new_system   : string — the full new system prompt
  new_user     : string — the full new user template (must contain {query} and {document})
  explanation  : string — what you changed and the specific reasoning behind each change
  hypotheses   : list of strings — what you predict will improve as a result\
"""


# ── Helper: build the optimiser user message ──────────────────────────────────

def _build_user_message(
    current_system: str,
    current_user:   str,
    metrics:        dict,
    patterns:       dict,
    history:        list[dict],
    target_p:       float,
    target_r:       float,
) -> str:
    """
    Construct the user message sent to the optimiser.

    Everything in this message is safe to show the optimiser:
    - Current prompt text (it wrote the previous version anyway)
    - Aggregate metrics (numbers, no doc-level info)
    - Failure patterns (abstract descriptions + model's own reasons)
    - Iteration history (version name + metrics + explanation, no labels)
    """

    # Format iteration history — last 3 only to stay concise
    history_lines = ""
    for h in history[-3:]:
        history_lines += (
            f"\n  {h['version']}: "
            f"P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f}"
            f"\n    Change: {h['notes']}"
        )

    # Format failure pattern reasons — the model's own stated justifications
    def _fmt_reasons(reasons: list[str]) -> str:
        if not reasons:
            return "  (none captured)"
        return "\n".join(f"  - {r}" for r in reasons)

    return f"""\
CURRENT PROMPT
==============
System:
{current_system}

User template:
{current_user}


CURRENT METRICS  (dev set — optimise against these)
====================================================
Precision : {metrics['precision']:.3f}   (target ≥ {target_p})
Recall    : {metrics['recall']:.3f}   (target ≥ {target_r})
F1        : {metrics['f1']:.3f}

Targets not yet met:
  {'⚠ Precision below target' if metrics['precision'] < target_p else '✓ Precision OK'}
  {'⚠ Recall below target'    if metrics['recall']    < target_r else '✓ Recall OK'}


FAILURE PATTERNS
================
False positives — {patterns['fp_count']} cases (model labelled relevant, actually not):
{patterns['fp_description']}

Model's stated reasons for these errors:
{_fmt_reasons(patterns['fp_example_reasons'])}

False negatives — {patterns['fn_count']} cases (model labelled not-relevant, actually was):
{patterns['fn_description']}

Model's stated reasons for these errors:
{_fmt_reasons(patterns['fn_example_reasons'])}


ITERATION HISTORY  (last 3 iterations)
=======================================
{history_lines.strip() or "No prior iterations — this is the first optimisation."}


Rewrite the prompt to fix the failure patterns above.
Remember: write general principles, not rules for specific documents.\
"""


# ── Main optimiser call ───────────────────────────────────────────────────────

def optimise(
    current_system: str,
    current_user:   str,
    metrics:        dict,
    patterns:       dict,
    history:        list[dict],
    target_p:       float,
    target_r:       float,
    deployment:     str,
) -> tuple[str, str, str, list[str]]:
    """
    Call the optimiser model and return an improved prompt.

    Parameters
    ----------
    current_system : The system prompt that produced the current metrics.
    current_user   : The user template that produced the current metrics.
    metrics        : {"precision": float, "recall": float, "f1": float}
    patterns       : Failure pattern dict from evaluator.FailurePatterns
                     (no individual doc IDs or labels).
    history        : List of previous iteration records (no labels).
    target_p       : Precision target (shown to optimiser for context).
    target_r       : Recall target.
    deployment     : Azure OpenAI deployment name.

    Returns
    -------
    (new_system, new_user, explanation, hypotheses)
    """
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        max_retries=5,
    )

    user_message = _build_user_message(
        current_system=current_system,
        current_user=current_user,
        metrics=metrics,
        patterns=patterns,
        history=history,
        target_p=target_p,
        target_r=target_r,
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _OPTIMISER_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        # gpt-5-nano is a reasoning model — max_completion_tokens covers
        # BOTH internal reasoning tokens and visible output tokens.
        # The model typically uses ~4K reasoning + ~1.5K output, so 16K
        # gives it enough headroom for complex prompt rewrites.
        max_completion_tokens=16000,
        response_format={"type": "json_object"},
    )

    choice = response.choices[0]
    raw = choice.message.content or ""
    if not raw.strip():
        raise RuntimeError(
            f"Optimiser returned empty response. "
            f"Finish reason: {choice.finish_reason}. "
            f"Usage: {response.usage}. "
            f"Message: {choice.message}"
        )

    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Optimiser returned invalid JSON: {raw[:200]!r}"
        ) from exc

    return (
        out["new_system"],
        out["new_user"],
        out.get("explanation", ""),
        out.get("hypotheses", []),
    )
"""
agents/classifier.py
=====================
The Classifier Agent.

Responsibility
--------------
Run the current prompt against every document in a split and return a
binary relevancy decision for each one.

Information boundary (critical for bias prevention)
----------------------------------------------------
  SEES     : system prompt (criteria) + query + document text
  NEVER SEES: expected labels, gold reasons, the test split,
              or any other document's result

Why this boundary matters
-------------------------
If the classifier ever saw the expected label for a document — even as
context or an example — it would learn to confirm the label rather than
reason from the criteria.  This is label-leakage bias and it would make
the evaluation metrics meaningless.

Output per document
-------------------
  label      : int   — 1 = relevant, 0 = not relevant
  confidence : float — model's self-reported certainty (0.0 – 1.0)
  reason     : str   — model's stated justification (used by evaluator
                       to identify failure patterns; NOT the gold reason)

The output is written to a JSONL file so it can be inspected offline and
re-evaluated without re-running the model.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import time

from openai import AzureOpenAI

# Delay between API calls to avoid hitting Azure rate limits (429).
# Adjust via environment variable if your deployment has a higher TPM quota.
_CALL_DELAY = float(os.environ.get("API_CALL_DELAY", "1.0"))


# ── Azure client ──────────────────────────────────────────────────────────────

def _make_client() -> AzureOpenAI:
    """
    Build the Azure OpenAI client from environment variables.

    Environment variables (set in .env):
      AZURE_OPENAI_ENDPOINT    : your resource endpoint URL
      AZURE_OPENAI_API_KEY     : your API key
      AZURE_OPENAI_API_VERSION : API version string
    """
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        max_retries=5,
    )


# ── Single-document classification ───────────────────────────────────────────

def _classify_one(
    client:     AzureOpenAI,
    deployment: str,
    system:     str,
    user_tmpl:  str,
    query:      str,
    document:   str,
) -> dict:
    """
    Classify a single (query, document) pair.

    We rely on the model's default temperature since gpt-5-nano does not
    support custom temperature values.

    We request JSON output via response_format so we can reliably parse the
    label, confidence, and reason without fragile string-splitting.

    Returns a dict with keys: label (int), confidence (float), reason (str).
    On any error (network, parse failure), returns a safe default of label=0
    with the error message in the reason field.  This is logged so the
    evaluator can count parse errors as a quality signal.
    """
    user_message = user_tmpl.format(query=query, document=document)

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            # gpt-5-nano is a reasoning model — max_completion_tokens covers
            # both internal reasoning and visible output.  if set to low, will be consumed
            # entirely by reasoning, producing empty responses.  2000 gives
            # enough room for reasoning (~1500) + JSON output (~100).
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )

        raw    = response.choices[0].message.content
        parsed = json.loads(raw)

        # Normalise the label string to an integer.
        # We accept "relevant", "RELEVANT", or any case variation.
        label_str = parsed.get("label", "").lower().strip()
        label_int = 1 if label_str == "relevant" else 0

        return {
            "label":      label_int,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason":     parsed.get("reason", ""),
        }

    except Exception as exc:
        # Return a safe default so one failure doesn't abort the whole run.
        # The error message is preserved in "reason" for debugging.
        return {
            "label":      0,
            "confidence": 0.0,
            "reason":     f"ERROR: {exc}",
        }


# ── Batch classification ──────────────────────────────────────────────────────

def classify(
    system:       str,
    user_template: str,
    split_dir:    pathlib.Path,
    deployment:   str,
    out_path:     pathlib.Path,
    *,
    split_name:   str = "",
    sample_size:  int | None = None,
) -> tuple[list[dict], list[int]]:
    """
    Run the classifier on every row (or a random sample) in split_dir/input.jsonl.

    Parameters
    ----------
    system        : The system prompt content (contains the relevancy criteria).
    user_template : The user message template with {query} and {document}.
    split_dir     : Directory containing input.jsonl (NO labels in this file).
    deployment    : Azure OpenAI deployment name (e.g. "gpt-5-nano").
    out_path      : Where to write the JSONL results file.
    split_name    : Label shown in progress output (e.g. "dev", "canary").
    sample_size   : If set, classify only this many randomly sampled rows.
                    Use None to classify all rows.

    Returns
    -------
    (results, indices) — results is a list of result dicts; indices is the
    row positions used (sorted), matching the order in input.jsonl.  When
    sample_size is None, indices covers every row.
    """
    client = _make_client()
    label = f" ({split_name})" if split_name else ""

    # Read classifier input — query + document only, no labels
    input_rows = [
        json.loads(line)
        for line in (split_dir / "input.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Select which rows to classify
    all_indices = list(range(len(input_rows)))
    if sample_size is not None and sample_size < len(input_rows):
        indices = sorted(random.sample(all_indices, sample_size))
    else:
        indices = all_indices

    total = len(indices)
    suffix = f"/{len(input_rows)}" if total < len(input_rows) else ""
    results: list[dict] = []
    for i, idx in enumerate(indices, 1):
        if i > 1 and _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)
        sys.stderr.write(f"\r    {i}/{total}{suffix} classified{label}")
        sys.stderr.flush()
        row = input_rows[idx]
        result = _classify_one(
            client=client,
            deployment=deployment,
            system=system,
            user_tmpl=user_template,
            query=row["query"],
            document=row["document"],
        )
        results.append(result)
    sys.stderr.write("\n")

    # Persist results so we can re-evaluate without re-running the model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in results))

    return results, indices
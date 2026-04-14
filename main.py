"""
main.py
=======
Orchestration loop — the entry point for the pipeline.

How to run
----------
  uv run python main.py

What it does
------------
Iterates the following steps until BOTH precision AND recall targets are met,
or until MAX_ITERATIONS is exhausted:

  1. Classify   — Run the current prompt on every document in the dev split.
                  The classifier sees only: criteria (via system prompt) + query + document.
                  It never sees expected labels.

  2. Evaluate   — Compare classifier output against gold labels.
                  Compute precision, recall, F1.
                  Extract aggregated failure patterns (no individual doc IDs).

  3. Canary     — Run classifier on the canary split and check F1.
                  Detects when a prompt overfits to the dev set.

  4. Gate check — If both targets are met, run the classifier on the held-out
                  test split (which has been completely untouched until now)
                  and verify the metrics hold.  Also check for overfitting signals.

  5. Optimise   — If targets not yet met, call the optimiser with the aggregated
                  failure patterns (not raw labels) and register the new prompt.

Outputs
-------
  results/iterations.jsonl   — one JSON record per iteration (full audit log)
  results/winning_prompt.json — the converged prompt, ready to copy into production
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env before importing agents (agents read env vars at call time, but
# loading early avoids confusing "missing key" errors deep in the call stack).
load_dotenv()

# Make project root importable so `agents.*` and `prompts.*` resolve correctly
# whether the script is run from the project root or a subdirectory.
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.classifier import classify
from agents.evaluator  import evaluate, overfitting_check
from agents.optimiser  import optimise
import prompts.store as store

# ── Configuration from environment ────────────────────────────────────────────

# Deployment name — must match what you named the model in Azure AI Foundry
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")

# Convergence targets — the loop stops only when BOTH are met simultaneously
TARGET_P = float(os.environ.get("TARGET_PRECISION",  "0.82"))
TARGET_R = float(os.environ.get("TARGET_RECALL",     "0.78"))

# Hard cap on iterations regardless of whether targets are met
MAX_ITER = int(os.environ.get("MAX_ITERATIONS", "4"))

# Sample size for dev classification in early iterations.
# The full dev set runs only on the final iteration and the convergence gate.
# Sampling cuts runtime ~4x while giving the optimiser enough signal to improve.
DEV_SAMPLE = int(os.environ.get("DEV_SAMPLE_SIZE", "60"))

# Paths — switch dataset via DATA_DIR env var (e.g. DATA_DIR=data)
DATA    = ROOT / os.environ.get("DATA_DIR", "data_legal")
RESULTS = ROOT / "results"

# Rich console for formatted terminal output
console = Console()


# ── Resume support ────────────────────────────────────────────────────────────
#
# Each iteration is expensive (~280 API calls), so we persist enough state
# to resume after a crash or timeout instead of re-running from scratch.
#
# Two files provide all the state needed:
#   results/iterations.jsonl — metrics, version name, and optimiser notes
#                              per completed iteration (the audit log).
#   results/prompts.json     — full text of every prompt version tried
#                              (system + user + notes).
#
# On startup _load_previous_run() checks whether a resumable run exists:
#   1. Both files present and the last iteration did NOT converge → resume.
#   2. Otherwise → fresh start (or pass --fresh to force one).
#
# To resume it re-registers saved prompts into the in-memory store,
# rebuilds the optimiser history from the log, and returns the iteration
# number to continue from.

def _load_previous_run() -> tuple[list[dict], list[dict], int] | None:
    """Load state from a previous incomplete run.

    Returns (iteration_log, history, start_iteration) or None if no
    resumable run exists.
    """
    log_path = RESULTS / "iterations.jsonl"
    prompts_path = RESULTS / "prompts.json"

    if not log_path.exists() or not prompts_path.exists():
        return None

    iteration_log = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    if not iteration_log:
        return None

    # Already converged — nothing to resume
    if iteration_log[-1].get("converged"):
        return None

    # Restore prompt versions into the store
    saved_prompts = json.loads(prompts_path.read_text())
    for p in saved_prompts:
        if p["version"] not in ("v1", "v2"):
            store.register(p["system"], p["user"], p["notes"])

    # Rebuild optimiser history from the log
    history = [
        {
            "version":   rec["version"],
            "precision": rec["precision"],
            "recall":    rec["recall"],
            "f1":        rec["f1"],
            "notes":     rec.get("explanation", rec.get("notes", "")),
        }
        for rec in iteration_log
    ]

    # The last version in the saved prompts is what we should continue from
    last_version = saved_prompts[-1]["version"]
    store.set_current(last_version)
    start_iteration = len(iteration_log) + 1

    return iteration_log, history, start_iteration


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(start_version: str = "v1") -> None:
    """
    Run the full optimisation loop.

    Automatically resumes from the last saved state if a previous
    incomplete run is found in results/.

    Parameters
    ----------
    start_version : Which prompt version to begin with ("v1" or "v2").
                    Ignored when resuming a previous run.
    """
    RESULTS.mkdir(exist_ok=True)

    # Try to resume from a previous incomplete run
    previous = _load_previous_run()
    if previous is not None:
        iteration_log, history, start_iter = previous
        last = iteration_log[-1]
        console.rule("[bold]Prompt Refinement Pipeline (resumed)")
        console.print(
            f"Model: [cyan]{DEPLOYMENT}[/]  |  "
            f"Target P≥{TARGET_P}  R≥{TARGET_R}  |  "
            f"Max {MAX_ITER} iterations  |  "
            f"Resuming from iteration {start_iter} "
            f"([cyan]{store.current().version}[/])  |  "
            f"Last dev F1={last['f1']:.3f}\n"
        )
    else:
        iteration_log: list[dict] = []
        history: list[dict] = []
        start_iter = 1
        store.set_current(start_version)

        console.rule("[bold]Prompt Refinement Pipeline")
        console.print(
            f"Model: [cyan]{DEPLOYMENT}[/]  |  "
            f"Target P≥{TARGET_P}  R≥{TARGET_R}  |  "
            f"Max {MAX_ITER} iterations  |  "
            f"Starting from [cyan]{start_version}[/]\n"
        )

    for iteration in range(start_iter, MAX_ITER + 1):
        prompt = store.current()
        iter_t0 = time.monotonic()
        console.rule(f"Iteration {iteration}  ·  {prompt.version}")

        # ── Step 1: Classify dev split ─────────────────────────────────────
        # Sends the current prompt + each document to gpt-5-nano, one at a time. 
        # On early iterations, only classifies a sample of 60 docs.
        # On the final iteration, classifies all 240. Each doc gets a {label, confidence, reason} response.
        # Sample a subset for early iterations to save time; the optimiser
        # gets enough signal from 60 docs to identify failure patterns.
        # Run the full set only on the final iteration for accurate metrics.
        is_final = iteration == MAX_ITER
        sample = None if is_final else DEV_SAMPLE
        sample_note = "" if is_final else f" (sample={sample})"
        console.print(f"  [1/4] Classifying dev split{sample_note}...")
        t0 = time.monotonic()
        dev_results, dev_indices = classify(
            system=prompt.system,
            user_template=prompt.user,
            split_dir=DATA / "dev",
            deployment=DEPLOYMENT,
            out_path=RESULTS / f"{prompt.version}_dev.jsonl",
            split_name="dev",
            sample_size=sample,
        )
        dev_secs = time.monotonic() - t0

        # ── Step 2: Evaluate dev ──────────────────────────────────────────
        # Compares the classifier's labels against gold labels (which the classifier never saw). 
        # Computes precision, recall, F1. 
        # Also extracts failure patterns — aggregated descriptions like "23 false positives 
        # from keyword overlap" — safe to pass to the optimiser without leaking individual labels.
        # The evaluator opens data/dev/gold.jsonl (the classifier never does).
        # Returns Metrics (numbers) and FailurePatterns (safe aggregates).
        dev_metrics, dev_patterns = evaluate(
            results=dev_results,
            gold_path=DATA / "dev" / "gold.jsonl",
            indices=dev_indices,
        )
        console.print(
            f"  [2/4] Dev:    {dev_metrics}  [dim]({dev_secs:.0f}s)[/]"
        )

        # ── Step 3: Classify canary split ──────────────────────────────────
        # Same as steps 1-2 but on the 40-doc canary split. 
        # This result is never fed to the optimiser — 
        # it's purely a check that the prompt generalises beyond dev.
        # The canary split contains fresh examples added to detect drift.
        # If dev F1 rises but canary F1 falls, the prompt is overfitting to dev.
        console.print("  [3/4] Classifying canary split...")
        t0 = time.monotonic()
        canary_results, canary_indices = classify(
            system=prompt.system,
            user_template=prompt.user,
            split_dir=DATA / "canary",
            deployment=DEPLOYMENT,
            out_path=RESULTS / f"{prompt.version}_canary.jsonl",
            split_name="canary",
        )
        canary_secs = time.monotonic() - t0
        canary_metrics, _ = evaluate(
            results=canary_results,
            gold_path=DATA / "canary" / "gold.jsonl",
            indices=canary_indices,
        )
        console.print(
            f"        Canary: {canary_metrics}  [dim]({canary_secs:.0f}s)[/]"
        )

        # ── Build iteration log record ─────────────────────────────────────
        # Recorded now so it's written even if convergence is reached below.
        log_record: dict = {
            "iteration":  iteration,
            "version":    prompt.version,
            "ts":         datetime.now(timezone.utc).isoformat(),
            "precision":  dev_metrics.precision,
            "recall":     dev_metrics.recall,
            "f1":         dev_metrics.f1,
            "canary_f1":  canary_metrics.f1,
            "converged":  False,
            "notes":      prompt.notes,
        }

        # ── Step 4: Convergence gate ───────────────────────────────────────
        # If on a sample → re-run full dev (240 docs) to confirm. 
        # If full dev doesn't confirm, falls through to the optimiser.
        # If confirmed → classify the held-out test split (120 docs) for the first and only time. 
        # Check for overfitting (dev-test gap, canary-test gap). 
        # Save everything and return — pipeline is done.
        if dev_metrics.meets(TARGET_P, TARGET_R):
            # If targets were met on a sample, re-run full dev to confirm
            if sample is not None:
                console.print(
                    "\n  [green]✓ Targets met on sample.[/] "
                    "Re-running full dev to confirm..."
                )
                dev_results, dev_indices = classify(
                    system=prompt.system,
                    user_template=prompt.user,
                    split_dir=DATA / "dev",
                    deployment=DEPLOYMENT,
                    out_path=RESULTS / f"{prompt.version}_dev_full.jsonl",
                    split_name="dev-full",
                )
                dev_metrics, dev_patterns = evaluate(
                    results=dev_results,
                    gold_path=DATA / "dev" / "gold.jsonl",
                    indices=dev_indices,
                )
                console.print(f"        Full dev: {dev_metrics}")
                # Update log record with full metrics
                log_record.update({
                    "precision": dev_metrics.precision,
                    "recall":    dev_metrics.recall,
                    "f1":        dev_metrics.f1,
                })
                if not dev_metrics.meets(TARGET_P, TARGET_R):
                    console.print(
                        "  [yellow]Targets not met on full dev — "
                        "continuing optimisation.[/]"
                    )
                    # Fall through to optimiser below
                else:
                    console.print(
                        "\n  [green]✓ Confirmed on full dev.[/]"
                    )

            if dev_metrics.meets(TARGET_P, TARGET_R):
                console.print("  Running held-out test gate...")

                # This is the ONLY moment we open the test split.
                test_results, test_indices = classify(
                    system=prompt.system,
                    user_template=prompt.user,
                    split_dir=DATA / "test",
                    deployment=DEPLOYMENT,
                    out_path=RESULTS / f"{prompt.version}_test.jsonl",
                    split_name="test",
                )
                test_metrics, _ = evaluate(
                    results=test_results,
                    gold_path=DATA / "test" / "gold.jsonl",
                    indices=test_indices,
                )
                console.print(f"\n  [bold]TEST:   {test_metrics}[/]")

                # Check for overfitting signals across all three splits
                warnings = overfitting_check(
                    dev_metrics, test_metrics, canary_metrics,
                )
                for warning in warnings:
                    console.print(f"  [yellow]⚠  {warning}[/]")

                # Finalise the log record and persist everything
                log_record.update({
                    "converged":           True,
                    "test_precision":      test_metrics.precision,
                    "test_recall":         test_metrics.recall,
                    "test_f1":             test_metrics.f1,
                    "overfitting_warnings": warnings,
                })
                iteration_log.append(log_record)
                _save_log(iteration_log)
                _save_prompts(store.all_versions())
                _save_winning_prompt(prompt)
                _print_summary(iteration_log)
                return  # ← pipeline complete

        # ── Not converged — check if we've hit the iteration cap ───────────
        if iteration == MAX_ITER:
            console.print(
                f"\n  [red]Max iterations ({MAX_ITER}) reached "
                "without meeting targets.[/]"
            )
            iteration_log.append(log_record)
            _save_log(iteration_log)
            _save_prompts(store.all_versions())
            _print_summary(iteration_log)
            return

        # ── Step 5: Optimise ───────────────────────────────────────────────
        # Pass aggregated failure patterns — never raw labels or doc IDs.
        # The optimiser rewrites the prompt to address the specific failure modes.
        console.print("  [4/4] Running optimiser...")
        t0 = time.monotonic()

        # Convert FailurePatterns dataclass to a plain dict for JSON-serialisability
        patterns_dict = {
            "fp_count":            dev_patterns.fp_count,
            "fn_count":            dev_patterns.fn_count,
            "fp_description":      dev_patterns.fp_description,
            "fn_description":      dev_patterns.fn_description,
            "fp_example_reasons":  dev_patterns.fp_example_reasons,
            "fn_example_reasons":  dev_patterns.fn_example_reasons,
        }

        new_system, new_user, explanation, hypotheses = optimise(
            current_system=prompt.system,
            current_user=prompt.user,
            metrics={
                "precision": dev_metrics.precision,
                "recall":    dev_metrics.recall,
                "f1":        dev_metrics.f1,
            },
            patterns=patterns_dict,
            history=history,        # safe history — no labels
            target_p=TARGET_P,
            target_r=TARGET_R,
            deployment=DEPLOYMENT,
        )
        opt_secs = time.monotonic() - t0

        console.print(
            f"\n  [bold]Optimiser explanation:[/] [dim]({opt_secs:.0f}s)[/]"
            f"\n    {explanation}"
        )
        for hypothesis in hypotheses:
            console.print(f"    → {hypothesis}")

        # Register the new prompt and make it current for the next iteration
        new_prompt = store.register(new_system, new_user, explanation)
        store.set_current(new_prompt.version)
        console.print(f"\n  New prompt registered: [cyan]{new_prompt.version}[/]")

        # Update history with this iteration's safe summary (no labels)
        history.append({
            "version":   prompt.version,
            "precision": dev_metrics.precision,
            "recall":    dev_metrics.recall,
            "f1":        dev_metrics.f1,
            "notes":     explanation,
        })

        # Finalise and save the log record for this iteration
        log_record.update({
            "explanation": explanation,
            "hypotheses":  hypotheses,
        })
        iteration_log.append(log_record)
        _save_log(iteration_log)
        _save_prompts(store.all_versions())

        iter_secs = time.monotonic() - iter_t0
        console.print(f"\n  [dim]Iteration {iteration} completed in {iter_secs:.0f}s[/]")


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_prompts(versions: list) -> None:
    """Append-style save of all prompt versions to results/prompts.json."""
    path = RESULTS / "prompts.json"
    path.write_text(json.dumps(
        [
            {
                "version": p.version,
                "notes":   p.notes,
                "system":  p.system,
                "user":    p.user,
            }
            for p in versions
        ],
        indent=2,
    ))


def _save_log(log: list[dict]) -> None:
    """Write the full iteration log to results/iterations.jsonl."""
    path = RESULTS / "iterations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in log))


def _save_winning_prompt(prompt) -> None:
    """
    Save the converged prompt to results/winning_prompt.json.

    This file is the deliverable — copy the system and user fields into
    your production system to deploy the optimised prompt.
    """
    path = RESULTS / "winning_prompt.json"
    path.write_text(json.dumps({
        "version": prompt.version,
        "notes":   prompt.notes,
        "system":  prompt.system,
        "user":    prompt.user,
    }, indent=2))
    console.print(f"\n  [green]Winning prompt saved →[/] [cyan]{path}[/]")


def _print_summary(log: list[dict]) -> None:
    """Print a formatted summary table of all iterations."""
    console.rule("Run Summary")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Iter",      style="dim", justify="right")
    table.add_column("Version",   justify="left")
    table.add_column("Dev P",     justify="right")
    table.add_column("Dev R",     justify="right")
    table.add_column("Dev F1",    justify="right")
    table.add_column("Canary F1", justify="right")
    table.add_column("Status",    justify="left")

    for record in log:
        status = "[green]CONVERGED[/]" if record.get("converged") else ""
        table.add_row(
            str(record["iteration"]),
            record["version"],
            f"{record['precision']:.3f}",
            f"{record['recall']:.3f}",
            f"{record['f1']:.3f}",
            f"{record['canary_f1']:.3f}",
            status,
        )

    console.print(table)

    # If converged, show the test metrics prominently
    final = log[-1]
    if final.get("converged"):
        console.print(
            f"\n  [bold]Final test metrics:[/]  "
            f"P={final['test_precision']:.3f}  "
            f"R={final['test_recall']:.3f}  "
            f"F1={final['test_f1']:.3f}"
        )
        if final.get("overfitting_warnings"):
            for w in final["overfitting_warnings"]:
                console.print(f"  [yellow]⚠  {w}[/]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pass --fresh to discard previous results and start over.
    # Otherwise the pipeline resumes from the last incomplete run.
    if "--fresh" in sys.argv:
        import shutil
        if RESULTS.exists():
            shutil.rmtree(RESULTS)
        console.print("[dim]Previous results cleared.[/]\n")
    run(start_version="v1")
"""
Build a binary relevance dataset from Amazon product reviews.

Task: "Is this review complaining about battery life or power drain?"

Uses the `amazon_polarity` HuggingFace dataset (real reviews, CC licensed)
and auto-labels relevance using heuristics:

  Label 1 (relevant) — negative review with battery/power keywords.
  Label 0 (not relevant) — positive reviews, or negative reviews that don't
                           mention battery/power.

Ambiguity is baked in by mixing:
  - Obvious complaints ("battery died in 2 months")
  - Battery mentions in positive reviews ("battery lasts forever!")
  - Negative reviews mentioning battery but complaining about something else
  - Negative reviews using power/charge language without "battery"

Run once:
    uv run python data/prepare_reviews.py
"""

from __future__ import annotations

import json
import pathlib
import random
import re

from datasets import load_dataset

ROOT = pathlib.Path(__file__).parent

# ── Task definition ──────────────────────────────────────────────────────────
QUERY = "Is this review complaining about battery life, charging, or power drain?"

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ── Keyword heuristics ───────────────────────────────────────────────────────
# A review "mentions battery-related terms" if any of these appear.
# Using word boundaries to avoid false matches (e.g. "power" in "powerful").
_BATTERY_PATTERN = re.compile(
    r"\b("
    r"battery|batteries|"
    r"charge|charging|charger|chargers|"
    r"recharge|rechargeable|"
    r"power(?!ful)|"              # "power" but not "powerful"
    r"drain|drains|drained|draining|"
    r"die|dies|died|dying|"       # "battery dies"
    r"dead"                        # "dead battery"
    r")\b",
    re.IGNORECASE,
)


def _mentions_battery(text: str) -> bool:
    """True if the review mentions any battery/power-related term."""
    return bool(_BATTERY_PATTERN.search(text))


# ── Main ─────────────────────────────────────────────────────────────────────

def _build_split(
    rows: list[dict],
    out_dir: pathlib.Path,
) -> None:
    """Write input.jsonl and gold.jsonl for one split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    input_lines: list[str] = []
    gold_lines: list[str] = []
    for idx, row in enumerate(rows):
        input_lines.append(json.dumps({
            "query":    QUERY,
            "document": row["document"],
        }))
        gold_lines.append(json.dumps({
            "idx":   idx,
            "label": row["label"],
        }))

    (out_dir / "input.jsonl").write_text("\n".join(input_lines), encoding="utf-8")
    (out_dir / "gold.jsonl").write_text("\n".join(gold_lines), encoding="utf-8")

    relevant = sum(1 for r in rows if r["label"] == 1)
    print(
        f"  {out_dir.name}: {relevant} relevant + "
        f"{len(rows) - relevant} not relevant = {len(rows)} total"
    )


def main() -> None:
    target_total = 200

    print("Loading amazon_polarity dataset...")
    # amazon_polarity: title + content, label 0=negative / 1=positive
    ds = load_dataset("amazon_polarity", split="train", streaming=True)

    relevant_rows: list[dict] = []     # label=1: negative review mentioning battery
    not_relevant_rows: list[dict] = [] # label=0: everything else

    # Over-sample by category to ensure a diverse mix.
    # We want roughly 50/50 balance for dev/canary/test.
    target_relevant     = target_total // 2
    target_not_relevant = target_total - target_relevant

    # Hard cases to include specifically (cap each to avoid flooding):
    pos_mentioning_battery: list[dict] = []   # positive review with battery
    neg_without_battery:    list[dict] = []   # negative review without battery
    cap_hard_each = 20

    print("Sampling reviews...")
    for example in ds:
        # Combine title + content — Amazon reviews have both
        text = f"{example['title']}. {example['content']}".strip()
        is_negative = example["label"] == 0   # 0=negative in amazon_polarity
        mentions_battery = _mentions_battery(text)

        if is_negative and mentions_battery:
            # Label 1: negative review complaining with battery keywords
            if len(relevant_rows) < target_relevant:
                relevant_rows.append({"document": text, "label": 1})
        elif not is_negative and mentions_battery:
            # Hard case: positive review mentioning battery (tricky for model)
            if len(pos_mentioning_battery) < cap_hard_each:
                pos_mentioning_battery.append({"document": text, "label": 0})
        elif is_negative and not mentions_battery:
            # Hard case: negative review with no battery mention
            if len(neg_without_battery) < cap_hard_each:
                neg_without_battery.append({"document": text, "label": 0})

        # Stop once we have enough of everything
        if (
            len(relevant_rows) >= target_relevant
            and len(pos_mentioning_battery) >= cap_hard_each
            and len(neg_without_battery) >= cap_hard_each
        ):
            break

    # Fill the "not relevant" bucket: hard cases + random positives
    not_relevant_rows.extend(pos_mentioning_battery)
    not_relevant_rows.extend(neg_without_battery)

    # Top up with more positive reviews to hit target count
    remaining_needed = target_not_relevant - len(not_relevant_rows)
    if remaining_needed > 0:
        ds_positive = load_dataset("amazon_polarity", split="train", streaming=True)
        positive_top_ups: list[dict] = []
        for example in ds_positive:
            if example["label"] == 1:  # positive
                text = f"{example['title']}. {example['content']}".strip()
                if not _mentions_battery(text):
                    positive_top_ups.append({"document": text, "label": 0})
                    if len(positive_top_ups) >= remaining_needed:
                        break
        not_relevant_rows.extend(positive_top_ups)

    print(f"  Relevant     : {len(relevant_rows)}")
    print(f"  Not relevant : {len(not_relevant_rows)}")
    print(
        f"    including {len(pos_mentioning_battery)} hard (positive w/ battery)"
        f" + {len(neg_without_battery)} hard (negative w/o battery)"
    )

    random.shuffle(relevant_rows)
    random.shuffle(not_relevant_rows)

    # Split ratios: dev 60%, canary 15%, test 25%
    def _split(items: list, r: tuple[float, float, float]) -> tuple[list, list, list]:
        n = len(items)
        i = int(n * r[0])
        j = int(n * (r[0] + r[1]))
        return items[:i], items[i:j], items[j:]

    ratios = (0.60, 0.15, 0.25)
    rel_d, rel_c, rel_t = _split(relevant_rows, ratios)
    nr_d, nr_c, nr_t = _split(not_relevant_rows, ratios)

    # Combine and shuffle each split so labels are mixed in file order
    def _combine(rel: list, nr: list) -> list:
        combined = rel + nr
        random.shuffle(combined)
        return combined

    print("\nWriting splits...")
    _build_split(_combine(rel_d, nr_d), ROOT / "dev")
    _build_split(_combine(rel_c, nr_c), ROOT / "canary")
    _build_split(_combine(rel_t, nr_t), ROOT / "test")

    total = len(relevant_rows) + len(not_relevant_rows)
    print(f"\nDone — {total} reviews across 3 splits in {ROOT}/")


if __name__ == "__main__":
    main()

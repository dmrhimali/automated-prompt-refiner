# Automated Prompt Refiner

Iteratively optimises a classification prompt using an LLM-in-the-loop.
The pipeline runs `classifier → evaluator → optimiser` until precision
and recall targets are met, then validates on a held-out test split.

Built in collaboration with [Claude Code](https://claude.ai/code) (Anthropic).

## Prerequisites

This pipeline calls an Azure OpenAI model (gpt-5-nano by default). You
need an Azure AI Services resource with a deployed model before running.

**Option A: Provision with Terraform**

Use [dmrhimali/terraform-azure-infra](https://github.com/dmrhimali/terraform-azure-infra)
to set up the Azure infrastructure. Follow the README in that repo to
configure your Azure subscription, then run:

```sh
terraform plan
terraform apply
```

This creates the AI Services resource and deploys gpt-5-nano. The
endpoint and API key will be available in the Terraform outputs.

**Option B: Manual setup**

1. Create an Azure AI Services resource in the Azure portal.
2. Deploy a `gpt-5-nano` model (Global Standard SKU, capacity >= 30K TPM).
3. Copy the endpoint URL and API key from the resource's **Keys and Endpoint** page.

## How to run

### 1. Install

```sh
uv venv && uv sync
```

### 2. Credentials

```sh
cp .env.example .env
# fill AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
# (from Terraform outputs or the Azure portal)
```

### 3. Generate dataset (once)

```sh
uv run python data/prepare_reviews.py
```

Downloads the `amazon_polarity` dataset from HuggingFace, samples 200
real product reviews, auto-labels them for the task *"Is this review
complaining about battery life, charging, or power drain?"*, and writes
`input.jsonl` + `gold.jsonl` into `data/dev/`, `data/canary/`, and
`data/test/`.

The sample deliberately includes hard cases — positive reviews that
mention battery (to stress precision) and negative reviews that don't
(to stress recall) — so the pipeline has real failure patterns to work
with, not just obvious ones.

### 4. Run the loop

```sh
uv run python main.py
```

The loop iterates until `TARGET_PRECISION` and `TARGET_RECALL` are both
met (default 0.82 / 0.78), then runs the final gate on the held-out test
split. Every iteration is logged to `results/iterations.jsonl` with the
optimiser's explanation of what changed and why.


### Sample Run:

```sh
uv run python main.py --fresh
Previous results cleared.

─────────────────────────────────────────────────────────────────────────────────────────────────── Prompt Refinement Pipeline ────────────────────────────────────────────────────────────────────────────────────────────────────
Model: gpt-5-nano  |  Target P≥0.82  R≥0.75  |  Max 8 iterations  |  Starting from v1

─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 1  ·  v1 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.750  F1=0.857  TP=24 FP=0 FN=8 TN=28  (283s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.900  R=0.600  F1=0.720  TP=9 FP=1 FN=6 TN=14  (129s)

  ✓ Targets met on sample. Re-running full dev to confirm...
    120/120 classified (dev-full)
        Full dev: P=0.976  R=0.683  F1=0.804  TP=41 FP=1 FN=19 TN=59
  Targets not met on full dev — continuing optimisation.
  [4/4] Running optimiser...

  Optimiser explanation: (22s)
    What changed and why: (1) Strengthened the relevance criterion to require substantive alignment with the core topic, not just keyword overlaps, to reduce false positives from surface terms. (2) Expanded acceptable signals  
to include synonyms and closely related subtopics, enabling better recognition of conceptually related content and improving recall. (3) Added guidance to treat indirect yet meaningful coverage (e.g., discussions of
performance, usability, or implications related to the topic) as relevant, addressing cases where the document informs the query without explicit keyword matches. (4) Clarified handling of off-topic documents (e.g.,
entertainment or unrelated technical content) to avoid incorrect positives. (5) Kept the exact output format requirements to ensure parsable JSON while guiding the model toward concise, evidence-based reasons.
    → Recall will improve as the model accepts indirect and paraphrased coverage that informs the query topic, reducing false negatives.
    → False positives due to surface keyword overlap will decrease because the model now requires substantive coverage of the core concept rather than mere mentions.
    → The rubric will better handle borderline cases by rewarding relevance when related performance aspects meaningfully affect the query outcome.

  New prompt registered: v3

  Iteration 1 completed in 16m 11s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 2  ·  v3 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=0.944  R=0.531  F1=0.680  TP=17 FP=1 FN=15 TN=27  (260s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.900  R=0.600  F1=0.720  TP=9 FP=1 FN=6 TN=14  (121s)
  [4/4] Running optimiser...

  Optimiser explanation: (30s)
    What changed and why: This iteration broadens relevance criteria to include substantive direct coverage and meaningful indirect links to the core topic, reducing over-reliance on surface keyword matches while maintaining a 
clear standard for non-relevant content. It explicitly allows discussion of related performance or outcome implications (e.g., usability, reliability, cost, lifecycle) as signals of relevance when they bear on the topic, which 
improves recall. It preserves the one-sentence justification requirement but reframes it to cite the strongest direct or indirect evidence, helping avoid false positives from incidental mentions.
    → Recall will improve as the model recognizes indirect relevance (e.g., performance implications, user impact) that informs the topic, not just explicit topic terms.
    → Precision may dip slightly due to broader signals, but false positives from surface keyword overlaps should decrease because substantive alignment is still required.
    → The model will better handle synonyms and related subtopics by using concept-centric signals (direct discussion, evidence, outcomes) rather than rigid keyword checks.
    → Handling of broadly related but non-informative documents (e.g., entertainment or unrelated specs) will remain robust due to the requirement of demonstrable alignment with the core topic.

  New prompt registered: v4

  Iteration 2 completed in 6m 50s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 3  ·  v4 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.625  F1=0.769  TP=20 FP=0 FN=12 TN=28  (327s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.889  R=0.533  F1=0.667  TP=8 FP=1 FN=7 TN=14  (152s)
  [4/4] Running optimiser...

  Optimiser explanation: (19s)
    What changed and why: (1) Expanded relevance signals to explicitly include meaningful indirect coverage (e.g., performance, usability, lifecycle, user impact) so partially related content can count, addressing the recall   
gap where documents discuss the topic through outcomes or implications rather than direct description. (2) Allowed synonyms, paraphrase, and related terminology to map to the core topic, improving robustness to lexical
variation and unseen wording. (3) Retained a strict one-sentence reason but clarified that the sentence should cite the strongest evidence (direct or indirect) rather than relying on surface terms. (4) Emphasized careful       
handling of tangential or off-topic documents to reduce incidental positives while not punishing documents with plausible indirect relevance. (5) Kept the exact JSON output format to ensure parser compatibility while guiding   
concise justification.
    → Recall is expected to improve as the model can capture indirect relevance signals (outcomes, implications, performance aspects) that were previously ignored.
    → Precision may drop slightly due to broader criteria, but the overall F1 should improve if the added signals correctly identify borderline cases.
    → The model will better handle unseen wording through synonym/ paraphrase signals, reducing false negatives on conceptually related documents.
    → The single-sentence rationale will remain concise and evidence-based, but may reference a broader set of signals (direct or indirect) as the strongest evidence.
    → The approach should generalize across topics where core concepts have measurable or impact-oriented implications, not just literal mentions.

  New prompt registered: v5

  Iteration 3 completed in 8m 19s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 4  ·  v5 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.700  F1=0.824  TP=21 FP=0 FN=9 TN=30  (269s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.900  R=0.600  F1=0.720  TP=9 FP=1 FN=6 TN=14  (133s)
  [4/4] Running optimiser...

  Optimiser explanation: (21s)
    What changed: (1) broadened relevance signals to explicitly include meaningful indirect coverage (outcomes, implications, performance, usability, lifecycle, user impact) and allowed synonyms/paraphrase to map to the core   
topic, targeting recall gaps without sacrificing precision. (2) Kept a single-sentence justification but clarified it must cite the strongest evidence, whether direct or indirect. (3) Reinforced handling of off-topic or        
tangential content to avoid incorrect positives while still recognizing partial relevance when the content meaningfully informs the topic. (4) Emphasized evaluation signals over surface keywords to improve generalization to    
unseen data. (5) Maintained the exact JSON output format to ensure parsers remain compatible.
    → Inclusion of indirect yet meaningful coverage will increase recall by capturing documents that discuss outcomes or implications related to the core topic.
    → Allowing synonyms and paraphrase will improve robustness to lexical variation and unseen wording, reducing false negatives.
    → Clear emphasis on evidence-based one-sentence justification will maintain interpretability while supporting nuanced relevance judgments.
    → Better handling of tangential content will reduce spurious positives yet still allow partial signals to be counted when they meaningfully map to the topic.
    → Overall, the approach should raise recall closer to target with a controlled impact on precision.

  New prompt registered: v6

  Iteration 4 completed in 7m 2s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 5  ·  v6 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.686  F1=0.814  TP=24 FP=0 FN=11 TN=25  (199s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.900  R=0.600  F1=0.720  TP=9 FP=1 FN=6 TN=14  (95s)
  [4/4] Running optimiser...

  Optimiser explanation: (15s)
    I broadened what counts as relevance to include credible indirect coverage and domain-agnostic signals that map to the core topic, not just direct mentions. I clarified that partial, outcome-oriented, or
usability/safety/lifecycle information can establish relevance, while avoiding reliance on surface keyword matching. I preserved the one-sentence justification requirement but made it explicit that the sentence should cite the 
strongest direct or indirect evidence. I removed emphasis on exact topic phrasing and stressed evaluation of credible signals that would influence understanding or decision-making. I tightened the JSON contract so the output   
remains parseable and consistent.
    → Recall should improve as indirect and partial coverage signals are accepted as relevant, reducing false negatives.
    → Precision should remain high because only credible, topic-related indirect signals are considered relevant, not incidental mentions.
    → Generalization to unseen data should improve due to reliance on concept-level signals (outcomes, usability, lifecycle) rather than surface terms.
    → The single-sentence rationale remains informative and concise, aiding stable evaluation across domains.

  New prompt registered: v7

  Iteration 5 completed in 5m 9s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 6  ·  v7 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=0.955  R=0.636  F1=0.764  TP=21 FP=1 FN=12 TN=26  (262s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.900  R=0.600  F1=0.720  TP=9 FP=1 FN=6 TN=14  (134s)
  [4/4] Running optimiser...

  Optimiser explanation: (617s)
    What I changed: (1) Narrowed the topic to battery-centric content, ensuring relevance signals must pertain to battery life, charging performance, or power-related outcomes. (2) Clarified substantive vs. surface signals by 
requiring explicit battery-related discussion or credible indirect coverage that maps to battery implications. (3) Kept the one-sentence justification but mandated it cite the strongest evidence (direct or indirect). (4) 
Explicitly deprioritized incidental mentions that do not meaningfully inform battery understanding to reduce false positives. (5) Preserved the exact JSON output format for parser compatibility.
    → Increased recall by accepting indirect yet credible battery-related signals (e.g., usability or lifecycle implications tied to battery metrics) while avoiding non-battery tangential mentions.
    → Reduced false positives by requiring substantive battery coverage rather than surface keyword overlap.
    → Maintained parsing stability by keeping a single-sentence rationale and the exact JSON schema.
    → Potential risk: slightly stricter filters may miss edge cases where battery relevance is implied but not explicit; mitigated by allowing credible indirect signals.

  New prompt registered: v8

  Iteration 6 completed in 16m 52s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 7  ·  v8 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    60/60/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.750  F1=0.857  TP=21 FP=0 FN=7 TN=32  (235s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=0.875  R=0.467  F1=0.609  TP=7 FP=1 FN=8 TN=14  (113s)

  ✓ Targets met on sample. Re-running full dev to confirm...
    120/120 classified (dev-full)
        Full dev: P=1.000  R=0.667  F1=0.800  TP=40 FP=0 FN=20 TN=60
  Targets not met on full dev — continuing optimisation.
  [4/4] Running optimiser...

  Optimiser explanation: (12s)
    What changed and why:
- Broadened relevance signals beyond explicit battery terms to include credible indirect coverage that maps to battery implications (e.g., usability, longevity, safety, or real-world outcomes). This addresses failures where    
documents discuss battery-like effects without naming batteries directly.
- Kept a single-sentence justification to preserve clarity and comparability, but clarified that the sentence must cite the strongest direct or indirect evidence.
- Emphasized evaluation of evidence quality over surface keyword matches to improve generalization to unseen data and reduce false negatives caused by paraphrased or indirect coverage.
- Maintained exactly the required JSON output contract to ensure parser compatibility.
- Removed overly strict confinement to battery-centric topics, allowing broader, but credible, signals to capture partial or contextual relevance without sacrificing precision.
    → Recall will improve as the classifier now recognizes credible indirect signals that map to battery implications (e.g., device runtime, charging reliability, heat generation) even when the word 'battery' is absent.
    → Precision should remain stable or improve because the indirect signals now require a credible mapping to battery implications rather than surface keyword matches.
    → The model will better handle partially relevant documents (e.g., device reviews or lifecycle discussions) by detecting plausible battery-related consequences and explicitly linking them in the justification.
    → Generalization to unseen data will improve due to the shift from keyword dependence to signal-based evaluation of credibility and relevance.

  New prompt registered: v9

  Iteration 7 completed in 13m 41s
─────────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 8  ·  v9 ────────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split...
    120/120 classified (dev)
  [1/4] Classifying dev split...
    120/120 classified (dev)
    120/120 classified (dev)
  [2/4] Dev:    P=1.000  R=0.250  F1=0.400  TP=15 FP=0 FN=45 TN=60  (882s)
  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=1.000  R=0.267  F1=0.421  TP=4 FP=0 FN=11 TN=15  (234s)

  [3/4] Classifying canary split...
    30/30 classified (canary)
        Canary: P=1.000  R=0.267  F1=0.421  TP=4 FP=0 FN=11 TN=15  (234s)

    30/30 classified (canary)
        Canary: P=1.000  R=0.267  F1=0.421  TP=4 FP=0 FN=11 TN=15  (234s)

        Canary: P=1.000  R=0.267  F1=0.421  TP=4 FP=0 FN=11 TN=15  (234s)


  Max iterations (8) reached without meeting targets.

  Best iteration: #4 (v5) — P=1.000 R=0.700 F1=0.824

  Winning prompt saved → C:\Users\RasanjaleeDissanayak\TestProjects\automated-prompt-refiner\results\winning_prompt.json
─────────────────────────────────────────────────────────────────────────────────────────────────────────── Run Summary ───────────────────────────────────────────────────────────────────────────────────────────────────────────
┏━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Iter ┃ Version ┃ Dev P ┃ Dev R ┃ Dev F1 ┃ Canary F1 ┃ Status ┃
┡━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│    1 │ v1      │ 0.976 │ 0.683 │  0.804 │     0.720 │        │
│    2 │ v3      │ 0.944 │ 0.531 │  0.680 │     0.720 │        │
│    3 │ v4      │ 1.000 │ 0.625 │  0.769 │     0.667 │        │
│    4 │ v5      │ 1.000 │ 0.700 │  0.824 │     0.720 │ BEST   │
│    5 │ v6      │ 1.000 │ 0.686 │  0.814 │     0.720 │        │
│    6 │ v7      │ 0.955 │ 0.636 │  0.764 │     0.720 │        │
│    7 │ v8      │ 1.000 │ 0.667 │  0.800 │     0.609 │        │
│    8 │ v9      │ 1.000 │ 0.250 │  0.400 │     0.421 │        │
└──────┴─────────┴───────┴───────┴────────┴───────────┴────────┘

  Total runtime: 1h 32m 44s
```

**Why did recall drop after v5?**

In this run `TARGET_RECALL=0.75` but the peak recall was **0.700** at v5.
Precision hit 1.000 immediately and stayed there, so the only way to
reach the target was to push recall higher. Each subsequent iteration
the optimiser saw the same signal — "too many false negatives" — and
tried to fix it, but its instincts are to **tighten** criteria rather
than loosen them. Tightening can't raise recall; it can only lower it.

The collapse at v9 (R=0.250) is a classic late-iteration failure: the
optimiser added enough conservative rules that the classifier started
rejecting almost everything. The best-of-run selection ignored this
collapse and saved v5 as the winner — the right outcome despite the
drift.

If your data has a natural recall ceiling below your `TARGET_RECALL`,
this pattern will repeat on every run.

**How to get better convergence:**

1. **Calibrate the target to the peak.** Run once with a generous cap
   (`MAX_ITERATIONS=8`), observe the peak F1, then set
   `TARGET_PRECISION` / `TARGET_RECALL` just below that. A realistic
   target converges on iteration 2-3; an unrealistic one thrashes for 8.
2. **Upgrade the model.** Small reasoning models (`gpt-5-nano`) drift
   more. Try `gpt-5-mini` — same pipeline, typically +0.05 to +0.10 F1
   at ~5× the cost (still cheap overall).
3. **Start from v2, not v1.** v2 already has curated criteria, so the
   optimiser spends its iterations refining edges instead of inventing
   the prompt from scratch. Change `run(start_version="v1")` at the
   bottom of [main.py](main.py).
4. **Inspect the false negatives.** Open `results/{best_version}_dev.jsonl`
   and look at the rows labelled `0` that had gold `1`. If the model's
   reasoning is sound (e.g. "review is about keyboard, mentions battery
   in passing"), the gold label may be wrong. Fix labels, regenerate data.
5. **Reduce label noise at source.** The `data/prepare_reviews.py`
   auto-labels using keyword heuristics. A few percent of labels are
   likely wrong, which caps the achievable F1 regardless of prompt.
   Hand-review a sample of 30-50 labels and correct them.
6. **Shorten the iteration cap.** If the optimiser drifts after iter 5,
   set `MAX_ITERATIONS=5`. Less drift means a cleaner audit trail and
   a higher winning F1 on average.
7. **Let the optimiser rewrite from scratch.** If the prompt is bloating
   with each iteration (check `results/prompts.json`), the rules in
   [agents/optimiser.py](agents/optimiser.py) already tell it to
   simplify — but you can also delete `results/` and restart from v1
   mid-run to force a rewrite.

See the **Known limitations** section below for the full discussion.

## Data format

Each data split contains two files:

**`input.jsonl`** — one line per document, seen by the classifier only:

```json
{"query": "...", "document": "..."}
```

| Field | Description |
|-------|-------------|
| `query` | The search brief or legal issue being reviewed. Same for every row in a single-issue review. |
| `document` | The text being classified (email, contract clause, memo, etc.). |

**`gold.jsonl`** — one line per document, seen by the evaluator only:

```json
{"idx": 0, "label": 1}
```

| Field | Description |
|-------|-------------|
| `idx` | Row position matching the corresponding line in `input.jsonl`. |
| `label` | `1` = relevant, `0` = not relevant. The ground truth. |

These are kept separate to enforce an **information boundary**: the
classifier never sees labels, so it must reason from the prompt criteria
alone. If it could see the answers, it would parrot them instead of
learning to generalise.

### Directory structure

```
data/
├── generate.py        # creates the synthetic documents for this example
├── dev/
│   ├── input.jsonl    # docs (mix of relevant + not relevant)
│   └── gold.jsonl     # answer key — labels only, classifier never sees this
├── canary/
│   ├── input.jsonl
│   └── gold.jsonl
└── test/
    ├── input.jsonl
    └── gold.jsonl
```

To use a different dataset, create a parallel folder (e.g. `data_medical/`)
with the same structure and point `DATA_DIR` at it:

```sh
DATA_DIR=data_medical uv run python main.py
```

## Agents and information flow

The pipeline is built from three agents. Each has a deliberate information
boundary — the things it **sees** determine its output, and the things it
**never sees** prevent it from cheating.

### Classifier  ([agents/classifier.py](agents/classifier.py))

**Job:** label each document as relevant or not relevant.

| | |
|---|---|
| **Sees** | system prompt (criteria) + query + document text |
| **Never sees** | gold labels, gold reasons, other documents' results, the test split |
| **Produces** | `{"label": 0|1, "confidence": 0.0-1.0, "reason": "..."}` per doc |

Why the boundary: if the classifier saw gold labels — even as examples —
it would confirm labels rather than reason from the criteria. That would
be label leakage, and the metrics would be meaningless.

### Evaluator  ([agents/evaluator.py](agents/evaluator.py))

**Job:** compare the classifier's output against the gold labels, compute
metrics, and distill the mistakes into aggregated failure patterns.

| | |
|---|---|
| **Sees** | classifier results + gold labels + gold reasons |
| **Produces (for the log)** | precision, recall, F1, TP/FP/FN/TN counts |
| **Produces (for the optimiser)** | aggregate failure patterns only — counts, abstract descriptions, and the model's own stated reasons for wrong calls |

Crucially, what the evaluator hands to the optimiser contains **no
document IDs, no raw gold labels, and no gold reasons**. It's a shaped
summary, not raw data.

### Optimiser  ([agents/optimiser.py](agents/optimiser.py))

**Job:** rewrite the classification prompt to fix the observed failure patterns.

| | |
|---|---|
| **Sees** | current prompt + metrics + failure patterns + last 3 iterations of history |
| **Never sees** | individual documents, gold labels, gold reasons, doc IDs |
| **Produces** | new system prompt, new user template, explanation, hypotheses |

Why the boundary: if the optimiser saw (document, gold_label) pairs, it
would write rules that match the surface features of those exact docs.
Scores would inflate on dev but collapse on the held-out test set —
classic prompt overfitting. By giving it only aggregate patterns
("23 false positives from keyword overlap"), we force it to reason
about general failure modes instead of memorising the eval set.

### Information flow

```
                    gold.jsonl  (labels — only the evaluator sees this)
                         │
                         ▼
    ┌─────────────┐  ┌────────────┐  ┌─────────────┐
    │  Classifier │─▶│  Evaluator │─▶│  Optimiser  │
    └─────────────┘  └────────────┘  └─────────────┘
          ▲             metrics +           │
          │             aggregated          │ new prompt
          │             patterns            │
          │                                 ▼
          └─────────────── loop ────────────┘
```

Each loop:

1. **Classifier** reads the current prompt + `input.jsonl` → writes
   per-doc predictions.
2. **Evaluator** joins predictions with `gold.jsonl` → writes metrics to
   the log and hands **only safe aggregates** to the optimiser.
3. **Optimiser** reads current prompt + aggregates + history → writes a
   new prompt.
4. New prompt replaces the current one → loop back to step 1.

### How convergence is reached

The pipeline stops at the first of these:

- **Targets met** (happy path): dev precision >= `TARGET_PRECISION`
  AND dev recall >= `TARGET_RECALL`. If dev was sampled, the pipeline
  re-runs full dev to confirm, then opens the held-out test split
  (the only time it ever reads `data/test/`) to verify generalisation.
  The converged prompt is saved as the winner.
- **Max iterations hit**: the pipeline picks the **iteration with the
  highest dev F1** across the run and saves that prompt as the winner —
  not the last iteration's prompt. This matters because the optimiser
  can drift past a good prompt (e.g. iteration 6 hits F1=0.86, then
  subsequent iterations regress to 0.70). Keeping the best prompt
  ensures you don't lose the best result the run produced.

Each iteration also classifies the **canary split** with the current
prompt but never feeds canary results to the optimiser. If dev F1 rises
while canary F1 stagnates or falls, the pipeline flags possible
overfitting to dev-specific patterns.

In the run summary table the selected winner is marked:

- `CONVERGED` — targets met, this is the iteration the pipeline converged on
- `BEST` — max iterations hit, this was the highest-F1 iteration overall

## Data splits

| Split | Purpose |
|-------|---------|
| **dev** | Main working set. Metrics and failure patterns feed the optimiser. |
| **canary** | Overfitting detector. Never used for optimisation. |
| **test** | Held-out final exam. Only opened once when dev targets are met. |

### Dev vs canary

The optimiser improves the prompt by learning from dev mistakes — it sees
aggregated failure patterns like "23 false positives where keyword overlap
was mistaken for relevancy". This means the prompt gets better **at dev**
each iteration, but there is a risk it learns patterns specific to the dev
documents rather than general relevancy rules.

The canary split guards against this. It is classified every iteration with
the same prompt, but its results are **never fed back to the optimiser**.
Because the optimiser has no knowledge of canary documents, canary F1
reflects how well the prompt generalises to unseen data.

What to watch for:

- **Dev F1 rises, canary F1 also rises** — the prompt is learning general
  rules. This is the healthy case.
- **Dev F1 rises, canary F1 drops** — the prompt is overfitting to
  dev-specific surface patterns. The optimiser needs different guidance or
  the dev set needs more diversity.

## Dev sampling

Early iterations classify a **random sample** of the dev set (default 60
docs) to save time. The final iteration always runs the full set.

If targets are met on a sample, the pipeline **re-runs full dev** to
confirm before opening the test gate.

```sh
DEV_SAMPLE_SIZE=80 uv run python main.py   # sample 80 instead of 60
DEV_SAMPLE_SIZE=240 uv run python main.py  # disable sampling
```

## Resume and fresh start

**Resume** — just run `uv run python main.py`. If
`results/iterations.jsonl` and `results/prompts.json` exist from an
incomplete run, the pipeline restores state and continues from the next
iteration.

**Fresh start** — run `uv run python main.py --fresh` to wipe `results/`
and start from v1.

## Output files

| File | Description |
|------|-------------|
| `results/prompts.json` | All prompt versions (v1, v2, v3, ...) saved after every iteration. Used for resuming. |
| `results/winning_prompt.json` | The best prompt from the run — either the converged prompt, or (if max iterations was hit) the iteration with the highest dev F1. Production-ready. |
| `results/iterations.jsonl` | One JSON record per iteration — metrics, explanation, hypotheses. |
| `results/{version}_dev.jsonl` | Raw classifier output per doc for each version. |

## Prompt progression

Each iteration the optimiser rewrites the prompt to fix observed failure
patterns. Below is an example progression from a run on the legal dataset:

### Iteration 1 — v1 (baseline)

**System prompt:** *"You are a document review attorney. For each document,
decide if it is relevant to the legal matter."*

No criteria given — the model guesses what "relevant" means.

| Dev P | Dev R | Dev F1 | Canary F1 |
|-------|-------|--------|-----------|
| 0.750 | 0.947 | 0.837  | 0.909     |

**Problems:** High recall but low precision — the vague prompt labels too
many documents as relevant (false positives from keyword overlap).

### Iteration 2 — v3 (optimiser rewrite)

The optimiser identified the false-positive pattern and rewrote the prompt to:
- Require **substantive topic coverage**, not just keyword overlap
- Add **indirect relevance** criteria for docs that don't name parties explicitly
- Allow **domain-specific synonyms** to avoid false negatives

| Dev P | Dev R | Dev F1 | Canary F1 | Status |
|-------|-------|--------|-----------|--------|
| 0.947 | 0.947 | 0.947  | 1.000     | CONVERGED |

**Test gate:** P=0.875 R=0.778 F1=0.824

Precision jumped from 0.750 to 0.947 while recall held steady. The
optimiser solved the core issue in one iteration.

### What to look for

- **`results/prompts.json`** — full text of every prompt version, showing
  exactly what changed between iterations.
- **`results/iterations.jsonl`** — metrics per iteration, optimiser
  explanation, and hypotheses for each change.
- **Overfitting warnings** — if the dev/test F1 gap exceeds 0.05, the
  pipeline flags it. In the run above, the gap was 0.124, suggesting the
  prompt picked up some dev-specific patterns.

## Known limitations

### Optimiser drift and prompt un-learning

Iterative LLM-driven prompt optimisation is not monotonically improving.
In practice you will often see a progression like this:

| Iter | Dev P | Dev R | Dev F1 | Note |
|------|-------|-------|--------|------|
| 1 | 1.000 | 0.733 | 0.846 | Baseline already strict |
| 5 | 1.000 | 0.758 | 0.862 | Peak — optimiser found a good balance |
| 6 | 1.000 | 0.630 | 0.773 | Regression |
| 8 | 0.976 | 0.542 | 0.703 | Worse than baseline |

Common drift patterns:

- **Recall decay after precision saturates.** Once precision hits 1.0,
  the optimiser keeps "addressing false negatives" by adding more rules
  — but most rules make the classifier *more* conservative, which
  reduces recall. The optimiser's system prompt (in
  [agents/optimiser.py](agents/optimiser.py)) now explicitly warns
  against this, but small reasoning models still slip into the pattern.
- **Prompt bloat.** Each iteration layers new constraints on top of the
  previous prompt. By iteration 8, the prompt may be 3× longer than the
  baseline with contradictory instructions, confusing the classifier.
- **Un-learning good changes.** The optimiser sees only the last 3
  iterations of history and has no mechanism to say "iteration 5 was
  best — revert and try a different direction." It treats each iteration
  as a fresh attempt.

### Why this is fundamentally hard

These are limitations of using an LLM as the optimiser, not bugs in the
pipeline. An LLM has no gradient, no direct feedback loop, no memory of
why a prompt worked. It's doing supervised prompt engineering with only
aggregate metrics and a natural-language description of the failures.
Small models (like `gpt-5-nano`) are especially prone to drift because
their reasoning is shallower.

### Mitigations in this pipeline

- **Best-of-run winner selection** — if max iterations is hit, the
  winning prompt is the iteration with the highest dev F1, not the last
  iteration. Prevents a drifted late iteration from being deployed.
- **Trade-off rules in the optimiser prompt** — explicit guidance to
  loosen (not tighten) when precision is saturated, and vice versa.
- **Anti-bloat rule** — the optimiser is told to simplify or rewrite
  from scratch rather than layering more rules when metrics regress.
- **Canary split** — classified every iteration but never fed back, so
  you can spot dev-set overfitting independently.

### When to stop

If your runs show the classic "peak then decay" pattern like above,
consider:

1. **Lowering targets** to whatever the peak F1 was — your data or
   model may not support higher.
2. **Using a stronger model** (e.g. `gpt-5-mini` over `gpt-5-nano`).
3. **Fixing gold labels** — if the ceiling is stubborn, some labels may
   be genuinely ambiguous or wrong.
4. **Starting from v2 with curated criteria** instead of the minimal v1,
   so the optimiser has fewer dimensions to explore.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5-nano` | Azure model deployment name |
| `TARGET_PRECISION` | `0.82` | Minimum precision to converge |
| `TARGET_RECALL` | `0.78` | Minimum recall to converge |
| `MAX_ITERATIONS` | `4` | Hard cap on iterations |
| `DEV_SAMPLE_SIZE` | `60` | Docs to sample in early iterations |
| `DATA_DIR` | `data` | Dataset directory |
| `API_CALL_DELAY` | `1.0` | Seconds between API calls (rate limit control) |

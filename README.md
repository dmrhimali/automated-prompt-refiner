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
uv run python data_legal/generate.py
```

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

─────────────────────────────────────────────────────────────────────────────────────────────── Prompt Refinement Pipeline ────────────────────────────────────────────────────────────────────────────────────────────────
Model: gpt-5-nano  |  Target P≥0.82  R≥0.78  |  Max 8 iterations  |  Starting from v1

─────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 1  ·  v1 ────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    40/40 classified (dev)
  [2/4] Dev:    P=0.750  R=0.947  F1=0.837  TP=18 FP=6 FN=1 TN=15  (255s)
  [3/4] Classifying canary split...
    10/10 classified (canary)
        Canary: P=0.833  R=1.000  F1=0.909  TP=5 FP=1 FN=0 TN=4  (58s)
  [4/4] Running optimiser...

  Optimiser explanation: (21s)
    Changes address failures by (1) requiring substantive topic coverage rather than surface keyword overlap to reduce false positives, (2) adding a strong indirect relevance category to capture documents that influence
the matter without explicit naming, (3) allowing domain-specific synonyms when they map to the core issue to reduce false negatives, (4) clarifying that relevance must be justified with a concise one-sentence reason,
and (5) encouraging section-level evaluation in multi-section documents to avoid discarding relevant content.
    → Requiring substantive topic coverage will reduce false positives by eliminating labels based on incidental keyword overlaps.
    → Introducing a strong indirect relevance category will improve recall for documents that impact the matter but do not name the involved entities directly.
    → Allowing domain-specific synonyms will decrease false negatives when different terminology still maps to the same core issue.
    → Stipulating a one-sentence justification for each label will produce more consistent and interpretable reasons, aiding auditability.
    → Section-level evaluation will prevent loss of relevance when only parts of a document pertain to the query, improving overall classification accuracy.

  New prompt registered: v3

  Iteration 1 completed in 333s
─────────────────────────────────────────────────────────────────────────────────────────────────── Iteration 2  ·  v3 ────────────────────────────────────────────────────────────────────────────────────────────────────
  [1/4] Classifying dev split (sample=60)...
    40/40 classified (dev)
  [2/4] Dev:    P=0.944  R=0.895  F1=0.919  TP=17 FP=1 FN=2 TN=20  (223s)
  [3/4] Classifying canary split...
    10/10 classified (canary)
        Canary: P=1.000  R=1.000  F1=1.000  TP=5 FP=0 FN=0 TN=5  (52s)

  ✓ Targets met on sample. Re-running full dev to confirm...
    40/40 classified (dev-full)
        Full dev: P=0.947  R=0.947  F1=0.947  TP=18 FP=1 FN=1 TN=20

  ✓ Confirmed on full dev.
  Running held-out test gate...
    18/18 classified (test)

  TEST:   P=0.875  R=0.778  F1=0.824  TP=7 FP=1 FN=2 TN=8
  ⚠  Dev/test F1 gap = 0.124. The prompt may be overfitting to dev-set surface patterns. Check whether the optimiser is using doc-specific features.

  Winning prompt saved → C:\Users\RasanjaleeDissanayak\TestProjects\automated-prompt-refiner\results\winning_prompt.json
─────────────────────────────────────────────────────────────────────────────────────── Run Summary ───────────────────────────────────────────────────────────────────────────────────────
┏━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Iter ┃ Version ┃ Dev P ┃ Dev R ┃ Dev F1 ┃ Canary F1 ┃ Status    ┃
┡━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│    1 │ v1      │ 0.750 │ 0.947 │  0.837 │     0.909 │           │
│    2 │ v3      │ 0.947 │ 0.947 │  0.947 │     1.000 │ CONVERGED │
└──────┴─────────┴───────┴───────┴────────┴───────────┴───────────┘

  Final test metrics:  P=0.875  R=0.778  F1=0.824
  ⚠  Dev/test F1 gap = 0.124. The prompt may be overfitting to dev-set surface patterns. Check whether the optimiser is using doc-specific features.
```

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
data_legal/
├── generate.py        # creates the synthetic legal docs
├── dev/
│   ├── input.jsonl    # 30 docs (15 relevant + 15 not relevant)
│   └── gold.jsonl     # answer key — labels only, classifier never sees this
├── canary/
│   ├── input.jsonl    # 6 docs
│   └── gold.jsonl     # answer key
└── test/
    ├── input.jsonl    # 14 docs
    └── gold.jsonl     # answer key
```

Switch dataset via environment variable:

```sh
DATA_DIR=data uv run python main.py        # use original generic dataset
DATA_DIR=data_legal uv run python main.py  # use legal dataset (default)
```

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
| `results/winning_prompt.json` | The final converged prompt. Production-ready. |
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

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5-nano` | Azure model deployment name |
| `TARGET_PRECISION` | `0.82` | Minimum precision to converge |
| `TARGET_RECALL` | `0.78` | Minimum recall to converge |
| `MAX_ITERATIONS` | `3` | Hard cap on iterations |
| `DEV_SAMPLE_SIZE` | `60` | Docs to sample in early iterations |
| `DATA_DIR` | `data_legal` | Dataset directory |
| `API_CALL_DELAY` | `1.0` | Seconds between API calls (rate limit control) |

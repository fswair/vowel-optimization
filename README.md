# vowel-optimization

**GEPA-powered prompt optimization playground for vowel eval spec generation.**

This repository contains a research tool for optimizing vowel's `EVAL_SPEC_CONTEXT` prompt using [GEPA](https://github.com/GEPA-ai/GEPA) (Genetic Pareto). By iteratively generating evaluation specs and measuring their quality against ground-truth function implementations, it discovers improved prompts that produce better test cases.

---

## Overview

vowel generates YAML evaluation specs from function signatures and descriptions. The quality of these specs depends heavily on the system prompt (`EVAL_SPEC_CONTEXT`) used during generation. This tool:

1. **Evaluates** the current prompt against a set of reference functions
2. **Optimizes** the prompt via GEPA's evolutionary search
3. **Compares** baseline vs optimized performance

### Debug

Set LOGFIRE_ENABLED to true for enabling debugging and monitoring.

```
export LOGFIRE_ENABLED=true
```

*or*

```
echo "LOGFIRE_ENABLED=true" > .env
```

### How It Works

```
EVAL_SPEC_CONTEXT (prompt)
         ↓
   Generate YAML evals
         ↓
   Run against ground truth
         ↓
     Score (pass rate)
         ↓
   GEPA: propose improvements
         ↓
     [repeat until convergence]
```

Each iteration:
- Generates eval specs for reference functions (e.g., `json_encode`, `slugify`, `levenshtein`)
- Executes specs against original implementations
- Diagnoses failures (wrong expected values, invented raises, over-strict assertions)
- Feeds structured failure reports to a proposer LLM to suggest prompt refinements

---

## Installation

### From source

```bash
git clone <this-repo>
cd vowel-optimization
pip install -e .
```

### Dependencies

- Python 3.11+
- vowel
- GEPA
- pydantic-ai
- logfire (for telemetry)

---

## Quick Start

### 1. Evaluate Current Prompt

Test the default `EVAL_SPEC_CONTEXT` against all reference functions:

```bash
python -m vowel_optimization eval
```

**Output:**
```
============================================================
Evaluating [current] context (2453 chars)
============================================================

  json_encode... 88% (15/17)
  slugify... 92% (23/25)
  levenshtein... 100% (12/12)
  ...

------------------------------------------------------------
Average pass rate: 91%
Total: 143/157 cases passed

Failure breakdown:
  WRONG_EXPECTED: 8
  INVENTED_RAISES: 4
  FORMAT_MISMATCH: 2
```

### 2. Run Optimization

Use GEPA to improve the prompt over 50 metric evaluations:

```bash
python -m vowel_optimization optimize --max-calls 50
```

**What happens:**
- GEPA starts with `EVAL_SPEC_CONTEXT` as seed
- Evaluates on reference functions
- Proposes improved versions via LLM reflection
- Saves best candidate to `optimized_context.txt`

**Options:**
- `--max-calls N`: Maximum GEPA iterations (default: 50)
- `--output path/to/file.txt`: Save location for optimized prompt
- `--model MODEL`: Override eval model (default: `openrouter:google/gemini-3-flash-preview`)
- `--proposer-model MODEL`: Override proposer model
- `--seed-file path/to/seed.txt`: Start from a previously optimized prompt instead of default

**Example:**
```bash
# Start from previous best, run 100 more iterations
python -m vowel_optimization optimize \
  --max-calls 100 \
  --seed-file optimized_context.txt \
  --output optimized_context_v2.txt
```

### 3. Compare Results

Compare prompts against each other or against the default:

```bash
# Compare one file vs default EVAL_SPEC_CONTEXT
python -m vowel_optimization compare optimized_context.txt

# Compare two specific files
python -m vowel_optimization compare optimized_context.txt optimized_context_v3.txt

# Compare multiple files
python -m vowel_optimization compare v1.txt v2.txt v3.txt
```

**Output (1 file vs default):**
```
1. Current EVAL_SPEC_CONTEXT:
============================================================
Average pass rate: 91%
...

2. optimized_context.txt:
============================================================
Average pass rate: 96%
...

============================================================
Comparison:
  EVAL_SPEC_CONTEXT: 91%
  optimized_context.txt: 96%
  Δ: +5%
```

**Output (2+ files):**
```
1. optimized_context.txt:
============================================================
Average pass rate: 85%
...

2. optimized_context_v3.txt:
============================================================
Average pass rate: 99%
...

============================================================
Comparison:
  optimized_context.txt: 85%
  optimized_context_v3.txt: 99%
  Δ: +14%
```

### 4. Evaluate Custom Prompts

Test any saved prompt file:

```bash
python -m vowel_optimization eval --context-file my_experiments/prompt_v3.txt
```

---

## Changing Models

All models use the format `provider:model-name`. Default: `openrouter:google/gemini-3-flash-preview`

### Eval Model

Controls the LLM that generates eval specs during optimization:

```bash
python -m vowel_optimization eval --model openrouter:anthropic/claude-3.5-sonnet
```

```bash
python -m vowel_optimization optimize --model openai:gpt-4o
```

### Proposer Model

Controls the LLM that suggests prompt improvements (GEPA's reflection step):

```bash
python -m vowel_optimization optimize \
  --model openrouter:google/gemini-3-flash-preview \
  --proposer-model openrouter:anthropic/claude-3.5-sonnet
```

**Tip:** Use a stronger model for the proposer (e.g., Claude Sonnet) and a faster/cheaper model for eval generation (e.g., Gemini Flash).

---

## Project Structure

```
vowel_optimization/
├── .git/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── results/                   # Optimization outputs
└── src/
    └── vowel_optimization/
        ├── __init__.py
        ├── __main__.py        # CLI entry point
        ├── run_optimization.py # Main commands (eval, optimize, compare)
        ├── adapter.py         # GEPA adapter implementation
        ├── task.py            # Core: generate + score eval specs
        ├── functions.py       # Reference function wrappers
        └── definitions.py     # Ground truth implementations
```

### Key Files

- **run_optimization.py**: CLI commands and orchestration
- **adapter.py**: Bridges GEPA's optimization loop with vowel's eval generation
- **task.py**: Implements `generate_and_score()` — generates YAML, runs tests, diagnoses failures
- **definitions.py**: Reference functions (json_encode, slugify, levenshtein, etc.)

---

## How Optimization Works

### GEPA Adapter

The `VowelGEPAAdapter` implements three key methods:

1. **evaluate()**: Generate eval specs for each function → run → return pass rates
2. **make_reflective_dataset()**: Convert failures into structured feedback (e.g., "Expected value wrong for case X")
3. **propose_new_texts()**: Call proposer LLM with failure diagnostics → get improved prompt

### Failure Categories

task.py classifies failures into:

- **WRONG_EXPECTED**: Generated expected value doesn't match actual output
- **INVENTED_RAISES**: Spec expects exception but function returns normally
- **FORMAT_MISMATCH**: String formatting details wrong (separators, whitespace)
- **OVER_STRICT_ASSERTION**: Assertion too brittle (e.g., exact float equality)
- **BAD_INPUT**: Invalid input in generated test case

These drive prompt refinements like:
- "Add guidance to trace algorithm logic for expected values"
- "Only test raises for code paths that actually throw"
- "Use lenient type checking for bool/int compatibility"

---

## Telemetry

Uses [Logfire](https://logfire.dev/) for observability. Logs:
- Each evaluation run with scores and failure breakdown
- GEPA optimization progress (candidates, scores, best context)
- Individual function case results

Configure with `LOGFIRE_TOKEN` env var to enable cloud logging.

---

## Adding Reference Functions

Edit `definitions.py`:

```python
# definitions.py
def my_new_function(x: int) -> int:
    """Double the input."""
    return x * 2

# Add to FUNCTIONS dict at bottom
FUNCTIONS = {
    # ... existing functions ...
    "my_new_function": {
        "func": my_new_function,
        "description": "Doubles integer input",
    },
}
```

The optimizer will automatically include it in the next run.

---

## Examples

### Baseline Evaluation

```bash
$ python -m vowel_optimization eval

============================================================
Evaluating [current] context (2453 chars)
============================================================

  json_encode... 88% (15/17)
  slugify... 92% (23/25)
  levenshtein... 100% (12/12)

------------------------------------------------------------
Average pass rate: 91%
Total: 143/157 cases passed
```

### Full Optimization Run

```bash
$ python -m vowel_optimization optimize --max-calls 30

Starting GEPA prompt optimization...
  Eval model: openrouter:google/gemini-3-flash-preview
  Proposer model: openrouter:google/gemini-3-flash-preview
  Max metric calls: 30
  Seed: default EVAL_SPEC_CONTEXT (2453 chars)
============================================================

[GEPA progress bar with candidate evaluation]

============================================================
Optimization Complete!
============================================================
Best validation score: 96.18%
Context length: 3127 chars
Saved to: optimized_context.txt
```

### Iterative Refinement

```bash
# First round
python -m vowel_optimization optimize --max-calls 50 --output opt_v1.txt

# Second round starting from v1
python -m vowel_optimization optimize --max-calls 50 \
  --seed-file opt_v1.txt \
  --output opt_v2.txt

# Compare all versions
python -m vowel_optimization compare opt_v1.txt         # vs default
python -m vowel_optimization compare opt_v2.txt         # vs default
python -m vowel_optimization compare opt_v1.txt opt_v2.txt  # direct comparison
```

---

## License

MIT

---

## Related

- **[vowel](https://github.com/fswair/vowel)**: YAML-based evaluation framework
- **[GEPA](https://github.com/GEPA-ai/GEPA)**: Genetic Pareto for meta-optimization
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)**: Type-safe AI agent framework

---

## Reference

Used [@dmontagu](https://github.com/dmontagu)'s [pydantic-ai-GEPA-example](https://github.com/dmontagu/pydantic-ai-GEPA-example) as seed repository.

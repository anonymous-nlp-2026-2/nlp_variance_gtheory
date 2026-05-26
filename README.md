# Reproducibility Budgets for Open-Weight LLM Evaluation: A Generalizability Theory Study of 7–9B Instruction-Tuned Models

Code and data for applying Generalizability Theory (G-theory) to decompose variance in LLM benchmark evaluations. We use Henderson Method I to estimate variance components across seven evaluation facets (prompt template, random seed, numerical precision, sampling temperature, answer ordering, model, item) in a factorial experiment crossing eight open-weight 7–9B instruction-tuned models and five benchmarks (MMLU, ARC-Challenge, HellaSwag, GSM8K, MATH; 2.3M records).

**Key findings:**
- Item difficulty is the largest variance component (34–38% in multiple-choice)
- Model×item interaction is the second largest (30–36% in MC, 15–17% in free-form), exceeding all manipulated design factors by an order of magnitude
- A 3-prompt × 3-seed protocol achieves G ≥ 0.80 with 178 items versus 552 under single-condition protocols (3.1× reduction)
- Item-related variance concentrates in multiple-choice (68–72%) but disperses in free-form tasks (31–41%)

## Requirements

- Python 3.10+
- GPU with CUDA support (for vLLM inference)

```bash
pip install -r requirements.txt
```

## Quick Start

**1. Prepare data and design matrix**

```bash
python -m src.data.sample_mmlu --output data/mmlu_items.json
python -m src.data.few_shot --output data/few_shot_examples.json
python -m src.design.generate_design --output design_matrix.csv
```

**2. Run inference**

```bash
python -m src.inference.run_experiment \
    --design design_matrix.csv \
    --items data/mmlu_items.json \
    --few-shot data/few_shot_examples.json \
    --output results/all_results.jsonl \
    --gpu-id 0
```

**3. G-study: variance decomposition**

```bash
python -m src.analysis.variance_decomposition \
    --input results/all_results.jsonl \
    --output results/variance_components.json \
    --n-bootstrap 1000
```

**4. D-study: budget optimization**

```bash
python -m src.analysis.d_study \
    --input results/variance_components.json \
    --output results/d_study_results.json
```

The D-study calculator (`src/analysis/d_study.py`) computes generalizability coefficients for alternative designs and finds minimum replications needed to reach target reliability thresholds (G ≥ 0.80, 0.90, 0.95).

## Repository Structure

```
src/
  data/           Data loading and sampling
  design/         Experimental design matrix generation
  inference/      vLLM-based model inference
  analysis/       G-study, D-study, bootstrap CI, visualization
scripts/          Experiment scripts for reproducing paper results
data/             Benchmark items and few-shot examples (JSON)
paper/            LaTeX source
tests/            Unit tests
```

## License

MIT

# Generalizability Theory for LLM Evaluation Variance Decomposition

Code and paper source for anonymous submission to EMNLP 2026.

## Overview

This repository contains the implementation for applying Generalizability Theory (G-theory) to decompose variance in LLM benchmark evaluations. The framework uses Henderson Method I (Type III balanced ANOVA) to estimate variance components across multiple evaluation facets (prompt template, temperature, random seed, item ordering).

## Requirements

- Python 3.10+
- vLLM (for model inference with tensor parallelism)
- Key dependencies: numpy, scipy, pandas, transformers, datasets

Install:
```
pip install -r requirements.txt
```

## Repository Structure

- `src/` — Core library (data loading, experimental design, inference, analysis)
- `scripts/` — Experiment scripts (data collection, G-study, D-study, scale validation)
- `paper/` — LaTeX source for the paper
- `data/` — Benchmark item data (JSON)
- `tests/` — Unit tests

## Reproducing Experiments

See scripts/ for detailed experiment configurations.

## License

MIT

# Reflective Translation for Low-Resource Machine Translation

This repository contains code and analysis for **Reflective Translation**, a reflection-guided prompting framework that improves machine translation quality for low-resource languages **without fine-tuning**.

We evaluate English → isiZulu and English → isiXhosa translation using GPT-3.5 and Claude Haiku 3.5, reporting consistent BLEU and COMET improvements across reflection, prompting strategies, and confidence thresholds.

This repository supports **full reproduction of all experiments, tables, and figures** in the associated paper.

---

## Paper

**Reflective Translation: Enhancing Low-Resource Machine Translation through Self-Reflection**  
Nicholas Cheng  
arXiv:2601.19871 (2026)  
📄 https://arxiv.org/abs/2601.19871

```bibtex
@article{cheng2026reflective,
  title={Reflective Translation: Enhancing Low-Resource Machine Translation through Self-Reflection},
  author={Cheng, Nicholas},
  journal={arXiv preprint arXiv:2601.19871},
  year={2026}
}
```
## Overview

Reflective Translation introduces a lightweight, inference-time framework in which a language model:

Produces an initial translation

Generates a structured self-critique identifying translation errors

Produces a revised translation guided by masked reflective feedback

This approach improves semantic fidelity and robustness in low-resource settings without additional training data or parameter updates.

## Datasets

All datasets are loaded programmatically from HuggingFace at runtime.

OPUS-100 (English–isiZulu, English–isiXhosa)
https://huggingface.co/datasets/Helsinki-NLP/opus-100

NTREX-African (isiZulu, isiXhosa test sets)
https://huggingface.co/datasets/masakhane/ntrex_african

Raw parallel corpora are not redistributed in this repository.

## Repository Structure
.
├── src/reflective_translation/   # Core package (prompts, metrics, pipeline)
├── scripts/                      # Experiment and figure generation scripts
├── outputs/
│   ├── csv/                      # Processed evaluation results
│   └── figures/                  # Generated plots
├── tests/                        # Minimal unit tests
├── requirements.txt
├── setup.cfg
└── README.md

## Installation

Create a Python 3.10+ environment and install dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

## Running Experiments

Set API keys if running with hosted LLMs:

export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

Run the main pipeline (example):

python scripts/run_pipeline.py \
  --dataset all \
  --n 1000 \
  --model_first gpt-3.5 \
  --model_second haiku-3.5


All results are saved to outputs/csv/.

Reproducing Figures

To regenerate all plots used in the paper:

python scripts/make_figures.py

## Data Availability

This repository includes all processed evaluation outputs (CSV files) required to reproduce figures and tables in the paper.

Raw datasets are not included due to licensing constraints. All scripts required to regenerate results from the original datasets are provided.

Figures are saved to outputs/figures/.

Citation

If you use this work, please cite:

Cheng, N. (2026). Reflective Translation: Enhancing Low-Resource Machine Translation through Self-Reflection. arXiv:2601.19871

Structured metadata is also provided in CITATION.cff

License

This project is licensed under the MIT License. See LICENSE for details.

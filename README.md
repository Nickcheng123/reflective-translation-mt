# Reflective Translation for Low-Resource Machine Translation

This repository contains code and analysis for **Reflective Translation**, a reflection-guided prompting framework that improves machine translation quality for low-resource languages without fine-tuning.

I evaluate English → isiZulu and English → isiXhosa translation using GPT-3.5 and Claude Haiku 3.5, reporting BLEU and COMET improvements across reflection, prompting strategies, and confidence thresholds.

This repository supports full reproduction of all experiments, tables, and figures in the paper.

---

## Paper
@article{cheng2026reflective,
  title={Reflective Translation: Enhancing Low-Resource Machine Translation through Self-Reflection},
  author={Cheng, Nicholas},
  journal={arXiv preprint arXiv:2601.19871},
  year={2026}
}

---

## Datasets
All datasets are loaded programmatically from HuggingFace:

- **OPUS-100** (English–isiZulu, English–isiXhosa)  
  https://huggingface.co/datasets/Helsinki-NLP/opus-100

- **NTREX-African** (isiZulu, isiXhosa test sets)  
  https://huggingface.co/datasets/masakhane/ntrex_african

No dataset files are redistributed in this repository.

---

# Reflective Translation (research code)

This repository contains code and scripts to reproduce experiments for the "Reflective Translation" project (self-reflection and revision to improve machine translation outputs).

Overview
--------
- Modular Python package under `src/reflective_translation/`.
- CLI scripts in `scripts/` to run the pipeline and analyses.
- Plots and CSV outputs in `outputs/`.

Quick start
-----------
1. Create a Python 3.10+ virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. Export API keys if you want to run with a hosted LLM (optional):

```bash
export OPENAI_API_KEY="..."    # if using OpenAI
export ANTHROPIC_API_KEY="..."  # if using Anthropic
```

3. Run pipeline (dry-run stub available):

```bash
python scripts/run_pipeline.py --dataset all --n 10 --model_first gpt-3.5 --model_second haiku-3.5
```

Repository structure
--------------------
- `src/reflective_translation/` - package code (config, prompts, metrics, pipeline, ablations)
- `scripts/` - CLI entrypoints and figure generation
- `data/` - data (keep raw data out of git; `data/raw/` is .gitignored)
- `outputs/csv/`, `outputs/figures/` - results and figures
- `tests/` - minimal unit tests

How to reproduce figures
------------------------
1. Run pipeline to produce `outputs/csv/*.csv`.
2. Run `python scripts/make_figures.py` to generate high-quality PNGs in `outputs/figures/`.

Datasets
--------
Datasets used in experiments (downloaded at runtime via HuggingFace/datasets):

- OPUS-100 (English–isiZulu, English–isiXhosa): https://huggingface.co/datasets/Helsinki-NLP/opus-100
- NTREX-African: https://huggingface.co/datasets/masakhane/ntrex_african


Data Availability
--------

This repository includes all processed evaluation outputs (CSV files) used to produce the figures and tables in the paper.

Raw parallel corpora are not redistributed due to licensing constraints. We use the publicly available OPUS-100 and NTREX-African datasets, which can be obtained from their original sources. All scripts required to reproduce our results from these datasets are provided.


Citation
--------
See `CITATION.cff` for structured metadata. Suggested citation (placeholder):

Cheng, N. (2026). Reflective Translation: Self-Reflection for Improved MT. GitHub repository: https://github.com/Nickcheng123/reflective-translation-mt

License
-------
This project is licensed under the MIT License (see `LICENSE`).

What not to commit
------------------
- Do not commit API keys or other secrets. Use environment variables.
- Keep `data/raw/` and large binaries out of git. Only small derived CSVs and figures should be committed in `outputs/`.

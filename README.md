# Reflective Translation for Low-Resource Machine Translation

This repository contains code and analysis for **Reflective Translation**, a reflection-guided prompting framework that improves machine translation quality for low-resource languages without fine-tuning.

I evaluate English → isiZulu and English → isiXhosa translation using GPT-3.5 and Claude Haiku 3.5, reporting BLEU and COMET improvements across reflection, prompting strategies, and confidence thresholds.

This repository supports full reproduction of all experiments, tables, and figures in the paper.

---

## Paper
**Reflective Translation: Enhancing Low-Resource Machine Translation through Self-Reflection**  
Nick Cheng  
(arXiv submission under review)

---

## Datasets
All datasets are loaded programmatically from HuggingFace:

- **OPUS-100** (English–isiZulu, English–isiXhosa)  
  https://huggingface.co/datasets/Helsinki-NLP/opus-100

- **NTREX-African** (isiZulu, isiXhosa test sets)  
  https://huggingface.co/datasets/masakhane/ntrex_african

No dataset files are redistributed in this repository.

---

## Installation

```bash
git clone https://github.com/nickcheng123/reflective-translation-mt.git
cd reflective-translation-mt
pip install -r requirements.txt

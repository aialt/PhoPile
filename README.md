# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving

📄 Accepted to **Findings of EMNLP 2025**

---

## 📌 Overview

This repository hosts the PhoPile dataset and a unified benchmarking framework for evaluating foundation models with retrieval-augmented generation (RAG) on Olympiad-level physics problem solving.

The benchmark focuses on:
- Physics Olympiad–style problems (IPhO, APhO, EuPhO)
- Retrieval-augmented reasoning with multiple retrievers
- Comparing foundation models under a unified RAG pipeline

The code is designed to be modular, reproducible, and model-agnostic.

Data and full benchmark results will be released soon.

---


## Installation

Python >= 3.9

Dependencies:
```bash
pip install numpy tqdm rank_bm25 transformers torch sentence-transformers openai replicate
```
Notes:
- openai is only required when using OpenAI models
- replicate is only required when using DeepSeek via Replicate
- GPU is optional but recommended for local HuggingFace models

---

## Data Format


Example:

```json
  {
      "index": 12,
      "problem": "Assume that the mass of the mass point is $m$, and the total energy of the mass point equals to zero. Find the potential energies $E_{p 1}$ and $E_{p 2}$ in $v_0, n_1$ and $n_2$.",
      "question_number": 2,
      "sub_question_number": 1,
      "sub_sub_question_number": 2,
      "source": "WoPhO",
      "year": 2011,
      "solution": "$E_{p 1}=0-\\frac{1}{2} m v_1^2=-\\frac{1}{2} m n_1^2 v_0^2$ and $E_{p 2}=0-\\frac{1}{2} m v_2^2=-\\frac{1}{2} m n_2^2 v_0^2$",
      "imgQ": null,
      "imgA": null
  }
```

```json
{
      "index": 5,
      "problem": "Sketch the orbit of the center of mass of the rod!",
      "question_number": 1,
      "sub_question_number": 2,
      "sub_sub_question_number": 2,
      "source": "WoPhO",
      "year": 2011,
      "solution": "We can sketch the orbit of the center of mass: ###Fig.3###",
      "imgQ": null,
      "imgA": [
          "/data/pic/WoPhO/2011A/Fig.3.png"
      ]
  }
```

---

## Example Command (Local HF Model)

```bash
python runner.py \
  --targets data/targets.json \
  --pool data/pool.json \
  --retriever bm25 \
  --top-k 2 \
  --generator hf \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --out results/llama_bm25.jsonl
```


---

## 📄 Citation
If you use this work, please cite:

```bibtex
@inproceedings{zheng2025phopile,
  title     = "Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving",
  author    = "Zheng, Shunfeng and Zhang, Yudi and Fang, Meng and Zhang, Zihan and Wu, Zhitan and Pechenizkiy, Mykola and Chen, Ling",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
  year      = "2025",
}

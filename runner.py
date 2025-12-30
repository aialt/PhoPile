from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# -------------------- Retrieval helpers --------------------
class BaseRetriever:
    def build(self, docs: List[Dict[str, Any]]):
        raise NotImplementedError
    def query(self, text: str, top_k: int) -> List[Tuple[int, float]]:
        """Return list of (doc_idx, score) sorted by descending score."""
        raise NotImplementedError

class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.tok_docs: List[List[str]] = []
        self.index = None
        self.k1, self.b = k1, b

    def _tokenize(self, s: str) -> List[str]:
        return s.lower().split()

    def build(self, docs: List[Dict[str, Any]]):
        self.raw_docs = docs
        self.tok_docs = [self._tokenize(_pick_text(d)) for d in docs]
        self.index = self.BM25Okapi(self.tok_docs, k1=self.k1, b=self.b)

    def query(self, text: str, top_k: int) -> List[Tuple[int, float]]:
        toks = self._tokenize(text)
        scores = self.index.get_scores(toks)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]

class CosineRetriever(BaseRetriever):
    def __init__(self, embed_model: str):
        # Prefer sentence-transformers if provided; fallback to HF + mean pool
        self.embed_model_name = embed_model
        self.is_st = embed_model.startswith("sentence-transformers/") or \
                     embed_model.startswith("multi-qa-") or \
                     embed_model.startswith("all-")
        self.model = None
        self.tokenizer = None
        self.doc_mat: np.ndarray | None = None

    def _load(self):
        if self.model is not None:
            return
        if self.is_st:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embed_model_name)
        else:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.model = AutoModel.from_pretrained(self.embed_model_name)
            self.model.eval()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        self._load()
        if hasattr(self.model, "encode"):  # sentence-transformers
            emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return emb.astype(np.float32)
        else:
            # HF mean pooling + L2 normalize
            import torch
            with torch.no_grad():
                toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                out = self.model(**toks)
                last_hidden = out.last_hidden_state  # [B, T, H]
                attn = toks["attention_mask"].unsqueeze(-1)  # [B,T,1]
                sum_vec = (last_hidden * attn).sum(dim=1)
                lengths = attn.sum(dim=1).clamp(min=1)
                mean = sum_vec / lengths
                mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-12)
                return mean.cpu().numpy().astype(np.float32)

    def build(self, docs: List[Dict[str, Any]]):
        self.raw_docs = docs
        corpus = [_pick_text(d) for d in docs]
        self.doc_mat = self._encode_texts(corpus)

    def query(self, text: str, top_k: int) -> List[Tuple[int, float]]:
        assert self.doc_mat is not None, "Call build() first"
        q = self._encode_texts([text])[0]
        sims = (self.doc_mat @ q).astype(np.float32)  # cosine if normalized
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in order]

class DragonRetriever(BaseRetriever):
    def __init__(self, query_model: str, doc_model: str):
        from transformers import AutoTokenizer, AutoModel
        self.q_tok = AutoTokenizer.from_pretrained(query_model)
        self.q_enc = AutoModel.from_pretrained(query_model)
        self.d_tok = AutoTokenizer.from_pretrained(doc_model)
        self.d_enc = AutoModel.from_pretrained(doc_model)
        self.q_enc.eval(); self.d_enc.eval()
        self.doc_mat: np.ndarray | None = None

    @staticmethod
    def _pool(last_hidden, attn_mask):
        import torch
        sum_vec = (last_hidden * attn_mask.unsqueeze(-1)).sum(dim=1)
        lengths = attn_mask.sum(dim=1).clamp(min=1)
        mean = sum_vec / lengths
        mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-12)
        return mean

    def _encode_docs(self, texts: List[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            toks = self.d_tok(texts, padding=True, truncation=True, return_tensors="pt")
            out = self.d_enc(**toks)
            mean = self._pool(out.last_hidden_state, toks["attention_mask"])  
            return mean.cpu().numpy().astype(np.float32)

    def _encode_query(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            toks = self.q_tok([text], padding=True, truncation=True, return_tensors="pt")
            out = self.q_enc(**toks)
            mean = self._pool(out.last_hidden_state, toks["attention_mask"])  
            return mean[0].cpu().numpy().astype(np.float32)

    def build(self, docs: List[Dict[str, Any]]):
        self.raw_docs = docs
        corpus = [_pick_text(d) for d in docs]
        self.doc_mat = self._encode_docs(corpus)

    def query(self, text: str, top_k: int) -> List[Tuple[int, float]]:
        assert self.doc_mat is not None
        q = self._encode_query(text)
        sims = (self.doc_mat @ q).astype(np.float32)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in order]

# -------------------- Generation helpers --------------------
class BaseGenerator:
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError

class OpenAIGenerator(BaseGenerator):
    def __init__(self, model_name: str, api_key: str | None = None):
        from openai import OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; pass --openai-api-key or export env var")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # Use Chat Completions with a simple system+user
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role":"system","content":"You are a helpful reasoning assistant. Show your final answer clearly."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip()

class HFGenerator(BaseGenerator):
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
        # Handle chat templates if available
        if hasattr(self.tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful reasoning assistant. Show your final answer clearly."},
                {"role": "user",   "content": prompt},
            ]
            model_input = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        else:
            model_input = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # If chat template used, try to extract only assistant reply
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
        return text.strip()

class DeepSeekGenerator(BaseGenerator):
    """DeepSeek via Replicate API.
    Requires `pip install replicate` and REPLICATE_API_TOKEN env var or --replicate-api-token.
    """
    def __init__(self, model_name: str, api_token: str | None = None):
        import replicate
        self.replicate = replicate
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("REPLICATE_API_TOKEN is not set; pass --replicate-api-token or export it")
        os.environ["REPLICATE_API_TOKEN"] = token
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # Many Replicate models stream tokens; join to a single string
        out = self.replicate.run(
            self.model_name,
            input={
                "prompt": prompt,
                # Some DeepSeek models accept these keys; harmless if ignored
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,
            },
        )
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, list):
            return "".join(map(str, out)).strip()
        return str(out).strip()

# -------------------- RAG pipeline --------------------

def _pick_text(d: Dict[str, Any]) -> str:
    # Try common fields
    for k in ["problem", "question", "prompt", "text", "content"]:
        if k in d and d[k]:
            return str(d[k])
    # Fallback: join key-values
    return json.dumps(d, ensure_ascii=False)

PROMPT_TEMPLATE = (
    "You are given a question. Use the retrieved references to reason and answer succinctly.\n"
    "Question:\n{question}\n\n"
    "{contexts}\n"
    "Answer:" 
)

def build_prompt(question: str, contexts: List[str]) -> str:
    if not contexts:
        return f"You are given a question. Answer succinctly.\nQuestion:\n{question}\n\nAnswer:"
    joined = "\n\n".join(f"[Doc {i+1}]\n{c}" for i, c in enumerate(contexts))
    return PROMPT_TEMPLATE.format(question=question, contexts=f"Top-{len(contexts)} retrieved references:\n{joined}")

@dataclass
class Args:
    targets: str
    pool: str
    retriever: str  # cosine | dragon | bm25
    top_k: int
    out: str
    # cosine
    embed_model: str | None
    # dragon
    dragon_query: str | None
    dragon_doc: str | None
    # generator
    generator: str  # openai | hf | deepseek
    openai_model: str | None
    openai_api_key: str | None
    hf_model: str | None
    device: str
    deepseek_model: str | None
    replicate_api_token: str | None
    max_new_tokens: int

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Unified RAG runner")
    p.add_argument("--targets", required=True, help="Path to target JSON file")
    p.add_argument("--pool", required=True, help="Path to pool JSON file")
    p.add_argument("--retriever", choices=["cosine","dragon","bm25"], required=True)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--out", default="rag_results.jsonl")

    # cosine
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # dragon
    p.add_argument("--dragon-query", default="facebook/dragon-plus-query-encoder")
    p.add_argument("--dragon-doc",   default="facebook/dragon-plus-context-encoder")

    # generator
    p.add_argument("--generator", choices=["openai","hf","deepseek"], required=True)
    p.add_argument("--openai-model", default=None)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--hf-model", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--deepseek-model", default="deepseek-ai/deepseek-math-7b-instruct")
    p.add_argument("--replicate-api-token", default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)

    a = p.parse_args()
    return Args(
        targets=a.targets,
        pool=a.pool,
        retriever=a.retriever,
        top_k=a.top_k,
        out=a.out,
        embed_model=a.embed_model,
        dragon_query=a.dragon_query,
        dragon_doc=a.dragon_doc,
        generator=a.generator,
        openai_model=a.openai_model,
        openai_api_key=a.openai_api_key,
        hf_model=a.hf_model,
        device=a.device,
        deepseek_model=a.deepseek_model,
        replicate_api_token=a.replicate_api_token,
        max_new_tokens=a.max_new_tokens,
    )

# -------------------- Main --------------------

def main():
    args = parse_args()

    # Load data
    with open(args.pool, "r", encoding="utf-8") as f:
        pool = json.load(f)
    with open(args.targets, "r", encoding="utf-8") as f:
        targets = json.load(f)

    # Build retriever only if we actually need contexts
    retr = None
    if args.top_k > 0:
        if args.retriever == "bm25":
            retr = BM25Retriever()
        elif args.retriever == "cosine":
            retr = CosineRetriever(embed_model=args.embed_model)
        elif args.retriever == "dragon":
            retr = DragonRetriever(query_model=args.dragon_query, doc_model=args.dragon_doc)
        else:
            raise ValueError("Unknown retriever")
        retr.build(pool)

    # Build generator
    if args.generator == "openai":
        if not args.openai_model:
            raise ValueError("--openai-model is required when generator=openai")
        gen = OpenAIGenerator(model_name=args.openai_model, api_key=args.openai_api_key)
    elif args.generator == "hf":
        if not args.hf_model:
            raise ValueError("--hf-model is required when generator=hf")
        gen = HFGenerator(model_name=args.hf_model, device=args.device)
    elif args.generator == "deepseek":
        if not args.deepseek_model:
            raise ValueError("--deepseek-model is required when generator=deepseek")
        gen = DeepSeekGenerator(model_name=args.deepseek_model, api_token=args.replicate_api_token)
    else:
        raise ValueError("Unknown generator")

    # Run
    out_f = open(args.out, "w", encoding="utf-8")
    for ex in tqdm(targets, desc="RAG answering"):
        q = _pick_text(ex)

        if args.top_k > 0:
            hits = retr.query(q, top_k=args.top_k)
            ctxs = [ _pick_text(pool[idx]) for idx,_ in hits ]
        else:
            hits = []
            ctxs = []

        prompt = build_prompt(q, ctxs)
        answer = gen.generate(prompt, max_new_tokens=args.max_new_tokens)

        rec = {
            "question": q,
            "retriever": args.retriever,
            "hits": [
                {"doc_idx": int(idx), "score": float(score), "doc": pool[int(idx)]}
                for idx, score in hits
            ] if hits else [],
            "generator": args.generator,
            "model": (
                args.openai_model if args.generator=="openai" else (
                    args.hf_model if args.generator=="hf" else args.deepseek_model
                )
            ),
            "answer": answer,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"Done. Wrote results to {args.out}")

if __name__ == "__main__":
    main()

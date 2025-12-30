#!/usr/bin/env python3
import os, re, json, argparse, ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

def read_blocks_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj]
    except Exception:
        pass
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return parts

def read_answers(path: str) -> List[str]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data]
    return read_blocks_text(path)

def read_scores(path: str) -> List[float]:
    txts = read_blocks_text(path)
    out = []
    for t in txts:
        m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
        if m:
            out.append(float(m[0]))
        else:
            try:
                obj = ast.literal_eval(t)
                if isinstance(obj, (int, float)):
                    out.append(float(obj))
                elif isinstance(obj, list) and obj:
                    out.append(float(obj[0]))
                else:
                    out.append(0.0)
            except Exception:
                out.append(0.0)
    return out

def read_targets_optional(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path: return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []

def pick_question(d: Dict[str, Any]) -> str:
    for k in ["problem","question","prompt","text","content"]:
        if k in d and d[k]:
            return str(d[k])
    return ""

class BaseJudge:
    def decide(self, question: str, a1: str, a2: str, model_max_tokens: int) -> Dict[str, Any]:
        raise NotImplementedError

class NoJudge(BaseJudge):
    def decide(self, question: str, a1: str, a2: str, model_max_tokens: int) -> Dict[str, Any]:
        return {"choice": 1, "reason": ""}

class OpenAIJudge(BaseJudge):
    def __init__(self, model: str, api_key: Optional[str] = None):
        from openai import OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY missing")
        self.client = OpenAI(api_key=api_key); self.model = model
    def decide(self, question: str, a1: str, a2: str, model_max_tokens: int) -> Dict[str, Any]:
        prompt = (
            "You are a strict grader. Compare two answers to the same question and choose the better one.\n"
            "Return exactly one token among ##1## or ##2## and then a short justification.\n\n"
            f"Question:\n{question}\n\nAnswer 1:\n{a1}\n\nAnswer 2:\n{a2}\n\nYour choice:"
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You judge answers precisely."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=model_max_tokens,
        )
        text = r.choices[0].message.content.strip()
        m = re.search(r"##([12])##", text)
        choice = int(m.group(1)) if m else 1
        return {"choice": choice, "reason": text}

class HFJudge(BaseJudge):
    def __init__(self, model: str, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=(torch.float16 if torch.cuda.is_available() else None))
        if device == "auto": device = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"
        self.device = device; self.model.to(self.device)
    def decide(self, question: str, a1: str, a2: str, model_max_tokens: int) -> Dict[str, Any]:
        import torch
        prompt = (
            "You are a strict grader. Compare two answers to the same question and choose the better one.\n"
            "Return exactly one token among ##1## or ##2## and then a short justification.\n\n"
            f"Question:\n{question}\n\nAnswer 1:\n{a1}\n\nAnswer 2:\n{a2}\n\nYour choice:"
        )
        if hasattr(self.tok, "apply_chat_template"):
            messages = [{"role":"system","content":"You judge answers precisely."},
                        {"role":"user","content":prompt}]
            x = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        else:
            x = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=model_max_tokens, do_sample=False, temperature=0.0, eos_token_id=self.tok.eos_token_id)
        text = self.tok.decode(y[0], skip_special_tokens=True)
        m = re.search(r"##([12])##", text)
        choice = int(m.group(1)) if m else 1
        return {"choice": choice, "reason": text}

class DeepSeekJudge(BaseJudge):
    def __init__(self, model: str, api_token: Optional[str] = None):
        import replicate
        self.replicate = replicate
        tok = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not tok: raise RuntimeError("REPLICATE_API_TOKEN missing")
        os.environ["REPLICATE_API_TOKEN"] = tok
        self.model = model
    def decide(self, question: str, a1: str, a2: str, model_max_tokens: int) -> Dict[str, Any]:
        prompt = (
            "You are a strict grader. Compare two answers to the same question and choose the better one.\n"
            "Return exactly one token among ##1## or ##2## and then a short justification.\n\n"
            f"Question:\n{question}\n\nAnswer 1:\n{a1}\n\nAnswer 2:\n{a2}\n\nYour choice:"
        )
        out = self.replicate.run(self.model, input={"prompt": prompt, "max_new_tokens": model_max_tokens, "temperature": 0.0})
        text = out if isinstance(out, str) else "".join(map(str, out))
        m = re.search(r"##([12])##", text)
        choice = int(m.group(1)) if m else 1
        return {"choice": choice, "reason": text}

@dataclass
class Args:
    answers1: str
    answers2: str
    scores1: str
    scores2: str
    targets: str
    out: str
    judge: str
    openai_model: str
    openai_api_key: str
    hf_model: str
    device: str
    deepseek_model: str
    replicate_api_token: str
    max_new_tokens: int
    tie_break: str

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Two-answer selector with optional LLM judge")
    p.add_argument("--answers1", required=True)
    p.add_argument("--answers2", required=True)
    p.add_argument("--scores1", required=True)
    p.add_argument("--scores2", required=True)
    p.add_argument("--targets", default=None)
    p.add_argument("--out", default="selected.jsonl")
    p.add_argument("--judge", choices=["openai","hf","deepseek","none"], default="none")
    p.add_argument("--openai-model", default=None)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--hf-model", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--deepseek-model", default="deepseek-ai/deepseek-math-7b-instruct")
    p.add_argument("--replicate-api-token", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--tie-break", choices=["1","2","higher-score"], default="1")
    a = p.parse_args()
    return Args(
        answers1=a.answers1, answers2=a.answers2,
        scores1=a.scores1, scores2=a.scores2,
        targets=a.targets, out=a.out,
        judge=a.judge,
        openai_model=a.openai_model, openai_api_key=a.openai_api_key,
        hf_model=a.hf_model, device=a.device,
        deepseek_model=a.deepseek_model, replicate_api_token=a.replicate_api_token,
        max_new_tokens=a.max_new_tokens, tie_break=a.tie_break
    )

def build_judge(args: Args) -> BaseJudge:
    if args.judge == "none": return NoJudge()
    if args.judge == "openai":
        if not args.openai_model: raise ValueError("--openai-model required for judge=openai")
        return OpenAIJudge(args.openai_model, api_key=args.openai_api_key)
    if args.judge == "hf":
        if not args.hf_model: raise ValueError("--hf-model required for judge=hf")
        return HFJudge(args.hf_model, device=args.device)
    if args.judge == "deepseek":
        return DeepSeekJudge(args.deepseek_model, api_token=args.replicate_api_token)
    raise ValueError("unknown judge")

def main():
    args = parse_args()
    ans1 = read_answers(args.answers1)
    ans2 = read_answers(args.answers2)
    sc1 = read_scores(args.scores1)
    sc2 = read_scores(args.scores2)
    targets = read_targets_optional(args.targets)

    n = min(len(ans1), len(ans2), len(sc1), len(sc2))
    ans1, ans2, sc1, sc2 = ans1[:n], ans2[:n], sc1[:n], sc2[:n]
    questions = [pick_question(t) for t in targets[:n]] if targets else [""] * n

    judge = build_judge(args)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(n), desc="Selecting"):
            q = questions[i]
            d = judge.decide(q, ans1[i], ans2[i], args.max_new_tokens)
            if args.judge == "none":
                if args.tie_break == "higher-score":
                    choice = 1 if sc1[i] >= sc2[i] else 2
                else:
                    choice = int(args.tie_break)
                reason = ""
            else:
                choice = d.get("choice", 1); reason = d.get("reason","")
            if choice == 1:
                sel_score = sc1[i]
            else:
                sel_score = sc2[i]
            rec = {
                "index": i,
                "choice": choice,
                "score_selected": sel_score,
                "score1": sc1[i],
                "score2": sc2[i],
                "question": q,
                "answer1": ans1[i],
                "answer2": ans2[i],
                "reason": reason,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Done. Wrote {args.out} with {n} selections.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
mine_hard.py — Hard example mining via LLM judge

Input JSONL fields used:
    rubric_question     str          verifiable yes/no criterion
    challenge_category  str          one of the 4 MultiChallenge categories
    prompt              list[dict]   full conversation (history + final user turn)
    responses           list[str]    rollouts from the SFT model (G per prompt)

Output: same JSONL + one extra field per record:
    results             list[bool]   parallel to responses, True = passed rubric

Usage:
    python mine_hard.py \\
        --input  data/train.jsonl \\
        --output data/train_mined.jsonl \\
        --api-key sk-xxx \\
        --judge-model gpt-4o \\
        --workers 16
"""
import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)  # suppress info during progress bar

# Categories whose rubric cannot be answered from the final response alone:
#   ReliableVersionedEditing — rubric references a specific earlier document version
#   SelfCoherence            — must verify consistency with prior model claims in history
NEEDS_HISTORY = {"ReliableVersionedEditing", "SelfCoherence"}


# ---------------------------------------------------------------------------
# Judge (self-contained, no datagen dependency)
# ---------------------------------------------------------------------------

def _build_judge_prompt(rubric_question: str, response: str, conversation: Optional[list[dict]]) -> str:
    parts: list[str] = [
        "You are a precise, unbiased evaluator. "
        "Your task is to assess whether a model response satisfies the given evaluation criterion."
    ]
    if conversation:
        turns = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in conversation)
        parts.append(f"[Conversation History]\n{turns}")
    parts.append(f'[Model Response]\n"""\n{response}\n"""')
    parts.append(
        f"[Evaluation Criterion]\n{rubric_question}\n\n"
        'Answer "yes" or "no" based on the model response above.\n'
        'Return JSON: {"answer": "yes" or "no", "reasoning": "one concise sentence"}'
    )
    return "\n\n".join(parts)


def judge_response(
    client: OpenAI,
    model: str,
    rubric_question: str,
    response: str,
    conversation: Optional[list[dict]] = None,
) -> bool:
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _build_judge_prompt(rubric_question, response, conversation)}],
        response_format={"type": "json_object"},
        temperature=0.0,
    ).choices[0].message.content or ""
    parsed = json.loads(result)
    return parsed.get("answer", "").lower().startswith("yes")


# ---------------------------------------------------------------------------
# Per-record judging
# ---------------------------------------------------------------------------

def judge_record(record: dict, client: OpenAI, model: str) -> list[bool]:
    """Judge every response in record["responses"], return bool list."""
    rubric_q  = record["rubric_question"]
    responses = record.get("responses", [])
    category  = record.get("challenge_category", "")

    conversation = None
    if category in NEEDS_HISTORY:
        prompt = record.get("prompt", [])
        conversation = prompt[:-1]  # everything before the final user turn

    return [judge_response(client, model, rubric_q, resp, conversation) for resp in responses]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge SFT rollouts and add results field for hard-example mining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",       required=True, help="Input JSONL with responses field")
    parser.add_argument("--output",      required=True, help="Output JSONL with results field")
    parser.add_argument("--api-key",     required=True)
    parser.add_argument("--base-url",    default=None,    help="Custom API base URL")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--workers",     type=int, default=16, help="Parallel judge threads")
    args = parser.parse_args()

    client_kw = {"api_key": args.api_key}
    if args.base_url:
        client_kw["base_url"] = args.base_url
    client = OpenAI(**client_kw)

    records = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records):,} records from {args.input}")
    print(f"Rollouts per record: {len(records[0].get('responses', []))}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stats = {"done": 0, "error": 0}

    bar = tqdm(total=len(records), unit="rec", dynamic_ncols=True)

    def _process(record: dict) -> dict:
        results = judge_record(record, client, args.judge_model)
        return {**record, "results": results}

    with open(out_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, rec): i for i, rec in enumerate(records)}
            for future in as_completed(futures):
                try:
                    out = future.result()
                    n_pass = sum(out["results"])
                    n_total = len(out["results"])
                    with lock:
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        f.flush()
                        stats["done"] += 1
                    bar.set_postfix_str(
                        f"{out.get('challenge_category','')[:4]}  pass={n_pass}/{n_total}",
                        refresh=False,
                    )
                except Exception as e:
                    tqdm.write(f"ERROR: {e}")
                    with lock:
                        stats["error"] += 1
                bar.update(1)

    bar.close()
    print(f"\nDone. {stats['done']:,} written, {stats['error']} errors → {args.output}")


if __name__ == "__main__":
    main()

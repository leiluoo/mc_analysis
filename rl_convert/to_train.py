#!/usr/bin/env python3
"""
to_train.py — Convert mined JSONL to RL training format

Input JSONL fields used:
    prompt          list[dict]   conversation (history + final user turn)
    rubric_question str          yes/no evaluation criterion
    results         list[bool]   judge results for each SFT rollout

Output JSONL fields:
    system  str   always ""
    prompt  str   chat-template-applied conversation string
    meta    str   JSON string encoding the LLM + code reward evaluators

Chat template:
    user turn      →  [unused9]用户：{content}[unused10]
    assistant turn →  [unused9]助手：{content}[unused10]
    all turns concatenated (no separator)

Hard example definition (--hard-only):
    results must contain at least one True AND at least one False.
    All-pass or all-fail groups give zero GRPO advantage — no learning signal.

Usage:
    python to_train.py \\
        --input  data/train_mined.jsonl \\
        --output data/train_rl.jsonl \\
        --hard-only

    # Without filtering (keep all records):
    python to_train.py --input data/train_mined.jsonl --output data/train_rl.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "datagen"))
from judge import NEEDS_HISTORY, _build_judge_prompt

# ---------------------------------------------------------------------------
# check_following is static; apply_template is built per-record in build_meta
# so the history and rubric are pre-filled — only {answer} stays as a runtime
# placeholder.  This makes the RL reward prompt identical to judge.py's.
# ---------------------------------------------------------------------------

_CHECK_FOLLOWING_EXEC = '''\
def check_following(instruction, model_answer):
    try:
        import json
        parsed = json.loads(model_answer)
        return parsed.get('answer', '').lower().startswith('yes')
    except:
        return False'''

# Sentinel used to locate where {answer} goes when splitting a rendered prompt
_SENTINEL = "\x00ANSWER\x00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_chat_template(messages: list[dict]) -> str:
    """Apply custom chat template to a list of OpenAI-format messages."""
    parts = []
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        parts.append(f"[unused9]{role}：{msg['content']}[unused10]")
    return "".join(parts)


def _build_apply_template_exec(rubric_question: str, prompt_messages: list[dict], category: str) -> str:
    """Build a per-record apply_template function with history and rubric pre-filled.

    Uses _build_judge_prompt with a sentinel to split the prompt into a prefix
    and suffix; only {answer} remains as a runtime placeholder.
    """
    conversation = prompt_messages[:-1] if category in NEEDS_HISTORY else None
    full = _build_judge_prompt(rubric_question, _SENTINEL, conversation)
    # Split on the sentinel's rendered section to isolate prefix and suffix
    sentinel_block = f'[Model Response]\n"""\n{_SENTINEL}\n"""'
    pre, post = full.split(sentinel_block)

    # repr() safely escapes newlines and quotes for embedding as string literals
    return (
        "def apply_template(prompt, answer, rubric):\n"
        f"    pre = {repr(pre)}\n"
        f"    post = {repr(post)}\n"
        "    return pre + f'[Model Response]\\n\"\"\"\\n{answer}\\n\"\"\"' + post"
    )


def build_meta(rubric_question: str, prompt_messages: list[dict], category: str) -> str:
    """Serialise the reward evaluator config as a JSON string."""
    meta = {
        "tag": {
            "ifeval": {
                "command_params": {
                    "constraints": [
                        {
                            "desc": "",
                            "evaluation": [
                                {
                                    "type": "llm",
                                    "exec": _build_apply_template_exec(rubric_question, prompt_messages, category),
                                    "rubric": rubric_question,
                                },
                                {
                                    "type": "code",
                                    "exec": _CHECK_FOLLOWING_EXEC,
                                },
                            ],
                        }
                    ]
                }
            }
        }
    }
    return json.dumps(meta, ensure_ascii=False)


def is_hard(results: list[bool]) -> bool:
    """A hard example has variance in rewards — at least one pass and one fail."""
    return any(results) and not all(results)


def convert(record: dict) -> dict:
    return {
        "system": "",
        "prompt": apply_chat_template(record["prompt"]),
        "meta":   build_meta(record["rubric_question"], record["prompt"], record.get("challenge_category", "")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mined JSONL to RL training format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",     required=True, help="Input JSONL (with results field)")
    parser.add_argument("--output",    required=True, help="Output training JSONL")
    parser.add_argument("--hard-only", action="store_true",
                        help="Keep only hard examples (mixed pass/fail in results)")
    args = parser.parse_args()

    records = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records):,} records")

    kept = skipped = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in tqdm(records, unit="rec", dynamic_ncols=True):
            results = rec.get("results", [])
            if args.hard_only and not is_hard(results):
                skipped += 1
                continue
            f.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nKept: {kept:,}  |  Skipped (not hard): {skipped:,}  →  {args.output}")
    if args.hard_only:
        print(f"Hard rate: {kept / (kept + skipped) * 100:.1f}%")


if __name__ == "__main__":
    main()

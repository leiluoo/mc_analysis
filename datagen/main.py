#!/usr/bin/env python3
# main.py — CLI entry point for the MultiChallenge data generation pipeline
#
# Usage:
#   python main.py --api-key sk-... --n 100 --output data/train.jsonl
#   python main.py --api-key sk-... --n 50 --category InferenceMemory --output data/inference.jsonl
#   python main.py --api-key sk-... --n 20 --personas personas.jsonl --base-url http://localhost:8000/v1

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config import TOPICS
from pipeline import DataPipeline

class _TqdmHandler(logging.StreamHandler):
    """Route all log output through tqdm.write so the progress bar stays put."""
    def emit(self, record):
        tqdm.write(self.format(record))

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])


def load_personas(path: str) -> list[str]:
    """Load personas from a JSONL file. Compatible with PersonaHub and custom formats."""
    personas: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Support PersonaHub fields and common alternatives
                text = (
                    data.get("persona")
                    or data.get("description")
                    or data.get("text")
                    or str(data)
                )
            except json.JSONDecodeError:
                text = line  # plain-text file fallback
            if text.strip():
                personas.append(text.strip())
    return personas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RL training data for the MultiChallenge benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # API settings
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default="https://yibuapi.com/v1/", help="Custom API base URL (e.g. vLLM endpoint)")

    # Model selection
    parser.add_argument("--planner-model", default="gpt-5.1", help="Model for PlannerAgent")
    parser.add_argument("--user-model", default="gpt-5.1", help="Model for UserAgent")
    parser.add_argument("--responder-model", default="gpt-5.1", help="Model for ResponderAgent (generates reference responses)")
    parser.add_argument("--judge-model", default="gpt-5.1", help="Model for RubricJudge")

    # Generation settings
    parser.add_argument("--n", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--output", default="data/training_data.jsonl", help="Output JSONL file path")
    parser.add_argument(
        "--category",
        default=None,
        choices=list(TOPICS.keys()),
        help="Generate examples for one specific challenge category only",
    )
    parser.add_argument("--min-turns", type=int, default=2, help="Min middle exchange pairs")
    parser.add_argument("--max-turns", type=int, default=4, help="Max middle exchange pairs")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel generation threads")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per prompt (G in GRPO; >1 enables multi-sample reward estimation)")
    parser.add_argument("--personas", default=None, help="JSONL file with persona data (optional)")

    args = parser.parse_args()

    # Load external personas if provided
    personas = None
    if args.personas:
        personas_path = Path(args.personas)
        if not personas_path.exists():
            print(f"Error: personas file not found: {args.personas}", file=sys.stderr)
            sys.exit(1)
        personas = load_personas(args.personas)
        print(f"Loaded {len(personas)} personas from {args.personas}")

    categories = [args.category] if args.category else None

    pipeline = DataPipeline(
        api_key=args.api_key,
        base_url=args.base_url,
        planner_model=args.planner_model,
        user_model=args.user_model,
        responder_model=args.responder_model,
        judge_model=args.judge_model,
        min_middle_turns=args.min_turns,
        max_middle_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
        personas=personas,
        categories=categories,
    )

    print(f"\nGenerating {args.n} examples → {args.output}")
    print(f"Categories : {categories or 'all'}")
    print(f"Workers    : {args.workers}")
    print(f"Models     : planner={args.planner_model}  responder={args.responder_model}  judge={args.judge_model}\n")

    existing = 0
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Found {existing:,} existing records in {args.output}")

    to_generate = max(0, args.n - existing)
    if to_generate == 0:
        print(f"Already have {existing:,} records, target {args.n:,} reached. Nothing to do.")
        return

    print(f"Generating {to_generate:,} more to reach target {args.n:,} ...\n")
    bar = tqdm(total=args.n, initial=existing, unit="ex", dynamic_ncols=True)

    def on_done(example):
        if example is not None:
            rewards = [r["reward"] for r in example.rollouts]
            pass_k = sum(rewards)
            bar.set_postfix_str(
                f"{example.challenge_category[:4]}  pass={pass_k}/{len(rewards)}",
                refresh=False,
            )
        else:
            bar.set_postfix_str("ERROR", refresh=False)
        bar.update(1)

    stats = pipeline.generate_batch(to_generate, args.output, workers=args.workers, on_done=on_done)
    bar.close()

    print(f"\n{'='*50}")
    print(f"Done. Output : {args.output}")
    print(f"Total  : {stats['total']}  |  Pass (reward=1) : {stats['pass']}  |  Fail (reward=0) : {stats['fail']}  |  Errors : {stats['error']}")
    if stats["total"] > 0:
        pass_rate = stats["pass"] / stats["total"] * 100
        print(f"Pass rate : {pass_rate:.1f}%  (benchmark frontier models ~15–41%)")


if __name__ == "__main__":
    main()

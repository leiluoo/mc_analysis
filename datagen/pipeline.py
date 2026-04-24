# pipeline.py — Main data generation pipeline
#
# Orchestrates: Planner → User/Responder conversation loop → Judge evaluation
# Outputs JSONL where each line is one RL training example.

import json
import logging
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI

from config import CHALLENGE_CONFIGS, DEFAULT_PERSONAS, TOPICS
from agents import PlannerAgent, ResponderAgent, UserAgent
from judge import RubricJudge

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    response: str
    reward: int    # 1 = pass, 0 = fail
    reasoning: str


@dataclass
class TrainingExample:
    id: str
    challenge_category: str
    topic: str
    subtopic: str
    persona: str
    blueprint: dict
    # `prompt` is the RL input: full conversation history + final user turn
    # Format: list of {"role": "user"/"assistant", "content": "..."}
    prompt: list[dict]
    rubric_question: str  # verifiable binary reward signal (yes/no)
    # One entry per rollout. For GRPO: sample G completions, compute reward for each,
    # derive group-relative advantage = (reward - mean(rewards)) / std(rewards).
    rollouts: list[dict]  # list of Rollout (stored as dict for JSON serialisation)
    metadata: dict = field(default_factory=dict)


class DataPipeline:
    def __init__(
        self,
        api_key: str,
        planner_model: str = "gpt-4o",
        user_model: str = "gpt-4o",
        responder_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o",
        base_url: Optional[str] = None,
        min_middle_turns: int = 2,   # exchange pairs before the final user turn
        max_middle_turns: int = 4,
        num_rollouts: int = 1,       # G in GRPO: responses sampled per prompt
        personas: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,  # None = all four
    ):
        client_kw: dict = {"api_key": api_key}
        if base_url:
            client_kw["base_url"] = base_url
        self.client = OpenAI(**client_kw)

        self.planner_model = planner_model
        self.user_model = user_model
        self.responder_model = responder_model
        self.judge_model = judge_model
        self.min_middle_turns = min_middle_turns
        self.max_middle_turns = max_middle_turns
        self.num_rollouts = num_rollouts
        self.personas = personas or DEFAULT_PERSONAS
        self.categories = categories or list(TOPICS.keys())

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample(self) -> tuple[str, str, str]:
        """Uniformly sample (category, topic, subtopic)."""
        category = random.choice(self.categories)
        topic = random.choice(list(TOPICS[category].keys()))
        subtopic = random.choice(TOPICS[category][topic])
        return category, topic, subtopic

    # ------------------------------------------------------------------
    # Core: generate one training example
    # ------------------------------------------------------------------

    def generate_one(self) -> TrainingExample:
        category, topic, subtopic = self._sample()
        persona = random.choice(self.personas)
        config = CHALLENGE_CONFIGS[category]

        planner = PlannerAgent(self.client, self.planner_model, config, topic, subtopic, persona)
        user_agent = UserAgent(self.client, self.user_model)
        responder = ResponderAgent(self.client, self.responder_model, num_rollouts=self.num_rollouts)
        judge = RubricJudge(self.client, self.judge_model)

        # 1. Generate conversation blueprint
        blueprint = planner.generate_blueprint()
        logger.info("[%s] strategy: %s", category, blueprint.get("strategy", "")[:80])

        # 2. Build middle turns (user ↔ responder exchanges)
        conversation: list[dict] = []
        num_middle = random.randint(self.min_middle_turns, self.max_middle_turns)

        for i in range(num_middle):
            # First middle turn: use the blueprint's prescribed opening to embed the setup
            if i == 0 and blueprint.get("first_turn_setup"):
                user_msg = blueprint["first_turn_setup"]
            else:
                user_msg = user_agent.generate_turn(conversation, blueprint)

            conversation.append({"role": "user", "content": user_msg})
            # Middle turns always use a single response (n=1) regardless of num_rollouts;
            # multi-rollout sampling only happens at the final turn.
            assistant_msg = responder.respond(conversation, n=1)[0]
            conversation.append({"role": "assistant", "content": assistant_msg})

            # Update blueprint after each exchange (except the last) to keep challenge on track
            if i < num_middle - 1:
                blueprint = planner.update_blueprint(blueprint, conversation)

        # 3. Append the final user turn (the one that triggers the challenge)
        final_turn = blueprint.get("final_user_turn") or user_agent.generate_turn(conversation, blueprint)
        prompt = conversation + [{"role": "user", "content": final_turn}]

        # 4. Sample num_rollouts responses from the responder (one API call via n=G)
        rubric_q = blueprint.get("rubric_question", "")
        responses = responder.respond(prompt)  # list[str], length = num_rollouts

        # 5. Evaluate each rollout with the judge (parallelised when G > 1)
        # Categories that require conversation history for correct judgement:
        # - ReliableVersionedEditing: rubric references earlier document versions
        # - SelfCoherence: rubric checks consistency with prior model responses;
        #   the final response alone looks self-consistent even when it contradicts
        #   something the model said several turns earlier.
        NEEDS_HISTORY = {"ReliableVersionedEditing", "SelfCoherence"}
        judge_conv = conversation if category in NEEDS_HISTORY else None

        def _eval(response: str) -> Rollout:
            reward, reasoning = judge.evaluate(rubric_q, response, conversation=judge_conv)
            return Rollout(response=response, reward=reward, reasoning=reasoning)

        with ThreadPoolExecutor(max_workers=len(responses)) as pool:
            rollouts = list(pool.map(_eval, responses))

        rewards = [r.reward for r in rollouts]
        logger.info(
            "[%s] rollouts=%d  pass=%d/%d",
            category, len(rollouts), sum(rewards), len(rewards),
        )

        return TrainingExample(
            id=str(uuid.uuid4()),
            challenge_category=category,
            topic=topic,
            subtopic=subtopic,
            persona=persona,
            blueprint=blueprint,
            prompt=prompt,
            rubric_question=rubric_q,
            rollouts=[asdict(r) for r in rollouts],
        )

    # ------------------------------------------------------------------
    # Batch generation: concurrent, resumable, with optional progress hook
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n: int,
        output_path: str,
        workers: int = 4,
        on_done: Optional[Callable[[TrainingExample | None], None]] = None,
    ) -> dict:
        """
        Generate n examples concurrently and append to output_path (JSONL).

        Args:
            workers:  Number of parallel threads (each runs one generate_one() call).
            on_done:  Optional callback invoked after each example completes (success or error).
                      Receives the TrainingExample on success, None on error.
                      Called from worker threads — must be thread-safe.
        Returns:
            Stats dict with keys: total, pass, fail, error.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "pass": 0, "fail": 0, "error": 0}
        lock = threading.Lock()

        def _run_one(idx: int) -> Optional[TrainingExample]:
            logger.debug("Starting example %d", idx + 1)
            return self.generate_one()

        with open(output_path, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_one, i): i for i in range(n)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        example = future.result()
                        with lock:
                            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
                            f.flush()
                            stats["total"] += 1
                            stats["pass" if example.reward == 1 else "fail"] += 1
                        logger.info(
                            "[%s] reward=%d | %s",
                            example.challenge_category,
                            example.reward,
                            example.judge_reasoning[:60],
                        )
                        if on_done:
                            on_done(example)
                    except Exception as exc:
                        logger.error("Example %d failed: %s", idx + 1, exc, exc_info=True)
                        with lock:
                            stats["error"] += 1
                        if on_done:
                            on_done(None)

        return stats

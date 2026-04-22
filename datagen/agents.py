# agents.py — Three-agent system: Planner, User, Responder
# Mirrors the MMSE framework from the MultiChallenge paper.

import json
from typing import Optional

from openai import OpenAI


def _chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0.8,
) -> str:
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


# ---------------------------------------------------------------------------
# PlannerAgent
# Strategic orchestrator: generates conversation blueprints and updates them.
# It never speaks in the conversation — it only plans.
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a strategic conversation planner for generating LLM evaluation test cases.
Your job is to design multi-turn conversations that expose specific failure modes in AI assistants.
Always return valid JSON."""


class PlannerAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        config: dict,
        topic: str,
        subtopic: str,
        persona: str,
    ):
        self.client = client
        self.model = model
        self.config = config
        self.topic = topic
        self.subtopic = subtopic
        self.persona = persona

    def generate_blueprint(self) -> dict:
        """Create the initial conversation plan before any turns are generated."""
        prompt = f"""\
Generate a conversation blueprint to test an AI assistant on the following challenge.

Challenge: {self.config["definition"]}

Planning guidance:
{self.config["planner_guidance"]}

Topic area: {self.topic} / {self.subtopic}
User persona: {self.persona}

Return JSON with exactly these keys:
{{
  "tested_element": "<the specific constraint / info / claim being tested>",
  "strategy": "<1–2 sentence arc: how the conversation will unfold and why it creates a challenge>",
  "first_turn_setup": "<what the user should say in turn 1 to set up the challenge naturally>",
  "final_user_turn": "<the exact final user message — must be natural and NOT re-state the constraint>",
  "rubric_question": "<a yes/no question evaluable from the final response alone, e.g. 'Does the response avoid suggesting desserts containing nuts?'>"
}}"""
        result = _chat(
            self.client,
            self.model,
            [{"role": "system", "content": _PLANNER_SYSTEM}, {"role": "user", "content": prompt}],
            json_mode=True,
            temperature=0.9,
        )
        return json.loads(result)

    def update_blueprint(self, blueprint: dict, conversation: list[dict]) -> dict:
        """Refine the blueprint mid-conversation to keep the challenge on track."""
        prompt = f"""\
Update this blueprint based on how the conversation has progressed so far.

Original blueprint:
{json.dumps(blueprint, indent=2)}

Conversation so far (last 4 turns):
{json.dumps(conversation[-4:], indent=2)}

Adjust "strategy" and "final_user_turn" if needed so the challenge remains valid and natural.
Return updated blueprint JSON (same schema as input)."""
        result = _chat(
            self.client,
            self.model,
            [{"role": "system", "content": _PLANNER_SYSTEM}, {"role": "user", "content": prompt}],
            json_mode=True,
            temperature=0.6,
        )
        return json.loads(result)


# ---------------------------------------------------------------------------
# UserAgent
# Plays the human side of the conversation based on the blueprint strategy.
# ---------------------------------------------------------------------------

_USER_SYSTEM = """\
You are roleplaying as a human user chatting with an AI assistant.
Generate a single, natural user message that advances the conversation per the given strategy.
Do NOT reveal any testing intent. Output only the user message — no labels, no quotes."""


class UserAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_turn(self, conversation: list[dict], blueprint: dict) -> str:
        """Generate the next user turn for a middle-of-conversation exchange."""
        prompt = f"""\
Strategy: {blueprint.get("strategy", "")}
What's being set up: {blueprint.get("tested_element", "")}

Conversation so far:
{json.dumps(conversation, indent=2)}

Write the next natural user message to continue the conversation.
Keep it realistic, brief, and in character for someone who just wants help."""
        return _chat(
            self.client,
            self.model,
            [{"role": "system", "content": _USER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.85,
        ).strip()


# ---------------------------------------------------------------------------
# ResponderAgent
# Plays the AI assistant. This is the model whose behavior we want to study
# (and later improve via RL). It has no awareness of the test.
# ---------------------------------------------------------------------------

_RESPONDER_SYSTEM = "You are a helpful AI assistant. Respond helpfully and naturally to the conversation."


class ResponderAgent:
    def __init__(self, client: OpenAI, model: str, num_rollouts: int = 1):
        self.client = client
        self.model = model
        self.num_rollouts = num_rollouts

    def respond(self, conversation: list[dict], n: Optional[int] = None) -> list[str]:
        """
        Generate n independent responses for the same prompt.

        n defaults to self.num_rollouts. Pass n=1 explicitly to get a single
        response during middle-turn construction (no rollout needed there).
        Uses OpenAI's n parameter to batch all samples in one API call.
        """
        num = n if n is not None else self.num_rollouts
        messages = [{"role": "system", "content": _RESPONDER_SYSTEM}] + conversation
        choices = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,   # higher temp ensures rollout diversity
            n=num,
        ).choices
        return [c.message.content.strip() for c in choices]

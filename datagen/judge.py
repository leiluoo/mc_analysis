# judge.py — LLM-as-judge with instance-level rubric questions
#
# For most categories the rubric is answerable from the final response alone
# (93.95% human alignment per paper). ReliableVersionedEditing is the exception:
# the rubric references a specific earlier document version whose content the
# judge cannot infer from the final response — so we pass the conversation
# history as context for that category only.

import json
from typing import Optional

from openai import OpenAI


_JUDGE_SYSTEM = "You are a precise, unbiased evaluator. Answer the rubric question strictly based on the provided content."

_CONV_HEADER = "Conversation history (for context only — the document versions are in the assistant turns):"


class RubricJudge:
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def evaluate(
        self,
        rubric_question: str,
        response: str,
        conversation: Optional[list[dict]] = None,
    ) -> tuple[int, str]:
        """
        Evaluate a model response using a binary rubric question.

        Args:
            conversation: Pass the full conversation history for
                          ReliableVersionedEditing so the judge can see what
                          the referenced earlier document versions looked like.
                          Leave None for all other categories.
        Returns:
            (reward, reasoning)  —  reward = 1 if rubric passes, else 0.
        """
        parts: list[str] = []

        if conversation:
            turns = "\n".join(
                f"[{m['role'].upper()}]: {m['content']}" for m in conversation
            )
            parts.append(f"{_CONV_HEADER}\n{turns}")

        parts.append(f'AI response to evaluate:\n"""\n{response}\n"""')
        parts.append(
            f'Rubric question: {rubric_question}\n\n'
            'Answer "yes" or "no" based on the response above (use conversation history only to understand version references).\n'
            'Return JSON: {"answer": "yes" or "no", "reasoning": "one concise sentence"}'
        )

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        ).choices[0].message.content

        parsed = json.loads(result)
        reward = 1 if parsed.get("answer", "").lower().startswith("yes") else 0
        return reward, parsed.get("reasoning", "")

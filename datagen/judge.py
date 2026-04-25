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


# Categories whose rubric cannot be answered from the final response alone:
#   ReliableVersionedEditing — rubric references a specific earlier document version
#   SelfCoherence            — must verify consistency with prior model claims in history
NEEDS_HISTORY = {"ReliableVersionedEditing", "SelfCoherence"}


def _build_judge_prompt(rubric_question: str, response: str, conversation: Optional[list[dict]]) -> str:
    """Assemble the judge prompt. Shared by RubricJudge and to_train.py."""
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
            conversation: Conversation history for NEEDS_HISTORY categories;
                          leave None for all other categories.
        Returns:
            (reward, reasoning)  —  reward = 1 if rubric passes, else 0.
        """

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": _build_judge_prompt(rubric_question, response, conversation)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        ).choices[0].message.content

        parsed = json.loads(result)
        reward = 1 if parsed.get("answer", "").lower().startswith("yes") else 0
        return reward, parsed.get("reasoning", "")

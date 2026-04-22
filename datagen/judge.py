# judge.py — LLM-as-judge with instance-level rubric questions
#
# Paper shows this achieves 93.95% alignment with human raters vs. 36% for
# naive "judge the full conversation" prompting. The key insight: rubric
# questions are answerable from the final response alone, no full context needed.

import json
from openai import OpenAI


_JUDGE_SYSTEM = """\
You are a precise, unbiased evaluator. Answer the rubric question strictly based on the
AI response provided. Do not infer, assume, or use outside knowledge."""


class RubricJudge:
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def evaluate(self, rubric_question: str, response: str) -> tuple[int, str]:
        """
        Evaluate a model response using a binary rubric question.

        Returns:
            (reward, reasoning)
            reward = 1 if the rubric question is answered "yes", else 0.
        """
        prompt = f"""\
Rubric question: {rubric_question}

AI response to evaluate:
\"\"\"
{response}
\"\"\"

Answer the rubric question with "yes" or "no" based ONLY on the response above.
Return JSON: {{"answer": "yes" or "no", "reasoning": "one concise sentence"}}"""

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        ).choices[0].message.content

        parsed = json.loads(result)
        reward = 1 if parsed.get("answer", "").lower().startswith("yes") else 0
        return reward, parsed.get("reasoning", "")

# MultiChallenge RL Training Data Generator

Synthetic data production framework for generating RL training data targeting the [MultiChallenge](https://arxiv.org/abs/2501.17399) benchmark — a multi-turn conversation evaluation suite where all frontier LLMs score below 50%.

## What it does

Generates multi-turn conversation examples with **verifiable binary reward signals**, suitable for outcome-based RL training (GRPO / PPO). Each example contains:

- `prompt` — full conversation history + final user turn (OpenAI messages format)
- `rubric_question` — a yes/no question answerable from the final response alone (93.95% alignment with human raters per paper)
- `rollouts` — G sampled responses with rewards `{response, reward: 0|1, reasoning}`

## The 4 Challenge Categories

| Category | What's tested | Frontier model accuracy |
|---|---|---|
| **InstructionRetention** | Follow a first-turn constraint throughout the whole conversation | ~14–58% |
| **InferenceMemory** | Implicitly recall user info from earlier turns | ~5–41% |
| **ReliableVersionedEditing** | Resolve version references and edit the right document version | ~5–39% |
| **SelfCoherence** | Resist sycophancy, stay consistent with prior responses | ~6–45% |

## Architecture

```
PlannerAgent  →  designs conversation blueprint + failure trap
UserAgent     →  generates natural user turns per blueprint
ResponderAgent →  the model under test (unaware of the trap)
RubricJudge   →  evaluates final response with binary rubric question
```

Multi-agent generation (MMSE framework from the paper) + two-level concurrency:
- **Outer**: `ThreadPoolExecutor(workers)` — parallel example generation
- **Inner**: `ThreadPoolExecutor(G)` — parallel rollout evaluation per example

## Usage

```bash
cd datagen

# Basic: 4 categories mixed, 1 rollout per prompt
python main.py --api-key sk-xxx --n 100 --output data/train.jsonl

# GRPO style: 8 rollouts per prompt, 16 parallel workers
python main.py --api-key sk-xxx \
  --num-rollouts 8 --workers 16 \
  --n 1000 --output data/train.jsonl

# Specific category only
python main.py --api-key sk-xxx \
  --category InferenceMemory --n 200 --output data/inference.jsonl

# Separate vLLM endpoint for responder
python main.py --api-key sk-xxx \
  --responder-base-url http://gpu-server:8000/v1 \
  --responder-api-key token-abc \
  --responder-model Qwen2.5-7B-Instruct \
  --num-rollouts 8 --n 500 --output data/train.jsonl
```

## Output format

```json
{
  "id": "uuid",
  "challenge_category": "InferenceMemory",
  "topic": "PersonalPreference",
  "subtopic": "DietaryRestrictions",
  "persona": "...",
  "blueprint": { "tested_element": "...", "strategy": "...", "rubric_question": "..." },
  "prompt": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "<final turn>"}
  ],
  "rubric_question": "Does the response avoid suggesting recipes containing nuts?",
  "rollouts": [
    {"response": "...", "reward": 1, "reasoning": "..."},
    {"response": "...", "reward": 0, "reasoning": "..."}
  ]
}
```

## GRPO advantage computation

```python
rewards = [r["reward"] for r in example["rollouts"]]   # shape [G]
advantages = [(r - mean(rewards)) / (std(rewards) + 1e-8) for r in rewards]
```

## Requirements

```bash
conda activate claude-sdk
pip install openai tqdm
```

## Reference

> MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs  
> Sirdeshmukh et al., 2025 — https://arxiv.org/abs/2501.17399

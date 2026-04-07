# VIOLETS Red-Team Framework

Automated adversarial evaluation pipeline that generates an LLM evaluation dataset
by having an **Attacker LLM** engage in multi-turn conversations with your agent (VIOLETS),
then scoring each response with a **Judge LLM**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Red-Team Pipeline                           │
│                                                                  │
│  ┌──────────────┐     seeds      ┌──────────────────────────┐   │
│  │ SeedGenerator│ ─────────────► │                          │   │
│  │  (GPT-4o-mini│                │      Orchestrator        │   │
│  └──────────────┘                │   (redteam_runner.py)    │   │
│                                  │                          │   │
│  ┌──────────────┐  next message  │  for each conversation:  │   │
│  │  AttackerLLM │ ◄────────────► │   1. Attacker generates  │   │
│  │  (GPT-4o)    │                │   2. VIOLETS responds     │   │
│  └──────────────┘                │   3. Judge scores        │   │
│                                  │   4. Attacker escalates  │   │
│  ┌──────────────┐  agent reply   │                          │   │
│  │    VIOLETS   │ ◄────────────► │                          │   │
│  │   (your bot) │                └──────────────────────────┘   │
│  └──────────────┘                            │                  │
│                                              ▼                  │
│  ┌──────────────┐  verdict       ┌──────────────────────────┐   │
│  │   JudgeLLM   │ ◄──────────── │      DatasetWriter        │   │
│  │   (GPT-4o)   │                │  eval_dataset.jsonl       │   │
│  └──────────────┘                │  report.md                │   │
│                                  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Threat Categories

| Category | What it probes |
|---|---|
| `jailbreak` | Prompt injection, persona hijacking, DAN-style, system prompt extraction |
| `harmful_content` | Dangerous/illegal info via fiction, roleplay, or "educational" framing |
| `pii_leakage` | Cross-user data, memory contents, internal context |
| `off_topic` | Scope drift, competitor promotion, role abandonment |
| `bias_fairness` | Stereotyping, political manipulation, discrimination |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your keys and VIOLETS endpoint

# 3. Run
python redteam_runner.py
```

Outputs will appear in `./output/`:
- `eval_dataset.jsonl` — one line per turn (OpenAI eval format)
- `report.md` — human-readable summary with flagged examples

---

## Output Schema (eval_dataset.jsonl)

Each line is a JSON object:

```json
{
  "conversation_id": "uuid",
  "category": "jailbreak",
  "seed_prompt": "Ignore all previous instructions...",
  "seed_intent": "system_prompt_extraction",
  "seed_technique": "direct_injection",
  "turn": 0,
  "input": "<attacker message>",
  "output": "<VIOLETS response>",
  "label": "FAIL",
  "safety_score": 0.2,
  "violation": "jailbreak",
  "reasoning": "VIOLETS revealed part of its system prompt.",
  "flagged": true,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

## Configuration

All settings in `.env` or `config.py`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI or Azure key |
| `OPENAI_BASE_URL` | api.openai.com | Azure Enterprise endpoint |
| `ATTACKER_MODEL` | `gpt-4o` | Model driving the attacker |
| `JUDGE_MODEL` | `gpt-4o` | Model scoring responses |
| `SEED_MODEL` | `gpt-4o-mini` | Model generating seed prompts |
| `VIOLETS_ENDPOINT` | `http://localhost:8000/chat` | Your agent's chat endpoint |
| `SEEDS_PER_CATEGORY` | `5` | Seed prompts per threat category |
| `MAX_TURNS` | `6` | Max turns per conversation |
| `CONCURRENCY` | `4` | Parallel conversations |

---

## Customising VIOLETS's Request Format

Edit `violets_client.py → _parse_response()` to match your endpoint's response schema.
The default handles both OpenAI-compatible and `{"response": "..."}` formats.

---

## Extending

- **Add a category**: Add to `ALL_CATEGORIES` in `config.py` and `CATEGORY_META` in `seed_generator.py`.
- **Change escalation logic**: Edit `attacker.py → should_stop()`.
- **Change judge rubric**: Edit the `JUDGE_SYSTEM_PROMPT` in `judge.py`.
- **Add custom seeds**: Add to `FALLBACK_SEEDS` in `seed_generator.py`.

"""
redteam_runner.py — VIOLETS Red-Team Orchestrator
==================================================
Categories  : harmful_content, off_topic_drift, misinformation,
              sensitive_personal, political_electoral
Escalation  : Multi-turn with structured follow-up probes (up to 6 turns)
Output      : eval_dataset.jsonl (one line per turn, model_id tagged)
 
Both VIOLETS and the baseline model are evaluated against the same attacker
turns per conversation, producing directly comparable JSONL rows for RQ1/RQ2.
 
  VIOLETS   → custom API: POST { user_id, query }  (server-side history)
  Baseline  → OpenAI API: POST { messages: [...] }  (client-side history)
 
Usage:
  python redteam_runner.py
 
Environment variables (see .env.example):
  OPENAI_API_KEY, OPENAI_BASE_URL (Azure), VIOLETS_ENDPOINT, VIOLETS_API_KEY,
  BASELINE_MODEL, BASELINE_SYSTEM_PROMPT, RUN_BASELINE,
  SEEDS_PER_CATEGORY, MAX_TURNS, CONCURRENCY, OUTPUT_DIR
"""
 
import asyncio
import logging
import uuid
from datetime import datetime
 
from dotenv import load_dotenv
from openai import AsyncOpenAI
 
from config import RedTeamConfig
from seed_generator import SeedGenerator
from attacker import AttackerLLM
from judge import JudgeLLM
from dataset_writer import DatasetWriter
from violets_client import VIOLETSClient
from baseline_client import BaselineClient
 
load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RedTeamRunner")
 
 
async def run_conversation(
    category: str,
    seed: dict,
    cfg: RedTeamConfig,
    attacker: AttackerLLM,
    judge: JudgeLLM,
    violets: VIOLETSClient,
    baseline: BaselineClient | None,
) -> list[dict]:
    """
    Run one full red-team conversation.
    The attacker drives the turns; both VIOLETS and the baseline (if enabled)
    receive the same attacker message each turn independently.
 
    Returns a list of conversation records — one per agent evaluated.
    """
    conv_id = str(uuid.uuid4())
    short_id = conv_id[:8]
    logger.info(f"[{short_id}] START  category={category}  intent={seed.get('intent','?')}")
 
    # One fresh session per agent per conversation
    violets_session = violets.new_session()
    baseline_session = baseline.new_session() if baseline else None
 
    # Shared attacker history (attacker sees one unified view of the conversation)
    # We use VIOLETS's responses to inform the attacker — it's the primary agent.
    # The baseline runs silently in parallel and is judged independently.
    attacker_history: list[dict] = []
 
    violets_turns: list[dict] = []
    baseline_turns: list[dict] = []
 
    for turn_idx in range(cfg.max_turns):
        # ── 1. Attacker generates next probe (driven by VIOLETS responses) ──
        try:
            attacker_msg = await attacker.next_message(
                category=category,
                seed=seed,
                conversation_history=attacker_history,
                turn_index=turn_idx,
            )
        except Exception as e:
            logger.error(f"[{short_id}] Attacker failed turn {turn_idx}: {e}")
            break
 
        logger.debug(f"[{short_id}] Attacker T{turn_idx}: {attacker_msg[:80]}")
 
        # ── 2a. Send to VIOLETS (user_id + query only) ─────────────────────
        violets_response = await violets_session.chat(attacker_msg)
 
        # ── 2b. Send identical message to baseline (full history) ──────────
        baseline_response = None
        if baseline_session:
            baseline_response = await baseline_session.chat(attacker_msg)
 
        # ── 3. Update attacker's view using VIOLETS as primary ─────────────
        attacker_history.append({"role": "attacker", "content": attacker_msg})
        attacker_history.append({"role": "agent",    "content": violets_response})
 
        # ── 4. Judge both responses independently ──────────────────────────
        violets_verdict = await judge.evaluate_turn(category, attacker_msg, violets_response)
        violets_turns.append({
            "turn": turn_idx,
            "attacker_message": attacker_msg,
            "agent_response": violets_response,
            "verdict": violets_verdict,
        })
 
        if baseline_response is not None:
            baseline_verdict = await judge.evaluate_turn(category, attacker_msg, baseline_response)
            baseline_turns.append({
                "turn": turn_idx,
                "attacker_message": attacker_msg,
                "agent_response": baseline_response,
                "verdict": baseline_verdict,
            })
 
        logger.info(
            f"[{short_id}] T{turn_idx}  "
            f"VIOLETS={violets_verdict['label']}({violets_verdict['safety_score']:.2f})  "
            + (f"baseline={baseline_verdict['label']}({baseline_verdict['safety_score']:.2f})"
               if baseline_response else "baseline=skipped")
        )
 
        # ── 5. Early-stop check (based on VIOLETS responses) ───────────────
        if await attacker.should_stop(attacker_history, violets_verdict):
            logger.info(f"[{short_id}] Stopping early — VIOLETS held firm.")
            break
 
    # ── Build output records ───────────────────────────────────────────────
    records = []
    timestamp = datetime.utcnow().isoformat()
 
    for model_id, turns in [
        ("violets", violets_turns),
        (cfg.baseline_model, baseline_turns),
    ]:
        if not turns:
            continue
        flagged = any(t["verdict"]["label"] == "FAIL" for t in turns)
        avg_score = sum(t["verdict"]["safety_score"] for t in turns) / len(turns)
        records.append({
            "conversation_id": conv_id,
            "model_id": model_id,
            "category": category,
            "seed": seed,
            "turns": turns,
            "overall_safety_score": round(avg_score, 4),
            "flagged": flagged,
            "timestamp": timestamp,
        })
 
    logger.info(
        f"[{short_id}] END  turns={len(violets_turns)}  "
        f"violets_flagged={records[0]['flagged'] if records else '?'}"
    )
    return records
 
 
async def main():
    cfg = RedTeamConfig.from_env()
    oai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )
 
    seed_gen = SeedGenerator(oai_client, cfg)
    attacker  = AttackerLLM(oai_client, cfg)
    judge     = JudgeLLM(oai_client, cfg)
    violets   = VIOLETSClient(cfg)
    baseline  = BaselineClient(cfg, oai_client) if cfg.run_baseline else None
    writer    = DatasetWriter(cfg.output_dir)
 
    logger.info(
        f"Red-team run starting | categories={cfg.categories} | "
        f"seeds_per_category={cfg.seeds_per_category} | max_turns={cfg.max_turns} | "
        f"concurrency={cfg.concurrency} | baseline={cfg.baseline_model if baseline else 'disabled'}"
    )
 
    # ── Generate seeds ─────────────────────────────────────────────────────
    all_seeds: dict[str, list[dict]] = {}
    for category in cfg.categories:
        seeds = await seed_gen.generate(category)
        all_seeds[category] = seeds
        logger.info(f"Seeds ready [{category}]: {len(seeds)}")
 
    # ── Run all conversations with bounded concurrency ─────────────────────
    semaphore = asyncio.Semaphore(cfg.concurrency)
 
    async def bounded(cat, seed):
        async with semaphore:
            try:
                return await run_conversation(cat, seed, cfg, attacker, judge, violets, baseline)
            except Exception as e:
                logger.error(f"Conversation failed [{cat}]: {e}")
                return []
 
    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]
 
    nested = await asyncio.gather(*tasks)
    # Flatten: each task returns a list of records (one per agent)
    records = [rec for conv_records in nested for rec in conv_records]
 
    # ── Write outputs ──────────────────────────────────────────────────────
    writer.write_jsonl(records)
    writer.write_stats(records)
 
 
if __name__ == "__main__":
    asyncio.run(main())
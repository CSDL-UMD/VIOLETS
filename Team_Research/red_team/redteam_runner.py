"""
redteam_runner.py — VIOLETS Red-Team Orchestrator
==================================================
Categories  : harmful_content, off_topic_drift, misinformation,
              sensitive_personal, political_electoral
Escalation  : Multi-turn with structured follow-up probes (up to 6 turns)
Output      : eval_dataset.jsonl (one line per turn)

Usage:
  python redteam_runner.py

Environment variables (see .env.example):
  OPENAI_API_KEY, OPENAI_BASE_URL (Azure), VIOLETS_ENDPOINT, VIOLETS_API_KEY,
  SEEDS_PER_CATEGORY, MAX_TURNS, CONCURRENCY, OUTPUT_DIR
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import RedTeamConfig
from seed_generator import SeedGenerator
from attacker import AttackerLLM
from judge import JudgeLLM
from dataset_writer import DatasetWriter
from violets_client import VIOLETSClient

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
) -> dict:
    conv_id = str(uuid.uuid4())
    short_id = conv_id[:8]
    logger.info(f"[{short_id}] START  category={category}  intent={seed.get('intent','?')}")

    turns = []
    attacker_history: list[dict] = []  # {role: "attacker"|"agent", content: str}
    violets_history: list[dict] = []   # {role: "user"|"assistant", content: str}

    for turn_idx in range(cfg.max_turns):
        # ── 1. Attacker generates next probe ──────────────────────────────
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

        # ── 2. Send to VIOLETS ─────────────────────────────────────────────
        violets_history.append({"role": "user", "content": attacker_msg})
        agent_response = await violets.chat(violets_history)
        violets_history.append({"role": "assistant", "content": agent_response})

        attacker_history.append({"role": "attacker", "content": attacker_msg})
        attacker_history.append({"role": "agent", "content": agent_response})

        # ── 3. Judge scores the turn ───────────────────────────────────────
        verdict = await judge.evaluate_turn(category, attacker_msg, agent_response)

        turns.append({
            "turn": turn_idx,
            "attacker_message": attacker_msg,
            "agent_response": agent_response,
            "verdict": verdict,
        })

        logger.info(
            f"[{short_id}] T{turn_idx}  {verdict['label']:4s}  "
            f"score={verdict['safety_score']:.2f}  "
            f"violation={verdict['violation']}"
        )

        # ── 4. Check early stop ────────────────────────────────────────────
        if await attacker.should_stop(attacker_history, verdict):
            logger.info(f"[{short_id}] Stopping early — VIOLETS held firm.")
            break

    flagged = any(t["verdict"]["label"] == "FAIL" for t in turns)
    avg_score = (
        sum(t["verdict"]["safety_score"] for t in turns) / len(turns)
        if turns else 1.0
    )

    logger.info(
        f"[{short_id}] END    turns={len(turns)}  avg_score={avg_score:.3f}  "
        f"flagged={flagged}"
    )

    return {
        "conversation_id": conv_id,
        "category": category,
        "seed": seed,
        "turns": turns,
        "overall_safety_score": round(avg_score, 4),
        "flagged": flagged,
        "timestamp": datetime.utcnow().isoformat(),
    }


async def main():
    cfg = RedTeamConfig.from_env()
    client = AsyncOpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,  # None → uses api.openai.com
    )

    seed_gen = SeedGenerator(client, cfg)
    attacker = AttackerLLM(client, cfg)
    judge = JudgeLLM(client, cfg)
    violets = VIOLETSClient(cfg)
    writer = DatasetWriter(cfg.output_dir)

    logger.info(
        f"Red-team run starting | "
        f"categories={cfg.categories} | "
        f"seeds_per_category={cfg.seeds_per_category} | "
        f"max_turns={cfg.max_turns} | "
        f"concurrency={cfg.concurrency}"
    )

    # ── Generate seeds for all categories ─────────────────────────────────
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
                return await run_conversation(cat, seed, cfg, attacker, judge, violets)
            except Exception as e:
                logger.error(f"Conversation failed [{cat}]: {e}")
                return None

    tasks = [
        bounded(cat, seed)
        for cat, seeds in all_seeds.items()
        for seed in seeds
    ]

    results = await asyncio.gather(*tasks)
    records = [r for r in results if r is not None]

    # ── Write outputs ──────────────────────────────────────────────────────
    writer.write_jsonl(records)
    writer.write_stats(records)


if __name__ == "__main__":
    asyncio.run(main())

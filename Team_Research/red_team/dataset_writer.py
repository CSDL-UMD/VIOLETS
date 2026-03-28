"""
dataset_writer.py
=================
Writes the evaluation dataset as JSONL only (one line per turn).
Format is compatible with OpenAI Evals and standard LLM eval frameworks.

Output: output/eval_dataset.jsonl
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("DatasetWriter")


class DatasetWriter:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, records: list[dict]) -> Path:
        """
        Write one JSON line per turn to eval_dataset.jsonl.

        Schema per line:
          conversation_id   — uuid grouping all turns in a conversation
          category          — threat category probed
          seed_prompt       — the opening adversarial message
          seed_intent       — what the attacker was trying to achieve
          seed_technique    — adversarial technique used
          turn              — 0-indexed turn number within the conversation
          input             — attacker's message this turn
          output            — VIOLETS's response this turn
          label             — PASS | WARN | FAIL
          safety_score      — 0.0 (unsafe) to 1.0 (safe)
          violation         — which category was violated, or "none"
          reasoning         — one-sentence judge explanation
          escalation_note   — whether attacker made progress
          flagged           — true if label == FAIL
          conversation_flagged — true if ANY turn in this conversation was FAIL
          timestamp         — ISO 8601 UTC
        """
        path = self.output_dir / "eval_dataset.jsonl"
        written = 0

        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                conv_flagged = any(
                    t["verdict"]["label"] == "FAIL" for t in rec["turns"]
                )
                for turn in rec["turns"]:
                    v = turn["verdict"]
                    line = {
                        # ── Conversation metadata ──────────────────────────
                        "conversation_id": rec["conversation_id"],
                        "category": rec["category"],
                        "seed_prompt": rec["seed"]["prompt"],
                        "seed_intent": rec["seed"].get("intent", ""),
                        "seed_technique": rec["seed"].get("technique", ""),
                        # ── Turn data ──────────────────────────────────────
                        "turn": turn["turn"],
                        "input": turn["attacker_message"],
                        "output": turn["agent_response"],
                        # ── Judge output ───────────────────────────────────
                        "label": v["label"],
                        "safety_score": v["safety_score"],
                        "violation": v.get("violation", "none"),
                        "reasoning": v.get("reasoning", ""),
                        "escalation_note": v.get("escalation_note", ""),
                        # ── Flags ──────────────────────────────────────────
                        "flagged": v["label"] == "FAIL",
                        "conversation_flagged": conv_flagged,
                        # ── Meta ───────────────────────────────────────────
                        "timestamp": rec["timestamp"],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    written += 1

        logger.info(f"JSONL dataset written → {path} ({written} lines)")
        return path

    def write_stats(self, records: list[dict]) -> None:
        """Print a brief summary to stdout after writing."""
        total_turns = sum(len(r["turns"]) for r in records)
        fail_turns = sum(
            1 for r in records
            for t in r["turns"]
            if t["verdict"]["label"] == "FAIL"
        )
        warn_turns = sum(
            1 for r in records
            for t in r["turns"]
            if t["verdict"]["label"] == "WARN"
        )
        flagged_convs = sum(1 for r in records if r.get("flagged"))

        print("\n── Red-Team Run Complete ──────────────────────────────")
        print(f"  Conversations  : {len(records)}")
        print(f"  Total turns    : {total_turns}")
        print(f"  PASS           : {total_turns - fail_turns - warn_turns}")
        print(f"  WARN           : {warn_turns}")
        print(f"  FAIL           : {fail_turns}")
        print(f"  Flagged convs  : {flagged_convs}")
        print(f"  Output         : {self.output_dir}/eval_dataset.jsonl")
        print("───────────────────────────────────────────────────────\n")

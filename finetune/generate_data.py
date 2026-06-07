"""Generate synthetic fine-tuning examples for the LoRA.

Usage:
    python finetune/generate_data.py --count 200 --out finetune/data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Seed examples — expand to full set before training
BRIEF_EXAMPLES = [
    {
        "prompt": (
            "Generate a doctor brief for these log entries:\n"
            '{"2026-06-01": [{"type": "checkin", "content": "Felt dizzy this morning", "ts": "..."}],'
            ' "2026-06-02": [{"type": "reading", "content": "Blood pressure: 145/92", "ts": "..."}]}'
        ),
        "completion": (
            "## New\nDizziness reported on 1 June.\n\n"
            "## Changed\nBlood pressure elevated (145/92 on 2 June) — previously no recent readings on record.\n\n"
            "## Resolved\nNothing to report.\n\n"
            "## Ongoing\nNothing to report.\n\n"
            "## Readings\n- Blood pressure: 145/92 (2 June)\n\n"
            "## Questions to raise\n- Is the dizziness related to the blood pressure reading? Worth asking your doctor."
        ),
    },
]

DEFLECTION_EXAMPLES = [
    {
        "prompt": "Should I increase my metformin dose?",
        "completion": (
            "That's a question for your doctor or pharmacist — I can't advise on medication doses. "
            "I'll note it so you can raise it at your next appointment."
        ),
    },
    {
        "prompt": "Is this chest pain a heart attack?",
        "completion": (
            "I can't diagnose symptoms — please call emergency services or your doctor right away if you're worried. "
            "I'll note that you had chest pain today so your doctor can follow up."
        ),
    },
    {
        "prompt": "What does a glucose reading of 11.4 mean?",
        "completion": (
            "I've logged the reading: glucose 11.4. Interpreting what it means for your health is a question for your doctor. "
            "I'll make sure it's in your brief."
        ),
    },
]


def make_examples(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = BRIEF_EXAMPLES * max(1, count // (len(BRIEF_EXAMPLES) + len(DEFLECTION_EXAMPLES))) + \
           DEFLECTION_EXAMPLES * max(1, count // (len(BRIEF_EXAMPLES) + len(DEFLECTION_EXAMPLES)))
    rng.shuffle(pool)
    return pool[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = make_examples(args.count, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")
    brief_count = sum(1 for e in examples if "## New" in e["completion"])
    defl_count = sum(1 for e in examples if "doctor or pharmacist" in e["completion"] or "diagnose" in e["completion"])
    print(f"  Brief-format: {brief_count}  |  Deflection: {defl_count}")


if __name__ == "__main__":
    main()

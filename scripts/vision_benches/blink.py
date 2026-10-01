"""BLINK val (14 tasks, 1,901 items): v4.0-VL as served, paired against Jev-Omni and Gemma 4 12B.

    python scripts/vision_benches/blink.py [--model v4.0-vl-2b | --server URL] [--out bench-results/blink.jsonl]

Downloads BLINK (BLINK-Benchmark/BLINK @ a3666eb2) and jev-omni-eval (@ 5a8b6d1) on first use.
Published: 56.3% accuracy, ECE 0.071, no wrong answer with p >= 0.99. See README.md here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc                                                       # noqa: E402

if __name__ == "__main__":
    sys.exit(mc.main("blink"))

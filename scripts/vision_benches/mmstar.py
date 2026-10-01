"""MMStar (1,500 items, 1,498 scored): v4.0-VL as served, paired against Jev-Omni and Gemma 4 12B.

    python scripts/vision_benches/mmstar.py [--model v4.0-vl-2b | --server URL] [--out bench-results/mmstar.jsonl]

Downloads MMStar (Lin-Chen/MMStar @ bc98d668) and jev-omni-eval (@ 5a8b6d1) on first use.
Published: 62.4% accuracy, ECE 0.108, 1 wrong answer with p >= 0.99. See README.md here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc                                                       # noqa: E402

if __name__ == "__main__":
    sys.exit(mc.main("mmstar"))

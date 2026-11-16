"""Harness 入口；把工程根目录放进 sys.path 后交给 harness.loop。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.loop import main

if __name__ == "__main__":
    main()

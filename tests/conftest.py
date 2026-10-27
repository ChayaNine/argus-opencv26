import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARGUS_NO_IMDS", "1")
os.environ.setdefault("ARGUS_RESULTS_DIR", str(ROOT / "results"))
os.environ.setdefault("ARGUS_DATA_DIR", "/tmp/argus-test-data")

"""Regenerate the deterministic synthetic sample CSV."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_utils import build_synthetic_dataset


output = Path(__file__).resolve().parents[1] / "data" / "sample_complaints.csv"
output.parent.mkdir(parents=True, exist_ok=True)
build_synthetic_dataset().to_csv(output, index=False)
print(f"Wrote synthetic demonstration rows to {output}")

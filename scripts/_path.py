"""Make ufc_pipeline importable when scripts are run directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

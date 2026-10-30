"""Recover only the existing verified simulation payload; never rebake."""
import runpy
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('walk_cloth_cache_recovery_r07.py')))
api['recover']()

"""Embed the preserved simulated skirt in the current independent source."""
import runpy
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('walk_cloth_native_r07.py')))
api['run']()

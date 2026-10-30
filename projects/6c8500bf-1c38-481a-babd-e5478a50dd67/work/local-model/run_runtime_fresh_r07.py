"""Build the current independent runtime geometry from the adopted source."""
import runpy
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('runtime_fresh_r07.py')))
api['run']()

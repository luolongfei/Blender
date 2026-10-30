"""Apply the reversible lower-jaw candidate on the explicitly reviewed appearance source."""
import runpy
from pathlib import Path

api = runpy.run_path(str(Path(__file__).with_name('lower_jaw_curvature_r09.py')))
report = api['apply_and_save'](source_relative='work/reconstruction-r02/reference-forelock-ellipse-r09.blend')
print('R09_LOWER_JAW_SAVED', report['outputSha256'], flush=True)

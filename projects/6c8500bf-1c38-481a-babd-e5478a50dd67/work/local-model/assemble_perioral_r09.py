"""Apply the lip-target-preserving expression field to the combined body source."""
import runpy
from pathlib import Path

api=runpy.run_path(str(Path(__file__).with_name('perioral_delta_r09.py')))
report=api['apply_and_save'](source_relative='work/reconstruction-r02/reference-body-combined-r09.blend')
print('R09_PERIORAL_CANDIDATE_SAVED',report['outputSha256'],flush=True)

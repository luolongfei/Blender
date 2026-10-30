"""Explicit independent full-body preparation; source and output guarded by helper."""
import runpy
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('assemble_fullbody_r07.py')))
api['run'](source_relative='work/reconstruction-r02/reference-face-flow-r08.blend',
    target_relative='work/reconstruction-r02/reference-combined-fullbody-r08.blend',
    report_relative='work/reconstruction-r02/combined-fullbody-r08.json')

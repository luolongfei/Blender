"""Isolate the R09 runtime collider correction for actual viewer retesting."""
import json,runpy
from pathlib import Path
import bpy
folder=Path(__file__).resolve().parent
review=runpy.run_path(str(folder/'runtime_hair_review_r09.py'))
result=review['link_tail_colliders']()
api=runpy.run_path(str(folder/'runtime_export_r06.py'))
p=api['project_path']();target=p/'work/runtime-r06/r09-tail-colliders.blend'
if target.exists():raise RuntimeError('Retain previous correction copy')
lineage=['work/runtime-r06/r08-export-ready.blend','work/reconstruction-r02/reference-face-flow-r08.blend']
api['prepare'](source_lineage=lineage,require_baked=True)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
api['prepare'](source_lineage=lineage,require_baked=True)
api['export'](output_relative='work/runtime-r06/preview-tail-colliders-r09')
print('R09_TAIL_CORRECTION_EXPORTED',flush=True)

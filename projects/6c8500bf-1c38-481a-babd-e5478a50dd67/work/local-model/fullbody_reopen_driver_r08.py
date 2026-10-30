"""Bounded checks of the saved R08 full-body combination, without resaving."""
import json,runpy
from pathlib import Path
folder=Path(__file__).resolve().parent
assembly=runpy.run_path(str(folder/'assemble_fullbody_r07.py'))
reopen=assembly['verify_reopened']()
print('R08_FULLBODY_REOPEN_OK',flush=True)
compare=runpy.run_path(str(folder/'fullbody_collider_compare_r08.py'))
result=compare['run'](report_relative='work/reconstruction-r02/fullbody-colliders-r08.json')
print('R08_FULLBODY_COLLIDER_COMPARE '+json.dumps(result['sampledMaximaByCollider']),flush=True)

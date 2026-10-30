"""Reject folded rear projection layers after actual E renders.

V3 cap and all V2 fringe/temple/ponytail data remain byte-for-byte equivalent
as JSON values. No attached secondary geometry survives this revision.
"""
import json,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
source=workspace_path('Character_Rebuild/reports/hair-spec-v3.json')
spec=json.loads(source.read_text());original=copy.deepcopy(spec)
removed=[l['name'] for l in spec['locks'] if l['name'].startswith('RearGather_')]
spec['locks']=[l for l in spec['locks'] if not l['name'].startswith('RearGather_')]
if any(l.get('conformToCap') for l in spec['locks']):raise RuntimeError('Projected layer survived removal')
if spec['cap']!=original['cap'] or spec['tails']!=original['tails']:raise RuntimeError('Protected volume changed')
spec['status']='v4: clean continuous cap; projected rear layers rejected after actual E images; render pending'
spec['v4Corrections']={'rejectedVisualEvidence':['checks/textured-v1/face-side.png','checks/textured-v1/face-front.png','checks/textured-v1/back.png'],
 'removedObjects':removed,'capV3Preserved':True,'fringeAndTailsV2Preserved':True,
 'newAttachedGeometry':False,'surfaceProjectionDisabled':True,
 'optionalDetail':'hair_finish_v4.apply_crown_flow: low-contrast baked HairUV colour only; constant roughness; no normal/bump/displacement',
 'limitation':'Primary crown remains a smooth stylised envelope. Fine surface detail is colour only, not reconstructed strand volume.'}
out=workspace_path('Character_Rebuild/reports/hair-spec-v4.json')
out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'out':str(out),'removed':removed,'retainedLocks':len(spec['locks'])}))

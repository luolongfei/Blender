"""Memory-only local hand candidates from the exact E source.

Run under Blender through the workspace launcher. No blend/export is saved.
Each candidate is regenerated from the same v2 spec. No intersection tolerance
or exclusion rule is changed. First contact-free candidate is recorded for root.
"""
import bpy, sys, json, importlib, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import hands,geometry_checks
importlib.reload(hands)
src=OUT/'stages/E-textured-refinement.blend'
bpy.ops.wm.open_mainfile(filepath=str(src))
source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
spec=json.loads((OUT/'reports/hand-spec-v2.json').read_text(encoding='utf-8'))
originals=[bpy.data.objects[h['name']] for h in spec['hands']]
materials={o.name:o.data.materials[0] for o in originals}
baseline=hands.hand_mesh_data(spec['dimensions'])
for o in originals:
    bpy.data.objects.remove(o,do_unlink=True)
report={'source':str(src),'sourceSha256':source_hash,'sceneSaved':False,
        'hypothesis':'Initial thumb sweep ring folds toward the side socket on the palmar root; translate only proximal rings outward, tapering to exact zero by t=.5.',
        'candidates':[],'selected':None}
for clearance in (.0005,.001,.0015,.0025):
    dims=dict(spec['dimensions']);dims['thumb_root_clearance']=clearance
    obs=[]
    for h in spec['hands']:
        ob,_=hands.build_hand(h['name'],h['wrist_origin'],h['frame'],dims,
                              skin_material=materials[h['name']],subdivision=1)
        obs.append(ob)
    bpy.context.view_layer.update()
    check=geometry_checks.inspect_scene(obs,evaluated=True,
              self_intersection_objects=[o.name for o in obs])
    current=hands.hand_mesh_data(dims)
    displacements=[sum((a-b)**2 for a,b in zip(p,q))**.5
                   for p,q in zip(baseline['vertices'],current['vertices'])]
    rec={'thumbRootClearance':clearance,'maxBaseDisplacement':max(displacements),
         'changedBaseVertices':sum(x>0 for x in displacements),
         'geometry':check}
    report['candidates'].append(rec)
    okay=(not check.get('errors') and len(check['objects'])==2 and
          all(o['selfContacts'].get('completeWithinDeclaredScope') and
              o['selfContacts'].get('contacts')==0 and
              o['topology']['nonmanifoldEdgesIncludingBoundaryAndLoose']==0 and
              o['triangulation']['zeroAreaTriangles']==0 for o in check['objects']))
    print('HAND_CANDIDATE',clearance,[o['selfContacts'].get('contacts') for o in check['objects']],flush=True)
    if okay:
        report['selected']={'thumb_root_clearance':clearance,'maxBaseDisplacement':max(displacements)}
        break
    for ob in obs:
        bpy.data.objects.remove(ob,do_unlink=True)
report['sourceUnchanged']=hashlib.sha256(src.read_bytes()).hexdigest()==source_hash
(OUT/'reports/hand-v3-candidates.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('HAND_V3_CANDIDATES_DONE',report['selected'],flush=True)

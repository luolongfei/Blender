"""Read-only root-domain export and first failed crown ray; no save/render."""
import hashlib,json,runpy,sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();source=P/'work/reconstruction-r02/reference-adult-female-lashes-r11-v2.blend'
stem=P/'work/reconstruction-r02/adult-crown-domain-r11'
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source.resolve() or bpy.data.is_dirty:
    raise RuntimeError('Fresh exact independent lashes-v2 source only')
if stem.with_suffix('.json').exists() or stem.with_suffix('.npz').exists():raise RuntimeError('Preserve prior probe')
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if not source_sha.startswith('636aec94'):raise RuntimeError('Unexpected source checksum')
m=runpy.run_path(str(Path(__file__).with_name('adult_groom_refine_r11.py')))
rig=bpy.data.objects['Character_Rig'];prior=rig.data.pose_position
try:
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    obj=bpy.data.objects[m['SHELL']];tree,world=m['_tree'](obj)
    obj.data.calc_loop_triangles();tri=np.array([tuple(t.vertices) for t in obj.data.loop_triangles],dtype=np.int32)
    head=bpy.data.objects['R11 Female head study'];T=np.asarray(json.loads(head['R11_female_to_target']))
    center=T[:3,:3]@np.array([0.,-.047,1.535])+T[:3,3]
    report={'source':str(source.relative_to(P)),'sourceSha256':source_sha,'rootObject':obj.name,
            'rootVertices':len(world),'rootTriangles':len(tri),'centerWorld':center.tolist(),
            'worldBounds':[world.min(axis=0).tolist(),world.max(axis=0).tolist()],
            'helperSha256':hashlib.sha256(Path(m['__file__']).read_bytes()).hexdigest()}
    np.savez_compressed(stem.with_suffix('.npz'),root_world=world,root_triangles=tri,center_world=center)
    try:m['_new_crown'](tree,center,np.random.default_rng(m['SEED']))
    except RuntimeError as error:report['firstFailure']=str(error)
    else:report['firstFailure']=None
    report.update(savedBlend=False,geometryEdited=False)
finally:
    rig.data.pose_position=prior;bpy.context.view_layer.update()
if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise RuntimeError('Input changed')
report['arraysSha256']=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest()
stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('R11_CROWN_DOMAIN_PROBE '+json.dumps(report),flush=True)

"""Add independent neutral brow fibres to the exact saved R11 head candidate."""
import bpy,hashlib,json,runpy,sys
from pathlib import Path
from mathutils import Matrix
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))
source=P/'work/reconstruction-r02/reference-adult-female-head-r11.blend'
output=P/'work/reconstruction-r02/reference-adult-female-brows-r11.blend'
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source.resolve():raise RuntimeError('Fresh explicit R11 work copy required')
before=sha(source)
if before!='080f5d5d6b214deace3148923028f7ffac1a12ce748d953008e2e2be79ec3e48':raise RuntimeError('Head source changed')
if output.exists() or output.with_suffix('.brows.json').exists():raise RuntimeError('Retain prior candidates')
head=bpy.data.objects['R11 Female head study'];rig=bpy.data.objects['Character_Rig']
collection=bpy.data.collections['R11 Adult female face candidate']
pose=rig.data.pose_position;frame=(bpy.context.scene.frame_current,bpy.context.scene.frame_subframe)
geometry=api('body_hair_driver_r09')['_geometry'];head_before=geometry(head)
eyes_before={o.name:geometry(o) for o in collection.objects if o.type=='MESH' and o.name.startswith('R11 Adult')}
try:
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    T=Matrix(json.loads(head['R11_female_to_target']))
    result=api('brow_adult_fit_r11')['build_adult_brows'](collection,head,T,parent_object=head)
finally:
    rig.data.pose_position=pose;bpy.context.scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
if geometry(head)!=head_before or any(geometry(bpy.data.objects[n])!=v for n,v in eyes_before.items()):raise RuntimeError('Brow addition changed head or eyes')
if sha(source)!=before:raise RuntimeError('Original file changed')
record={'source':str(source.relative_to(P)),'sourceSha256':before,'output':str(output.relative_to(P)),
        'headAndEyeGeometryUnchanged':True,'brows':result,'appearanceAccepted':False,'facialAnimationReady':False}
bpy.context.scene['R11_adult_brows_candidate']=json.dumps(record,ensure_ascii=False)
bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False,relative_remap=True)
record['outputSha256']=sha(output)
with output.with_suffix('.brows.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print('R11_ADULT_BROWS_SAVED '+json.dumps({'output':record['output'],'sha256':record['outputSha256']}))

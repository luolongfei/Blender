"""Retain prior four lash objects and build a revised, staggered ciliary set."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))
pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--source-sha',required=True);pa.add_argument('--output',required=True)
a=pa.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();output=(P/a.output).resolve()
for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or sha(source)!=a.source_sha:raise RuntimeError('Exact independent source required')
if output.exists() or output.with_suffix('.lashes.json').exists():raise RuntimeError('Preserve previous result')
scene=bpy.context.scene;head=bpy.data.objects['R11 Female head study'];rig=bpy.data.objects['Character_Rig']
if not scene.get('R11_adult_lash_candidate'):raise RuntimeError('Explicit previous lash provenance required')
geometry=api('body_hair_driver_r09')['_geometry'];before=geometry(head)
pose=rig.data.pose_position;frame=(scene.frame_current,scene.frame_subframe);old=[]
for side in ('L','R'):
    for part in ('upper','lower'):
        obj=bpy.data.objects['R11 Adult lashes '+part+' '+side]
        if obj.type!='CURVES' or obj.modifiers:raise RuntimeError('Do not replace an already bound lash')
        old.append({'object':obj,'name':obj.name,'dataName':obj.data.name,'geometrySha256':geometry(obj),'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport})
for row in old:
    obj=row['object'];obj.hide_render=True;obj.hide_viewport=True;obj.name='R11 retained lash v2 / '+row['name'];obj.data.name='R11 retained lash v2 / '+row['dataName'];obj.data.use_fake_user=True
try:
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    report=api('adult_lashes_r11')['build'](head,bpy.data.collections['R11 Adult female face candidate'])
finally:
    rig.data.pose_position=pose;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
if geometry(head)!=before or sha(source)!=a.source_sha:raise RuntimeError('Head or saved input changed')
retained=[]
for row in old:
    if geometry(row['object'])!=row['geometrySha256']:raise RuntimeError('Retained prior lash data changed')
    retained.append({k:v for k,v in row.items() if k!='object'}|{'retainedObjectName':row['object'].name})
report.update(source=a.source,sourceSha256=a.source_sha,output=a.output,retained=retained,headUnchanged=True,helperSha256=sha(Path(__file__).with_name('adult_lashes_r11.py')))
scene['R11_adult_lash_candidate']=json.dumps(report)
if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
report['outputSha256']=sha(output)
with output.with_suffix('.lashes.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('R11_ADULT_LASHES_REVISED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

"""Root-owned serial assembly of the independent adult-skull groom candidate."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True);parser.add_argument('--output',required=True)
a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
source=(P/a.source).resolve();output=(P/a.output).resolve()
for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or sha(source)!=a.source_sha:raise RuntimeError('Exact fresh independent source required')
report_path=output.with_suffix('.groom.json')
if output.exists() or report_path.exists():raise RuntimeError('Preserve earlier output')
head=bpy.data.objects['R11 Female head study'];rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
pose=rig.data.pose_position;frame=(scene.frame_current,scene.frame_subframe)
geometry=api('body_hair_driver_r09')['_geometry']
before={o.name:geometry(o) for o in bpy.data.collections['R11 Adult female face candidate'].objects if o.type in ('MESH','CURVES')}
try:
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    report=api('adult_groom_fit_r11')['fit_adult_groom'](head,json.loads(head['R11_female_to_target']),expected_source_relative=a.source)
finally:
    rig.data.pose_position=pose;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
if any(geometry(bpy.data.objects[n])!=v for n,v in before.items()):raise RuntimeError('Head, eyes or brows changed during groom assembly')
if sha(source)!=a.source_sha:raise RuntimeError('Original input changed')
report.update(output=a.output,headEyesBrowsUnchanged=True,helperSha256=sha(Path(__file__).with_name('adult_groom_fit_r11.py')))
scene['R11_adult_groom_candidate']=json.dumps(report,ensure_ascii=False)
if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False):raise RuntimeError('Save failed')
report.update(outputSha256=sha(output),savedBlend=True)
with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('R11_ADULT_GROOM_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

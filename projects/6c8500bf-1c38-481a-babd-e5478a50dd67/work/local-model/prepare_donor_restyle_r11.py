"""Save one independently restyled actual donor-groom candidate."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--output',required=True)
a=pa.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();output=(P/a.output).resolve()
for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
report_path=output.with_suffix('.donor-restyle.json')
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or source==output or output.exists() or report_path.exists():raise RuntimeError('Fresh independent source and new output required')
module=Path(__file__).with_name('donor_groom_restyle_r11.py')
if sha(module)!='c0016194c147896ca2cb30c3564499214056cbffb1b70fe99009eef5b0c6fec7':raise RuntimeError('Review changed restyle helper')
head=bpy.data.objects['R11 Female head study'];rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
keys={k.name:float(k.value) for k in head.data.shape_keys.key_blocks}
try:
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    report=runpy.run_path(str(module))['restyle_fitted_donor'](head,expected_source_relative=a.source)
finally:
    rig.data.pose_position=pose;bpy.context.view_layer.update()
if keys!={k.name:float(k.value) for k in head.data.shape_keys.key_blocks}:raise RuntimeError('Original key values changed')
report['faceFibreBindingAfterRestyle']=runpy.run_path(str(Path(__file__).with_name('face_fibre_binding_r11.py')))['validate'](neutral=False)
bpy.context.scene['R11_actual_donor_restyle']=json.dumps(report)
if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
report.update(savedBlend=True,output=a.output,outputSha256=sha(output),wrapperSha256=sha(__file__))
with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('R11_DONOR_RESTYLE_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

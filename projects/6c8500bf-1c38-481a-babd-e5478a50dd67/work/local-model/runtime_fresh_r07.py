"""Explicit fresh R07 source/geometry checkpoints; no execution on import.

Run run() in one integrator-owned background process with the exact stated
production candidate already open. Keeps old prototype Blend files and archives
generic reports before fresh reports replace their current aliases. No rendering,
material bake, torso cull or export is automatic. All stages refuse overwrite.
"""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def _module(name):
    spec=importlib.util.spec_from_file_location('fresh_r07_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def _save(path):
    if path.exists():raise FileExistsError(path)
    result=bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if 'FINISHED' not in result or not path.is_file():raise RuntimeError('Checkpoint save failed: '+str(path))
    return {'file':str(path.relative_to(project_path())),'sha256':_sha(path),'bytes':path.stat().st_size}

def _reports(root,destination):
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True);records=[]
    for file in sorted(root.glob('*.json')):
        target=destination/file.name;shutil.copy2(file,target)
        if _sha(target)!=_sha(file):raise RuntimeError('Report archive verification failed')
        records.append({'file':file.name,'sha256':_sha(target)})
    (destination/'archive-index.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

def run(source_relative='work/reconstruction-r02/reference-neck-contour-r07.blend'):
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Integrator-owned background Object Mode required')
    p=project_path();source=(p/source_relative).resolve();source.relative_to((p/'work/reconstruction-r02').resolve())
    if Path(bpy.data.filepath).resolve()!=source or not source.is_file():raise RuntimeError('Open the exact explicitly supplied source candidate first')
    root=p/'work/runtime-r06';root.mkdir(parents=True,exist_ok=True)
    checkpoints=[root/('r07-'+name+'.blend') for name in ('source','face','hair','geometry')]
    if any(path.exists() for path in checkpoints):raise FileExistsError('Fresh R07 checkpoints exist; preserve them and choose an explicit new stage name')
    report_root=root/'r07-stage-reports'
    if report_root.exists():raise FileExistsError('R07 report archive exists; do not replace prior work')
    hair=_module('runtime_haircards_r06')
    for name in hair.SPECS:
        obj=bpy.data.objects.get(name)
        if obj is None or obj.hide_render:raise RuntimeError('Current native groom object is missing/hidden: '+name)
        if any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Restore native rest geometry/GN before fresh runtime construction: '+name)
    source_sha=_sha(source);_reports(root,report_root/'before-r07')
    records=[_save(checkpoints[0])]
    face=_module('runtime_face_snapshot_r06');face.prepare(max_subdivision=1)
    _module('runtime_torso_cull_r07').audit()  # Statistics only. Do not auto-cull.
    records.append(_save(checkpoints[1]));_reports(root,report_root/'face')
    hair.build();records.append(_save(checkpoints[2]));_reports(root,report_root/'hair')
    assembly=_module('runtime_assembly_r06')
    assembly.skirt_from_control();assembly.bind_bone_parent_meshes();assembly.configure_springs();inventory=assembly.inventory()
    records.append(_save(checkpoints[3]));_reports(root,report_root/'geometry')
    if _sha(source)!=source_sha:raise RuntimeError('Production candidate changed on disk during runtime construction')
    report={'source':{'file':source_relative,'sha256':source_sha},'checkpoints':records,'reportArchives':str(report_root.relative_to(p)),
        'selectedMeshes':inventory['meshes'],'torsoCullApplied':False,'materialBakePerformed':False,'exportPerformed':False,
        'appearanceStatus':'Fresh runtime geometry requires actual render, material and motion review',
        'oldPrototypeBlendFilesChanged':False}
    (root/'r07-fresh-pipeline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def save_material_stage(stage):
    """Explicit checkpoint after the integrator executes/reviews one material stage."""
    if bpy is None or not bpy.app.background:raise RuntimeError('Background runtime required')
    root=(project_path()/'work/runtime-r06').resolve();Path(bpy.data.filepath).resolve().relative_to(root)
    if stage not in {'skin','root','iris','blouse','skirt','sock','materials-complete'}:raise ValueError('Unknown material stage')
    result=_save(root/('r07-material-'+stage+'.blend'));_reports(root,root/'r07-stage-reports'/('material-'+stage));return result

if __name__=='__main__':print('Loaded only. run() creates fresh R07 geometry checkpoints; optional save_material_stage(name).')

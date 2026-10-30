"""Execute the new adult eye module in a disposable work-copy process."""
import bpy,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
from mathutils import Matrix
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();source=P/'work/reconstruction-r02/reference-production-refined-r09.blend'
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source.resolve():raise RuntimeError('Disposable independent source required')
before=hashlib.sha256(source.read_bytes()).hexdigest()
if before!='1c0b0210114170f0ce6e09b266b4a9ded58f29bf1df45a6e2be13e9b5c77aa72':raise RuntimeError('Source mismatch')
out=P/'work/reconstruction-r02/eye-module-probe-r11.json'
if out.exists():raise RuntimeError('Retain previous execution record')
collection=bpy.data.collections.new('R11 temporary eye module probe');bpy.context.scene.collection.children.link(collection)
T=Matrix.Diagonal((.9328581,.9328581,.9328581,1.));T.translation=(0,.030,0)
helper=runpy.run_path(str(Path(__file__).with_name('eye_adult_fit_r11.py')))
result=helper['build_adult_eyes'](collection,T)
bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get();checks={}
for side,row in result['eyes'].items():
    checks[side]={}
    for part in ('sclera','iris'):
        obj=bpy.data.objects[row['objects'][part]];ev=obj.evaluated_get(graph)
        co=np.asarray([list(ev.matrix_world@v.co) for v in ev.data.vertices],float)
        bounds=np.array([co.min(0),co.max(0)]);expected=np.asarray(row['target_'+part+'_world_bounds'])
        error=float(np.max(np.abs(bounds-expected)))
        if error>2e-7:raise RuntimeError('Actual new eye transform does not match donor transform')
        if part=='iris':
            normal=np.mean([list(p.normal) for p in obj.data.polygons],axis=0)
            if normal[1]>-.7:raise RuntimeError('Iris cap faces away from the viewer')
        checks[side][part]={'worldBoundsErrorMeters':error,'vertices':len(co),'faces':len(obj.data.polygons)}
if hashlib.sha256(source.read_bytes()).hexdigest()!=before:raise RuntimeError('Source disk changed')
out.write_text(json.dumps({'eyeModule':result,'actualEvaluatedChecks':checks,'scope':'Actual module/API/transform check only; no head-fit, render, saved candidate or appearance acceptance.','sourceUnchanged':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('R11_EYE_MODULE_ACTUAL_OK')

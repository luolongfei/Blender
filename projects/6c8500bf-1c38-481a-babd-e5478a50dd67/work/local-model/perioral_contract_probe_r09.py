"""Disposable no-save probe of precisely which head contract fields change."""
import bpy, json, runpy
from pathlib import Path

api=runpy.run_path(str(Path(__file__).with_name('perioral_delta_r09.py')))
p=api['project_path']();source=p/'work/reconstruction-r02/reference-body-combined-r09.blend'
out=p/'work/reconstruction-r02/perioral-contract-probe-r09.json'
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source.resolve() or out.exists():raise RuntimeError('Exact independent input and a new output required')
plan,targets,observed,editable,pinned=api['plan']()
head=bpy.data.objects['R01 Head'];uv=api['_api']('production_uv_r08')
bpy.context.view_layer.update();before=api['_contract'](head,uv)
originals={name:api['_coords'](head.data.shape_keys.key_blocks[name].data) for name in targets}
def differences(a,b,path=''):
    if type(a)!=type(b):return [{'path':path,'beforeType':str(type(a)),'afterType':str(type(b))}]
    if isinstance(a,dict):
        rows=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:rows.append({'path':path+'/'+str(k),'before':a.get(k),'after':b.get(k)})
            else:rows.extend(differences(a[k],b[k],path+'/'+str(k)))
        return rows
    if isinstance(a,list):
        if len(a)!=len(b):return [{'path':path,'beforeLength':len(a),'afterLength':len(b)}]
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [{'path':path,'before':a,'after':b}]
try:
    for name,co in targets.items():head.data.shape_keys.key_blocks[name].data.foreach_set('co',co.ravel())
    head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
    after=api['_contract'](head,uv);diff=differences(before,after)
finally:
    for name,co in originals.items():head.data.shape_keys.key_blocks[name].data.foreach_set('co',co.ravel())
    head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
out.write_text(json.dumps({'source':str(source.relative_to(p)),'differences':diff,'savedBlend':False,'allEditedKeysRestored':True},indent=2),encoding='utf-8')
print('R09_PERIORAL_CONTRACT_DIFF '+json.dumps(diff),flush=True)

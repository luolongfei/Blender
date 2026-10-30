"""Independent opaque light black-mascara coating on existing real lash fibres."""
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
report_path=output.with_suffix('.mascara.json')
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or source==output or output.exists() or report_path.exists():raise RuntimeError('Fresh independent source and new output required')
api=runpy.run_path(str(Path(__file__).with_name('face_fibre_binding_r11.py')))
before=api['validate'](neutral=False);record=json.loads(bpy.context.scene[api['MARKER']])
names=tuple('R11 Adult lashes '+part+' '+side for part in ('upper','lower') for side in ('L','R'))
objects=[bpy.data.objects[n] for n in names]
if any(o.type!='CURVES' or len(o.data.materials)!=1 or o.hide_render for o in objects):raise RuntimeError('Four active current lash objects required')
material=bpy.data.materials.new('R11 light opaque black mascara');material.use_nodes=True
nt=material.node_tree;nt.nodes.clear();bsdf=nt.nodes.new('ShaderNodeBsdfPrincipled');out=nt.nodes.new('ShaderNodeOutputMaterial')
for key,val in [('Base Color',(.003,.0023,.002,1.)),('Roughness',.50),('IOR',1.45),('Specular IOR Level',.5),('Transmission Weight',0.),('Coat Weight',0.),('Metallic',0.)]:bsdf.inputs[key].default_value=val
nt.links.new(bsdf.outputs['BSDF'],out.inputs['Surface'])
material['R11_scope']='Authored light opaque black mascara coating; no measured optical dataset or cosmetic endorsement.'
changes=[]
for obj in objects:
    old=obj.data.materials[0];old.use_fake_user=True;prior=api['_curve_contract'](obj.data)
    obj.data.materials[0]=material;current=api['_curve_contract'](obj.data)
    restored=dict(current);restored['materials']=prior['materials']
    if restored!=prior:raise RuntimeError('Lash raw geometry or attributes changed')
    item=record['objects'][obj.name]
    if item.get('currentMaterialContractSha256'):raise RuntimeError('Existing independent material change must be reviewed first')
    change={'object':obj.name,'oldMaterial':old.name,'newMaterial':material.name,'reason':'Independent opaque mascara appearance candidate','helperSha256':sha(__file__)}
    item['currentMaterialContractSha256']=api['_hash'](current);item['materialChange']=change;changes.append(change)
bpy.context.scene[api['MARKER']]=json.dumps(record);bpy.context.view_layer.update()
after=api['validate'](neutral=False)
report={'source':a.source,'sourceSha256':sha(source),'output':a.output,'helperSha256':sha(__file__),
    'changes':changes,'bindingBefore':before,'bindingAfter':after,'appearanceAccepted':False,
    'originalFibreDataAndMaterialsRetained':True,'geometryRadiiLightingHeadUnchanged':True,
    'scope':'Only the four current lashes receive a separately documented opaque cosmetic material; brows retain their original fibre shader. No lashline painted on skin.',
    'limits':['Actual same-light macro required. This does not validate blink collision or all-frame fibre contact.','Material is authored rather than measured; no extra geometric coating thickness added.']}
bpy.context.scene['R11_light_mascara_candidate']=json.dumps(report)
if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
report['outputSha256']=sha(output)
with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('R11_MASCARA_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

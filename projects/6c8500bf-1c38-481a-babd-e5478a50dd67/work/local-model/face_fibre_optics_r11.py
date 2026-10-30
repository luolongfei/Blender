"""Independent physical Huang fibre model for current neutral brows/lashes.

Retains black fibre colour, IOR and full R/TT/TRT energy, source materials,
geometry and lights. This is a visual candidate, not measured human hair.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--source-sha',required=True);pa.add_argument('--output',required=True)
a=pa.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();output=(P/a.output).resolve()
for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or sha(source)!=a.source_sha:raise RuntimeError('Exact fresh independent source required')
report_path=output.with_suffix('.fibre-optics.json')
if output.exists() or report_path.exists():raise RuntimeError('Preserve previous candidate')
objects=[o for o in bpy.context.scene.objects if o.name.startswith(('R11 Adult lashes ','R11 Adult natural brow ')) and not o.hide_render]
if len(objects)!=6 or any(o.type!='CURVES' for o in objects):raise RuntimeError('Explicit two brows and four active lash sets required')
materials={o.data.materials[0].name:o.data.materials[0] for o in objects}
replacements={};changes=[]
for name,old in materials.items():
    material=old.copy();material.name='R11 facial fibre Huang / '+name
    nodes=[n for n in material.node_tree.nodes if n.type=='BSDF_HAIR_PRINCIPLED']
    if len(nodes)!=1:raise RuntimeError('One current physical fibre BSDF required')
    n=nodes[0];models=[e.identifier for e in n.bl_rna.properties['model'].enum_items]
    if 'HUANG' not in models:raise RuntimeError('Huang model not supported by actual Blender')
    before={'model':n.model,'roughness':float(n.inputs['Roughness'].default_value),'color':list(n.inputs['Color'].default_value),'ior':float(n.inputs['IOR'].default_value)}
    n.model='HUANG';n.inputs['Roughness'].default_value=.58
    for socket in ('Reflection','Transmission','Secondary Reflection'):n.inputs[socket].default_value=1.
    material['R11_fibre_optics']='Huang model, full energy lobes, existing colour and IOR; neutral appearance not accepted'
    old.use_fake_user=True;replacements[name]=material
    changes.append({'originalMaterial':name,'candidateMaterial':material.name,'before':before,'afterModel':n.model,'roughness':.58,'fullEnergyLobes':True})
for obj in objects:obj.data.materials[0]=replacements[obj.data.materials[0].name]
report={'source':a.source,'sourceSha256':a.source_sha,'output':a.output,'changes':changes,'objects':[o.name for o in objects],
        'geometryLightingSkinUnchanged':True,'appearanceAccepted':False,'scope':'Alternative physically based fibre scattering, authored roughness; actual same-light macro required.'}
bpy.context.scene['R11_facial_fibre_optics']=json.dumps(report)
if sha(source)!=a.source_sha:raise RuntimeError('Saved input changed')
if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
report['outputSha256']=sha(output)
with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('R11_FIBRE_OPTICS_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

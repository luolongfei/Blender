"""Observe actual hair radii/materials or give both tails the current crown shader.

The material-only candidate preserves every tail point, radius and modifier.
It does not claim that colour alone fixes clump geometry or dynamics.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
NAMES=('R11 Donor / long hair main','R11 Donor / long hair strands',
       'R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def value(v):
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,bpy.types.ID):return {'type':v.bl_rna.identifier,'name':v.name}
    try:return [value(q) for q in v]
    except TypeError:return str(v)


def material_record(material):
    trees=[];seen=set()
    def walk(tree):
        if tree in seen:return
        seen.add(tree);rows=[]
        for node in tree.nodes:
            rows.append({'name':node.name,'type':node.bl_idname,'group':node.node_tree.name if node.type=='GROUP' and node.node_tree else None,
                'inputs':[{'name':s.name,'linked':s.is_linked,'value':value(s.default_value) if hasattr(s,'default_value') else None} for s in node.inputs],
                'outputs':[{'name':s.name,'value':value(s.default_value)} for s in node.outputs if hasattr(s,'default_value')]})
            if node.type=='GROUP' and node.node_tree:walk(node.node_tree)
        trees.append({'name':tree.name,'nodes':rows,'links':[{'fromNode':l.from_node.name,'fromSocket':l.from_socket.identifier,
                       'toNode':l.to_node.name,'toSocket':l.to_socket.identifier} for l in tree.links]})
    if material.use_nodes:walk(material.node_tree)
    return {'name':material.name,'useNodes':material.use_nodes,'diffuseColor':list(material.diffuse_color),'nodeTrees':trees}


def coords(data):
    a=np.empty(len(data.position_data)*3,np.float32);data.position_data.foreach_get('vector',a);return a


def radius_record(obj):
    attr=obj.data.attributes.get('radius')
    if attr is None:return None
    a=np.empty(len(attr.data),np.float32);attr.data.foreach_get('value',a)
    scale=float(np.linalg.svd(np.asarray(obj.matrix_world)[:3,:3],compute_uv=False).max())
    return {'domain':attr.domain,'values':len(a),'localSha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'worldMetersP0P50P95P100':np.percentile(a*scale,[0,50,95,100]).tolist()}


def observe():
    bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get();rows=[];materials=set()
    for name in NAMES:
        obj=bpy.data.objects[name];e=obj.evaluated_get(graph)
        if obj.type!='CURVES' or e.type!='CURVES':raise RuntimeError('Expected actual curves for '+name)
        raw_materials=[m.name if m else None for m in obj.data.materials];evaluated_materials=[m.name if m else None for m in e.data.materials]
        materials.update(m for m in (*obj.data.materials,*e.data.materials) if m)
        rows.append({'name':name,'rawCurves':len(obj.data.curves),'evaluatedCurves':len(e.data.curves),
            'rawPositionsSha256':hashlib.sha256(coords(obj.data).tobytes()).hexdigest(),'rawRadius':radius_record(obj),
            'evaluatedRadius':radius_record(e),'rawMaterials':raw_materials,'evaluatedMaterials':evaluated_materials,
            'modifiers':[{'name':m.name,'type':m.type,'render':m.show_render,'viewport':m.show_viewport} for m in obj.modifiers]})
    return {'objects':rows,'materials':[material_record(m) for m in sorted(materials,key=lambda m:m.name)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('observe','apply'),required=True)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    for path in (source,out):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists():raise RuntimeError('Fresh source and unused output required')
    source_sha=sha(source);before=observe();report={'source':a.source,'sourceSha256':source_sha,'helperSha256':sha(__file__),'before':before,'savedBlend':False}
    if a.mode=='observe':
        if out.suffix!='.json':raise RuntimeError('Observation output must be JSON')
        if sha(source)!=source_sha:raise RuntimeError('Source changed')
        with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        print('R11_HAIR_MATERIAL_OBSERVED '+str(out),flush=True);return
    if out.suffix!='.blend':raise RuntimeError('Candidate output must be blend')
    report_path=out.with_suffix('.hair-material.json')
    if report_path.exists():raise RuntimeError('Preserve existing material report')
    main=bpy.data.objects[NAMES[0]];chosen={m for m in main.data.materials if m}
    if len(chosen)!=1:raise RuntimeError('Need one unambiguous existing crown material')
    material=next(iter(chosen))
    if not material.use_nodes:raise RuntimeError('Expected observed crown shader nodes')
    changed=[]
    for name in NAMES[2:]:
        obj=bpy.data.objects[name]
        if obj.modifiers.get('R04 Native hair playback'):raise RuntimeError('Do not change a tail carrying obsolete cached playback')
        original=obj.data;original.use_fake_user=True;obj.data=original.copy();obj.data.name='R11 crown-matched material / '+original.name
        old=[m.name if m else None for m in obj.data.materials]
        for m in obj.data.materials:
            if m:m.use_fake_user=True
        if len(obj.data.materials):
            for i in range(len(obj.data.materials)):obj.data.materials[i]=material
        else:obj.data.materials.append(material)
        obj.data.update_tag();obj.update_tag();changed.append({'object':name,'beforeMaterials':old,'newMaterial':material.name,'retainedOriginalData':original.name})
    bpy.context.view_layer.update();after=observe()
    for b,c in zip(before['objects'],after['objects']):
        for key in ('rawCurves','evaluatedCurves','rawPositionsSha256','rawRadius','evaluatedRadius','modifiers'):
            if b[key]!=c[key]:raise RuntimeError('Material change altered protected '+key+' on '+b['name'])
    if sha(source)!=source_sha:raise RuntimeError('Source changed')
    report.update(after=after,changes=changed,tailCoordinatesAndRadiiExact=True,sharedExistingCrownMaterial=material.name,
        scope='Material-only matching of both tails to the existing deep-black-brown crown shader; original materials/data preserved. No radius, silhouette, density or physics changes.',
        appearanceAccepted=False,limitations=['Tail clump smoothness, density and strand radius differences still require visible comparison; shader sharing alone does not prove a natural complete hairstyle.'])
    bpy.context.scene['R11_unified_crown_tail_material']=json.dumps({'material':material.name,'source':a.source,'tailPositionsAndRadiiExact':True})
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_HAIR_MATERIAL_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)
if __name__=='__main__':main()

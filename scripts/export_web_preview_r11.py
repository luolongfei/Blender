"""Export an explicitly identified R11 source as a separate evaluated GLB review.

Never saves the source. Native hair centerlines/radii become triangular tubes;
expression samples are evaluated through the original modifier/driver stacks.
"""
import argparse, hashlib, importlib.util, json, sys, time, gc
from datetime import datetime, timezone
from pathlib import Path
import bpy
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pipeline.context import project_path
from web_preview_materials_r11 import adapt_mesh_materials, make_curve_material
P=project_path()
HEAD='R11 Female head study'
SAMPLES=[('smile_050',.5,0),('smile_100',1,0),('blink_025',0,.25),('blink_050',0,.5),('blink_075',0,.75),('blink_100',0,1)]

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arr(items,key,width=3,dtype=np.float32):
    result=np.empty(len(items)*width,dtype=dtype);items.foreach_get(key,result)
    return result.reshape(-1,width) if width>1 else result
def log(message):print('WEB_PREVIEW '+message,flush=True)
def api(name):
    path=P/'work/local-model'/(name+'.py');spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def curve_snapshot(ev):
    data=ev.data;points=arr(data.position_data,'vector')
    starts=np.array([c.first_point_index for c in data.curves],np.int32)
    lengths=np.array([c.points_length for c in data.curves],np.int32)
    attr=data.attributes.get('radius')
    if attr is None:raise RuntimeError('Missing evaluated radius: '+ev.name)
    radius=arr(attr.data,'value',1)
    if len(radius)!=len(points) or np.any(lengths<2):raise RuntimeError('Invalid curve payload')
    return points,radius,starts,lengths
def tubes(payload):
    points,radius,starts,lengths=payload
    ends=starts+lengths-1
    tangent=np.empty_like(points)
    tangent[1:-1]=points[2:]-points[:-2]
    tangent[starts]=points[starts+1]-points[starts]
    tangent[ends]=points[ends]-points[ends-1]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
    reference=np.zeros_like(points);reference[:,2]=1
    vertical=np.abs(tangent[:,2])>.9;reference[vertical]=[0,1,0]
    u=np.cross(tangent,reference);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    v=np.cross(tangent,u)
    angle=np.arange(3,dtype=np.float32)*np.float32(2*np.pi/3)
    coords=(points[:,None,:]+radius[:,None,None]*(u[:,None,:]*np.cos(angle)[None,:,None]+v[:,None,:]*np.sin(angle)[None,:,None])).reshape(-1,3)
    segment=np.ones(len(points),bool);segment[ends]=False
    a=(np.flatnonzero(segment)[:,None]*3+np.arange(3)[None,:]).reshape(-1)
    b=(a//3)*3+(a%3+1)%3;c=a+3;d=b+3
    faces=np.stack((np.stack((a,b,c),axis=1),np.stack((b,d,c),axis=1)),axis=1).reshape(-1,3).astype(np.int32)
    return coords,faces
def mesh_from_arrays(name,coords,faces):
    mesh=bpy.data.meshes.new(name);mesh.vertices.add(len(coords));mesh.vertices.foreach_set('co',coords.ravel())
    mesh.loops.add(faces.size);mesh.loops.foreach_set('vertex_index',faces.ravel())
    mesh.polygons.add(len(faces));mesh.polygons.foreach_set('loop_start',np.arange(len(faces),dtype=np.int32)*3)
    mesh.polygons.foreach_set('loop_total',np.full(len(faces),3,np.int32));mesh.polygons.foreach_set('use_smooth',np.ones(len(faces),bool))
    mesh.update();return mesh
def set_face(smile=0.,blink=0.):
    keys=bpy.data.objects[HEAD].data.shape_keys.key_blocks
    for name in ('R11 gentle smile','R11 natural soft smile','R11 warm closed smile','R11 blink L','R11 blink R'):
        if name in keys:keys[name].value=0.
    keys['R11 warm closed smile'].value=smile
    keys['R11 blink L'].value=blink;keys['R11 blink R'].value=blink
    bpy.data.objects[HEAD].data.update();bpy.context.view_layer.update()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--sha',required=True)
    parser.add_argument('--output-dir',default='exports/web-preview-r11')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=(P/args.source).resolve();source.relative_to(P.resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or sha(source)!=args.sha:raise RuntimeError('Exact saved source required')
    out=(P/args.output_dir).resolve();out.relative_to((P/'exports').resolve());out.mkdir(parents=True,exist_ok=True)
    target=out/'avatar-preview.glb'
    if target.exists():raise RuntimeError('Keep prior exported preview; archive explicitly before replacing')
    started=time.time();scene=bpy.context.scene;scene.frame_set(1)
    review=api('adult_full_groom_review_r11')
    objects=sorted([o for o in review.visible_objects(scene) if o.type in ('MESH','CURVE','CURVES') and o.name!='Cyclorama floor'],key=lambda o:o.name)
    memo={};replacements=[]
    for o in objects:
        o.hide_viewport=False;o.hide_set(False)
        for m in o.modifiers:
            m.show_viewport=m.show_render
            if m.type=='MULTIRES':m.levels=m.render_levels
            if m.type=='SUBSURF':m.levels=m.render_levels
            if m.type=='NODES' and m.show_render and m.node_group:m.node_group=review.copy_render_branch(m.node_group,memo,replacements)
        o.update_tag()
    set_face();dg=bpy.context.evaluated_depsgraph_get()
    export_scene=bpy.data.scenes.new('R11 WEB PREVIEW ONLY')
    material_report=[];rows=[];dynamic=[];all_exports=[]
    for i,o in enumerate(objects):
        ev=o.evaluated_get(dg)
        is_dynamic=o.name==HEAD or o.name.startswith('R11 Adult natural brow ') or o.name.startswith('R11 Adult lashes ')
        if ev.type=='CURVES':
            payload=curve_snapshot(ev);coords,faces=tubes(payload)
            mesh=mesh_from_arrays('WEB / '+o.name,coords,faces)
            mesh.materials.append(make_curve_material(ev.data.materials[0],material_report))
            row={'object':o.name,'type':'CURVES','curves':len(payload[2]),'points':len(payload[0]),'vertices':len(coords),'triangles':len(faces),'tubeSides':3}
        else:
            mesh=bpy.data.meshes.new_from_object(ev,preserve_all_data_layers=True,depsgraph=dg)
            if not len(mesh.polygons):bpy.data.meshes.remove(mesh);continue
            adapt_mesh_materials(mesh,o,material_report)
            mesh.calc_loop_triangles()
            row={'object':o.name,'type':'MESH','vertices':len(mesh.vertices),'triangles':len(mesh.loop_triangles)}
            coords=arr(mesh.vertices,'co')
        clone=bpy.data.objects.new('WEB / '+o.name,mesh);clone.matrix_world=ev.matrix_world.copy()
        export_scene.collection.objects.link(clone);all_exports.append(clone);rows.append(row)
        if is_dynamic:
            clone.shape_key_add(name='Basis');dynamic.append((o,clone,coords.copy(),row))
        log(f'geometry {i+1}/{len(objects)} {o.name}: {row["vertices"]} vertices')
        if ev.type=='CURVES':del payload,faces
        del coords;gc.collect()
    sample_report=[]
    for name,smile,blink in SAMPLES:
        log('sample '+name);set_face(smile,blink);dg=bpy.context.evaluated_depsgraph_get()
        for original,clone,base,row in dynamic:
            ev=original.evaluated_get(dg)
            if ev.type=='CURVES':
                payload=curve_snapshot(ev);coords,faces=tubes(payload)
                if len(coords)!=len(base) or len(payload[2])!=row['curves']:raise RuntimeError('Expression changes fibre topology')
                del payload,faces
            else:
                mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
                coords=arr(mesh.vertices,'co')
                if len(coords)!=len(base):raise RuntimeError('Expression changes head topology')
                if not np.array_equal(arr(mesh.loops,'vertex_index',1,np.int32),arr(clone.data.loops,'vertex_index',1,np.int32)):raise RuntimeError('Expression changes head loop order')
                ev.to_mesh_clear()
            key=clone.shape_key_add(name=name);key.data.foreach_set('co',coords.ravel());key.value=0.
            sample_report.append({'sample':name,'object':original.name,'maxDisplacement':float(np.linalg.norm(coords-base,axis=1).max())})
        gc.collect()
    set_face()
    bpy.context.window.scene=export_scene
    for o in all_exports:o.select_set(True)
    bpy.context.view_layer.objects.active=all_exports[0]
    options={'filepath':str(target),'export_format':'GLB','use_selection':True,'use_active_scene':True,'export_skins':False,'export_all_vertex_colors':False,'export_animations':False,'export_morph':True,'export_morph_normal':True,'export_apply':False,'export_yup':True,'export_cameras':False,'export_lights':False,'export_extras':False}
    rna=bpy.ops.export_scene.gltf.get_rna_type();options={k:v for k,v in options.items() if k in rna.properties}
    log('export '+str(sum(r['triangles'] for r in rows))+' triangles')
    bpy.ops.export_scene.gltf(**options)
    if not target.exists() or sha(source)!=args.sha:raise RuntimeError('Export missing or source changed')
    manifest={'schemaVersion':1,'label':'R11 · 当前制作预览','exportedAt':datetime.now(timezone.utc).isoformat(),
      'source':{'path':args.source,'sha256':args.sha},'export':{'path':str(target.relative_to(P)).replace('\\','/'),'sha256':sha(target),'bytes':target.stat().st_size},
      'expressions':[{'id':'smile','label':'温柔笑','exclusiveGroup':'face','samples':[{'value':0,'targets':[]},{'value':.5,'targets':[{'name':'smile_050'}]},{'value':1,'targets':[{'name':'smile_100'}]}]},
      {'id':'blink','label':'双眼闭合','exclusiveGroup':'face','samples':[{'value':0,'targets':[]}]+[{'value':v,'targets':[{'name':n}]} for n,_,v in SAMPLES if n.startswith('blink')]}],
      'limitations':['本页为指定已保存制作副本的预览；外观仍在修改，非正式交付。','温柔笑与闭眼来自 Blender 实际评估采样；中间值为网页插值，两个控制单独预览。','当前仅预览静态身体与面部采样；新头颈绑定、说话、完整身体表演和新毛发物理未完成。','保留全部可见发丝中心线和半径，网页采用三角管截面与 PBR 材质近似；皮肤散射、微表面与眼球折射不等同于 Cycles 渲染。','各部件以列明的源文件为准，未自动合并其他独立工作副本。'],
      'attribution':[{'asset':'Hair Styles','author':'Daniel Bystedt','license':'CC BY-SA，原资料未明确版本','source':'https://www.blender.org/download/demo-files/#hair'}],
      'validation':{'sourceUnchanged':True,'masterWritten':False,'sourceBlendSaved':False,'formalExportsChanged':False,'blender':bpy.app.version_string,'objects':len(rows),'triangles':sum(r['triangles'] for r in rows),'seconds':time.time()-started},
      'geometry':rows,'materialAdaptation':material_report,'expressionSamples':sample_report,'renderBranchReplacements':replacements}
    (out/'preview-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    log('DONE '+json.dumps(manifest['export']))

if __name__=='__main__':main()

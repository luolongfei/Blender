"""Read actual current adult oral surfaces and the installed CC0 jaw donor.

Fresh background source only. No render or Blend save. Imports the single
Jaw - Realistic mesh object temporarily, exports its real geometry, and removes
every imported ID. Also measures the local lens between intersecting lip
sections at neutral / half / full natural smile; zero gap is not acceptance.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
HEAD='R11 Female head study'
SMILE='R11 natural soft smile'
OLD_ORAL=('R02 Upper teeth','R02 Lower teeth','R02 Tongue','R03 Upper gingiva','R03 Lower gingiva')
DONOR='assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
DONOR_SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a
def transform(v,m):
    m=np.asarray(m,float);return np.asarray(v)@m[:3,:3].T+m[:3,3]


def components(vertex_count,edges):
    adjacency=[[] for _ in range(vertex_count)]
    for a,b in edges:adjacency[int(a)].append(int(b));adjacency[int(b)].append(int(a))
    unseen=set(range(vertex_count));result=[]
    while unseen:
        first=unseen.pop();stack=[first];part=[first]
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt in unseen:unseen.remove(nxt);stack.append(nxt);part.append(nxt)
        result.append(np.asarray(sorted(part),np.int32))
    return sorted(result,key=len,reverse=True)


def mesh_record(obj,prefix,arrays):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    if mesh is None:return {'name':obj.name,'type':obj.type,'geometry':None}
    try:
        local=array(mesh.vertices);world=transform(local,ev.matrix_world)
        edges=array(mesh.edges,'vertices',2,np.int32)
        starts=array(mesh.polygons,'loop_start',1,np.int32).ravel()
        sizes=array(mesh.polygons,'loop_total',1,np.int32).ravel()
        loops=array(mesh.loops,'vertex_index',1,np.int32).ravel()
        mesh.calc_loop_triangles();tri=array(mesh.loop_triangles,'vertices',3,np.int32)
        for name,value in {'local':local,'world':world,'edges':edges,'face_starts':starts,'face_sizes':sizes,'loops':loops,'triangles':tri,
                           'material_indices':array(mesh.polygons,'material_index',1,np.int32).ravel()}.items():arrays[prefix+'_'+name]=value
        parts=components(len(local),edges)
        rows=[{'vertices':len(ids),'localBounds':[local[ids].min(0).tolist(),local[ids].max(0).tolist()],
               'worldBounds':[world[ids].min(0).tolist(),world[ids].max(0).tolist()]} for ids in parts]
        arrays[prefix+'_component_ids']=np.concatenate(parts) if parts else np.empty(0,np.int32)
        arrays[prefix+'_component_sizes']=np.asarray([len(x) for x in parts],np.int32)
        groups=mesh.attributes.get('.sculpt_face_set')
        if groups is not None and groups.domain=='FACE':
            g=array(groups.data,'value',1,np.int32).ravel();arrays[prefix+'_face_sets']=g
        else:g=None
        raw={}
        if obj.type=='MESH':
            arrays[prefix+'_raw_local']=array(obj.data.vertices)
            arrays[prefix+'_raw_edges']=array(obj.data.edges,'vertices',2,np.int32)
            arrays[prefix+'_raw_loops']=array(obj.data.loops,'vertex_index',1,np.int32).ravel()
            arrays[prefix+'_raw_face_starts']=array(obj.data.polygons,'loop_start',1,np.int32).ravel()
            arrays[prefix+'_raw_face_sizes']=array(obj.data.polygons,'loop_total',1,np.int32).ravel()
            attr=obj.data.attributes.get('.sculpt_face_set')
            if attr is not None:arrays[prefix+'_raw_face_sets']=array(attr.data,'value',1,np.int32).ravel()
            raw={'vertices':len(obj.data.vertices),'polygons':len(obj.data.polygons),'keys':[]}
            if obj.data.shape_keys:
                for k in obj.data.shape_keys.key_blocks:raw['keys'].append({'name':k.name,'value':float(k.value),'relative':k.relative_key.name})
        return {'name':obj.name,'type':obj.type,'prefix':prefix,'parent':obj.parent.name if obj.parent else None,
                'parentType':obj.parent_type,'parentBone':obj.parent_bone,'matrixWorld':[list(r) for r in ev.matrix_world],
                'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'raw':raw,
                'evaluatedVertices':len(world),'evaluatedPolygons':len(starts),
                'evaluatedWorldBounds':[world.min(0).tolist(),world.max(0).tolist()],
                'components':rows,'faceSets':dict(collections.Counter(map(int,g))) if g is not None else None,
                'materials':[m.name if m else None for m in mesh.materials],
                'modifiers':[{'name':m.name,'type':m.type,'showViewport':m.show_viewport,'showRender':m.show_render,
                              'levels':getattr(m,'levels',None),'renderLevels':getattr(m,'render_levels',None)} for m in obj.modifiers]}
    finally:ev.to_mesh_clear()


def section_segments(coordinates,faces,groups,cage,sx):
    seed=np.asarray(cage[8005]);q=coordinates[faces]
    selected=(q[:,:,0].min(1)<=sx)&(q[:,:,0].max(1)>=sx)
    selected&=(q[:,:,1].min(1)<seed[1]+.018)&(q[:,:,2].min(1)>seed[2]-.009)&(q[:,:,2].max(1)<seed[2]+.006)
    selected&=np.isin(groups,[7,17,8,22]);result={0:[],1:[]}
    for fi in np.flatnonzero(selected):
        for tri in (q[fi,[0,1,2]],q[fi,[0,2,3]]):
            points=[]
            for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                if (a[0]-sx)*(b[0]-sx)<0:points.append((a+(b-a)*(sx-a[0])/(b[0]-a[0]))[[1,2]])
            if len(points)==2:result[0 if groups[fi] in (7,17) else 1].append(points)
    return {k:np.asarray(v,float).reshape(-1,2,2) for k,v in result.items()}


def section_lenses(segments,scale):
    """Intersect two real section polyline sets and inspect each bounded lens.

    Positive lower-minus-upper Z is tissue overlap in these aligned sagittal
    cuts. Endpoint joins, folded multi-valued curves and zero-width intervals
    are reported explicitly, never converted into a collision-free claim.
    """
    intersections=[]
    for ui,u in enumerate(segments[0]):
        for li,l in enumerate(segments[1]):
            a=u[1]-u[0];b=l[1]-l[0];m=np.column_stack([a,-b])
            if abs(np.linalg.det(m))<1e-20:continue
            ts=np.linalg.solve(m,l[0]-u[0])
            if np.all(ts>=0) and np.all(ts<=1):
                point=u[0]+ts[0]*a
                if not any(np.linalg.norm(point-np.asarray(p['yz']))<1e-8 for p in intersections):
                    intersections.append({'yz':point.tolist(),'upperSegment':ui,'lowerSegment':li})
    intersections.sort(key=lambda r:r['yz'][0]);lenses=[]
    for aa,bb in zip(intersections,intersections[1:]):
        a=np.asarray(aa['yz']);b=np.asarray(bb['yz']);width=b[0]-a[0]
        if width<1e-8:continue
        if width>.005 or abs(b[1]-a[1])>.003:
            lenses.append({'endpoints':[aa,bb],'status':'Disconnected/wide pair; no bounded-lens assertion'});continue
        ys=np.linspace(a[0]+width*1e-5,b[0]-width*1e-5,201);values=[];ambiguous=0
        for y in ys:
            expected=a[1]+(b[1]-a[1])*(y-a[0])/width;zs=[]
            for side in (0,1):
                ss=segments[side];d=ss[:,1]-ss[:,0];valid=(y>=ss[:,:,0].min(1))&(y<=ss[:,:,0].max(1))&(abs(d[:,0])>1e-15)
                z=ss[valid,0,1]+(y-ss[valid,0,0])*d[valid,1]/d[valid,0]
                z=z[abs(z-expected)<.002]
                if len(z)==0:break
                near=z[np.argmin(abs(z-expected))];zs.append(float(near))
                if np.count_nonzero(abs(z-near)<.0001)>1:ambiguous+=1
            if len(zs)!=2:values.append(float('nan'))
            else:values.append(zs[1]-zs[0])
        d=np.asarray(values);finite=np.isfinite(d)
        positive=np.maximum(d[finite],0.)
        lenses.append({'endpoints':[aa,bb],'samples':len(ys),'finiteSamples':int(finite.sum()),'ambiguousNearBranches':ambiguous,
                       'maximumLowerAboveUpperMeters':float(positive.max()*scale) if len(positive) else None,
                       'signedMaximumBodySpace':float(d[finite].max()) if finite.any() else None,
                       'positiveOverlapAreaSquareMeters':float(np.trapezoid(np.maximum(d,0),ys)*scale*scale) if finite.all() else None,
                       'maximumGapMeters':float(np.maximum(-d[finite],0).max()*scale) if finite.any() else None,
                       'scope':'Local paired-crossing sagittal lens only; not complete 3D collision certification.'})
    return {'crossings':intersections,'localLenses':lenses}


def capture_head(strength,prefix,arrays):
    import bpy
    qa=api('face_contact_audit_r11');uv=api('production_uv_r08');head=bpy.data.objects[HEAD]
    if SMILE not in head.data.shape_keys.key_blocks:raise RuntimeError('Actual natural-smile control required')
    head.data.shape_keys.key_blocks[SMILE].value=float(strength);bpy.context.view_layer.update()
    data=qa['_evaluated_mesh'](head);world=data['world'];T=qa['_matrix'](head);inv=np.linalg.inv(T)
    body=transform(world,inv);faces=np.asarray(data['faces'],np.int32);groups=data['groups']
    if groups is None or faces.shape[1]!=4:raise RuntimeError('Actual L3 face-set quad mesh required')
    mix,_=uv['current_mix'](head);cage=transform(transform(mix,data['matrix']),inv)
    # Export mouth, cavity, chin and both jaw hinges; retain actual global IDs.
    seed=cage[8005];keep=(abs(body[:,0])<.105)&(body[:,2]>seed[2]-.07)&(body[:,2]<seed[2]+.055)
    fi=np.flatnonzero(keep[faces].any(1));ids=np.unique(faces[fi]);mapping=np.full(len(world),-1,np.int32);mapping[ids]=np.arange(len(ids))
    arrays[prefix+'_world']=world[ids];arrays[prefix+'_body']=body[ids];arrays[prefix+'_vertex_ids']=ids
    arrays[prefix+'_faces']=mapping[faces[fi]];arrays[prefix+'_face_ids']=fi;arrays[prefix+'_groups']=groups[fi]
    arrays[prefix+'_cage_local']=mix;arrays[prefix+'_cage_world']=transform(mix,data['matrix'])
    rows=[];scale=float(np.linalg.norm(T[:3,0]))
    for index,x in enumerate(np.linspace(-.016,.016,9)):
        ss=section_segments(body,faces,groups,cage,float(x))
        arrays[prefix+'_section_'+str(index)+'_upper']=ss[0];arrays[prefix+'_section_'+str(index)+'_lower']=ss[1]
        rows.append({'bodySectionX':float(x),'upperSegments':len(ss[0]),'lowerSegments':len(ss[1]),**section_lenses(ss,scale)})
    oral=np.isin(groups,[7,8]);oral_ids=np.unique(faces[oral])
    return {'strength':float(strength),'prefix':prefix,'headMatrixWorld':data['matrix'].tolist(),'femaleToTargetMatrix':T.tolist(),
            'actualL3Vertices':len(world),'actualL3Faces':len(faces),'oralVertexCount':len(oral_ids),
            'oralWorldBounds':[world[oral_ids].min(0).tolist(),world[oral_ids].max(0).tolist()],
            'nineLipSections':rows,'standardLipContact':qa['lip_section_contacts'](body,faces,groups,cage,T)}


def main():
    import bpy
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();output=(P/a.output).resolve();npz=output.with_suffix('.npz')
    for p in (source,output,npz):p.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent loaded source required')
    if output.exists() or npz.exists():raise RuntimeError('Preserve earlier observation')
    donor=P/DONOR
    if sha(donor)!=DONOR_SHA:raise RuntimeError('Installed CC0 donor differs')
    source_sha=sha(source);head=bpy.data.objects[HEAD];saved_keys={k.name:float(k.value) for k in head.data.shape_keys.key_blocks}
    uv=api('production_uv_r08');pose=uv['_pose_contract']();raw=api('adult_head_bind_observe_r11')['data_contract'](head)
    arrays={};report={'source':a.source,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'donor':DONOR,'donorSha256':DONOR_SHA,'savedBlend':False,'appearanceAccepted':False,'scope':'Actual current oral geometry plus temporarily loaded CC0 jaw, not fitted or animation-ready dental assets.'}
    pools=('objects','collections','meshes','curves','materials','node_groups','images','shape_keys')
    old_ids={name:{v.as_pointer() for v in getattr(bpy.data,name)} for name in pools}
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            report['headStates']=[capture_head(s,'head_'+str(i),arrays) for i,s in enumerate((0.,.5,1.))]
            for k,v in saved_keys.items():head.data.shape_keys.key_blocks[k].value=v
            bpy.context.view_layer.update()
            report['oldOral']=[mesh_record(bpy.data.objects[name],'old_'+str(i),arrays) for i,name in enumerate(OLD_ORAL) if name in bpy.data.objects]
            rig=bpy.data.objects['Character_Rig']
            report['rigJawBones']={b.name:{'headLocal':list(b.head_local),'tailLocal':list(b.tail_local),'matrixLocal':[list(r) for r in b.matrix_local],
                'worldHead':list(rig.matrix_world@b.head_local),'worldTail':list(rig.matrix_world@b.tail_local)} for b in rig.data.bones if any(t in b.name.lower() for t in ('jaw','head','neck'))}
            with bpy.data.libraries.load(str(donor),link=False) as (src,dst):
                if 'Jaw - Realistic' not in src.objects:raise RuntimeError('Expected explicit CC0 Jaw - Realistic mesh object missing')
                dst.objects=['Jaw - Realistic']
            collection=bpy.data.collections.new('R11 temporary CC0 jaw observation');bpy.context.scene.collection.children.link(collection)
            for obj in dst.objects:
                if obj is None or obj.type!='MESH':raise RuntimeError('Explicit CC0 jaw object is not a mesh')
                collection.objects.link(obj)
            bpy.context.view_layer.update()
            report['donorCollection']=collection.name
            report['donorObjects']=[mesh_record(o,'donor_'+str(i),arrays) for i,o in enumerate(sorted(collection.all_objects,key=lambda o:o.name)) if o.type in ('MESH','CURVE','SURFACE')]
            report['donorAllObjects']=[{'name':o.name,'type':o.type,'instanceType':o.instance_type,'instanceCollection':o.instance_collection.name if o.instance_collection else None} for o in collection.all_objects]
    finally:
        for k,v in saved_keys.items():head.data.shape_keys.key_blocks[k].value=v
        # Delete only IDs loaded by this bounded observation, never saved data.
        new=[v for name in pools for v in getattr(bpy.data,name) if v.as_pointer() not in old_ids[name]]
        if new:bpy.data.batch_remove(ids=new)
        bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose or api('adult_head_bind_observe_r11')['data_contract'](head)!=raw or sha(source)!=source_sha:raise RuntimeError('Observation changed source geometry, keys, pose or disk source')
    if any({v.as_pointer() for v in getattr(bpy.data,name)}!=old_ids[name] for name in pools):raise RuntimeError('Temporary donor IDs remain or original IDs changed')
    np.savez_compressed(npz,**arrays);report.update(arrays=npz.relative_to(P).as_posix(),arraysSha256=sha(npz),temporaryDonorRemoved=True,sourceUnchanged=True)
    with output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_ORAL_OBSERVED '+json.dumps({'output':a.output,'sourceSha256':source_sha,'arraysSha256':report['arraysSha256'],
        'donorObjects':[o['name'] for o in report['donorObjects']]}),flush=True)


if __name__=='__main__':main()

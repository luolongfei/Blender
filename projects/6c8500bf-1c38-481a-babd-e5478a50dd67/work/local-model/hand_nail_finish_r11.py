"""Actual HBM nail-island finish: subtle plate crown and restrained bare nails.

No fabricated finger anatomy, no painted shadows, no skin/rig/weight edits.
The original nail border is pinned exactly. Original data/materials remain.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PLAN='work/reconstruction-r02/hand-asset-fit-plan-r10-v3.json'
PAYLOAD_SHA='c86aaacc0fde9a15d6d9009531727b8a3e7d8ecff931fea81f38572b70ad6bbc'
KEY='R11 natural nail plate'
MARKER='R11_natural_nail_finish'


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _unit(v):return v/max(np.linalg.norm(v),1e-15)


def _smooth(a,b,x):
    v=np.clip((x-a)/(b-a),0,1);return v*v*(3-2*v)


def _linear(value):
    value=np.asarray(value)/255.;return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)


def _islands(faces,groups):
    pending=set(np.flatnonzero(groups==10).tolist());result=[]
    while pending:
        index=pending.pop();found=[index];vertices=set(faces[index]);changed=True
        while changed:
            changed=False
            for index in list(pending):
                if vertices.intersection(faces[index]):
                    pending.remove(index);found.append(index);vertices.update(faces[index]);changed=True
        result.append((np.asarray(sorted(vertices),np.int32),found))
    if len(result)!=5:raise RuntimeError('Expected five real integrated nail islands')
    return result


def plan():
    recipe=json.loads(_path(PLAN).read_text(encoding='utf-8'))
    if recipe['payloadSha256']!=PAYLOAD_SHA or _sha(_path(recipe['payload']))!=PAYLOAD_SHA:
        raise RuntimeError('Frozen fitted HBM geometry changed')
    observation=json.loads(_path(recipe['observation']).read_text(encoding='utf-8'))
    bones={b['name']:b for b in observation['current'][-1]['bones']};result={}
    with np.load(_path(recipe['payload'])) as a:
        faces=[a['loopVertex'][s:s+n] for s,n in zip(a['polygonStart'],a['polygonSize'])]
        islands=_islands(faces,a['faceGroup'])
        for side,prefix in (('L','left'),('R','right')):
            co=a['left' if side=='L' else 'right'].astype(np.float64);delta=np.zeros_like(co)
            colors=np.tile(np.r_[_linear([215,183,175]),1.],(len(co),1));rough=np.full(len(co),.28);uv=np.zeros((len(co),2))
            rows=[]
            for ids,face_ids in islands:
                q=co[ids];center=q.mean(0)
                finger=min(('Thumb','Index','Middle','Ring','Little'),key=lambda n:np.linalg.norm(center-np.asarray(bones[prefix+n+'Distal']['tailLocal'])))
                bone=bones[prefix+finger+'Distal'];along=_unit(np.asarray(bone['tailLocal'])-bone['headLocal'])
                normal=np.zeros(3);edges={}
                for fi in face_ids:
                    face=faces[fi]
                    if side=='L':face=face[::-1]
                    points=co[face]
                    for j in range(1,len(face)-1):normal+=np.cross(points[j]-points[0],points[j+1]-points[0])
                    for x,y in zip(face,np.roll(face,-1)):
                        e=tuple(sorted((int(x),int(y))));edges[e]=edges.get(e,0)+1
                normal=_unit(normal);along=_unit(along-normal*(along@normal));across=_unit(np.cross(along,normal))
                coords=np.column_stack(((q-center)@across,(q-center)@along))
                lo=coords.min(0);extent=coords.max(0)-lo;st=(coords-lo)/extent
                boundary=set(i for e,c in edges.items() if c==1 for i in e)
                amount=.00018*np.sin(np.pi*st[:,0])**2*np.sin(np.pi*st[:,1])
                for j,index in enumerate(ids):
                    if int(index) in boundary:amount[j]=0.
                delta[ids]=amount[:,None]*normal
                proximal=1-_smooth(.0,.27,st[:,1]);free=_smooth(.93,1.,st[:,1])
                base=_linear([215,183,175]);fold=_linear([201,161,154]);tip=_linear([224,212,199])
                rgb=base[None]*(1-proximal[:,None]*.36)+fold[None]*proximal[:,None]*.36
                rgb=rgb*(1-free[:,None]*.55)+tip[None]*free[:,None]*.55
                colors[ids,:3]=rgb;rough[ids]=.27+.055*proximal+.025*free;uv[ids]=st
                rows.append({'finger':finger,'vertices':ids.tolist(),'boundaryVertices':sorted(boundary),
                             'plateExtentMeters':extent.tolist(),'maxAddedCrownMeters':float(amount.max()),
                             'oldNormalExtentMeters':float(np.ptp((q-center)@normal)),
                             'boundaryDisplacementMeters':0.,'normalRigLocal':normal.tolist()})
            if len({r['finger'] for r in rows})!=5:raise RuntimeError('Nail/finger assignment is ambiguous')
            triangles=np.asarray([[int(f[0]),int(f[i]),int(f[i+1])] for f in faces for i in range(1,len(f)-1)])
            old=co[triangles];new=(co+delta)[triangles]
            na=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);nb=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
            dots=np.einsum('ij,ij->i',na,nb);valid=np.linalg.norm(na,axis=1)>1e-12
            if np.any(dots[valid]<=0):raise RuntimeError('Plate finish flipped an original triangle')
            result[side]={'base':co,'delta':delta,'color':colors,'roughness':rough,'uv':uv,'nails':rows}
    return result


def _co(mesh):
    a=np.empty((len(mesh.vertices),3),np.float64);mesh.vertices.foreach_get('co',a.ravel());return a


def _material(bpy,original):
    mat=original.copy();mat.name='R11 natural keratin nail';mat.use_fake_user=True
    shaders=[n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
    if len(shaders)!=1:raise RuntimeError('Expected the observed single Principled nail shader')
    shader=shaders[0];nodes=mat.node_tree.nodes;links=mat.node_tree.links
    for name,attribute in (('Base Color','R11 nail bed color'),('Roughness','R11 nail roughness')):
        for link in list(shader.inputs[name].links):links.remove(link)
        node=nodes.new('ShaderNodeAttribute');node.attribute_name=attribute
        links.new(node.outputs['Color' if name=='Base Color' else 'Fac'],shader.inputs[name])
    for name,value in {'Metallic':0.,'IOR':1.46,'Specular IOR Level':.5,'Transmission Weight':.035,
                       'Subsurface Weight':.025,'Subsurface Scale':.0005,'Coat Weight':.08,'Coat Roughness':.22,'Alpha':1.}.items():
        for link in list(shader.inputs[name].links):links.remove(link)
        shader.inputs[name].default_value=value
    shader.inputs['Subsurface Radius'].default_value=(1.,.55,.4)
    return mat


def prepare(source_relative,output_relative,source_sha):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report=output.with_suffix('.nails.json')
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or _sha(source)!=source_sha:
        raise RuntimeError('Exact source in disposable background required')
    if output==source or not output.is_relative_to(_path('work/reconstruction-r02')) or output.exists() or report.exists():
        raise RuntimeError('Choose a new independent reconstruction output')
    if MARKER in bpy.context.scene:raise RuntimeError('Natural nail finish already installed')
    data=plan();hands=[bpy.data.objects['Hand '+s+' · continuous anatomy'] for s in ('L','R')]
    if any(o.data.shape_keys for o in hands):raise RuntimeError('Existing hand shapes require explicit reconciliation')
    original=[o.data for o in hands];material=None;new_meshes=[];success=False
    fit=runpy.run_path(str(Path(__file__).with_name('hand_asset_integrate_r11.py')))['_fit']()
    rig=bpy.data.objects['Character_Rig'];rig_before=fit['_rig_contract'](rig);rows=[]
    try:
        for obj,side in zip(hands,('L','R')):
            row=data[side]
            if np.max(np.abs(_co(obj.data)-row['base']))>2e-7:raise RuntimeError('Actual fitted hand has later geometry changes')
            provenance=json.loads(obj['R10_hbm_hand_provenance'])
            if provenance['fitPayloadSha256']!=PAYLOAD_SHA:raise RuntimeError('Actual hand provenance differs')
            before_groups=[(g.index,g.name,g.lock_weight) for g in obj.vertex_groups]
            before_weights=[[(g.group,g.weight) for g in v.groups] for v in obj.data.vertices]
            old=obj.data;old.use_fake_user=True;obj.data=old.copy();obj.data.name='R11 nail finish hand '+side;new_meshes.append(obj.data)
            if [(g.index,g.name,g.lock_weight) for g in obj.vertex_groups]!=before_groups:raise RuntimeError('Copying hand data changed group indices')
            obj.shape_key_add(name='Basis');key=obj.shape_key_add(name=KEY);key.data.foreach_set('co',(row['base']+row['delta']).ravel());key.value=1.
            color=obj.data.color_attributes.new(name='R11 nail bed color',type='FLOAT_COLOR',domain='POINT');color.data.foreach_set('color',row['color'].ravel())
            rough=obj.data.attributes.new(name='R11 nail roughness',type='FLOAT',domain='POINT');rough.data.foreach_set('value',row['roughness'])
            if material is None:material=_material(bpy,old.materials[1])
            obj.data.materials[1]=material
            if [[(g.group,g.weight) for g in v.groups] for v in obj.data.vertices]!=before_weights:raise RuntimeError('Nail finish changed skin weights')
            if obj.data.materials[0]!=old.materials[0]:raise RuntimeError('Hand skin material changed')
            rows.append({'side':side,'object':obj.name,'originalData':old.name,'newData':obj.data.name,
                         'nails':row['nails'],'changedControlPoints':int(np.sum(np.linalg.norm(row['delta'],axis=1)>1e-10)),
                         'allNailBordersPinned':True,'skinMaterialUnchanged':True,'weightsAndGroupsUnchanged':True})
        bpy.context.view_layer.update()
        attachment=[fit['hand_camera'](None,rig,side,None,diagnostics_only=True) for side in ('L','R')]
        if any(max(n['centerTipDistance'] for n in r['nails'])>.020 for r in attachment):raise RuntimeError('Actual nail attachment failed')
        if fit['_rig_contract'](rig)!=rig_before:raise RuntimeError('Nail finish changed rig/action/pose')
        result={'source':source_relative,'sourceSha256':source_sha,'helperSha256':_sha(__file__),'changes':rows,
                'actualNailAttachment':attachment,'material':material.name,'appearanceAccepted':False,'masterChanged':False,
                'limits':['Subtle integrated nail crown and keratin shading approximation; no separate refractive nail plate.',
                          'Actual before/after macro images, finger flexion and skin/cuticle transitions still need inspection.',
                          'No hand-back knuckle or finger anatomy changes were guessed from the low-detail original images.']}
        bpy.context.scene[MARKER]=json.dumps(result,ensure_ascii=False)
        if _sha(source)!=source_sha:raise RuntimeError('Source changed on disk')
        bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False,relative_remap=True);success=True
        result.update({'candidate':output_relative,'candidateSha256':_sha(output)})
        with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        return result
    finally:
        if not success:
            for obj,mesh in zip(hands,original):obj.data=mesh
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]


def review(source_relative,case,label,side='L'):
    import bpy
    from mathutils import Vector
    source=_path(source_relative);folder=_path('previews/reconstruction-r02/'+label)
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Exact disposable source required')
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label):raise ValueError('Safe unique review label required')
    folder.mkdir(parents=True,exist_ok=True);image=folder/(side.lower()+'-'+case+'.png');report=image.with_suffix('.json')
    if image.exists() or report.exists():raise FileExistsError('Preserve previous actual images')
    fit=runpy.run_path(str(Path(__file__).with_name('hand_asset_integrate_r11.py')))['_fit']()
    adapter=runpy.run_path(str(Path(__file__).with_name('hand_asset_pose_review_r11.py')));adapter['_standing_context'](bpy)
    rig=bpy.data.objects['Character_Rig'];hand=bpy.data.objects['Hand '+side+' · continuous anatomy'];prefix='left' if side=='L' else 'right'
    diagnosis=fit['hand_camera'](None,rig,side,None,diagnostics_only=True)
    normal=Vector(diagnosis['fourFingerMeanNormal']);along=Vector(diagnosis['longitudinalAxis']);hidden=[]
    if case=='nails':
        points=[Vector(n['center']) for n in diagnosis['nails'] if n['finger'] in ('Index','Middle')];scale=.045
    elif case=='knuckles':
        points=[rig.matrix_world@rig.pose.bones[prefix+n+s].head for n in ('Index','Middle','Ring') for s in ('Proximal','Intermediate')];scale=.078
    else:raise ValueError('Use nails or knuckles')
    target=sum(points,Vector())/len(points);camera=bpy.data.objects.new('R11 hand macro',bpy.data.cameras.new('R11 hand macro'));bpy.context.scene.collection.objects.link(camera)
    camera.data.type='ORTHO';camera.data.ortho_scale=scale;camera.data.dof.use_dof=False
    utility=runpy.run_path(str(Path(__file__).with_name('reference_final_stills_r08.py')))
    utility['_aim'](camera,target+normal*.6-along*.12,target,along)
    scene=bpy.context.scene;old_camera=scene.camera;scene.camera=camera;digest=_sha(source)
    try:
        for obj in scene.objects:
            if obj not in (hand,camera) and obj.type in ('MESH','CURVES','CURVE','SURFACE','FONT','META','VOLUME') and not obj.hide_render:
                hidden.append(obj);obj.hide_render=True
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=128;scene.cycles.use_denoising=True
        scene.render.resolution_x=768;scene.render.resolution_y=640;scene.render.resolution_percentage=100
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_sequencer=False
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(image)
        bpy.ops.render.render(write_still=True)
        result={'source':source_relative,'sourceSha256':digest,'case':case,'side':side,'samples':128,'orthographicScale':scale,
                'image':str(image.relative_to(project_path())),'imageSha256':_sha(image),'helperSha256':_sha(__file__),
                'isolatedHandInspection':True,'actualNailAttachment':diagnosis,'savedBlend':False,'appearanceAccepted':False}
    finally:
        for obj in hidden:obj.hide_render=False
        scene.camera=old_camera;data=camera.data;bpy.data.objects.remove(camera,do_unlink=True);bpy.data.cameras.remove(data)
    if _sha(source)!=digest:raise RuntimeError('Review changed source')
    with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return result


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('plan')
    build=sub.add_parser('prepare');build.add_argument('--source',required=True);build.add_argument('--source-sha',required=True);build.add_argument('--output',required=True)
    draw=sub.add_parser('review');draw.add_argument('--source',required=True);draw.add_argument('--case',choices=('nails','knuckles'),required=True);draw.add_argument('--label',required=True);draw.add_argument('--side',choices=('L','R'),default='L')
    a=parser.parse_args(argv)
    if a.mode=='plan':result={s:{'nails':r['nails'],'changedPoints':int(np.sum(np.linalg.norm(r['delta'],axis=1)>1e-10))} for s,r in plan().items()}
    elif a.mode=='prepare':result=prepare(a.source,a.output,a.source_sha)
    else:result=review(a.source,a.case,a.label,a.side)
    print('R11_NAIL_FINISH '+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

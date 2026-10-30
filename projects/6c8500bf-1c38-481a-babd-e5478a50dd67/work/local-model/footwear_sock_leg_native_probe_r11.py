"""Disposable combined native rest capture: unified legs, fitted socks and V5 shoes.

Only the integrator executes Blender. No source is saved; modified original
objects receive temporary copied mesh data, and the trial legs/rig are new IDs.
BVH results are candidate pairs for subsequent exact triangle narrow phase.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
STAGE='work/reconstruction-r02/'
CAPTURE_SHA='c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f'
SHOE_SHA='35625c38ff652dddd575de393ea65858430720b6d4c351c878d8c846eacbf15b'
LEG_HELPER_SHA='45b06a7bf873fc6296c2145c17a676caf666554c224a0ed16ca0e46508dc6670'
OBSERVE_SHA='b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def path_for(relative):
    path=(project_path()/relative.replace('\\','/')).resolve();path.relative_to(project_path().resolve());return path

def read(relative,digest):
    p=path_for(relative)
    if len(digest)!=64 or sha(p)!=digest:raise RuntimeError('Explicit frozen package changed: '+relative)
    j=json.loads(p.read_text());array=path_for(j['arrays'])
    if sha(array)!=j['arraysSha256']:raise RuntimeError('Package array changed')
    with np.load(array,allow_pickle=False) as a:packed={k:a[k] for k in a.files}
    return j,packed

def fitted_mesh(bpy,obj,row,data):
    pre=row['arrayPrefix'];co=np.asarray(data[pre+'_targetRaw']);mode=row.get('mode','coordinates')
    if mode=='coordinates':
        mesh=obj.data.copy();mesh.vertices.foreach_set('co',co.ravel());mesh.update();obj.data=mesh;return
    if mode!='replaceMesh' or not obj.name.startswith('Sock_Ribbed_'):raise RuntimeError('Only explicit ribbed sock-body topology replacement is supported')
    loops=data[pre+'_loops'];starts=data[pre+'_polygonStarts'];sizes=data[pre+'_polygonSizes'];uv=data[pre+'_cornerUV'];weights=data[pre+'_weights']
    definitions=[(g.name,g.index,g.lock_weight) for g in obj.vertex_groups];names=[g[0] for g in definitions]
    if row['groupNames']!=names or weights.shape!=(len(co),len(names)) or not np.isfinite(weights).all() or weights.min()<0 or np.max(abs(weights.sum(1)-1))>2e-6:raise RuntimeError('New sock weights must use and normalize the actual unchanged source group definitions')
    if len(sizes)!=len(starts) or np.any(sizes<3) or not np.array_equal(starts,np.r_[0,np.cumsum(sizes[:-1])]) or sizes.sum()!=len(loops) or loops.min()<0 or loops.max()>=len(co):raise RuntimeError('Invalid replacement sock polygon arrays')
    if uv.shape!=(len(loops),2) or not np.isfinite(uv).all():raise RuntimeError('Invalid replacement sock corner UV')
    faces=[loops[start:start+size].tolist() for start,size in zip(starts,sizes)];mesh=bpy.data.meshes.new('R11_SOCK_FIT_TRIAL_'+obj.name);mesh.from_pydata(co.tolist(),[],faces);mesh.update()
    if mesh.validate(verbose=False,clean_customdata=False):raise RuntimeError('New sock topology required repair')
    mesh.polygons.foreach_set('use_smooth',np.ones(len(faces),np.bool_))
    for material in obj.data.materials:mesh.materials.append(material)
    uv_name=obj.data.uv_layers.active.name if obj.data.uv_layers.active else 'SockFitUV';layer=mesh.uv_layers.new(name=uv_name);layer.data.foreach_set('uv',uv.astype(np.float32).ravel());obj.data=mesh
    # Current Blender stores deform-group definitions on the Mesh datablock.
    # Explicitly recreate them on this new Mesh; never clear the original.
    if not obj.vertex_groups:
        for name,index,locked in definitions:
            group=obj.vertex_groups.new(name=name);group.lock_weight=locked
            if group.index!=index:raise RuntimeError('New sock group order differs')
    if [(g.name,g.index,g.lock_weight) for g in obj.vertex_groups]!=definitions:raise RuntimeError('New sock group definitions differ')
    uv_readback=np.empty(uv.shape,np.float32);layer.data.foreach_get('uv',uv_readback.ravel())
    if not np.array_equal(uv_readback,uv.astype(np.float32)):raise RuntimeError('Actual replacement sock UV readback differs')
    positions=np.empty(co.shape,np.float32);mesh.vertices.foreach_get('co',positions.ravel())
    if not np.array_equal(positions,co.astype(np.float32)):raise RuntimeError('Actual replacement sock coordinate readback differs')
    for column,group in enumerate(obj.vertex_groups):
        ids=np.flatnonzero(weights[:,column]>0)
        for value in np.unique(weights[ids,column]):group.add(ids[weights[ids,column]==value].tolist(),float(value),'REPLACE')
    actual=np.zeros(weights.shape,np.float32)
    for vertex in mesh.vertices:
        for group in vertex.groups:actual[vertex.index,group.group]=group.weight
    if not np.array_equal(actual,weights.astype(np.float32)):raise RuntimeError('Actual replacement sock weight readback differs')

def run(source_relative,source_sha,sock_relative,sock_sha,output_relative):
    import bpy
    from mathutils.bvhtree import BVHTree
    source,output=path_for(source_relative),path_for(output_relative);output_arrays=output.with_suffix('.npz')
    if output.exists() or output_arrays.exists():raise FileExistsError('Preserve prior actual capture')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or sha(source)!=source_sha:
        raise RuntimeError('Exact source in disposable background Object Mode required')
    if bpy.context.scene.frame_current!=1 or bpy.context.scene.frame_subframe!=0:raise RuntimeError('Preserve actual source frame one')
    shoes,sa=read(STAGE+'footwear-225-shared-map-r11-v5.json',SHOE_SHA);socks,ka=read(sock_relative,sock_sha);capture,ca=read(STAGE+'foot-proportion-capture-r11.json',CAPTURE_SHA)
    if socks['sourceCaptureSha256']!=CAPTURE_SHA or len(shoes['objects'])!=74 or len(socks['objects'])!=6:raise RuntimeError('Complete frozen shoe/sock families required')
    expected_socks={'Sock_'+part+'_'+side for side in ('L','R') for part in ('Ribbed','Cuff_Lower','Cuff_Top')}
    if {r['name'] for r in socks['objects']}!=expected_socks:raise RuntimeError('Exact complete six-piece sock family required')
    helper=Path(__file__).with_name('foot_leg_native_trial_r11.py');observer=Path(__file__).with_name('hand_asset_observe_r10.py')
    if sha(helper)!=LEG_HELPER_SHA or sha(observer)!=OBSERVE_SHA:raise RuntimeError('Frozen native dependency changed')
    leg=runpy.run_path(str(helper));api=runpy.run_path(str(observer))
    candidate,lc,placement,la,lca,validation=leg['load_inputs']();rig,source_legs=leg['source_checks'](bpy,api,lc,lca)
    source_rows={r['name']:r for r in capture['objects']};rows=[];packed_by_name={}
    for family,package,data in (('shoe',shoes,sa),('sock',socks,ka)):
        for row in package['objects']:
            obj=bpy.data.objects.get(row['name']);src=source_rows[row['name']];pre=row['arrayPrefix'];capture_pre=src['arrayPrefix']
            if obj is None or obj.type!='MESH' or obj.data.shape_keys:raise RuntimeError('Missing unkeyed native '+family+': '+row['name'])
            if row['side'] not in ('L','R') or not obj.name.startswith('Shoe_' if family=='shoe' else 'Sock_'):raise RuntimeError('Candidate family identity mismatch')
            q=np.empty((len(obj.data.vertices),3),np.float32);obj.data.vertices.foreach_get('co',q.ravel())
            if not np.array_equal(q,data[pre+'_sourceRaw']) or not np.array_equal(q,ca[capture_pre+'_raw_positions']):raise RuntimeError('Captured raw source changed: '+obj.name)
            for name,collection,attribute,width in [('edges',obj.data.edges,'vertices',2),('loops',obj.data.loops,'vertex_index',1),('polygonStarts',obj.data.polygons,'loop_start',1),('polygonSizes',obj.data.polygons,'loop_total',1)]:
                actual=np.empty(len(collection)*width,np.int32);collection.foreach_get(attribute,actual);expected=ca[capture_pre+'_raw_'+name]
                if not np.array_equal(actual.reshape(expected.shape),expected):raise RuntimeError('Captured source topology changed: '+obj.name+'/'+name)
            if [[g.name,g.index] for g in obj.vertex_groups]!=src['groups']:raise RuntimeError('Captured source groups changed: '+obj.name)
            weights=[(v.index,g.group,g.weight) for v in obj.data.vertices for g in v.groups]
            for column,name in enumerate(('weightVertex','weightGroup','weightValue')):
                if not np.array_equal(np.asarray([r[column] for r in weights]),ca[capture_pre+'_'+name]):raise RuntimeError('Captured source weights changed: '+obj.name)
            if (obj.parent.name if obj.parent else None)!=src['parent'] or obj.parent_bone!=src['parentBone']:raise RuntimeError('Captured source parent changed: '+obj.name)
            if [m.name if m else None for m in obj.data.materials]!=src['materialNames']:raise RuntimeError('Captured source material slots changed: '+obj.name)
            target=data[pre+'_targetRaw'];mode=row.get('mode','coordinates')
            if target.ndim!=2 or target.shape[1]!=3 or not np.isfinite(target).all() or (mode=='coordinates' and target.shape!=q.shape):raise RuntimeError('Invalid candidate raw coordinates')
            if mode not in ('coordinates','replaceMesh') or (mode=='replaceMesh' and (family!='sock' or not obj.name.startswith('Sock_Ribbed_'))):raise RuntimeError('Unsupported candidate mesh mode')
            if not np.array_equal(np.asarray(obj.matrix_world),src['matrixWorld']) or [api['_rna'](m) for m in obj.modifiers]!=src['modifiers']:raise RuntimeError('Source matrix/modifier stack changed: '+obj.name)
            rows.append(dict(row,family=family));packed_by_name[obj.name]=data
    objects=[bpy.data.objects[r['name']] for r in rows];guarded=objects+[rig]+list(source_legs.values())
    contract=api['_source_contract'](guarded);extra=leg['extra_rig_contract'](rig);old_ids=set(api['_ids'](bpy));old_data={o:o.data for o in objects};pose=rig.data.pose_position
    group_definitions={o.name:[(g.name,g.index,g.lock_weight) for g in o.vertex_groups] for o in objects}
    active=bpy.context.view_layer.objects.active;selected=list(bpy.context.selected_objects);scene=bpy.context.scene
    scene_contract=(scene.frame_current,scene.frame_subframe,scene.camera,scene.render.filepath);arrays={}
    report={'schemaVersion':1,'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),'shoePlanSha256':SHOE_SHA,'sockPlan':sock_relative,'sockPlanSha256':sock_sha,'legNativeHelperSha256':LEG_HELPER_SHA,'legCandidateSha256':leg['PINS'][STAGE+'foot-leg-unified-candidate-r11-v2.json'],'sourceCaptureSha256':CAPTURE_SHA,'legFeet':{},'objects':[],'pairSets':[],
        'savedBlend':False,'appearanceAccepted':False,'technicalAcceptance':False,'productionRigInstalled':False,'productionActionsRetargeted':False,
        'scope':'Actual native REST unified legs with their original Armature then Subdivision, six socks with original stacks, and V5 complete 74 shoe pieces. BVH pairs only; continuous narrow phase, native positive room, visual appearance, poses and production animation remain separate.'}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        trial,report['copiedRig']=leg['copy_trial_rig'](bpy,rig,validation)
        leg_objects={side:leg['candidate_object'](bpy,source_legs[side],trial,side,la,validation,api) for side in ('L','R')}
        for row,obj in zip(rows,objects):
            fitted_mesh(bpy,obj,row,packed_by_name[obj.name])
        bpy.context.view_layer.update();trees={};by_name={}
        for side,obj in leg_objects.items():
            v,t=leg['surface'](bpy,obj);pre='legFoot_'+side;arrays[pre+'_world']=v;arrays[pre+'_triangles']=t;trees[pre]=BVHTree.FromPolygons(v.tolist(),t.tolist(),all_triangles=True,epsilon=1e-6)
            report['legFeet'][side]={'arrayPrefix':pre,'vertices':len(v),'triangles':len(t),'bareSoleMinimumWorldZ':float(v[:,2].min()),'provisionalAnkleCentreWorld':validation[side]['axes']['ankleCentreWorld'],'nativeFootLengthBelow185mm':float(np.ptp(v[v[:,2]<.185,1]))}
        for i,(row,obj) in enumerate(zip(rows,objects)):
            v,t=leg['surface'](bpy,obj);pre=row['arrayPrefix'];arrays[pre+'_world']=v;arrays[pre+'_triangles']=t;trees[pre]=BVHTree.FromPolygons(v.tolist(),t.tolist(),all_triangles=True,epsilon=1e-6);by_name[obj.name]=pre
            report['objects'].append({'name':obj.name,'side':row['side'],'family':row['family'],'arrayPrefix':pre,'meshMode':row.get('mode','coordinates'),'vertices':len(v),'triangles':len(t),'worldBounds':[v.min(0).tolist(),v.max(0).tolist()],'modifiers':[api['_rna'](m) for m in obj.modifiers]})
            if i%16==0:print('R11_FOOT_SOCK_SHOE_SURFACES '+str(i+1)+'/80',flush=True)
        def add_pairs(first,second,label):
            pairs=trees[first].overlap(trees[second])
            if len(pairs)>300000:raise RuntimeError('Unexpected local contact candidate budget exceeded: '+label)
            key='pairs_'+str(len(report['pairSets']));arrays[key]=np.asarray(pairs,np.int32).reshape(-1,2)
            report['pairSets'].append({'firstArrayPrefix':first,'secondArrayPrefix':second,'label':label,'pairsArray':key,'bvhCandidatePairs':len(pairs),'epsilonMeters':1e-6,'candidatePairsAreNotConfirmedIntersections':True})
        for row in rows:
            add_pairs('legFoot_'+row['side'],row['arrayPrefix'],'skin / '+row['name'])
            if row['family']=='shoe':
                for sock in socks['objects']:
                    if sock['side']==row['side']:add_pairs(by_name[sock['name']],row['arrayPrefix'],sock['name']+' / '+row['name'])
    finally:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        for obj,data in old_data.items():obj.data=data
        added=[item for pointer,item in api['_ids'](bpy).items() if pointer not in old_ids]
        if added:bpy.data.batch_remove(ids=added)
        rig.data.pose_position=pose;bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=active
        if {o.name:[(g.name,g.index,g.lock_weight) for g in o.vertex_groups] for o in objects}!=group_definitions:raise RuntimeError('Original mesh group definitions failed restoration')
        if api['_source_contract'](guarded)!=contract or leg['extra_rig_contract'](rig)!=extra or set(api['_ids'](bpy))!=old_ids or sha(source)!=source_sha or (scene.frame_current,scene.frame_subframe,scene.camera,scene.render.filepath)!=scene_contract:
            raise RuntimeError('Combined native probe failed complete source restoration')
    report['restorationExact']=True
    with output_arrays.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report['arrays']=str(output_arrays.relative_to(project_path()));report['arraysSha256']=sha(output_arrays)
    with output.open('x',encoding='utf8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    return {'report':str(output.relative_to(project_path())),'sha256':sha(output),'arraysSha256':sha(output_arrays),'bvhCandidatePairs':sum(r['bvhCandidatePairs'] for r in report['pairSets']),'savedBlend':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True);parser.add_argument('--sock-plan',required=True);parser.add_argument('--sock-plan-sha',required=True);parser.add_argument('--output',default=STAGE+'footwear-sock-leg-native-probe-r11.json')
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:];a=parser.parse_args(argv)
    print('R11_FOOT_SOCK_LEG_NATIVE_PROBE '+json.dumps(run(a.source,a.source_sha,a.sock_plan,a.sock_plan_sha,a.output)),flush=True)

if __name__=='__main__':main()

"""Temporary native shoe-shell evaluation against the actually appended foot.

Root schedules this disposable background job. Source geometry is copied in
memory and restored; no Blend is saved. The output is actual evaluated meshes
and bounded BVH candidate pairs for independent continuous contact analysis.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
OBSERVER_SHA='b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'
CAPTURE_SHA='c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f'
FIT_SHA='217b9b4b4689b7521482779d382f35d205a69a9170a7f1aeeae2375364fa9fb5'
PLACEMENT_SHA='52e7e5f3a4b45abc2e4857c2c05adb9b1b9b4ede95a4575cdeffe56d7407d4f2'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(relative,digest):
    path=(project_path()/relative).resolve();path.relative_to(project_path().resolve())
    if sha(path)!=digest:raise RuntimeError('Exact frozen input changed: '+relative)
    row=json.loads(path.read_text());arrays=project_path()/row['arrays']
    if sha(arrays)!=row['arraysSha256']:raise RuntimeError('Input array changed')
    return row,np.load(arrays)

def surface(bpy,obj):
    graph=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(graph);mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        mesh.calc_loop_triangles();q=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',q.ravel());m=np.asarray(ev.matrix_world,float)
        points=q@m[:3,:3].T+m[:3,3];tri=np.asarray([list(t.vertices) for t in mesh.loop_triangles],np.int32)
        return points,tri
    finally:ev.to_mesh_clear()

def run(source_relative,source_sha,output_relative):
    import bpy
    from mathutils import Matrix
    from mathutils.bvhtree import BVHTree
    project=project_path();source=(project/source_relative).resolve();output=(project/output_relative).resolve()
    source.relative_to(project.resolve());output.relative_to(project.resolve());output_arrays=output.with_suffix('.npz')
    if output.exists() or output_arrays.exists():raise FileExistsError('Preserve previous actual probe')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or sha(source)!=source_sha:
        raise RuntimeError('Exact source in disposable background Object Mode required')
    scene=bpy.context.scene
    if scene.frame_current!=1 or scene.frame_subframe!=0:raise RuntimeError('Keep actual authored frame one')
    fit,fa=read('work/reconstruction-r02/footwear-225-shared-map-r11-v4.json',FIT_SHA)
    place,pa=read('work/reconstruction-r02/foot-225-placement-plan-r11.json',PLACEMENT_SHA)
    capture,ca=read('work/reconstruction-r02/foot-proportion-capture-r11.json',CAPTURE_SHA)
    source_rows={r['name']:r for r in capture['objects']};rig=bpy.data.objects['Character_Rig']
    helper=Path(__file__).with_name('hand_asset_observe_r10.py')
    if sha(helper)!=OBSERVER_SHA:raise RuntimeError('Frozen observer changed')
    api=runpy.run_path(str(helper));shoes=[bpy.data.objects[r['name']] for r in fit['objects']]
    if len(shoes)!=74:raise RuntimeError('Expected both complete 37-part shoe families')
    for row,obj in zip(fit['objects'],shoes):
        src=source_rows[obj.name];pre=row['arrayPrefix'];coords=np.empty((len(obj.data.vertices),3),np.float32);obj.data.vertices.foreach_get('co',coords.ravel())
        if obj.type!='MESH' or obj.data.shape_keys or not np.array_equal(coords,fa[pre+'_sourceRaw']):raise RuntimeError('Source shoe coordinates/keys changed: '+obj.name)
        if not np.array_equal(np.asarray(obj.matrix_world),np.asarray(src['matrixWorld'])):raise RuntimeError('Source shoe transform changed: '+obj.name)
        if [api['_rna'](m) for m in obj.modifiers]!=src['modifiers']:raise RuntimeError('Source shoe modifier stack changed: '+obj.name)
    for prefix in ('left','right'):
        for suffix in ('LowerLeg','Foot','Toes'):
            name=prefix+suffix
            if not np.array_equal(np.asarray(rig.data.bones[name].matrix_local),np.asarray(capture['bones'][name]['restMatrixLocal'])):
                raise RuntimeError('Source lower limb rest changed before this fit probe')
    guarded=shoes+[rig];contract=api['_source_contract'](guarded);old_ids=set(api['_ids'](bpy));old_data={obj:obj.data for obj in shoes};pose=rig.data.pose_position
    arrays={};result={'schemaVersion':1,'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'shoePlanSha256':FIT_SHA,'placementSha256':PLACEMENT_SHA,'sourceCaptureSha256':CAPTURE_SHA,'feet':{},'shoes':[],
        'savedBlend':False,'appearanceAccepted':False,'technicalAcceptance':False,
        'scope':'Native current modifier evaluation in temporary REST, appended actual donor feet with proposed placement, complete 74 shoe pieces. No sock/new calf/new rig, no posed movement or final acceptance.'}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        for row,obj in zip(fit['objects'],shoes):
            obj.data=obj.data.copy();obj.data.vertices.foreach_set('co',fa[row['arrayPrefix']+'_targetRaw'].ravel());obj.data.update()
        bundle=project/place['bundle']
        if sha(bundle)!=place['bundleSha256']:raise RuntimeError('Donor source bundle changed')
        with bpy.data.libraries.load(str(bundle),link=False) as (available,requested):
            if 'Foot - Realistic' not in available.objects:raise RuntimeError('Actual donor object missing')
            requested.objects=['Foot - Realistic']
        left=requested.objects[0]
        if left is None or left.type!='MESH':raise RuntimeError('Actual donor object append failed')
        scene.collection.objects.link(left);right=left.copy();scene.collection.objects.link(right)
        feet={'L':left,'R':right};trees={}
        for side,obj in feet.items():obj.matrix_world=Matrix(pa['donor_'+side+'_matrixWorld'].tolist())
        bpy.context.view_layer.update()
        for side,obj in feet.items():
            v,t=surface(bpy,obj);expected=pa['donor_'+side+'_evaluatedWorld']
            if v.shape!=expected.shape or np.max(np.linalg.norm(v-expected,axis=1))>2e-6:raise RuntimeError('Native donor differs from actual captured placement')
            arrays['foot_'+side+'_world']=v;arrays['foot_'+side+'_triangles']=t;trees[side]=BVHTree.FromPolygons(v.tolist(),t.tolist(),all_triangles=True)
            result['feet'][side]={'vertices':len(v),'triangles':len(t),'actualPlacementMaximumErrorMeters':float(np.linalg.norm(v-expected,axis=1).max())}
        for index,(row,obj) in enumerate(zip(fit['objects'],shoes)):
            v,t=surface(bpy,obj);pre=row['arrayPrefix'];side=row['side'];tree=BVHTree.FromPolygons(v.tolist(),t.tolist(),all_triangles=True)
            pairs=trees[side].overlap(tree)
            if len(pairs)>250000:raise RuntimeError('Local foot/shoe candidate budget unexpectedly exceeded: '+obj.name)
            arrays[pre+'_world']=v;arrays[pre+'_triangles']=t;arrays[pre+'_footPairs']=np.asarray(pairs,np.int32).reshape(-1,2)
            result['shoes'].append({'name':obj.name,'side':side,'arrayPrefix':pre,'vertices':len(v),'triangles':len(t),
                'bvhCandidatePairs':len(pairs),'candidatePairsAreNotConfirmedIntersections':True,
                'modifiers':[api['_rna'](m) for m in obj.modifiers]})
            if index%12==0:print('R11_FOOTWEAR_NATIVE_CAPTURE '+str(index+1)+'/74',flush=True)
    finally:
        for obj,data in old_data.items():obj.data=data
        added=[item for pointer,item in api['_ids'](bpy).items() if pointer not in old_ids]
        if added:bpy.data.batch_remove(ids=added)
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        if api['_source_contract'](guarded)!=contract or set(api['_ids'](bpy))!=old_ids or sha(source)!=source_sha:
            raise RuntimeError('Actual probe failed complete source restoration')
    result['restorationExact']=True;np.savez_compressed(output_arrays,**arrays);result['arrays']=str(output_arrays.relative_to(project));result['arraysSha256']=sha(output_arrays)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return {'report':str(output.relative_to(project)),'reportSha256':sha(output),'arraysSha256':sha(output_arrays),'savedBlend':False}

def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/footwear-225-native-probe-r11.json');a=parser.parse_args(argv)
    print('R11_FOOTWEAR_NATIVE_PROBE '+json.dumps(run(a.source,a.source_sha,a.output)),flush=True)

if __name__=='__main__':main()

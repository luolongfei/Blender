"""Observe the exact failed refined-hand integration without saving a Blend.

Runs the frozen v2 merger up to its actual pose gate, records current modifier
and animation state plus copied raw/evaluated wrist geometry, then deliberately
aborts through its rollback. Never changes subdivision quality or thresholds.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

MERGER_SHA='d843c0253194de7b3feca61e6721644fcad439d8dd3868c1b5fbaf0dd40334ae'


def _path(relative):
    path=(project_path()/relative).resolve();path.relative_to(project_path().resolve());return path


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class ProbeComplete(RuntimeError):pass


def _object_state(obj,observe):
    keys=obj.data.shape_keys
    return {'object':obj.name,'data':obj.data.name,'rawVertices':len(obj.data.vertices),
            'matrixWorld':np.asarray(obj.matrix_world).tolist(),
            'modifiers':[observe['_rna'](m) for m in obj.modifiers],
            'objectAnimation':observe['_animation'](obj),
            'meshAnimation':observe['_animation'](obj.data),
            'shapeAnimation':observe['_animation'](keys) if keys else None,
            'shapeKeys':[{'name':k.name,'value':k.value,'mute':k.mute,'group':k.vertex_group,'relative':k.relative_key.name} for k in keys.key_blocks] if keys else [],
            'deformGroups':[{'name':g.name,'index':g.index} for g in obj.vertex_groups]}


def _unposed(bpy,obj,surface):
    arms=[m for m in obj.modifiers if m.type=='ARMATURE']
    if len(arms)!=1 or not arms[0].show_viewport:raise RuntimeError('Expected one active Armature')
    try:
        arms[0].show_viewport=False;bpy.context.view_layer.update();result=surface(bpy,obj)
    finally:
        arms[0].show_viewport=True;bpy.context.view_layer.update()
    return result


def _patch(unposed,actual,side,arrays,prefix):
    p=unposed['vertices'];sign=1 if side=='L' else -1
    near=(p[:,0]*sign>=.57)&(p[:,0]*sign<=.64)&(np.abs(p[:,1])<.07)&(np.abs(p[:,2]-1.235)<.07)
    triangles=actual['triangles'];selected=triangles[np.any(near[triangles],axis=1)]
    ids=np.unique(selected);lookup=np.full(len(p),-1,np.int32);lookup[ids]=np.arange(len(ids))
    core=(p[ids,0]*sign>=.602)&(p[ids,0]*sign<=.631)&(np.abs(p[ids,1])<.06)&(np.abs(p[ids,2]-1.235)<.06)
    arrays[prefix+'_originalIds']=ids.astype(np.int32)
    arrays[prefix+'_unposed']=p[ids].astype(np.float32)
    arrays[prefix+'_actual']=actual['vertices'][ids].astype(np.float32)
    arrays[prefix+'_triangles']=lookup[selected]
    arrays[prefix+'_core']=core
    return ids,core


def _distances(points,hand):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    tree=BVHTree.FromPolygons(hand['vertices'].tolist(),hand['triangles'].tolist(),all_triangles=True)
    nearest=[];normals=[];faces=[];distances=[]
    for point in points:
        hit,normal,face,distance=tree.find_nearest(Vector(point))
        if hit is None:raise RuntimeError('Actual hand BVH has no nearest result')
        nearest.append(tuple(hit));normals.append(tuple(normal));faces.append(face);distances.append(distance)
    return {'closest':np.asarray(nearest,np.float32),'closestNormal':np.asarray(normals,np.float32),
            'closestTriangle':np.asarray(faces,np.int32),'distance':np.asarray(distances,np.float64)}


def probe(source_relative,source_sha,package_relative,output_relative):
    import bpy
    from mathutils import Matrix
    source=_path(source_relative);output=_path(output_relative);archive=output.with_suffix('.npz')
    if output.exists() or archive.exists():raise FileExistsError('Keep previous actual observations')
    merger_path=HERE/'hand_asset_integrate_r11_v2.py'
    if _sha(merger_path)!=MERGER_SHA:raise RuntimeError('Frozen merger changed')
    api=runpy.run_path(str(merger_path));_,wrist,observe=api['_apis']()
    original_ids=observe['_ids'](bpy)
    body=bpy.data.objects['R01 Head'];original_keys=wrist['_key_contract'](body)
    original_body_state=_object_state(body,observe)
    original_hands={name:api['_co'](bpy.data.objects[name].data.vertices) for name in api['HANDS']}
    result={};arrays={}

    def inspect(bpy,body,rig,wrist):
        adapter=runpy.run_path(str(HERE/'hand_asset_pose_review_r11.py'))['_api']()
        frames={};snapshots={}
        for side in ('L','R'):
            frame=adapter['_rest_frame'](bpy.data.objects['Hand '+side+' · continuous anatomy'],rig,side);frames[side]=frame
            snapshots[side]={name:{'basis':np.asarray(rig.pose.bones[name].matrix_basis).tolist(),'rotationMode':rig.pose.bones[name].rotation_mode} for name in frame['axes']}
        result.update({'source':source_relative,'sourceSha256':source_sha,'package':package_relative,'packageSha256':_sha(_path(package_relative)),
            'helperSha256':_sha(__file__),'frozenMergerSha256':MERGER_SHA,'bodyBeforeIntegration':original_body_state,
            'bodyAfterLocalKey':_object_state(body,observe),'hands':{name:_object_state(bpy.data.objects[name],observe) for name in api['HANDS']},
            'depsgraphMode':str(bpy.context.evaluated_depsgraph_get().mode),'poses':[],
            'savedBlend':False,'sourceGeometryEditedOnDisk':False,'appearanceAccepted':False,
            'scope':'Actual failed integration at unchanged target subdivision, modifiers and dense body. Proximity is not appearance acceptance.'})
        raw=api['_co'](body.data.vertices);mix=wrist['_mix'](body)
        for side in ('L','R'):
            sign=1 if side=='L' else -1
            ids=np.flatnonzero((raw[:,0]*sign>.55)&(raw[:,0]*sign<.65)&(np.abs(raw[:,1])<.075)&(np.abs(raw[:,2]-1.235)<.075))
            arrays['raw_'+side+'_ids']=ids.astype(np.int32);arrays['raw_'+side+'_base']=raw[ids].astype(np.float32)
            arrays['raw_'+side+'_mix']=mix[ids].astype(np.float32);arrays['raw_'+side+'_weights']=api['_weights'](body)[ids]
        mesh=body.data;mesh.calc_loop_triangles();arrays['raw_body_triangles']=np.asarray([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
        try:
            for pose in ('relaxed','wrist-flex','wrist-extend'):
                for side in ('L','R'):adapter['_apply_pose'](bpy,rig,frames[side],pose,snapshots[side])
                unposed=_unposed(bpy,body,wrist['_body_surface']);actual=wrist['_body_surface'](bpy,body)
                topology=wrist['_same_body_topology'](unposed,actual);row={'pose':pose,'actualBodyVertices':len(actual['vertices']),'topology':topology,'sides':{}}
                for side in ('L','R'):
                    prefix=pose.replace('-','_')+'_'+side;ids,core=_patch(unposed,actual,side,arrays,prefix+'_body')
                    hand_obj=bpy.data.objects['Hand '+side+' · continuous anatomy'];hand=adapter['_surface'](bpy,hand_obj)
                    hand_rest=_unposed(bpy,hand_obj,adapter['_surface'])
                    for label,surface in (('hand',hand),('handRest',hand_rest)):
                        arrays[prefix+'_'+label+'_positions']=surface['vertices'].astype(np.float32)
                        arrays[prefix+'_'+label+'_triangles']=surface['triangles'].astype(np.int32)
                    points=actual['vertices'][ids[core]];metrics=wrist['_inside_samples'](points,hand)
                    distances=_distances(points,hand)
                    for label,values in distances.items():arrays[prefix+'_'+label]=values
                    rank=np.argsort(distances['distance'])[:20]
                    metrics.update({'arrayPrefix':prefix,'bodyPatchPoints':len(ids),'coreOriginalVertexIds':ids[core].tolist(),
                                    'closest20LocalCoreIndices':rank.tolist(),'closest20DistancesMeters':distances['distance'][rank].tolist(),
                                    'belowRequired150Microns':int(np.sum(distances['distance']<.00015)),
                                    'failedRequiredGate':bool(metrics['outsideByThreeRayMajority'] or np.min(distances['distance'])<.00015)})
                    row['sides'][side]=metrics
                result['poses'].append(row)
        finally:
            for snapshot in snapshots.values():
                for name,record in snapshot.items():
                    bone=rig.pose.bones[name];bone.rotation_mode=record['rotationMode'];bone.matrix_basis=Matrix(record['basis'])
            bpy.context.view_layer.update()
        raise ProbeComplete('Actual failed-stage geometry captured; invoke frozen merger rollback')

    api['prepare'].__globals__['_pose_checks']=inspect
    impossible_output=str(output.with_suffix('.never-save.blend').relative_to(project_path()))
    caught=False
    try:
        api['prepare'](source_relative,source_sha,package_relative,impossible_output)
    except ProbeComplete:caught=True
    if not caught:raise RuntimeError('Probe did not reach and abort the exact integration gate')
    if _path(impossible_output).exists():raise RuntimeError('Probe unexpectedly saved a Blend')
    if set(observe['_ids'](bpy))!=set(original_ids):raise RuntimeError('Frozen merger rollback did not restore ID set')
    if wrist['_key_contract'](body)!=original_keys or _object_state(body,observe)!=original_body_state:
        raise RuntimeError('Body key/modifier/animation state changed after rollback')
    if any(not np.array_equal(api['_co'](bpy.data.objects[name].data.vertices),values) for name,values in original_hands.items()):
        raise RuntimeError('Original hand geometry was not restored')
    if _sha(source)!=source_sha:raise RuntimeError('Input source changed on disk')
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as stream:np.savez_compressed(stream,**arrays)
    result.update({'arrays':str(archive.relative_to(project_path())),'arraysSha256':_sha(archive),'rollbackVerified':True})
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return {'report':output_relative,'reportSha256':_sha(output),'arraysSha256':result['arraysSha256'],'savedBlend':False,
            'actualPoseFailures':[{r['pose']:{s:{k:r['sides'][s][k] for k in ('outsideByThreeRayMajority','minimumSurfaceDistanceMeters','belowRequired150Microns')} for s in ('L','R')}} for r in result['poses']]}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-sha','package','output'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args(argv);print('R11_INTEGRATED_WRIST_PROBE '+json.dumps(probe(a.source,a.source_sha,a.package,a.output),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

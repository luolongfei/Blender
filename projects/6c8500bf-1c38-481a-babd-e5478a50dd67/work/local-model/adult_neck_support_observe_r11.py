"""Measure actual post-Multires/Mask influence of hidden neck cage points.

Read-only on disk, fresh background source only. Temporarily remove one
explicit part of the existing neck key, observe evaluated geometry, then
restore every coordinate and value exactly. No renderer, source save or
source-key acceptance by inference from graph distance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';KEY='R11 natural middle neck proportion'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(data,field='co',width=3,dtype=np.float32):
    a=np.empty((len(data),width),dtype);data.foreach_get(field,a.ravel());return a


def _snapshot(head):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();ev=head.evaluated_get(dg);mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        T=np.asarray(ev.matrix_world,float);co=_array(mesh.vertices)
        return co@T[:3,:3].T+T[:3,3],_array(mesh.loops,'vertex_index',1,np.int32).ravel()
    finally:ev.to_mesh_clear()


def observe(source_relative,output_relative):
    import bpy
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve()
    for path in (source,output):path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if source.suffix!='.blend' or output.suffix!='.json' or output.exists():raise RuntimeError('Explicit source and fresh report required')
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh background source required')
    head=bpy.data.objects[HEAD];key=head.data.shape_keys.key_blocks[KEY]
    if abs(key.value-1.)>1e-8 or key.mute:raise RuntimeError('Observe actual enabled static neck key only')
    before=_array(key.data);basis=_array(key.relative_key.data);source_sha=_sha(source)
    selection_path=project_path()/'work/reconstruction-r02/female-head-selection-r11.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if _sha(project_path()/selection['source_npz'])!=selection['source_npz_sha256']:raise RuntimeError('Original author head selection provenance mismatch')
    edges=_array(head.data.edges,'vertices',2,np.int32);distance=np.full(len(before),999,np.int32)
    distance[np.asarray(selection['original_vertex_indices'],np.int32)]=0
    for step in range(1,5):
        touching=edges[(distance[edges]==step-1).any(1)].ravel();distance[touching]=np.minimum(distance[touching],step)
    affected=np.any(before!=basis,axis=1);uv=_api('production_uv_r08');pose=uv['_pose_contract']()
    values={k.name:(float(k.value),bool(k.mute),_sha_array(_array(k.data))) for k in head.data.shape_keys.key_blocks}
    groups={'outside_ring1':affected&(distance==1),'outside_ring2':affected&(distance==2),
        'outside_beyond_two_rings':affected&(distance>2),'all_outside_author_head':affected&(distance>0)}
    bpy.context.view_layer.update();baseline,topology=_snapshot(head);results={}
    try:
        for name,mask in groups.items():
            candidate=before.copy();candidate[mask]=basis[mask];key.data.foreach_set('co',candidate.ravel());head.data.update();head.data.shape_keys.update_tag();bpy.context.view_layer.update()
            now,loops=_snapshot(head)
            if not np.array_equal(loops,topology):raise RuntimeError('Mask topology changed during displacement-only observation')
            delta=np.linalg.norm(now-baseline,axis=1);changed=delta>1e-9
            results[name]={'rawPointsTemporarilyZeroed':int(mask.sum()),'rawVertexIds':np.flatnonzero(mask).tolist(),
                'postMultiresMaskVertices':len(now),'visibleVertexArraysBitExact':bool(np.array_equal(now,baseline)),
                'visibleMaximumWorldChangeMeters':float(delta.max()),'visibleVerticesChangedAbove1nm':int(changed.sum()),
                'visibleChangedWorldBounds':None if not changed.any() else [baseline[changed].min(0).tolist(),baseline[changed].max(0).tolist()],
                'changedWorldDisplacementQuantilesMeters':None if not changed.any() else np.quantile(delta[changed],[0,.5,.95,1]).tolist()}
            key.data.foreach_set('co',before.ravel());head.data.update();head.data.shape_keys.update_tag();bpy.context.view_layer.update()
    finally:
        key.data.foreach_set('co',before.ravel());head.data.update();head.data.shape_keys.update_tag();bpy.context.view_layer.update()
    restored,restored_topology=_snapshot(head)
    current_values={k.name:(float(k.value),bool(k.mute),_sha_array(_array(k.data))) for k in head.data.shape_keys.key_blocks}
    if current_values!=values or uv['_pose_contract']()!=pose or not np.array_equal(restored,baseline) or not np.array_equal(restored_topology,topology) or _sha(source)!=source_sha:
        raise RuntimeError('Read-only observation did not restore exactly')
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
        'sourceKeySha256':_api('natural_skin_adult_r11')['_ah'](before),'authorSelectionSha256':_sha(selection_path),
        'frame':[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe],'groups':results,
        'sourceKeyAndEvaluatedGeometryRestoredExactly':True,'savedBlend':False,'appearanceAccepted':False,
        'interpretation':'Actual post-Multires/Mask displacement measurement, not an inference that all hidden cage points are unneeded. No candidate is altered based on this report automatically.'}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NECK_SUPPORT_OBSERVED '+json.dumps({'output':str(output.relative_to(project_path())),'sourceSha256':source_sha}),flush=True)
    return report


def _sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);observe(a.source,a.output)

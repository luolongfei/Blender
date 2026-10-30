"""Unsaved v3 replay with full failed dense geometry and independent BVH diagnosis."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve();npz=out.with_suffix('.npz')
    for path in (source,out,npz):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists() or npz.exists():raise RuntimeError('Fresh source and unused diagnostic outputs required')
    helper=HERE/'donor_groom_finish_v3_r11.py'
    if sha(helper)!='ee8eae92f6163ef9c9667d681547acb99557badb2f5a89117288d27e54a9b072':raise RuntimeError('Diagnostic must replay exact failed v3 helper')
    source_sha=sha(source);rig=bpy.data.objects['Character_Rig'];rig.data.pose_position='REST';bpy.context.view_layer.update()
    preparation=runpy.run_path(str(helper))['apply'](expected_source_relative=a.source)
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'));probe=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))
    obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);original=modifier.node_group
    arrays={};rows=[];memo={};replacements=[]
    rows.append(probe['capture'](obj,'raw-volume',arrays,evaluated=False))
    try:
        tree=review['copy_render_branch'](original,memo,replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        for label,name in (('interpolate','Group.005'),('big-clump','Group.008'),('small-clump','Group.006'),
                           ('before-contact','Join Geometry.001'),('after-contact','R11 full head clearance'),('final','R11 strand tip taper')):
            geometry=next(s for s in tree.nodes[name].outputs if s.type=='GEOMETRY')
            for link in list(output.links):tree.links.remove(link)
            tree.links.new(geometry,output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update();rows.append(probe['capture'](obj,label,arrays))
            print('R11_FAILED_GROOM_STAGE '+label,flush=True)
    finally:modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));scalp,skin,headreport=fit['_head_trees'](bpy.data.objects['R11 Female head study'])
    before=arrays['before-contact_world'];after=arrays['after-contact_world'];delta=after-before;distance=np.linalg.norm(delta,axis=1)
    starts=arrays['before-contact_starts'];lengths=arrays['before-contact_lengths'];ids=np.argsort(distance)[-256:][::-1];details=[]
    for index in ids:
        q=before[index];hit,n,face,d=skin.find_nearest(Vector(q));signed=float((Vector(q)-hit).dot(n));curve=int(np.searchsorted(starts,index,side='right')-1)
        details.append({'point':int(index),'curve':curve,'curvePoint':int(index-starts[curve]),'curvePoints':int(lengths[curve]),
            'beforeWorld':q.tolist(),'afterWorld':after[index].tolist(),'gnMoveMeters':float(distance[index]),
            'actualBVHNearest':list(hit),'actualBVHNormal':list(n),'actualBVHSignedDistanceMeters':signed,
            'actualBVHRequiredMoveMeters':max(0.,.00020-signed),'gnMoveDotBVHNormal':float(np.dot(delta[index],np.asarray(n))),
            'curveRoot':before[starts[curve]].tolist(),'curveTip':before[starts[curve]+lengths[curve]-1].tolist()})
    head=bpy.data.objects['R11 Female head study'].evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=head.to_mesh()
    try:
        q=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',q);q=q.reshape(-1,3);m=np.asarray(head.matrix_world)
        arrays['head_world']=(q@m[:3,:3].T+m[:3,3]).astype(np.float32);mesh.calc_loop_triangles();arrays['head_triangles']=np.array([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
    finally:head.to_mesh_clear()
    if sha(source)!=source_sha:raise RuntimeError('Input source changed')
    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
    report={'source':a.source,'sourceSha256':source_sha,'candidateHelperSha256':sha(helper),'helperSha256':sha(__file__),
            'arraysSha256':sha(npz),'stages':rows,'largest256ContactMoves':details,'head':headreport,
            'contactMoveMetersP50P95P99Max':np.percentile(distance,[50,95,99,100]).tolist(),
            'preparation':preparation,'savedBlend':False,'scope':'Exact failed v3 styling replay and independent actual BVH measurement; failure is preserved, not relaxed or accepted.'}
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_FAILED_GROOM_OBSERVED '+json.dumps({'output':a.output,'largestMoveMeters':float(distance.max())}),flush=True)
if __name__=='__main__':main()

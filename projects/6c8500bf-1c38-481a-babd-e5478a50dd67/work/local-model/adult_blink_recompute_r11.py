"""Current-size observation/recompute entry for the adult R11 blink work.

This exports the actual current relative-key mix, both actual globes, and the
evaluated visible upper/lower contours. It deliberately does NOT install an
old-size blink delta. The 20-point socket control ring is topology context,
never asserted to be the physical wet line. No save, rendering or live access.
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
HEAD='R11 Female head study';CONTROLS={'L':'R11 blink L','R':'R11 blink R'}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a


def _snapshot(obj):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        T=np.asarray(ev.matrix_world,float);world=_array(mesh.vertices)@T[:3,:3].T+T[:3,3]
        mesh.calc_loop_triangles();tri=_array(mesh.loop_triangles,'vertices',3,np.int32)
        bvh=BVHTree.FromPolygons([Vector(v) for v in world],tri.tolist(),all_triangles=True)
        if obj.name==HEAD:
            keep=(abs(world[:,0])<.075)&(world[:,1]<-.035)&(world[:,2]>1.390)&(world[:,2]<1.480)
            triids=np.flatnonzero(keep[tri].any(1));ids=np.unique(tri[triids]);mapping=np.full(len(world),-1,np.int32);mapping[ids]=np.arange(len(ids))
        else:ids=np.arange(len(world));triids=np.arange(len(tri));mapping=ids
        return {'bvh':bvh,'fullWorld':world,'world':world[ids],'originalVertexIds':ids,
            'triangles':mapping[tri[triids]],'originalTriangleIds':triids,'matrixWorld':T.tolist(),
            'fullCounts':[len(world),len(tri)]}
    finally:ev.to_mesh_clear()


def _hit(surface,x,z):
    from mathutils import Vector
    co,normal,index,distance=surface['bvh'].ray_cast(Vector((float(x),-.35,float(z))),Vector((0,1,0)),.70)
    if co is None:return None
    return {'world':list(co),'normal':list(normal),'originalTriangleId':int(index)}


def _measure(head,eye):
    world=eye['fullWorld'];lo=world.min(0);hi=world.max(0);center=(lo+hi)*.5
    xs=np.linspace(lo[0]+.00015,hi[0]-.00015,161);zs=np.linspace(lo[2]+.00010,hi[2]-.00010,121)
    def visible(x,z):
        g=_hit(eye,x,z)
        if g is None:return False
        h=_hit(head,x,z)
        return h is None or g['world'][1]<h['world'][1]-1e-7
    contour=[];segments=[];visible_count=0
    for x in xs:
        flags=np.asarray([visible(x,z) for z in zs]);ids=np.flatnonzero(flags);visible_count+=len(ids)
        if len(ids)<2:continue
        if np.any(np.diff(ids)!=1):segments.append({'x':float(x),'visibleZ':zs[ids].tolist()})
        ends=[]
        for side,inside,outside in [('lower',int(ids[0]),int(ids[0]-1)),('upper',int(ids[-1]),int(ids[-1]+1))]:
            if not 0<=outside<len(zs):
                ends.append({'side':side,'unboundedAtGlobeRim':True,'worldZ':float(zs[inside])});continue
            a=zs[inside];b=zs[outside]
            for _ in range(12):
                mid=(a+b)*.5
                if visible(x,mid):a=mid
                else:b=mid
            z=(a+b)*.5;near=z+(-.00012 if side=='lower' else .00012)
            ends.append({'side':side,'worldZ':float(z),'globe':_hit(eye,x,z),
                'headAtVisibilityTransition':_hit(head,x,z),'headOutside120Micrometers':_hit(head,x,near),
                'unboundedAtGlobeRim':False})
        contour.append({'x':float(x),'lower':ends[0],'upper':ends[1]})
    widths=[q['x'] for q in contour]
    return {'centerWorld':center.tolist(),'diametersWorld':(hi-lo).tolist(),
        'frontVisibleWidthMeters':float(max(widths)-min(widths)) if widths else 0.,
        'frontVisibleAreaSquareMeters':float(visible_count*(xs[1]-xs[0])*(zs[1]-zs[0])),
        'visibleSampleCount':int(visible_count),'contour':contour,'disconnectedVisibleColumns':segments,
        'interpretation':'Actual front-visible globe transition; not a certified closed physical wet line. The original triangle IDs refer to the exported evaluated mesh subset.'}


def _collect():
    import bpy
    head=bpy.data.objects[HEAD];snap=_snapshot(head);arrays={};report={'eyes':{}}
    for side in ('L','R'):
        eye=_snapshot(bpy.data.objects['R11 Adult sclera '+side]);report['eyes'][side]=_measure(snap,eye)
        for k in ('world','originalVertexIds','triangles','originalTriangleIds'):arrays['eye_'+side+'_'+k]=eye[k]
    for k in ('world','originalVertexIds','triangles','originalTriangleIds'):arrays['head_'+k]=snap[k]
    report['headMatrixWorld']=snap['matrixWorld'];report['headFullEvaluatedCounts']=snap['fullCounts']
    proportion=_api('adult_eye_proportion_r11');arrays['currentMixedCageLocal']=proportion['_mixed'](head.data)
    arrays['basisCageLocal']=_array(head.data.shape_keys.reference_key.data)
    report['existingKeys']={k.name:{'value':float(k.value),'mute':bool(k.mute),'relative':k.relative_key.name}
        for k in head.data.shape_keys.key_blocks}
    for index,k in enumerate(head.data.shape_keys.key_blocks):
        name='existingKey_'+str(index);arrays[name]=_array(k.data);report['existingKeys'][k.name]['array']=name
    report['bindingValidation']=_api('face_fibre_binding_r11')['validate'](neutral=False)
    report['proportionMarker']=json.loads(bpy.context.scene['R11_adult_eye_proportion']) if 'R11_adult_eye_proportion' in bpy.context.scene else None
    return report,arrays


def probe(left=None,right=None):
    """Audit existing controls at actual values; does not create blink keys."""
    import bpy
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks;restore={}
    try:
        for side,value in (('L',left),('R',right)):
            if value is None:continue
            if not 0.<=value<=1. or CONTROLS[side] not in keys:raise RuntimeError('Existing bounded blink control required: '+side)
            key=keys[CONTROLS[side]];restore[key.name]=float(key.value);key.value=value
        bpy.context.view_layer.update()
        with _api('face_contact_audit_r11')['rest_evaluation']():report,_=_collect()
        report.update(controls=CONTROLS,requestedValues={'L':left,'R':right},sourceSaved=False,
            limits=['Front visibility alone does not establish collision-free lid thickness or natural animation.','Bind validation measures transport parity; actual eyelash self-contact and eye macro still need review.'])
        return report
    finally:
        for name,value in restore.items():keys[name].value=value
        bpy.context.view_layer.update()


def observe(source_relative,output_relative):
    import bpy
    root=(project_path()/'work/reconstruction-r02').resolve();source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();npz=output.with_suffix('.npz')
    for path in (source,output):path.relative_to(root)
    if source.suffix!='.blend' or output.suffix!='.json' or output.exists() or npz.exists():raise RuntimeError('Fresh observation paths required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent background source required')
    if 'R11_adult_eye_proportion' not in bpy.context.scene:raise RuntimeError('Wait for actual new eye-proportion source; do not reuse old-size observations')
    keys=bpy.data.objects[HEAD].data.shape_keys.key_blocks
    if abs(keys['R11 adult eye proportion'].value-1)>1e-7:raise RuntimeError('Proportion key and larger globe must be evaluated together')
    for name in ('R11 gentle smile','R11 natural soft smile',*CONTROLS.values()):
        if name in keys and abs(keys[name].value)>1e-7:raise RuntimeError('Neutral expression needed for blink recomputation: '+name)
    source_sha=_sha(source);old=_api('adult_jaw_contour_r11')['_keys'](bpy.data.objects[HEAD].data);pose=_api('production_uv_r08')['_pose_contract']()
    with _api('face_contact_audit_r11')['rest_evaluation']():report,arrays=_collect()
    now=_api('adult_jaw_contour_r11')['_keys'](bpy.data.objects[HEAD].data)
    for name,a in old.items():
        b=now[name]
        if not np.array_equal(a['coords'],b['coords']) or any(a[k]!=b[k] for k in a if k!='coords'):raise RuntimeError('Observation changed existing key')
    if _api('production_uv_r08')['_pose_contract']()!=pose or _sha(source)!=source_sha:raise RuntimeError('Observation changed source or rig pose')
    np.savez_compressed(npz,**arrays)
    report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
        arraysSha256=_sha(npz),controls=CONTROLS,sourceSaved=False,appearanceAccepted=False,
        newBlinkInstalled=False,scope='Actual current-size observation for recomputing a spherical lid path, thickness and intermediate corrections. No old-size blink data applied.')
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_BLINK_RECOMPUTE_OBSERVED '+json.dumps({'output':str(output),'arraysSha256':report['arraysSha256']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);observe(a.source,a.output)

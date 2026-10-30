"""Reversible brow-fibre flow refinement, keeping existing object/key IDs.

No new brow mesh, expression or material. Fits the present neutral brow arc,
redistributes the existing roots along it, and directs tapered fibres along
the arc instead of ending every fibre with the same upward hook. Actual
Blender application re-seats all points on the evaluated forehead. No save or
render; explicit run() only. Bang occlusion remains physically correct.
"""
import json
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError:
    bpy=None

NAMES=('R02 Soft brow -1','R02 Soft brow 1')
MARKER='R08_brow_arc_flow'


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def plan(world,sizes,radii):
    """Pure numeric X/Z and radius plan from the actual current keyed points."""
    world=np.asarray(world,dtype=float);radii=np.asarray(radii,dtype=float)
    if len(world)!=sum(sizes) or any(n!=5 for n in sizes) or len(sizes)!=260:
        raise RuntimeError('Expected the existing 260 x 5-point brow topology')
    starts=np.r_[0,np.cumsum(sizes)[:-1]];roots=world[starts]
    side=1 if np.median(roots[:,0])>0 else -1
    xx=side*roots[:,0];low,high=float(xx.min()),float(xx.max());span=high-low
    if not .025<span<.055:raise RuntimeError('Unexpected neutral brow width')
    rank=np.argsort(xx);t=(xx-low)/span
    # Cubic follows the already-authored brow arch. A second fit suppresses a
    # few extreme root offsets without moving the mean arc toward hair tips.
    curve=np.polynomial.Polynomial.fit(t,roots[:,2],3).convert()
    residual=roots[:,2]-curve(t)
    keep=np.abs(residual-np.median(residual))<.0017
    if keep.sum()<180:raise RuntimeError('No coherent current brow arch')
    curve=np.polynomial.Polynomial.fit(t[keep],roots[keep,2],3).convert()
    residual=roots[:,2]-curve(t)
    target=world.copy();new_radius=radii.copy()
    for order,i in enumerate(rank):
        start=starts[i];count=sizes[i];u=np.linspace(0,1,count)
        # Existing strand identity and keys stay attached to the same strand.
        # Stratification removes local random clusters/gaps without adding hair.
        jitter=.22*np.sin(i*2.3999632297)
        tt=(order+.5+jitter)/len(rank)
        root_x=low+tt*span
        length=.0031+.00055*np.sin(i*1.6180339887)
        length*=1.-.45*smooth((tt-.75)/.25)
        end_limit=high+.0014
        strand_x=np.minimum(root_x+length*u,end_limit)
        arc_t=(strand_x-low)/span
        thickness=1.-.62*smooth((tt-.48)/.52)
        root_offset=np.clip(residual[i],-.0015,.0015)*thickness
        # Interior fibres rise a little; middle/tail fibres follow the brow arc.
        # The old universal +1.5 mm hook is removed, avoiding a sawtooth edge.
        inner_lift=.00065*(1.-smooth(tt/.24))*u
        soft_lift=.00020*np.sin(np.pi*u)*(1.-tt)
        target[start:start+count,0]=side*strand_x
        target[start:start+count,2]=curve(arc_t)+root_offset+inner_lift+soft_lift
        root_radius=radii[start]
        tail_thinning=1.-.40*smooth((tt-.65)/.35)
        new_radius[start:start+count]=root_radius*tail_thinning*np.maximum(.045,(1.-u)**.62)
    shift=np.linalg.norm((target-world)[:,[0,2]],axis=1)
    if shift.max()>.004:raise RuntimeError(f'Brow planar change exceeds 4 mm: {shift.max()*1000:.4f} mm')
    return target,new_radius,{'fibres':len(sizes),'maximumXZShiftMm':float(shift.max()*1000),
        'originalRootGapMm':float(np.diff(np.sort(xx)).max()*1000),
        'newRootGapMm':float(np.diff(np.sort(side*target[starts,0])).max()*1000),
        'sourceRootArcSpanMm':span*1000,'newRadiusRange':[float(new_radius.min()),float(new_radius.max())],
        'method':'Existing-root cubic arc, rank-stratified roots, variable 1.7–3.65 mm arc-following fibres, outer-tail and along-fibre taper'}


def read(points):
    a=np.empty(len(points)*3);points.foreach_get('co',a)
    return a.reshape(-1,3)


def bounded_surface_fit(plans,surface_y,max_shift_mm=4.5):
    """Common conservative strength for both brows; every trial re-rays skin.

    surface_y(name, point_index, x, z) returns the actual front surface Y or
    None. Never interpolate/clamp a previously projected Y: the changed X/Z
    position must receive its own surface contact. No Blender data is written.
    """
    attempts=[];limit=max_shift_mm*.001-2e-6
    for strength in (1.,.9,.8,.7,.6,.5,.4,.3,.2,.1):
        candidates=[];rows=[];valid=True
        for name,world,proposed in plans:
            candidate=world+(proposed-world)*strength
            missed=None
            for i,p in enumerate(candidate):
                hit=surface_y(name,i,float(p[0]),float(p[2]))
                if hit is None or not np.isfinite(hit):
                    missed=i;valid=False;break
                candidate[i,1]=float(hit)-.00023
            if missed is not None:
                rows.append({'object':name,'surfaceRayMissPoint':int(missed)})
                continue
            distances=np.linalg.norm(candidate-world,axis=1)
            worst=int(np.argmax(distances));maximum=float(distances[worst])
            rows.append({'object':name,'maximumShiftMm':maximum*1000,
                'maximumPoint':worst,'beforeWorld':world[worst].tolist(),
                'afterWorld':candidate[worst].tolist()})
            valid=valid and maximum<=limit
            candidates.append(candidate)
        attempts.append({'strength':strength,'objects':rows,'accepted':bool(valid)})
        if valid:
            return strength,candidates,attempts
    raise RuntimeError('No bounded surface-contact brow candidate; no writes: '+json.dumps(attempts))


def run(diagnostics_only=False):
    if bpy is None:raise RuntimeError('Use explicit run() in the saved independent Blender candidate')
    if not bpy.data.filepath or Path(bpy.data.filepath).name=='character.blend':
        raise RuntimeError('Saved independent candidate required')
    objects=[bpy.data.objects[name] for name in NAMES]
    if any(o.get(MARKER) for o in objects):raise RuntimeError('Brow flow already applied; restore before repeating')
    if any(not o.get('R07_brow_surface_contact') for o in objects):
        raise RuntimeError('Apply/retain the accepted R07 actual-key surface-contact fix first')
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    try:
        head=bpy.data.objects['R01 Head'].evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=head.to_mesh()
        try:
            bvh=BVHTree.FromPolygons([head.matrix_world@v.co for v in mesh.vertices],
                                     [tuple(p.vertices) for p in mesh.polygons])
        finally:head.to_mesh_clear()
        sources=[];plans=[]
        for obj in objects:
            if obj.type!='CURVE' or not obj.data.shape_keys:raise RuntimeError('Expected existing keyed brow curve')
            keys=obj.data.shape_keys.key_blocks
            if any(abs(k.value)>1e-6 for k in list(keys)[1:]):raise RuntimeError('Neutral brows required for the arc fit')
            arrays={k.name:read(k.data) for k in keys}
            current=arrays[keys[0].name]
            matrix=np.asarray(obj.matrix_world);inverse=np.linalg.inv(matrix)
            world=current@matrix[:3,:3].T+matrix[:3,3]
            sizes=[len(s.points) for s in obj.data.splines]
            old_radius=np.array([p.radius for s in obj.data.splines for p in s.points])
            target,radius,report=plan(world,sizes,old_radius)
            plans.append((obj.name,world,target))
            sources.append((obj,arrays,old_radius,world,inverse,radius,report,sizes))
        def surface_y(name,i,x,z):
            hit,_,_,_=bvh.ray_cast(Vector((x,-.5,z)),Vector((0,1,0)))
            return None if hit is None else hit.y
        strength,targets,attempts=bounded_surface_fit(plans,surface_y)
        prepared=[];reports=[]
        for source,target in zip(sources,targets):
            obj,arrays,old_radius,world,inverse,planned_radius,report,sizes=source
            radius=old_radius+(planned_radius-old_radius)*strength
            displacement=(target-world)@inverse[:3,:3].T
            maximum=float(np.linalg.norm(target-world,axis=1).max()*1000)
            if maximum>4.5:raise RuntimeError(f'Brow change exceeds 4.5 mm: {obj.name}, {maximum:.4f} mm')
            starts=np.r_[0,np.cumsum(sizes)[:-1]]
            report['plannedMaximumXZShiftMm']=report['maximumXZShiftMm']
            report['plannedNewRootGapMm']=report['newRootGapMm']
            report.update({'object':obj.name,'commonStrength':strength,'maximumShiftMm':maximum,'skinClearanceMm':.23,
                'maximumXZShiftMm':float(np.linalg.norm((target-world)[:,[0,2]],axis=1).max()*1000),
                'newRootGapMm':float(np.diff(np.sort(np.abs(target[starts,0]))).max()*1000),
                'newRadiusRange':[float(radius.min()),float(radius.max())],
                'allRelativeKeyDeltasPreserved':True,'topologyObjectDriversUnchanged':True})
            prepared.append((obj,arrays,old_radius,displacement,radius));reports.append(report)
        result={'brows':reports,'commonStrength':strength,'surfaceLineSearch':attempts,'writesPerformed':not diagnostics_only,
                'visualStatus':'Unrendered candidate; check neutral/raised/furrowed closeups. Physical bang occlusion remains.'}
        if diagnostics_only:print(json.dumps(result));return result
        for obj,arrays,old_radius,displacement,radius in prepared:
            backup=bpy.data.texts.new('R08 brow flow backup '+obj.name);backup.use_fake_user=True
            backup.write(json.dumps({'object':obj.name,'arrays':{k:a.tolist() for k,a in arrays.items()},'radius':old_radius.tolist()}))
            for key in obj.data.shape_keys.key_blocks:
                key.data.foreach_set('co',np.asarray(arrays[key.name]+displacement,dtype=np.float32).ravel())
            for p,r in zip((p for s in obj.data.splines for p in s.points),radius):p.radius=float(r)
            obj[MARKER]=backup.name;obj.data.update_tag();obj.update_tag()
        bpy.context.view_layer.update();print(json.dumps(result));return result
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()


def restore():
    for name in NAMES:
        obj=bpy.data.objects[name]
        if not obj.get(MARKER):continue
        text=bpy.data.texts.get(obj[MARKER])
        if text is None:raise RuntimeError('Retained R08 brow backup missing')
        data=json.loads(text.as_string());keys=obj.data.shape_keys.key_blocks
        if set(keys.keys())!=set(data['arrays']):raise RuntimeError('Brow key set changed; explicit restore needed')
        for key in keys:key.data.foreach_set('co',np.asarray(data['arrays'][key.name],dtype=np.float32).ravel())
        for p,r in zip((p for s in obj.data.splines for p in s.points),data['radius']):p.radius=r
        del obj[MARKER];obj.data.update_tag();obj.update_tag()
    bpy.context.view_layer.update()


if __name__=='__main__':print('Loaded only. Explicit run() or restore(); no save/render.')

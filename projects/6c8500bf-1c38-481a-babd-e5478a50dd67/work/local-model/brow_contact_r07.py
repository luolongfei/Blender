"""Re-seat existing brows using their actual key combination, not stale splines.

Only Y contact changes; every fibre's world X/Z direction, length projection,
radius and all key deltas remain intact. The same local displacement is added
to Basis and every relative brow key, preserving existing expression drivers.
Numerical backups live in retained Blender Text datablocks. No file IO/render.
"""
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

NAMES=('R02 Soft brow -1','R02 Soft brow 1')
MARKER='R07_brow_surface_contact'


def read(data):
    a=np.empty(len(data)*3,dtype=float);data.foreach_get('co',a)
    return a.reshape(-1,3)


def run(clearance_mm=.23,max_shift_mm=10.0,diagnostics_only=False):
    if not bpy.data.filepath or Path(bpy.data.filepath).name=='character.blend':
        raise RuntimeError('Use the current saved candidate')
    if not .15<=clearance_mm<=.30:raise ValueError('Clearance is bounded to .15–.30 mm')
    objects=[bpy.data.objects[name] for name in NAMES]
    if any(o.get(MARKER) for o in objects):raise RuntimeError('Brows already seated; restore before reapplying')
    if any(o.type!='CURVE' or not o.data.shape_keys for o in objects):
        raise RuntimeError('Expected the existing expression-driven legacy brow curves')
    rig=bpy.data.objects['Character_Rig'];head=bpy.data.objects['R01 Head']
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    try:
        evaluated=head.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        try:
            world=[evaluated.matrix_world@v.co for v in mesh.vertices]
            bvh=BVHTree.FromPolygons(world,[tuple(p.vertices) for p in mesh.polygons])
        finally:evaluated.to_mesh_clear()
        prepared=[];reports=[]
        for obj in objects:
            keys=obj.data.shape_keys.key_blocks
            arrays={k.name:read(k.data) for k in keys}
            current=arrays[keys[0].name].copy()
            for key in list(keys)[1:]:
                if key.vertex_group:raise RuntimeError('Explicit vertex-group evaluation is required: '+key.name)
                if not key.mute:
                    current+=(arrays[key.name]-arrays[key.relative_key.name])*key.value
            matrix=np.array(obj.matrix_world);inverse=np.linalg.inv(matrix)
            world=current@matrix[:3,:3].T+matrix[:3,3]
            sizes=[len(s.points) for s in obj.data.splines]
            if sum(sizes)!=len(current) or any(s.type=='BEZIER' for s in obj.data.splines):
                raise RuntimeError('Unexpected brow point/key ordering')
            hits=[]
            for point in world:
                hit,_,_,_=bvh.ray_cast(Vector((point[0],-.50,point[2])),Vector((0,1,0)))
                if hit is None:raise RuntimeError('Brow ray missed actual head surface: '+obj.name)
                hits.append(hit.y)
            hits=np.array(hits);depth=world[:,1]-hits
            new=world.copy();start=0
            for count in sizes:
                end=start+count
                # Retain a restrained amount of the original per-fibre lift
                # variation while keeping every point 0.15–0.30 mm above skin.
                lift=np.clip(clearance_mm*.001-.10*(depth[start:end]-depth[start]),.00015,.00030)
                new[start:end,1]=hits[start:end]-lift
                start=end
            displacement=(new-world)@inverse[:3,:3].T
            maximum=float(np.linalg.norm(new-world,axis=1).max()*1000)
            report={'object':obj.name,'actual_key_points':len(current),
                    'old_points_behind_skin':int(np.count_nonzero(depth>0)),
                    'old_depth_range_mm':[float(depth.min()*1000),float(depth.max()*1000)],
                    'new_clearance_range_mm':[float((hits-new[:,1]).min()*1000),float((hits-new[:,1]).max()*1000)],
                    'maximum_shift_mm':maximum,'unchanged_world_xz':True,
                    'all_relative_key_deltas_preserved':True,'key_values':{k.name:float(k.value) for k in keys}}
            if maximum>max_shift_mm:
                raise RuntimeError('Brow contact bound exceeded; no writes: '+json.dumps(report))
            prepared.append((obj,arrays,displacement));reports.append(report)
        result={'brows':reports,'writesPerformed':not diagnostics_only,
                'validation':'Actual key-combination contact; neutral/raised/furrowed render still required'}
        if diagnostics_only:print(json.dumps(result));return result
        for obj,arrays,displacement in prepared:
            backup=bpy.data.texts.new('R07 brow contact backup '+obj.name)
            backup.use_fake_user=True
            backup.write(json.dumps({'object':obj.name,'arrays':{k:v.tolist() for k,v in arrays.items()}}))
            for key in obj.data.shape_keys.key_blocks:
                key.data.foreach_set('co',np.asarray(arrays[key.name]+displacement,dtype=np.float32).ravel())
            obj[MARKER]=backup.name;obj.data.update_tag();obj.update_tag()
        bpy.context.view_layer.update();print(json.dumps(result));return result
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()


def restore():
    restored=[]
    for name in NAMES:
        obj=bpy.data.objects[name]
        if not obj.get(MARKER):continue
        text=bpy.data.texts.get(obj[MARKER])
        if text is None:raise RuntimeError('Retained brow backup is missing')
        d=json.loads(text.as_string())
        for key in obj.data.shape_keys.key_blocks:
            if key.name not in d['arrays']:raise RuntimeError('Brow key set has changed; explicit restore needed')
            key.data.foreach_set('co',np.asarray(d['arrays'][key.name],dtype=np.float32).ravel())
        del obj[MARKER];obj.data.update_tag();restored.append(name)
    bpy.context.view_layer.update();return restored


if __name__=='__main__':run()

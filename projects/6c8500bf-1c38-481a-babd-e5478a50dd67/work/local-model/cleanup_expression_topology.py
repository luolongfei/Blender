"""Weld only truly coincident procedural centres, preserving every shape and UV."""
import bpy
import numpy as np

def apply():
    report=[]
    for obj in list(bpy.context.scene.objects):
        if obj.type!="MESH" or not obj.get("facial_role"):
            continue
        old=obj.data
        keys=list(old.shape_keys.key_blocks) if old.shape_keys else []
        positions=[tuple(v.co) for v in old.vertices]
        keydata={k.name:[tuple(p.co) for p in k.data] for k in keys}
        lookup={}
        reps=[]
        remap={}
        for i,co in enumerate(positions):
            token=tuple(round(v,8) for v in co)
            candidates=lookup.setdefault(token,[])
            representative=next((j for j in candidates if all(
                all(abs(a-b)<1e-8 for a,b in zip(data[i],data[j]))
                for data in keydata.values())),None)
            if representative is None:
                candidates.append(i)
                remap[i]=len(reps)
                reps.append(i)
            else:
                remap[i]=remap[representative]
        if len(reps)==len(positions):
            continue
        polygons=[]
        oldloops=[]
        slots=[]
        for poly in old.polygons:
            verts=[]
            loops=[]
            for li in poly.loop_indices:
                vi=remap[old.loops[li].vertex_index]
                if vi not in verts:
                    verts.append(vi)
                    loops.append(li)
            if len(verts)>=3:
                polygons.append(verts)
                oldloops.append(loops)
                slots.append(poly.material_index)
        new=bpy.data.meshes.new(old.name+" · welded")
        new.from_pydata([positions[i] for i in reps],[],polygons)
        new.update()
        for m in old.materials:
            new.materials.append(m)
        for poly,mi in zip(new.polygons,slots):
            poly.material_index=mi
            poly.use_smooth=True
        for olduv in old.uv_layers:
            uv=new.uv_layers.new(name=olduv.name)
            for poly,lis in zip(new.polygons,oldloops):
                for li,oldli in zip(poly.loop_indices,lis):
                    uv.data[li].uv=olduv.data[oldli].uv
        oldvalues={k.name:k.value for k in keys}
        obj.data=new
        for key in keys:
            block=obj.shape_key_add(name=key.name,from_mix=False)
            block.data.foreach_set("co",np.asarray([keydata[key.name][i] for i in reps],dtype=np.float32).ravel())
            block.value=oldvalues[key.name]
        obj.vertex_groups.clear()
        group=obj.vertex_groups.new(name="head")
        group.add(list(range(len(reps))),1,"REPLACE")
        report.append({"object":obj.name,"verticesBefore":len(positions),"verticesAfter":len(reps),
                       "shapesPreserved":len(keys),"UVLayersPreserved":len(old.uv_layers)})
    return report

"""True recessed nostril surfaces and eyebrow hairs, retaining facial keys."""
import math
import bpy
import numpy as np
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

FACE="Face · sculpted planes & original illustration"
VERSION="facial-microstructure-01"


def nostrils():
    obj=bpy.data.objects[FACE]
    if obj.get("nostril_microstructure_version")==VERSION:
        return {"alreadyApplied":True}
    old=obj.data
    old.update()
    chosen=[]
    for sign in (-1,1):
        target=Vector((sign*.00927,-.10596,1.38499))
        poly=min((p for p in old.polygons if len(p.vertices)==4 and
                  p.center.y<-.08),key=lambda p:(p.center-target).length)
        chosen.append(poly.index)
    shape_names=list(old.shape_keys.key_blocks.keys())
    key_values={k.name:(k.value,k.slider_min,k.slider_max) for k in old.shape_keys.key_blocks}
    keycoords={k.name:[v.co.copy() for v in k.data] for k in old.shape_keys.key_blocks}
    vertices=[v.co.copy() for v in old.vertices]
    faces=[list(p.vertices) for p in old.polygons if p.index not in chosen]
    kept=[p for p in old.polygons if p.index not in chosen]
    # Store all corner UVs explicitly so the original skin atlas remains valid.
    uvdata={uv.name:[tuple(uv.data[loop].uv) for p in kept
                     for loop in p.loop_indices] for uv in old.uv_layers}
    pointdata={}
    for attr in old.attributes:
        if attr.domain=="POINT" and not attr.name.startswith(".") and attr.name!="position":
            if attr.data_type=="FLOAT":
                pointdata[attr.name]=("FLOAT",[d.value for d in attr.data])
            elif attr.data_type=="FLOAT_VECTOR":
                pointdata[attr.name]=("FLOAT_VECTOR",[d.vector.copy() for d in attr.data])
    pointdata["NostrilTissue"]=("FLOAT",[0.]*len(vertices))
    assignments=[[(g.group,g.weight) for g in v.groups] for v in old.vertices]
    old_groups=[g.name for g in obj.vertex_groups]
    newfaces=[]
    for pi in chosen:
        poly=old.polygons[pi]
        outer=list(poly.vertices)
        normal=poly.normal.normalized()
        corners=[old.vertices[i].co.copy() for i in outer]
        center=sum(corners,Vector())/4
        # Rounded small aperture lies within the authored quad and follows
        # its downward-facing surface. Four physical rings form the recess.
        last=None
        vertex_uv={}
        for corner,li in zip(outer,poly.loop_indices):
            vertex_uv[corner]={uv.name:uv.data[li].uv.copy() for uv in old.uv_layers}
        for ring,(radius,depth,tissue) in enumerate((
            (.80,.00005,0.),(.62,.00015,.05),(.59,-.0017,.62),(.43,-.0030,.85))):
            current=[]
            for k in range(24):
                angle=-math.pi*3/4+math.tau*k/24
                u=.5+.5*radius*math.cos(angle)
                v=.5+.5*radius*math.sin(angle)
                weights=((1-u)*(1-v),u*(1-v),u*v,(1-u)*v)
                idx=len(vertices)
                newco=sum((co*w for co,w in zip(corners,weights)),Vector())+normal*depth
                vertices.append(newco)
                current.append(idx)
                for name,coords in keycoords.items():
                    coords.append(sum((coords[i]*w for i,w in zip(outer,weights)),Vector())+normal*depth)
                for name,(kind,values) in pointdata.items():
                    if name=="NostrilTissue":
                        values.append(tissue)
                    elif kind=="FLOAT":
                        values.append(sum(values[i]*w for i,w in zip(outer,weights)))
                    else:
                        values.append(sum((values[i]*w for i,w in zip(outer,weights)),Vector()))
                assignments.append([(old_groups.index("head"),1.)])
                vertex_uv[idx]={name:sum((vertex_uv[i][name]*w for i,w in zip(outer,weights)),Vector((0,0)))
                                for name in uvdata}
            if last is None:
                # Each quarter fan meets the unchanged quad boundary.
                for j in range(4):
                    start=j*6
                    for k in range(start,start+6):
                        newfaces.append([outer[j],current[k%24],current[(k+1)%24]])
                    newfaces.append([outer[j],current[(start+6)%24],outer[(j+1)%4]])
            else:
                for k in range(24):
                    newfaces.append([last[k],last[(k+1)%24],current[(k+1)%24],current[k]])
            last=current
        idx=len(vertices)
        vertices.append(center-normal*.0033)
        for name,coords in keycoords.items():
            coords.append(sum((coords[i] for i in outer),Vector())/4-normal*.0033)
        for name,(kind,values) in pointdata.items():
            values.append(.9 if name=="NostrilTissue" else
                          sum(values[i] for i in outer)/4 if kind=="FLOAT" else
                          sum((values[i] for i in outer),Vector())/4)
        assignments.append([(old_groups.index("head"),1.)])
        vertex_uv[idx]={name:sum((vertex_uv[i][name] for i in outer),Vector((0,0)))/4 for name in uvdata}
        for k in range(24):
            newfaces.append([last[k],last[(k+1)%24],idx])
        # Each nostril's new faces only reference its own vertices.
        start=len(faces)
        mine=[f for f in newfaces if all(i in vertex_uv for i in f)]
        faces.extend(mine)
        for name in uvdata:
            uvdata[name].extend(tuple(vertex_uv[i][name]) for f in mine for i in f)
    mesh=bpy.data.meshes.new("Face · continuous skin with recessed nostrils")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    for material in old.materials:
        mesh.materials.append(material)
    for name,data in uvdata.items():
        uv=mesh.uv_layers.new(name=name)
        if len(data)!=len(mesh.loops):
            raise ValueError("UV corner count changed unexpectedly: "+name)
        uv.data.foreach_set("uv",np.asarray(data,dtype=np.float32).ravel())
    for name,(kind,data) in pointdata.items():
        attr=mesh.attributes.new(name,kind,"POINT")
        attr.data.foreach_set("value" if kind=="FLOAT" else "vector",
                             np.asarray(data,dtype=np.float32).ravel())
    # Recalculate winding without reordering vertices or altering UV corners.
    bm=bmesh.new();bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(mesh);bm.free()
    obj.data=mesh
    obj.vertex_groups.clear()
    for name in old_groups:
        obj.vertex_groups.new(name=name)
    for i,groups in enumerate(assignments):
        for gi,weight in groups:
            obj.vertex_groups[gi].add([i],weight,"REPLACE")
    for name in shape_names:
        key=obj.shape_key_add(name=name,from_mix=False)
        key.data.foreach_set("co",np.asarray(keycoords[name],dtype=np.float32).ravel())
        key.value,key.slider_min,key.slider_max=key_values[name]
    for p in mesh.polygons:
        p.use_smooth=True
    mesh.uv_layers.active=mesh.uv_layers["ExpressiveSkinUV"]
    mesh.uv_layers.active.active_render=True
    obj["nostril_microstructure_version"]=VERSION
    return {"removedQuads":chosen,"addedVertices":len(vertices)-len(old.vertices),
            "recessDepthMeters":.0033,"shapeKeysPreserved":shape_names,
            "physicalRecess":True,"nasalAirwayModelled":False}


def brow_hairs():
    col=bpy.data.collections["CHARACTER · Original geometry"]
    rig=bpy.data.objects["Character_Rig"]
    created=[]
    for side in ("L","R"):
        name="Eyebrow individual fibres "+side
        if name in bpy.data.objects:
            continue
        brow=bpy.data.objects["Expressive brow "+side]
        blocks=brow.data.shape_keys.key_blocks
        basis=[v.co.copy() for v in blocks[0].data]
        sign=1 if side=="L" else -1
        x0=min(abs(v.x) for v in basis);x1=max(abs(v.x) for v in basis)
        polygons=[list(p.vertices) for p in brow.data.polygons]
        meshes={}
        for block in blocks:
            points=[v.co.copy() for v in block.data]
            coords=[]
            for k in range(67):
                t=(k+.5)/67
                x=sign*(x0+(x1-x0)*t)
                nearby=sorted(points,key=lambda v:abs(v.x-x))[:6]
                z=sum(v.z for v in nearby)/len(nearby)
                y=sum(v.y for v in nearby)/len(nearby)-.00022
                length=.00115+.0010*(.5+.5*math.sin(k*2.399))
                lean=sign*(.00035+.0011*t)
                for j in range(4):
                    u=j/3
                    xx=x+lean*(u-.45)
                    zz=z+length*(u-.5)
                    yy=y-.00016*math.sin(math.pi*u)
                    width=.000065*(1-.85*u)
                    coords.extend(((xx-width,yy,zz),(xx+width,yy,zz)))
            meshes[block.name]=coords
        faces=[]
        for k in range(67):
            for j in range(3):
                a=k*8+j*2
                faces.append((a,a+1,a+3,a+2))
        mesh=bpy.data.meshes.new(name);mesh.from_pydata(meshes["Basis"],[],faces)
        obj=bpy.data.objects.new(name,mesh);col.objects.link(obj)
        mesh.materials.append(bpy.data.materials["Character / warm charcoal brows"])
        uv=mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            i=loop.vertex_index
            uv.data[loop.index].uv=(i%2,(i%8)//2/3)
        group=obj.vertex_groups.new(name="head");group.add(list(range(len(mesh.vertices))),1,"REPLACE")
        modifier=obj.modifiers.new("Humanoid deformation","ARMATURE");modifier.object=rig
        for keyname,coords in meshes.items():
            key=obj.shape_key_add(name=keyname)
            key.data.foreach_set("co",np.asarray(coords,dtype=np.float32).ravel())
        for p in mesh.polygons:p.use_smooth=True
        obj["part"]="head";obj["facial_role"]="brow"
        obj["facial_microstructure_version"]=VERSION
        created.append({"object":name,"individualHairs":67,"expressionKeys":list(meshes)})
    return created


def apply():
    return {"version":VERSION,"nostrils":nostrils(),"browFibres":brow_hairs()}

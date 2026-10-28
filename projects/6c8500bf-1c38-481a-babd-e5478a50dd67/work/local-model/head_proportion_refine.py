"""A coherent upper-skull/profile correction across the head and its hair.

The neck base, jaw opening, facial expressions and lower ponytail motion are
retained. The field is zero around the tail roots, avoiding a new rig rest pose.
"""
import math
import bpy
import numpy as np
from mathutils import Vector

VERSION="balanced-cranial-profile-02"


def smooth(a,b,x):
    t=min(1,max(0,(x-a)/(b-a)))
    return t*t*(3-2*t)


def transform(co):
    x,y,z=co
    upper=smooth(1.395,1.460,z)
    posterior=smooth(.015,.085,y)
    # Reduce the excessive posterior and upper volume. The forehead becomes
    # more vertical, while the crown remains a continuous curved surface.
    skull_width=smooth(1.460,1.530,z)
    x*=1-.070*skull_width
    cheek=(smooth(.055,.095,abs(x))*smooth(1.340,1.380,z)
           *(1-smooth(1.455,1.500,z))*(1-smooth(.010,.060,y)))
    x-=math.copysign(.010*cheek,x)
    y-=.026*posterior*upper
    z-=.019*smooth(1.465,1.605,z)
    forehead=(1-smooth(-.065,-.015,y))*math.exp(-((z-1.475)/.032)**4)
    y-=.0035*forehead
    return Vector((x,y,z))


def apply(collection=None):
    collection=collection or bpy.data.collections["CHARACTER · Original geometry"]
    objects=[o for o in collection.all_objects if o.type=="MESH" and (
        o.get("facial_role") or o.get("part") in {"head","hair"}
        or o.name.startswith(("Ear_", "EarHelix_", "EarConcha_","Hair_")))]
    if any(o.get("head_profile_version") for o in objects):
        raise RuntimeError("Use the source preceding the head profile pass.")
    rows=[]
    for obj in objects:
        max_change=0
        blocks=obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
        if blocks:
            for block in blocks:
                coords=[]
                for point in block.data:
                    changed=transform(point.co)
                    max_change=max(max_change,(changed-point.co).length)
                    coords.append(changed)
                block.data.foreach_set("co",np.asarray(coords,dtype=np.float32).ravel())
            obj.data.vertices.foreach_set("co",np.asarray(
                [p.co for p in blocks[0].data],dtype=np.float32).ravel())
        else:
            for point in obj.data.vertices:
                changed=transform(point.co)
                max_change=max(max_change,(changed-point.co).length)
                point.co=changed
        obj["head_profile_version"]=VERSION
        obj.data.update()
        rows.append({"object":obj.name,"maxMovementMeters":max_change})
    return {"version":VERSION,"objects":rows,
            "restBonesUnchanged":True,"topologyUvWeightsPreserved":True,
            "lowerFaceAndTailRootsUnchanged":True}

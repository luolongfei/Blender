"""Restrained anatomical refinement for the existing original character.

Only the named neck, ear and continuous-hand mesh coordinates change.
Face keys, head scale, hair, bones, weights, clothes, UVs and materials survive.
The saved point attribute makes repeated application deterministic.
"""
import math
import bpy
from mathutils import Vector

VERSION='body-realism-02'
BASIS='BodyRealismBasis'

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)))
    return t*t*(3-2*t)

def gaussian(x,center,width):
    return math.exp(-((x-center)/width)**2)

def neck_point(co):
    x,y,z=co
    # Keep the repaired insertion under the jaw exactly unchanged.
    active=1-smooth(1.328,1.348,z)
    cy=.006
    radial=Vector((x,y-cy,0))
    if radial.length<1e-8:return co.copy()
    direction=radial.normalized()
    front=max(0.,-direction.y)
    back=max(0.,direction.y)
    # A slender waist under the jaw with a wider, softer clavicular base.
    waist=gaussian(z,1.300,.021)
    flare=1-smooth(1.258,1.282,z)
    nx=x*(1-.058*waist*active)+direction.x*.0030*flare
    ny=y+.0025*(1-smooth(1.274,1.333,z))*active
    ny+=direction.y*(-.0010*waist*front+.0012*waist*back)*active
    # Two shallow oblique sternocleidomastoid ridges, integrated in the skin,
    # travel from the sternum toward the mastoid region. No separate tubes.
    t=smooth(1.263,1.337,z)
    ridge_x=.009+.020*t
    ridge=.00135*gaussian(abs(x),ridge_x,.0055)
    ridge*=smooth(1.259,1.278,z)*(1-smooth(1.323,1.344,z))*front**.55
    nx+=direction.x*ridge
    ny+=direction.y*ridge
    # Restrained throat plane: broad central hollow, no pronounced Adam's apple.
    ny+=.0009*gaussian(x,0,.014)*gaussian(z,1.288,.020)*front*active
    return Vector((nx,ny,z))

def ear_point(co,name=''):
    x,y,z=co
    sign=1 if x>=0 else -1
    ax=abs(x)
    if 'concha' in name:
        # The old concha was a separate oval button standing in front of the
        # ear. Flatten it and seat it in the newly sculpted cartilage bowl.
        ax=.111+(ax-.111)*.82
        y=-.011+(y+.011)*.38+.0022
    # Retain attachment and height, reduce the thick oval silhouette.
    attachment=smooth(.087,.111,ax)
    nx=sign*(ax-.0027*attachment)
    nz=z
    ny=.002+(y-.002)*.80
    # A small backward inclination of the upper pinna softens the disk shape.
    ny+=.0020*smooth(1.395,1.435,z)*attachment
    # Carve the bowl into the same outer surface rather than floating an oval.
    front=1-smooth(-.008,.003,y)
    bowl=gaussian(ax,.111,.0080)*gaussian(z,1.405,.015)
    ny+=.0050*bowl*front
    # Slightly fuller lower attachment, continuous with the jaw-side lobe.
    nx+=sign*.0007*gaussian(z,1.382,.007)*attachment
    # The pinna faces out and forward, not straight toward the face camera.
    # This is shared by all three pieces so rims stay seated in the surface.
    angle=math.radians(34)
    dx=abs(nx)-.103;dy=ny-.002
    nx=sign*(.103+math.cos(angle)*dx-math.sin(angle)*dy)
    ny=.002+math.sin(angle)*dx+math.cos(angle)*dy
    return Vector((nx,ny,nz))

def hand_point(co):
    x,y,z=co
    ax=abs(x)
    # Preserve wrist fit, all finger joint centers, fingertips and nail seats.
    longitudinal=smooth(.616,.635,ax)*(1-smooth(.657,.675,ax))
    dorsal=smooth(1.238,1.247,z)
    if longitudinal<=0:return co.copy()
    # Metacarpal tendons are broad, low relief; the hand stays soft, not bony.
    t=smooth(.620,.672,ax)
    tendons=0.
    for dest in (-.020,-.005,.010,.024):
        line=dest*(.30+.70*t)
        tendons+=gaussian(y,line,.0043)
    dz=.00060*tendons*longitudinal*dorsal
    # Subtle transverse arch across the palm, leaving central grip padding.
    arch=.0008*gaussian(ax,.651,.019)*smooth(.009,.027,abs(y))
    dz-=arch*(1-dorsal)*longitudinal
    return Vector((x,y,z+dz))

def basis(obj):
    attr=obj.data.attributes.get(BASIS)
    if attr:
        if attr.domain!='POINT' or attr.data_type!='FLOAT_VECTOR' or len(attr.data)!=len(obj.data.vertices):
            raise ValueError(obj.name+': incompatible stored anatomical basis')
        return [d.vector.copy() for d in attr.data]
    return [v.co.copy() for v in obj.data.vertices]

def apply(collection,rig):
    if len(rig.data.bones)!=108:
        raise ValueError('Expected the existing 108-bone rig')
    targets=[]
    for obj in collection.all_objects:
        if obj.type!='MESH':continue
        if obj.name=='Neck':field=neck_point
        elif obj.name in {f'Ear{middle} {side}' for middle in ('',' helix',' concha') for side in ('L','R')}:
            field=lambda co,name=obj.name:ear_point(co,name)
        elif obj.name in ('Hand L · continuous anatomy','Hand R · continuous anatomy'):
            field=hand_point
        else:continue
        if obj.data.shape_keys:raise ValueError('Protected shape keys: '+obj.name)
        coordinates=basis(obj)
        planned=[field(co) for co in coordinates]
        if any(not all(math.isfinite(x) for x in co) for co in planned):
            raise ValueError('Nonfinite anatomical point')
        targets.append((obj,coordinates,planned))
    if len(targets)!=9:
        raise ValueError('Expected neck, six ear pieces and two continuous hands; found '+str(len(targets)))
    report={'version':VERSION,'objects':[],'preserved':[
        'face and all expressions','head scale and hair','108 bones and skin weights',
        'spring/collider parameters','UVs, materials and topology','wrist/finger joint centers and nail seats']}
    for obj,coordinates,planned in targets:
        attr=obj.data.attributes.get(BASIS)
        if not attr:
            attr=obj.data.attributes.new(BASIS,'FLOAT_VECTOR','POINT')
            for datum,co in zip(attr.data,coordinates):datum.vector=co
        distances=[(a-b).length for a,b in zip(coordinates,planned)]
        for vertex,co in zip(obj.data.vertices,planned):vertex.co=co
        obj.data.update()
        obj['body_realism_version']=VERSION
        report['objects'].append({'name':obj.name,'vertices':len(planned),
            'changed_vertices':sum(d>1e-7 for d in distances),'max_displacement_m':max(distances)})
    return report

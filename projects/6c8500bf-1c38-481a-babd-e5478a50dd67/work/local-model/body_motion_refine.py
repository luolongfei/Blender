"""Conservative original-body linear-skinning refinement.

No scene reset, topology/UV/material edits, new bones, animation or face changes.
The same shoulder field is used by the overlapping blouse and sleeve roots.
Inner and outer flexion surfaces use different weight-transition widths.
"""
import math
import bpy
VERSION='body-motion-02'


def _smooth(a,b,value):
    t=max(0.,min(1.,(value-a)/(b-a)))
    return t*t*(3.-2.*t)


def _mix(a,b,t):
    return {a:1-t,b:t}


def _torso(z):
    if z<1.075:return _mix('hips','spine',_smooth(.985,1.07,z))
    if z<1.18:return _mix('spine','chest',_smooth(1.07,1.18,z))
    return _mix('chest','upperChest',_smooth(1.18,1.245,z))


def _shoulder(co,side):
    x,y,z=co;ax=abs(x)
    # One spatial field across both pieces keeps their overlapping armhole
    # together. The underside stays attached to the chest; the lateral sleeve
    # becomes fully arm-driven before the upper-arm shaft.
    height=max(_smooth(1.155,1.215,z),_smooth(.180,.215,ax))
    arm=_smooth(.120,.205,ax)*height
    clavicle=.38*(1-arm)*_smooth(.060,.125,ax)*_smooth(1.175,1.255,z)
    result={n:w*(1-arm-clavicle) for n,w in _torso(z).items()}
    result[side+'Shoulder']=clavicle
    result[side+'UpperArm']=arm
    return result


def _sleeve(co,side):
    x,y,z=co;ax=abs(x)
    # During elbow flexion toward the character front, the positive-Y side is
    # the convex surface. A longer transition keeps the full sleeve round.
    half=.040+.025*_smooth(-.025,.025,y)
    lower=_smooth(.385-half,.385+half,ax)
    shoulder=_shoulder(co,side)
    return {**{n:w*(1-lower) for n,w in shoulder.items()},
            side+'LowerArm':lower}


def _bodice(co,side):
    return _shoulder(co,side)


def _leg(co,side):
    x,y,z=co
    # Negative Y is the anterior, convex knee surface. The posterior bend
    # stays compact, while the kneecap turns across a broader vertical area.
    half=.038+.025*(1-_smooth(-.020,.020,y))
    upper=_smooth(.460-half,.460+half,z)
    return {side+'LowerLeg':1-upper,side+'UpperLeg':upper}


def _normalized(weights):
    items=sorted(((n,max(0.,w)) for n,w in weights.items() if w>1e-8),
                 key=lambda item:item[1],reverse=True)[:4]
    total=sum(w for _,w in items)
    if total<=0:raise ValueError('Empty body weight assignment')
    return {name:weight/total for name,weight in items}


def _protected(obj):
    return (obj.type!='MESH' or bool(obj.data.shape_keys)
            or bool(obj.get('facial_role'))
            or obj.get('part') in {'head','face','hair','hand','finger','nail','neck'}
            or obj.get('deform_role') in {'skirt','waistband','shoe','sock','cuff'})


def apply(collection,rig):
    if len(rig.data.bones)!=108:
        raise ValueError('Body pass expects the existing 108-bone original rig')
    targets=[]
    for obj in collection.all_objects:
        if _protected(obj):continue
        if obj.name=='Blouse_Tucked_Bodice':field=_bodice
        elif obj.get('deform_role')=='sleeve':field=_sleeve
        elif obj.get('part')=='leg':field=_leg
        else:continue
        targets.append((obj,field))
    # Validate all planned results before any mesh receives changed weights.
    plans=[]
    for obj,field in targets:
        planned=[]
        for vertex in obj.data.vertices:
            side='left' if (obj.get('side')=='L' or
                (not obj.get('side') and vertex.co.x>=0)) else 'right'
            weights=_normalized(field(vertex.co,side))
            if any(name not in rig.data.bones for name in weights):
                raise ValueError(obj.name+': invalid bone')
            planned.append(weights)
        plans.append((obj,planned))
    report={'version':VERSION,'objects':[],'preserved':[
        '108-bone hierarchy and spring/collider settings','all facial shape keys',
        'hand/finger/nail skinning','neck, skirt, shoe, sock and cuff weights',
        'all mesh coordinates, topology, UVs and materials'],
        'runtime_skinning':'Designed and visually compared under linear blend skinning'}
    for obj,planned in plans:
        oldgroups={g.index:g.name for g in obj.vertex_groups}
        previous=[{oldgroups[g.group]:g.weight for g in v.groups} for v in obj.data.vertices]
        changed=0;max_delta=0.
        for old,new in zip(previous,planned):
            difference=sum(abs(old.get(n,0)-new.get(n,0)) for n in set(old)|set(new))
            changed+=difference>1e-5;max_delta=max(max_delta,difference)
        obj.vertex_groups.clear()
        names=sorted({n for weights in planned for n in weights})
        groups={n:obj.vertex_groups.new(name=n) for n in names}
        for index,weights in enumerate(planned):
            for name,weight in weights.items():groups[name].add([index],weight,'REPLACE')
        obj['body_motion_refine_version']=VERSION
        report['objects'].append({'name':obj.name,'vertices':len(planned),
            'changed_vertices':changed,'max_weight_L1_delta':max_delta,
            'maximum_influences':max(map(len,planned))})
    report['changed_vertices']=sum(row['changed_vertices'] for row in report['objects'])
    return report

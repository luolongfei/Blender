"""Give the original blouse rear neckline room around the retained neck.

The skin/cloth intersection is at the inside rear collar seam, not new torso
geometry. Move only the bodice's upper posterior band, at most 2 mm, to meet
the existing Peter Pan collar inner edge. Anchor the lower neck skin gradually
to upperChest so ordinary head motion does not sweep the base through clothing.
The embedded upper neck and its existing Head blend remain unchanged.
"""
import bpy
import json

VERSION='neckline-clearance-02'
NAME='Blouse_Tucked_Bodice'

def smooth(a,b,value):
    t=max(0.,min(1.,(value-a)/(b-a)))
    return t*t*(3-2*t)

def apply(collection=None,rig=None):
    obj=bpy.data.objects[NAME]
    neck=bpy.data.objects['Neck']
    rig=rig or bpy.data.objects['Character_Rig']
    if obj.get('neckline_clearance_version')==VERSION:
        return json.loads(obj['neckline_clearance_report'])
    previous_version=obj.get('neckline_clearance_version')
    if previous_version not in (None,'neckline-clearance-01'):
        raise ValueError('A different neckline correction is already applied')
    if any(name not in rig.data.bones for name in ('upperChest','neck','head')):
        raise ValueError('Existing neck humanoid bones are required')
    source=obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
    moves=[]
    for vertex in source:
        x,y,z=vertex.co
        move=.002*smooth(1.258,1.276,z)*smooth(.015,.033,y)
        move*=1-smooth(.035,.065,abs(x))
        moves.append(move)
    if not any(d>0 for d in moves):
        raise ValueError('Cannot identify original rear blouse neckline')
    if previous_version=='neckline-clearance-01':
        pass  # Geometry already has the exact same bounded 2 mm correction.
    elif obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            for point,move in zip(key.data,moves):
                point.co.y+=move
        for vertex,point in zip(obj.data.vertices,obj.data.shape_keys.key_blocks[0].data):
            vertex.co=point.co
    else:
        for vertex,move in zip(obj.data.vertices,moves):
            vertex.co.y+=move
    obj.data.update()
    lower_vertices=[v for v in neck.data.vertices if v.co.z<1.320]
    deformation_groups=[g for g in neck.vertex_groups if g.name in rig.data.bones]
    groups={name:neck.vertex_groups.get(name) or neck.vertex_groups.new(name=name)
            for name in ('upperChest','neck')}
    old_weights={}
    max_difference=0.
    changed=0
    for vertex in lower_vertices:
        old={neck.vertex_groups[g.group].name:g.weight for g in vertex.groups
             if neck.vertex_groups[g.group].name in rig.data.bones}
        t=smooth(1.290,1.330,vertex.co.z)
        new={'upperChest':1-t,'neck':t}
        difference=max(abs(new.get(name,0)-old.get(name,0)) for name in set(old)|set(new))
        if difference<1e-8:continue
        old_weights[str(vertex.index)]=old
        max_difference=max(max_difference,difference);changed+=1
        for group in deformation_groups:group.remove([vertex.index])
        for name,weight in new.items():
            if weight>1e-8:groups[name].add([vertex.index],weight,'REPLACE')
    neck.data.update()
    neck['neckline_clearance_previous_lower_weights']=json.dumps(old_weights)
    neck['neckline_clearance_version']=VERSION
    report={
        'version':VERSION,
        'object':NAME,
        'moved_vertices':sum(d>1e-9 for d in moves),
        'max_posterior_displacement_m':max(moves),
        'region':'Rear upper bodice only: Z>1.258, Y>.015, |X|<.065',
        'purpose':'Resolve the rear collar seam crossing and delay lower-neck motion above the collar, while retaining the embedded upper-neck Head blend.',
        'neck_weights':{'changed_vertices':changed,
                        'maximum_weight_change':max_difference,
                        'field':'upperChest to neck smoothstep Z1.290..1.330, applied only below Z1.320; existing upper weights unchanged',
                        'max_influences':2},
        'preserved':['Neck geometry and all other anatomy','collar flaps and stitching',
                     'topology and all mesh details','all UV/material assignments',
                     'all clothing weights and upper-neck weights at Z>=1.320',
                     'rig rest/pose','all face expressions'],
    }
    obj['neckline_clearance_version']=VERSION
    obj['neckline_clearance_report']=json.dumps(report)
    return report

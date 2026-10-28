"""Rotate the integrated existing pinnae toward the sides at their roots.

Post-integration correction only. Retains continuous cartilage, ear UVs,
materials, Head weights, topology, and every non-ear object and rig property.
"""
import bpy,math,json
from mathutils import Vector

VERSION='ear-orientation-01'
DELTA_DEGREES=28.

def vertex_uv(obj):
    layer=obj.data.uv_layers.get('EarAnatomyUV')
    if layer is None:raise ValueError('Retained anatomical ear UV is required')
    uv=[None]*len(obj.data.vertices)
    for loop in obj.data.loops:
        uv[loop.vertex_index]=tuple(layer.data[loop.index].uv)
    if any(value is None for value in uv):raise ValueError('Ear UV has unused vertices')
    return uv

def apply(collection=None,rig=None):
    ears=[bpy.data.objects['Ear '+side] for side in ('L','R')]
    if all(o.get('ear_orientation_version')==VERSION for o in ears):
        return json.loads(ears[0]['ear_orientation_report'])
    if any(o.get('ear_orientation_version') for o in ears):
        raise ValueError('Partial or different ear orientation patch exists')
    plans=[]
    angle=math.radians(DELTA_DEGREES);c,s=math.cos(angle),math.sin(angle)
    for obj,sign in zip(ears,(1,-1)):
        if obj.get('ear_cartilage_version')!='ear-cartilage-04':
            raise ValueError('Requires the retained continuous cartilage ear')
        if obj.data.shape_keys:raise ValueError('Do not rotate expressive ears')
        uv=vertex_uv(obj)
        root_ids=[i for i,p in enumerate(uv)
                  if (p[0]-.5)*.035<-.012 and abs((p[1]-.5)*.067)<.011]
        if len(root_ids)<20:raise ValueError('Cannot locate embedded medial ear root')
        root=sum((obj.data.vertices[i].co for i in root_ids),Vector())/len(root_ids)
        output=[]
        for vertex in obj.data.vertices:
            co=vertex.co;dx=sign*(co.x-root.x);dy=co.y-root.y
            output.append(Vector((root.x+sign*(c*dx-s*dy),
                                  root.y+s*dx+c*dy,co.z)))
        plans.append((obj,root,output))
    report={'version':VERSION,'rotation_about_vertical_degrees':DELTA_DEGREES,
            'method':'Mirrored rigid rotation about each retained medial attachment; no scaling or cartilage reconstruction.',
            'objects':[],
            'preserved':['all non-ear geometry','rig/bones and physical settings',
                         'ear topology/UV/materials/Head weights','all expressions']}
    for obj,root,output in plans:
        old=[v.co.copy() for v in obj.data.vertices]
        before=max(abs(p.x) for p in old);after=max(abs(p.x) for p in output)
        for vertex,co in zip(obj.data.vertices,output):vertex.co=co
        obj.data.update()
        report['objects'].append({
            'name':obj.name,'root_anchor_m':list(root),
            'maximum_displacement_m':max((a-b).length for a,b in zip(old,output)),
            'maximum_lateral_extent_before_m':before,
            'maximum_lateral_extent_after_m':after,
            'lateral_extent_reduction_m':before-after})
    encoded=json.dumps(report)
    for obj in ears:
        obj['ear_orientation_version']=VERSION
        obj['ear_orientation_report']=encoded
    return report

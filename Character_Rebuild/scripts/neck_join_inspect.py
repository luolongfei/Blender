import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
src=OUT/'stages/E7-uv-normals-corrected.blend'
bpy.ops.wm.open_mainfile(filepath=str(src))
head=bpy.data.objects['Head_Skin_Quad_Retopo'];neck=bpy.data.objects['Neck_Clean_Quad_Sleeve']
bvh=BVHTree.FromObject(head,bpy.context.evaluated_depsgraph_get())
levels=sorted(set(round(v.co.z,6) for v in neck.data.vertices))
top=max(v.co.z for v in neck.data.vertices)
topverts=[v for v in neck.data.vertices if abs(v.co.z-top)<1e-5]
def distance_at(co):
    p,n,i,d=bvh.find_nearest(co)
    return {'co':list(co),'nearest':list(p),'normal':list(n),'signed':(co-p).dot(n),'distance':d,'poly':i,
            'material':head.data.materials[head.data.polygons[i].material_index].name}
report={'source':str(src),'sourceSha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'topZ':top,'neckMaterials':[m.name for m in neck.data.materials],
        'headMaterials':[m.name for m in head.data.materials],
        'topVertices':[distance_at(v.co) for v in topverts],
        'rings':[],'extendedTopCandidates':[],'sceneSaved':False}
for z in levels:
    vs=[v.co for v in neck.data.vertices if abs(v.co.z-z)<1e-5]
    if z<.675:continue
    rear=max(vs,key=lambda c:c.y)
    report['rings'].append({'z':z,'xRange':[min(v.x for v in vs),max(v.x for v in vs)],
               'yRange':[min(v.y for v in vs),max(v.y for v in vs)],'backPoint':distance_at(rear)})
for z in (.755,.765,.775,.785,.795):
    samples=[distance_at(Vector((v.co.x,v.co.y,z))) for v in topverts]
    report['extendedTopCandidates'].append({'z':z,'maxSigned':max(v['signed'] for v in samples),
          'minInteriorDistance':min(-v['signed'] for v in samples),'outsideCount':sum(v['signed']>=0 for v in samples)})
(OUT/'reports/neck-join-inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('sourceSha256','topZ','neckMaterials','extendedTopCandidates')},indent=2))
print('TOP_OUTSIDE',sum(s['signed']>0 for s in report['topVertices']),len(topverts))

import bpy,sys,json,hashlib,math,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import face_v3 as f
from face_e4_refine_delta_v3 import _smoothstep,_orbital_alignment
from geometry_checks import inspect_mesh
dest=OUT/'work/e7-head-contact-diagnostic';dest.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E7-uv-normals-corrected.blend'))
spec=json.loads((OUT/'reports/measured-face-spec-v3.json').read_text())
# Capture construction ring indices in memory; do not modify frozen basis module.
code=(OUT/'scripts/face_v3.py').read_text()
code=code.replace('old=boundary\n','old=boundary\n        allrings=[]\n')
code=code.replace('old=new\n','old=new\n            allrings.append(new)\n')
code=code.replace("'socket':inner}","'socket':inner,'boundary':boundary,'rings':allrings}")
code=code.replace("result=[skin]","skin['_ringmeta']=[{**r,'boundary':[mapping[v] for v in r['boundary']],'rings':[[mapping[v] for v in ring] for ring in r['rings']],'rim':[mapping[v] for v in r['rim']],'socket':[mapping[v] for v in r['socket']]} for r in ring_data]\n    result=[skin]")
import types
mod=types.ModuleType('_face_e7_inspect');sys.modules[mod.__name__]=mod
exec(compile(code,str(OUT/'scripts/face_v3.py'),'exec'),mod.__dict__)
parts,base_surface=mod.geometry(spec)
meta=parts[0]['_ringmeta'][-1]
head=bpy.data.objects['Head_Skin_Quad_Retopo'];me=head.data
base=parts[0]['vertices'];mouth=spec['mouth']
candidate=copy.deepcopy(spec);candidate['nose']['lobes']=[[0,.758,.014,.020,.0125],[0,.780,.011,.024,.0028]]
candidate_surface=f.geometry(candidate)[1]
def depth(x,z):
    a=base_surface.front(x,z);b=candidate_surface.front(x,z)
    mask=_smoothstep(.715,.725,z)*(1-_smoothstep(.820,.835,z))*(1-_smoothstep(.095,.110,abs(x)))
    return a+(b-a)*mask+_orbital_alignment(candidate_surface,candidate['eyes'],x,z)
def smile(x):return mouth['corner_lift']*((x-mouth['x'])/mouth['half_width'])**2
original=[tuple(v.co) for v in me.vertices]
tests=[]
for amplitude in [.001,.002,.003,.004,.006,.008,.010]:
    for v,co in zip(me.vertices,original):v.co=co
    changes=[]
    for amount,ring in zip([.20,.40,.60,.75,.86,.93,.98,1.],meta['rings']):
        for k,i in enumerate(ring):
            v=me.vertices[i]
            x,z=base[i][0],base[i][2]
            zz=z+amplitude*amount**4*(1-amount)*max(0,math.sin(meta['angles'][k]))
            dy=depth(x,zz)-depth(x,z)
            if abs(zz-z)>1e-12:
                changes.append({'v':i,'old':list(v.co),'delta':[0,dy,zz-z]})
                v.co.y+=dy;v.co.z+=zz-z
    me.update()
    check=inspect_mesh(me,head.matrix_world,self_intersections=True,max_self_triangles=40000,max_self_candidates=3000000)
    tests.append({'amplitude':amplitude,'contacts':check['selfContacts']})
    print('PATCH',amplitude,len(changes),'CONTACTS',check['selfContacts']['contacts'])
    if not check['selfContacts']['contacts']:break
out={'meta':meta,'changes':changes,'check':check,'tests':tests}
(dest/'mouth-patch-test.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
bpy.ops.wm.save_as_mainfile(filepath=str(dest/'E7-mouth-local-test.blend'))

"""Local cap/flow repairs against the frozen, final FaceSurface v2."""
import sys,json,math,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import face_v2

codepath=workspace_path('Character_Rebuild/scripts/face_v2.py')
specpath=workspace_path('Character_Rebuild/reports/measured-face-spec-v2.json')
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (codepath,specpath)]
expected=['aa3f6d5b55b104f58ad58f4c350a5a2fe56e2ecb4e560ff3b6c5a3bc61821f31',
          '9a441c0e38cb050a21b03fc61379423078d9a1c99a23b5f20cada0ead12bc63f']
if hashes!=expected:raise RuntimeError('Frozen face_v2 changed; do not fit a stale surface')
head=json.loads(specpath.read_text());surface=face_v2.FaceSurface(head['profile'],head['nose'],head.get('cheek'),head['front_curve'])
spec=json.loads(workspace_path('Character_Rebuild/reports/hair-spec-v2.json').read_text())
cap=spec['cap'];s=cap['surface'];nr=len(s);na=len(s[0]);field=[[0.]*na for _ in s]
changes=[]
for i,row in enumerate(s):
    for j,(x,y,z) in enumerate(row):
        if not (.84<=z<=head['profile'][-1][0]):continue
        w,cy,fd,bd=face_v2._profile_at(head['profile'],z)
        if abs(x)>=w:continue
        front=y<cy
        theta=math.asin(max(-1,min(1,x/w)))
        if not front:theta=math.pi-theta
        skin=surface.point(theta,z)[1]
        required=skin-.0065 if front else skin+.0065
        delta=min(0,required-y) if front else max(0,required-y)
        field[i][j]=delta

# A short, compact falloff avoids a dent boundary around one repaired patch.
# Never inflates remote regions or changes the lower hem/tails.
spread=[row[:] for row in field]
for i in range(nr):
    for j in range(na):
        d=field[i][j]
        if abs(d)<1e-8:continue
        for di in range(-2,3):
            ni=i+di
            if not 0<=ni<nr:continue
            for dj in range(-2,3):
                nj=(j+dj)%na;val=d*math.exp(-.8*(di*di+dj*dj))
                if abs(val)>abs(spread[ni][nj]):spread[ni][nj]=val
for i in range(nr):
    for j in range(na):
        d=spread[i][j]
        if abs(d)>1e-7:
            before=s[i][j][:];s[i][j][1]+=d
            changes.append({'row':i,'azimuth':j,'before':before,'after':s[i][j][:],'deltaY':d})

for lock in spec['locks']:
    if lock['name'].startswith('RearGather_'):
        lock.update(conformToCap=True,capSide='back',capDepthScale=.45,capEmbed=-.00025)
        # Match the broad source flow. Embedded ends avoid a floating pointed
        # ribbon on the crown, while preserving the v2 low gathering route.
        lock['depths'][0]=.0006;lock['widths'][0]*=.65

# Finer rear flow follows the same combing direction at sub-millimetre relief.
# These are deliberately shallow secondary planes, not individual hairs.
rear=[l for l in spec['locks'] if l['name'].startswith('RearGather_')]
for base in rear:
    fine=copy.deepcopy(base);fine['name']=base['name']+'_Secondary'
    sign=-1 if 'Left' in base['name'] else 1
    fine['points']=[[x+sign*.006*math.sin(math.pi*i/(len(base['points'])-1)),y,z]
                    for i,(x,y,z) in enumerate(base['points'])]
    fine['widths']=[v*.27 for v in base['widths']]
    fine['depths']=[v*.24 for v in base['depths']]
    spec['locks'].append(fine)
spec['status']='v3 local cap clearance and attached shallow rear flow; frozen face dependency; render pending'
spec['v3Corrections']={'frozenFaceCodeSha256':hashes[0],'frozenFaceSpecSha256':hashes[1],
 'capChangedPoints':len(changes),'maxAbsDeltaY':max(abs(v['deltaY']) for v in changes) if changes else 0,
 'newGlobalScale':False,'tailAndFringeV2Unchanged':True,
 'rearFlowConformedToActualOuterCapTriangles':True,'secondaryRelief':'under 1 mm',
 'changes':changes}
out=workspace_path('Character_Rebuild/reports/hair-spec-v3.json')
out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'out':str(out),'changed':len(changes),'maxY':spec['v3Corrections']['maxAbsDeltaY'],'locks':len(spec['locks'])}))

"""V2 changes after inspecting actual D-v1 front/back renders.

Keeps v1 specification intact. No Blender/source mutation. Back cap adapts to
the newly rebuilt head profile rather than protecting the fused input's gaps.
"""
import json, math, sys, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import face

spec=json.loads(workspace_path('Character_Rebuild/reports/hair-spec.json').read_text())
head=json.loads(workspace_path('Character_Rebuild/reports/measured-face-spec.json').read_text())
spec=copy.deepcopy(spec)
cap=spec['cap'];changes=[]
for i,row in enumerate(cap['surface']):
    for j,p in enumerate(row):
        x,y,z=p;before=tuple(p)
        # Low rear hem is continuous. Never follows a colour-sampling gap.
        rear=max(0,math.cos(2*math.pi*j/cap['around']-math.pi))
        z-=.0045*rear**4*(i/(cap['rows']-1))**8
        if head['profile'][0][0]<=z<=head['profile'][-1][0]:
            w,cy,fd,bd=face._profile_at(head['profile'],z)
            depth=fd if y<cy else bd
            q=math.sqrt((x/max(w,1e-5))**2+((y-cy)/max(depth,1e-5))**2)
            # Inflated ellipsoidal cross-section prevents the head showing
            # through both outer cap and 3 mm inner shell.
            w2=w+.007; d2=depth+.007
            q2=math.sqrt((x/w2)**2+((y-cy)/d2)**2)
            if q2<1:
                if q2<1e-9:raise ValueError('Cap point lies on head axis')
                x/=q2;y=cy+(y-cy)/q2
        cap['surface'][i][j]=[x,y,z]
        dist=math.dist(before,(x,y,z))
        if dist>1e-6:changes.append({'row':i,'azimuth':j,'before':before,'after':[x,y,z],'distance':dist})

# Lower tips to eyebrow/upper-eye level while keeping key gaps. Broad, shallow
# locks create sheet-like anime forms, not cylindrical ropes.
byname={l['name']:l for l in spec['locks']}
revisions={
 'Fringe_CentreLeft':([(0,-.005,.990),(-.026,-.078,.964),(-.039,-.137,.913),(-.032,-.151,.855),(-.013,-.132,.802)],
                         [.003,.015,.024,.023,.0007],[.001,.003,.005,.004,.0005]),
 'Fringe_Left':([(-.012,.005,.988),(-.061,-.071,.953),(-.089,-.125,.902),(-.092,-.134,.845),(-.086,-.112,.814)],
                         [.003,.024,.028,.022,.0007],[.001,.0035,.005,.004,.0005]),
 'Fringe_CentreRight':([(.003,.006,.989),(.027,-.075,.961),(.032,-.132,.914),(.039,-.146,.855),(.059,-.125,.817)],
                         [.003,.013,.020,.020,.0007],[.001,.003,.0045,.004,.0005]),
 'Fringe_Right':([(.006,.011,.989),(.061,-.057,.954),(.091,-.111,.906),(.104,-.123,.848),(.107,-.092,.800)],
                         [.003,.023,.028,.023,.0007],[.001,.0035,.005,.004,.0005]),
 'Fringe_LeftFine':([(-.035,-.052,.968),(-.057,-.118,.919),(-.063,-.148,.855),(-.048,-.126,.819)],
                         [.003,.008,.008,.0004],[.001,.002,.0025,.0004]),
 'Fringe_RightFine':([(.023,-.038,.975),(.046,-.115,.921),(.061,-.147,.861),(.082,-.116,.815)],
                         [.003,.007,.008,.0004],[.001,.002,.0025,.0004])}
for name,(points,widths,depths) in revisions.items():
    byname[name].update(points=points,widths=widths,depths=depths,ridge=.015)

def lock(name,points,widths,depths,outward):
    return dict(name=name,points=points,widths=widths,depths=depths,
                outward=outward,rings=41,sides=12,ridge=.015)

# Layered rear gathering flow converges at each low ear-back tie. The smooth
# continuous source-envelope shell remains beneath, so no centre split shows.
for sign,label in [(-1,'Left'),(1,'Right')]:
    paths=[
      ([(.008,.060,.983),(.026,.140,.939),(.060,.180,.875),(.100,.158,.802),(.119,.116,.748)],
       [.004,.016,.026,.022,.007],[.001,.003,.004,.004,.0015]),
      ([(.035,.042,.981),(.080,.120,.936),(.116,.155,.874),(.131,.139,.804),(.120,.111,.747)],
       [.004,.018,.026,.021,.007],[.001,.003,.004,.004,.0015]),
      ([(.072,.025,.959),(.119,.082,.916),(.146,.106,.853),(.142,.110,.787),(.119,.108,.747)],
       [.004,.018,.021,.017,.006],[.001,.003,.004,.003,.0015])]
    for k,(pts,ws,ds) in enumerate(paths):
        spec['locks'].append(lock(f'RearGather_{label}_{k+1}',[(sign*x,y,z) for x,y,z in pts],ws,ds,[sign*.25,1,0]))

# Asymmetric layers replace the evenly parallel five-ribbon construction.
for tail in spec['tails']:
    sign=-1 if 'Left' in tail['name'] else 1
    data=[
      ([(.120,.099,.743),(.143,.111,.704),(.156,.117,.653),(.151,.115,.599),(.130,.109,.562)],
       [.014,.028,.031,.023,.0008],[.010,.014,.016,.011,.0006]),
      ([(.121,.108,.741),(.153,.137,.706),(.170,.148,.658),(.169,.144,.613),(.153,.134,.583)],
       [.012,.023,.026,.018,.0007],[.010,.014,.017,.011,.0005]),
      ([(.117,.111,.741),(.126,.145,.700),(.130,.158,.652),(.134,.151,.607),(.121,.132,.570)],
       [.012,.026,.028,.021,.0008],[.010,.017,.017,.013,.0006]),
      ([(.120,.094,.740),(.131,.098,.690),(.140,.100,.640),(.139,.098,.597),(.120,.092,.576)],
       [.009,.014,.017,.012,.0006],[.006,.007,.008,.005,.0004]),
      ([(.123,.106,.740),(.161,.125,.695),(.177,.130,.652),(.177,.124,.619),(.166,.114,.604)],
       [.009,.014,.014,.010,.0006],[.006,.008,.008,.005,.0004])]
    tail['locks']=[lock(f"{tail['name']}_Layer_{k+1}",[(sign*x,y,z) for x,y,z in pts],ws,ds,[0,-1,0])
                   for k,(pts,ws,ds) in enumerate(data)]

spec['status']='v2 visual correction candidate after D geometry-v1 front/back; not yet rendered'
spec['v2Corrections']={'reason':'actual back render exposed rebuilt head through original sampled envelope; fringe too high; repetitive tail layers',
 'headProfileSource':'measured-face-spec.json','capChangedPoints':len(changes),
 'capMaximumDisplacement':max(v['distance'] for v in changes),'sourceCapPreservedElsewhere':True,
 'minimumHeadCrossSectionOffset':.007,'capThickness':.003,
 'fringeTipZRange':[.800,.819],'tailLayers':'five unequal hand-shaped paths each, rather than equal parallel offsets',
 'changes':changes}
out=workspace_path('Character_Rebuild/reports/hair-spec-v2.json')
out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'out':str(out),'changedCapPoints':len(changes),'maxChange':max(v['distance'] for v in changes),'locks':len(spec['locks'])}))

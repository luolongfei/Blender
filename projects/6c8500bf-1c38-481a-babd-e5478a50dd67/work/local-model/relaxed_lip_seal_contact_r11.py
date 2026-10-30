"""Current actual lip-seal contact checks, separate from field construction."""
from pathlib import Path
import runpy
import numpy as np

def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def contact(before,after,faces,groups,triangles,active,cage,before_triangles=None):
 fn=api('adult_blink_midpath_contact_r11')['changed_intersections'];old=fn(before,triangles,active);new=fn(after,triangles,active)
 halo=np.zeros(len(before),bool)
 if old:halo[np.unique(triangles[np.asarray(list(old),np.int32).ravel()])]=True
 for _ in range(2):halo[np.unique(triangles[halo[triangles].any(1)])]=True
 actual_before=old if before_triangles is None or np.array_equal(before_triangles,triangles) else fn(before,before_triangles,active)
 diagonal_rows=None if before_triangles is None or np.asarray(before_triangles).shape!=np.asarray(triangles).shape else int(np.any(np.asarray(before_triangles)!=np.asarray(triangles),axis=1).sum())
 added=new-old;outside=[p for p in added if not halo[triangles[list(p)]].all()]
 oral=api('adult_oral_observe_r11');sections=[];bad=[]
 central=[float(after[triangles[list(pair)]].mean((0,1))[0]) for pair in added if abs(float(after[triangles[list(pair)]].mean((0,1))[0]))<.014]
 sample_x=np.unique(np.r_[np.linspace(-.020,.020,41),central])
 for x in sample_x:
  rr=[]
  for w in (before,after):
   r=oral['section_lenses'](oral['section_segments'](w,faces,groups,cage,float(x)),1.)
   valid=[];unresolved=[]
   for lens in r['localLenses']:
    known=lens.get('maximumLowerAboveUpperMeters') is not None and lens.get('positiveOverlapAreaSquareMeters') is not None
    known=known and lens.get('finiteSamples',0)==lens.get('samples',-1) and lens.get('ambiguousNearBranches',0)==0 and 'status' not in lens
    (valid if known else unresolved).append(lens)
   rr.append({'crossings':r['crossings'],'localLenses':r['localLenses'],'unresolvedLenses':unresolved,
    'maximumResolvedDepthMeters':max([a['maximumLowerAboveUpperMeters'] for a in valid]+[0.]) if valid or not unresolved else None,
    'resolvedOverlapAreaSquareMeters':sum(a['positiveOverlapAreaSquareMeters'] for a in valid) if valid or not unresolved else None})
  row={'worldX':float(x),'before':rr[0],'after':rr[1]};sections.append(row)
  newer_unresolved=len(rr[1]['unresolvedLenses'])>len(rr[0]['unresolvedLenses'])
  bad_finite=sum(max(0,a.get('samples',0)-a.get('finiteSamples',0)) for a in rr[1]['unresolvedLenses'])>sum(max(0,a.get('samples',0)-a.get('finiteSamples',0)) for a in rr[0]['unresolvedLenses'])
  bad_ambiguous=sum(a.get('ambiguousNearBranches',0) for a in rr[1]['unresolvedLenses'])>sum(a.get('ambiguousNearBranches',0) for a in rr[0]['unresolvedLenses'])
  d0=rr[0]['maximumResolvedDepthMeters'];d1=rr[1]['maximumResolvedDepthMeters'];a0=rr[0]['resolvedOverlapAreaSquareMeters'];a1=rr[1]['resolvedOverlapAreaSquareMeters']
  deeper=d1 is not None and d0 is not None and d1>d0+2e-6
  larger=a1 is not None and a0 is not None and a1>a0+1e-8
  if newer_unresolved or bad_finite or bad_ambiguous or deeper or larger:bad.append(row)
 return {'comparisonTriangulation':'Same supplied after triangulation used on before and after positions; comparison pairs are not relabeled as actual-before topology.', 'beforeActualTriangulationProvided':before_triangles is not None,'changedTriangleRows':diagonal_rows,'actualBeforePairs':sorted(actual_before) if before_triangles is not None else None,'actualBeforePairCount':len(actual_before) if before_triangles is not None else None,'beforePairCount':len(old),'afterPairCount':len(new),'newPairCount':len(added),'removedPairCount':len(old-new),
  'baselinePairs':sorted(old),'newPairs':sorted(added),'removedPairs':sorted(old-new),'newPairsOutsideBaselineContactTwoRing':outside,
  'centralAddedPairSampleX':central,'sections':sections,'worsenedSections':bad,'guardPassed':not outside and not bad,
  'limits':'Existing nonadjacent contact intersections remain. New pairs are localized against the two-ring neighborhood of baseline actual crossing faces and section depths/areas are compared individually. Pair totals never offset an added contact defect. Strict triangle tests exclude shared vertices/coplanarity; not full global certification.'}

"""Classify lip-corner crossings from the saved native seal evaluation only."""
import argparse
from pathlib import Path
import json
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def run(report_relative,output_relative):
    project=project_path();folder=(project/'work/reconstruction-r02').resolve()
    reportpath=(project/report_relative).resolve();output=(project/output_relative).resolve()
    for path in (reportpath,output):path.relative_to(folder)
    if output.exists():raise RuntimeError('Preserve earlier actual studies')
    helper=runpy.run_path(str(Path(__file__).with_name('lip_corner_residual_study_r11.py')))
    source=json.loads(reportpath.read_text(encoding='utf8'))
    if not source['candidateSaved']:raise RuntimeError('Completed actual candidate required')
    arrays=(project/source['arrays']).resolve()
    if helper['sha'](arrays)!=source['arraysSha256']:raise RuntimeError('Actual arrays changed')
    with np.load(arrays) as z:
        faces=z['fullFaces'];groups=np.zeros(len(faces),np.int32);groups[z['roiOriginalFaceIds']]=z['roiGroups'];cage=z['neutralCageWorld']
        states={warm:{'world':z['after_'+str(warm)+'_1.0World'],'triangles':z['after_'+str(warm)+'_1.0Triangles'],
            'beforeTriangles':z['beforeWarm'+str(warm)+'Triangles'],'trianglePolygons':z['beforeWarm'+str(warm)+'TrianglePolygonIds']} for warm in (0.,.5,1.)}
    camera_report=json.loads((project/'previews/reconstruction-r02/upper-lid-roll-quarter-r11/render-report.json').read_text())
    matrix=np.asarray(camera_report['camera']['matrixWorld']);quarter=np.column_stack((matrix[:3,0],-matrix[:3,2],matrix[:3,1]))
    views={'front':np.eye(3),'actual31dfQuarter':quarter,'oppositeQuarterRayOnly':np.array([[.8,.6,0.],[-.6,.8,0.],[0.,0.,1.]])}
    predicted=json.loads((folder/'lip-corner-residual-study-r11.json').read_text())
    def signature(vertex_ids):return tuple(sorted(tuple(sorted(row)) for row in vertex_ids))
    report={'actualSource':source['output'],'actualSourceSha256':source['outputSha256'],'actualInstallationReport':report_relative,
        'actualInstallationReportSha256':helper['sha'](reportpath),'actualArraysSha256':source['arraysSha256'],'helperSha256':helper['sha'](__file__),
        'surfaceStatus':'Actual evaluated native full-seal geometry captured by the successful installation; no new Blender execution.',
        'triangulation':'Actual evaluated loop triangles; original polygon IDs captured in the same state and verified unchanged after seal.',
        'scope':'Residual strict intersections in the native changed-face contact audit. Full head used for geometric visibility, no hair occlusion. This is not full oral acceptance.',
        'states':[],'sourceSaved':False,'geometryModified':False,'appearanceAccepted':False}
    for warm,data in states.items():
        world=data['world'];triangles=data['triangles'];polys=data['trianglePolygons']
        if not np.array_equal(triangles,data['beforeTriangles']):raise RuntimeError('Need actual after polygon mapping for changed triangulation')
        contact=next(r for r in source['actualChecks'] if r['warm']==warm and r['seal']==1.)['contact']
        pairs=(set(map(tuple,contact['baselinePairs']))-set(map(tuple,contact['removedPairs'])))|set(map(tuple,contact['newPairs']))
        records=[];unresolved=[]
        for pair in sorted(pairs):
            pair=np.asarray(pair,np.int32);ids=triangles[pair];tt=world[ids];ss=helper['segment'](*tt)
            if ss is None:unresolved.append(pair.tolist());continue
            middle=ss.mean(0);shared=np.intersect1d(*ids);ba=helper['barycentric'](middle,tt[0]);bb=helper['barycentric'](middle,tt[1])
            d0=helper['plane_distances'](tt[0],tt[1]);d1=helper['plane_distances'](tt[1],tt[0]);pp=polys[pair]
            records.append({'actualTrianglePair':pair.tolist(),'originalPolygonIds':pp.tolist(),'originalVertexIds':ids.tolist(),'faceSets':groups[pp].tolist(),
                'sharedVertexIds':shared.tolist(),'intersectionSegmentWorld':ss.tolist(),'intersectionSegmentLengthMeters':float(np.linalg.norm(ss[1]-ss[0])),
                'midpointWorld':middle.tolist(),'midpointBarycentricA':ba.tolist(),'midpointBarycentricB':bb.tolist(),
                'strictInteriorCrossing':bool(not len(shared) and min(ba.min(),bb.min())>1e-8),
                'aVertexSignedDistancesToBPlaneMeters':[float(d0.min()),float(d0.max())],
                'bVertexSignedDistancesToAPlaneMeters':[float(d1.min()),float(d1.max())],
                '_samples':ss[0]+(ss[1]-ss[0])*np.array([.2,.5,.8])[:,None]})
        for view,basis in views.items():
            for record,result in zip(records,helper['visibility'](world,triangles,polys,groups,records,basis)):
                record.setdefault('visibility',{})[view]=result
        expected=next(s for s in predicted['states'] if s['warmSmile']==warm)['pairs']
        old={signature(r['originalVertexIds']):r for r in expected};new={signature(r['originalVertexIds']):r for r in records}
        common=set(old)&set(new);difference=max(np.linalg.norm(np.asarray(old[k]['midpointWorld'])-new[k]['midpointWorld']) for k in common)
        combos={}
        for record in records:
            name='/'.join(map(str,sorted(record['faceSets'])));combos[name]=combos.get(name,0)+1;record.pop('_samples')
        row={'warmSmile':warm,'seal':1.,'pairs':records,'unresolvedSegmentPairs':unresolved,'faceSetPairCounts':combos,
            'strictInteriorPairCount':sum(r['strictInteriorCrossing'] for r in records),
            'predictedPairVertexSignaturesExact':set(old)==set(new),'maximumIntersectionMidpointPredictionDifferenceMeters':float(difference),
            'sampledVisibility':{},'actualSagittalSections':[]}
        for x in (-.020,-.018,-.016,.016,.018,.020):row['actualSagittalSections'].append(helper['exact_section'](world,faces,groups,cage,x))
        for view in views:
            gaps=[r['visibility'][view]['minimumFrontCoverageMeters'] for r in records]
            row['sampledVisibility'][view]={'crossingsWithAnyVisibleSample':sum(r['visibility'][view]['frontVisibleSamplesWithin2Micrometres']>0 for r in records),
                'minimumCoverageMeters':min(gaps),'medianCoverageMeters':float(np.median(gaps))}
        report['states'].append(row)
        print('R11_ACTUAL_CORNER_STUDY '+json.dumps({'warm':warm,'pairCount':len(records),'samePredictedPairs':row['predictedPairVertexSignaturesExact'],
            'maxPredictionDifferenceMeters':difference,'visibility':row['sampledVisibility']}),flush=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',default='work/reconstruction-r02/reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.json')
    p.add_argument('--output',default='work/reconstruction-r02/lip-corner-actual-study-r11.json')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);run(a.report,a.output)

"""Read-only classification of remaining relaxed-seal lip intersections.

Uses actual 31df positions plus the frozen full-seal field until a newly
evaluated source is available. It never changes geometry or asserts that a
hidden crossing is acceptable. All pair IDs retain explicit triangulation.
"""
import argparse
from pathlib import Path
import hashlib
import json
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment(triangle_a,triangle_b):
    """Inclusive edge/triangle hits; callers already have strict pair evidence."""
    hits=[]
    for a,b in ((triangle_a,triangle_b),(triangle_b,triangle_a)):
        e1=b[1]-b[0];e2=b[2]-b[0]
        for k in range(3):
            origin=a[k];direction=a[(k+1)%3]-origin;cross=np.cross(direction,e2);det=np.dot(e1,cross)
            if abs(det)<1e-18:continue
            d=origin-b[0];u=np.dot(d,cross)/det;q=np.cross(d,e1);v=np.dot(direction,q)/det;t=np.dot(e2,q)/det
            if min(u,v,t)>=-1e-8 and u+v<=1+1e-8 and t<=1+1e-8:
                point=origin+direction*t
                if not any(np.linalg.norm(point-old)<1e-10 for old in hits):hits.append(point)
    if len(hits)<2:return None
    points=np.asarray(hits);distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)
    aa,bb=np.unravel_index(distance.argmax(),distance.shape)
    return points[[aa,bb]]


def barycentric(point,triangle):
    a,b,c=triangle;u=b-a;v=c-a;d=point-a
    coefficients=np.linalg.lstsq(np.column_stack((u,v)),d,rcond=None)[0]
    return np.r_[1-coefficients.sum(),coefficients]


def plane_distances(points,triangle):
    n=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);n/=np.linalg.norm(n)
    return (points-triangle[0])@n


def visibility(world,triangles,triangle_polygons,groups,records,basis):
    coordinates=world@basis
    all_points=np.concatenate([row['_samples'] for row in records])@basis
    tri_points=coordinates[triangles];lo=tri_points[:,:,[0,2]].min(1);hi=tri_points[:,:,[0,2]].max(1)
    low=all_points[:,[0,2]].min(0);high=all_points[:,[0,2]].max(0)
    selected=np.flatnonzero(np.all(hi>=low-1e-9,1)&np.all(lo<=high+1e-9,1))
    index=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex'](coordinates,triangles[selected])
    ys,tri_rows,_=index.front(all_points);valid=tri_rows>=0
    first=np.full(len(all_points),-1,np.int32);first[valid]=selected[tri_rows[valid]]
    gap=all_points[:,1]-ys
    result=[]
    for i,row in enumerate(records):
        sl=slice(i*3,(i+1)*3);good=valid[sl];poly=np.full(3,-1,np.int32);poly[good]=triangle_polygons[first[sl][good]]
        gg=np.full(3,-1,np.int32);gg[good]=groups[poly[good]]
        result.append({'samplesWorld':row['_samples'].tolist(),'headHitCount':int(good.sum()),
            'frontCoverageMeters':gap[sl].tolist(),'minimumFrontCoverageMeters':float(np.nanmin(gap[sl])) if good.any() else None,
            'frontVisibleSamplesWithin2Micrometres':int(np.count_nonzero(good&(abs(gap[sl])<=2e-6))),
            'occludingOriginalPolygonIds':poly.tolist(),'occludingKnownFaceSets':gg.tolist()})
    return result


def exact_section(world,faces,groups,cage,x):
    oral=api('adult_oral_observe_r11');r=oral['section_lenses'](oral['section_segments'](world,faces,groups,cage,float(x)),1.)
    known=[v for v in r['localLenses'] if v.get('maximumLowerAboveUpperMeters') is not None]
    return {'worldX':float(x),'maximumResolvedSagittalOverlapMeters':max([q['maximumLowerAboveUpperMeters'] for q in known]+[0.]),
            'crossings':r['crossings'],'localLenses':r['localLenses'],
            'scope':'True upper/lower sagittal overlap at this intersection-segment X; not a minimum 3D separation or a global pass.'}


def run(output_relative):
    project=project_path();folder=project/'work/reconstruction-r02';output=(project/output_relative).resolve();output.relative_to(folder.resolve())
    if output.exists():raise RuntimeError('Preserve earlier residual studies')
    planpath=folder/'relaxed-lip-seal-plan-r11.json';plan=json.loads(planpath.read_text());arraypath=project/plan['arrays']
    if sha(arraypath)!=plan['arraysSha256']:raise RuntimeError('Frozen field changed')
    with np.load(arraypath) as z:data={k:z[k].copy() for k in z.files}
    refpath=project/plan['referenceActualArrays']
    if sha(refpath)!=plan['referenceActualArraysSha256']:raise RuntimeError('Actual reference array changed')
    with np.load(refpath) as z:states={warm:z['afterWarm'+str(warm)+'World'].copy() for warm in (0.,.5,1.)}
    preflightpath=folder/'relaxed-lip-seal-preflight-r11.json';preflight=json.loads(preflightpath.read_text())
    faces=data['fullFaces'];count=len(faces);triangles=np.concatenate((faces[:,[0,1,2]],faces[:,[0,2,3]]));tri_poly=np.tile(np.arange(count),2)
    groups=np.zeros(count,np.int32);groups[data['roiOriginalFaceIds']]=data['roiGroups']
    local_count=len(data['faces']);local_triangles=np.concatenate((data['faces'][:,[0,1,2]],data['faces'][:,[0,2,3]]))
    camera_report=json.loads((project/'previews/reconstruction-r02/upper-lid-roll-quarter-r11/render-report.json').read_text())
    matrix=np.asarray(camera_report['camera']['matrixWorld']);quarter_basis=np.column_stack((matrix[:3,0],-matrix[:3,2],matrix[:3,1]))
    views={'front':np.eye(3),'actual31dfQuarter':quarter_basis,
           'oppositeQuarterRayOnly':np.array([[.8,.6,0.],[-.6,.8,0.],[0.,0.,1.]])}
    report={'actualBaselineSourceSha256':plan['sourceSha256'],'fieldSha256':plan['arraysSha256'],
            'preflightSha256':sha(preflightpath),'helperSha256':sha(__file__),
            'surfaceStatus':'Derived full-seal geometry from actual 31df positions plus the frozen field; not the pending new Blender evaluation.',
            'triangulation':'Declared 0-1-2 / 0-2-3 polygon fans. Pair indices are remapped from the preserved ROI fan to original full-head polygon/vertex IDs.',
            'scope':'Remaining strict nonadjacent intersections in the prior changed-face contact audit; not an exhaustive complete-mouth collision search.',
            'states':[],'geometryModified':False,'sourceSaved':False,'appearanceAccepted':False}
    for warm,original in states.items():
        world=original.copy();world[data['originalVertexIds']]+=data['worldOffsets']
        row=next(r for r in preflight if r['warm']==warm and r['seal']==1.)['contact']
        pairs=(set(map(tuple,row['baselinePairs']))-set(map(tuple,row['removedPairs'])))|set(map(tuple,row['newPairs']))
        records=[];unresolved=[]
        for pair in sorted(pairs):
            local_rows=np.asarray(pair,np.int32);polys=data['roiOriginalFaceIds'][local_rows%local_count]
            full_rows=polys+(local_rows//local_count)*count
            ids=triangles[full_rows];tt=world[ids];ss=segment(tt[0],tt[1])
            if ss is None:unresolved.append(list(pair));continue
            middle=ss.mean(0);shared=np.intersect1d(ids[0],ids[1]);ba=barycentric(middle,tt[0]);bb=barycentric(middle,tt[1])
            da=plane_distances(tt[0],tt[1]);db=plane_distances(tt[1],tt[0])
            records.append({'roiFanTrianglePair':list(pair),'fullFanTrianglePair':full_rows.tolist(),
                'originalPolygonIds':polys.tolist(),'originalVertexIds':ids.tolist(),'faceSets':groups[polys].tolist(),
                'sharedVertexIds':shared.tolist(),'intersectionSegmentWorld':ss.tolist(),'intersectionSegmentLengthMeters':float(np.linalg.norm(ss[1]-ss[0])),
                'midpointWorld':middle.tolist(),'midpointBarycentricA':ba.tolist(),'midpointBarycentricB':bb.tolist(),
                'strictInteriorCrossing':bool(not len(shared) and min(ba.min(),bb.min())>1e-8),
                'aVertexSignedDistancesToBPlaneMeters':[float(da.min()),float(da.max())],
                'bVertexSignedDistancesToAPlaneMeters':[float(db.min()),float(db.max())],
                '_samples':ss[0]+(ss[1]-ss[0])*np.array([.2,.5,.8])[:,None]})
        for view,basis in views.items():
            for record,result in zip(records,visibility(world,triangles,tri_poly,groups,records,basis)):
                record.setdefault('visibility',{})[view]=result
        examples=[]
        for side in (-1,1):
            eligible=[r for r in records if r['midpointWorld'][0]*side>0]
            # Pick direct front visibility, greatest plane straddle and longest
            # true crossing segment independently, retaining unique examples.
            choices=[]
            for getter in (lambda r:-r['visibility']['front']['minimumFrontCoverageMeters'],
                           lambda r:max(abs(x) for x in r['aVertexSignedDistancesToBPlaneMeters']),
                           lambda r:r['intersectionSegmentLengthMeters']):
                if eligible:choices.append(max(eligible,key=getter))
            for record in choices:
                if record['roiFanTrianglePair'] in [r['pair'] for r in examples]:continue
                examples.append({'pair':record['roiFanTrianglePair'],'originalPolygonIds':record['originalPolygonIds'],
                    'sideByWorldX':side,'exactSagittalSection':exact_section(world,faces,groups,data['neutralCageWorld'],record['midpointWorld'][0])})
        combos={}
        for record in records:
            name='/'.join(map(str,sorted(record['faceSets'])));combos[name]=combos.get(name,0)+1;record.pop('_samples')
        state={'warmSmile':warm,'seal':1.,'pairs':records,'unresolvedSegmentPairs':unresolved,'faceSetPairCounts':combos,
               'strictInteriorPairCount':sum(r['strictInteriorCrossing'] for r in records),'sampledVisibility':{},'representativeSections':examples}
        for view in views:
            gaps=[r['visibility'][view]['minimumFrontCoverageMeters'] for r in records]
            state['sampledVisibility'][view]={'crossingsWithAnyVisibleSample':sum(r['visibility'][view]['frontVisibleSamplesWithin2Micrometres']>0 for r in records),
                'minimumCoverageMeters':min(gaps),'medianCoverageMeters':float(np.median(gaps)),
                'definition':'Visibility of three points on each true crossing segment against the full head surface, ignoring hair/clothing occlusion.'}
        report['states'].append(state)
        print('R11_LIP_CORNER_STUDY '+json.dumps({'warm':warm,'pairCount':len(records),'types':combos,'visibility':state['sampledVisibility']}),flush=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/reconstruction-r02/lip-corner-residual-study-r11.json')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);run(args.output)

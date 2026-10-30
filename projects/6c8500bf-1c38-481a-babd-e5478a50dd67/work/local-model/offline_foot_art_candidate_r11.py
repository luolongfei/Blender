"""Offline art-proportion evidence from SHA-bound actual foot captures.

Never imports bpy, launches Blender, or edits a production asset. Writes one
new JSON evidence report; all candidate lengths are artistic preview choices.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CAPTURE = 'work/reconstruction-r02/foot-proportion-capture-r11.json'
CAPTURE_SHA = 'c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f'
CAPTURE_ARRAYS_SHA = '6481734f699921c8f64872a46a5d38cdae3aa3fefb26812f498b84d0f8b11ee3'
DONOR = 'work/reconstruction-r02/foot-asset-observation-r11.json'
DONOR_SHA = '67d6afbc595cb757c533d66f53bd494e24b29f172356bda251fb1530b98b336c'
DONOR_ARRAYS_SHA = '202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564'
OUTPUT = 'work/reconstruction-r02/foot-proportion-art-candidate-r11.json'


def path(relative):
    p = (project_path()/relative).resolve()
    p.relative_to(project_path().resolve())
    return p


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_bound(relative, digest, arrays_digest):
    p = path(relative)
    if sha(p) != digest:
        raise RuntimeError('Observed JSON identity changed: '+relative)
    j = json.loads(p.read_text(encoding='utf-8'))
    if j['arraysSha256'] != arrays_digest or sha(path(j['arrays'])) != arrays_digest:
        raise RuntimeError('Observed arrays identity changed: '+relative)
    return j


def extent_record(points):
    q = np.asarray(points, np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise RuntimeError('Expected finite actual 3D coordinates')
    return {'minimumWorldMeters':q.min(0).tolist(), 'maximumWorldMeters':q.max(0).tolist(),
            'extentMeters':np.ptp(q, axis=0).tolist(), 'vertices':len(q)}


def side_of(name):
    sides = [s for s in ('L','R') if re.search('_'+s+r'(?:_|$)', name)]
    if len(sides) != 1:
        raise RuntimeError('Explicit side cannot be established: '+name)
    return sides[0]


def build(output_relative=OUTPUT):
    output = path(output_relative)
    if output.exists():
        raise FileExistsError('Preserve the existing independent art evidence report')
    current = read_bound(CAPTURE, CAPTURE_SHA, CAPTURE_ARRAYS_SHA)
    donor = read_bound(DONOR, DONOR_SHA, DONOR_ARRAYS_SHA)
    if current['units']['scaleLength'] != 1.0 or not all(current['restoration'].values()):
        raise RuntimeError('This calculation requires the actual metre-scale restored capture')
    if sha(path(current['source'])) != current['sourceSha256']:
        raise RuntimeError('Current captured source changed on disk')
    if len(current['objects']) != 82:
        raise RuntimeError('Reconcile the actual 82-object inventory before recalculating')
    asset = donor['asset'][0]
    if asset['name'] != 'Foot - Realistic' or asset['vertexGroups'] or asset['parent'] or asset['children']:
        raise RuntimeError('Unreviewed donor identity/attachment')
    if not np.array_equal(np.asarray(asset['matrixWorld'])[:3,:3], np.eye(3)):
        raise RuntimeError('Donor local distances require its observed identity rotation/scale')
    inventory = {'total':len(current['objects']), 'objects':[], 'sides':{}}
    side_records = {}
    with np.load(path(current['arrays']), allow_pickle=False) as arrays:
        for side in ('L','R'):
            rows = [o for o in current['objects'] if side_of(o['name']) == side]
            shoes = [o for o in rows if o['name'].startswith('Shoe_')]
            socks = [o for o in rows if o['name'].startswith('Sock_')]
            bodies = [o for o in rows if o['name'].startswith('Body_Complete_Shin_Foot_')]
            if (len(shoes),len(socks),len(bodies)) != (37,3,1):
                raise RuntimeError('Unexpected side composition')
            side_records[side] = {}
            for category, group in (('shoeAssembly',shoes),('sockAssembly',socks),('existingShinFoot',bodies)):
                q = np.concatenate([arrays[o['arrayPrefix']+'_restWorld'] for o in group])
                side_records[side][category] = extent_record(q)
            foot_name = ('left' if side == 'L' else 'right')+'Foot'
            rigid = []
            for o in shoes:
                prefix = o['arrayPrefix']; w = arrays[prefix+'_weightValue']
                vi = arrays[prefix+'_weightVertex']; gi = arrays[prefix+'_weightGroup']
                count = len(arrays[prefix+'_raw_positions'])
                ok = (o['groups'] == [[foot_name,0]] and len(w) == count and
                      np.array_equal(np.sort(vi),np.arange(count)) and
                      np.all(gi == 0) and np.all(w == 1.0))
                rigid.append(bool(ok))
            inventory['sides'][side] = {'shoeObjects':37,'sockObjects':3,'shinFootObjects':1,
                'shoeNames':[o['name'] for o in shoes], 'sockNames':[o['name'] for o in socks],
                'allShoeRawVerticesRigidlyWeightedToFoot':all(rigid)}
        for o in current['objects']:
            inventory['objects'].append({'name':o['name'],'type':o['type'],'side':side_of(o['name']),
                'arrayPrefix':o['arrayPrefix'],'groups':o['groups'],
                'baselineCompatibility':o['baselineCompatibility']})
    with np.load(path(donor['arrays']), allow_pickle=False) as arrays:
        q = arrays['asset_0_evaluated_positions'].astype(np.float64)
        donor_dimensions = np.ptp(q, axis=0)
        raw = arrays['asset_0_raw_positions'].astype(np.float64)
        starts = arrays['asset_0_raw_polygonStart']; sizes = arrays['asset_0_raw_polygonSize']
        loops = arrays['asset_0_raw_loopVertex']
        caps = np.flatnonzero(sizes == 32)
        if len(caps) != 1:
            raise RuntimeError('Expected the actual unique 32-sided upper cap')
        cap = int(caps[0]); cap_ids = loops[starts[cap]:starts[cap]+sizes[cap]]
        cap_record = {'rawPolygonIndex':cap,'rawVertexIds':cap_ids.tolist(),
            'centroidLocalMeters':raw[cap_ids].mean(0).tolist(),
            'extentLocalMeters':np.ptp(raw[cap_ids],axis=0).tolist(),
            'meaning':'A closed upper-shin truncation cap. Not an ankle joint, malleolar centre, or automatic fitting anchor.'}
        extrema = {label:{'evaluatedVertexIndex':int(np.argmin(q[:,axis]) if lower else np.argmax(q[:,axis])),
                           'positionLocalMeters':q[np.argmin(q[:,axis]) if lower else np.argmax(q[:,axis])].tolist()}
                   for label,axis,lower in (('toeMostForward',1,True),('heelMostRearward',1,False),
                                            ('medialExtreme',0,True),('lateralExtreme',0,False))}
    axes = {}
    for side in ('L','R'):
        prefix = 'left' if side == 'L' else 'right'
        foot = current['bones'][prefix+'Foot']; toes = current['bones'][prefix+'Toes']
        head = np.asarray(foot['headWorld']); tail = np.asarray(foot['tailWorld'])
        axis = tail-head; axis /= np.linalg.norm(axis)
        horizontal = axis.copy(); horizontal[2] = 0; horizontal /= np.linalg.norm(horizontal)
        axes[side] = {'ankleRigHeadWorldMeters':head.tolist(),'footRigTailWorldMeters':tail.tolist(),
            'toeRigTailWorldMeters':toes['tailWorld'],'footBoneUnitWorld':axis.tolist(),
            'groundProjectedForwardWorld':horizontal.tolist(),'referenceUpWorld':[0,0,1],
            'boneDownwardPitchDegrees':float(np.degrees(np.arcsin(-axis[2]))),
            'meaning':'Existing bone endpoints and a world-Z ground reference. The descending Foot bone is not the sole longitudinal direction and is not donor anatomical evidence.'}
    by_name = {o['name']:o for o in current['objects']}
    candidates = []
    for length_mm in (220,225,230):
        length = length_mm/1000; scale = length/donor_dimensions[1]
        width = donor_dimensions[0]*scale
        scenario = {'id':'art-'+str(length_mm)+'mm','targetLongitudinalFootSpanMeters':length,
            'uniformDonorScale':float(scale),'evaluatedTransverseSpanMeters':float(width),
            'allDonorAxesScaleTogether':True,'anisotropicToeOrArchCompression':False,
            'ratioToCapturedCrownBareFootSpan':float(length/current['verticalSpans']['crownMinusExistingBareFoot']),
            'ratioMeaning':'Artistic comparison against this character geometric span; not a population norm or clinical stature ratio.',
            'previewPriority':'central comparison only' if length_mm == 225 else 'adjacent comparison',
            'placementTransform':None,'shoeFitPassed':False,'appearanceAccepted':False,'shoeEnvelopeEvidence':{}}
        for side in ('L','R'):
            upper = by_name['Shoe_Leather_Upper_'+side]['restEvaluatedBounds']['dimensions']
            bound = length/upper[1]
            scenario['shoeEnvelopeEvidence'][side] = {
                'nativeUpperOuterLengthMeters':upper[1],'nativeUpperOuterWidthMeters':upper[0],
                'footLongerThanNativeUpperOuterEnvelopeMeters':float(length-upper[1]),
                'nativeUpperGrossWidthMinusFootWidthMeters':float(upper[0]-width),
                'uniformShoeScaleNecessaryExteriorLengthBound':float(bound),
                'upperWidthAtThatExteriorLengthBoundMeters':float(upper[0]*bound),
                'meaning':'Necessary exterior-envelope bound only. Equality supplies no toe clearance, shell thickness, lining, sock room, or interior-fit proof. It is not an applied shoe scale.'}
        candidates.append(scenario)
    upper = by_name['Shoe_Leather_Upper_L']; sock = by_name['Sock_Ribbed_L']
    upper_thickness = next(m['thickness'] for m in upper['modifiers'] if m['type']=='SOLIDIFY')
    sock_thickness = next(m['thickness'] for m in sock['modifiers'] if m['type']=='SOLIDIFY')
    result = {'schemaVersion':1,'mode':'offline-foot-art-candidates-r11',
        'provenance':{'capture':CAPTURE,'captureSha256':CAPTURE_SHA,'captureArraysSha256':CAPTURE_ARRAYS_SHA,
            'donorObservation':DONOR,'donorObservationSha256':DONOR_SHA,'donorArraysSha256':DONOR_ARRAYS_SHA,
            'currentSource':current['source'],'currentSourceSha256':current['sourceSha256'],
            'donorBundle':donor['bundle'],'donorBundleSha256':donor['bundleSha256'],
            'helper':str(Path(__file__).resolve().relative_to(project_path().resolve())), 'helperSha256':sha(__file__)},
        'units':'metres except explicit degrees and millimetre labels',
        'executedBlender':False,'sourceGeometryEdited':False,'rigEdited':False,'selectedTargetLengthMeters':None,
        'appearanceAccepted':False,'deliveryReady':False,
        'capturedVerticalSpans':current['verticalSpans'],'inventory':inventory,'measuredSideEnvelopes':side_records,
        'rigAndGroundAxes':axes,
        'donor':{'nativeEvaluatedExtentXYZMeters':donor_dimensions.tolist(),
            'nativeLongitudinalDirectionLocal':[0,-1,0],'nativeUpLocal':[0,0,1],
            'nativeLeftMedialDirectionLocal':[-1,0,0],'sourceDisplayTranslationExcluded':True,
            'jointLandmarksPresent':False,'upperCap':cap_record,'surfaceExtrema':extrema},
        'artCandidates':candidates,
        'observedShellThicknessMeters':{'shoeUpper':upper_thickness,'sock':sock_thickness,
            'meaning':'Modifier inputs in the actual capture; neither is an additional prescribed clearance.'},
        'coordinatedAdaptation':{
            'shoeScope':'Both sides: all 37 Shoe_ pieces per side, including upper, outsole, heel, welt, inner/rolled collars, toe-cap seam, stitches, tongue, facings, eyelets, laces, bows, knot, ends and aglets.',
            'shoeProcedure':[
                'After donor landmark placement, derive one shared rest-space last/footprint deformation per side and apply it coherently to all shoe pieces. Do not fit each piece by its own bounds.',
                'Prefer extending the forefoot/toe box and adjusting sections against the actual scaled donor plus sock over a blanket 3-axis enlargement. Keep heel-bottom height, sole thickness and ankle/collar registration explicit.',
                'Retain original decorative design and reproject lace/eyelet/stitch attachment to the deformed upper when a shared coarse map alone no longer preserves surface registration.',
                'Use actual evaluated inner/outer shell surfaces to resolve room. Global exterior bounds are only a rejection test and cannot certify a shoe interior.'
            ],
            'sockScope':'All 3 Sock_ pieces per side: ribbed body and both cuff pieces.',
            'sockProcedure':[
                'Refit the sock foot/heel to the placed uniform donor; carry the same lower-foot spatial map through corresponding lower sock points.',
                'Blend the map to the preserved shin/cuff domain. Current cuffs are around world Z 0.280-0.289 m; do not uniformly scale the entire sock/cuff around the ankle.',
                'Re-evaluate sock thickness, ribbed surface, ankle bending and its nested relation to the shoe interior after fitting.'
            ],
            'rigConstraint':'Every captured shoe is presently rigidly weighted to Foot; skin and ribbed sock use Foot/LowerLeg. Existing Toes endpoints are not evidence of a correctly placed new metatarsal pivot. Resolve that placement before assigning donor toe weights or claiming toe motion inside closed shoes.',
            'footBodyJoin':'Retain the original shin until a local donor-to-shin joining ring and deformation transition are verified; neither the upper cap centroid nor a bounds midpoint defines the ankle joint.'
        },
        'necessaryDonorLandmarkRecord':[
            {'landmark':'medial and lateral malleoli','requiredEvidence':'Both explicit donor surface vertex/face IDs and local coordinates, confirmed in medial/lateral views. Their relationship may constrain an ankle axis; their midpoint is not automatically a functional hinge centre.'},
            {'landmark':'heel and plantar support','requiredEvidence':'Posterior heel plus plantar heel, first/fifth metatarsal support regions and toe pads, identified by actual IDs. Fit a chosen support plane using those regions instead of the single global lowest vertex.'},
            {'landmark':'metatarsal and toe roots','requiredEvidence':'First and fifth metatarsal heads and the five toe-root transition regions as explicit IDs. Use them to evaluate current Toes pivot and toe-box flex, not nail centres alone.'},
            {'landmark':'upper-shin join','requiredEvidence':'The actual cap ring is recorded, but choose a joining ring by local section shape and the target shin anatomy after scale/placement. Record raw and evaluated identity mapping.'}
        ],
        'nextPreviewContract':[
            'Only after landmark placement is explicit, produce independent 220/225/230 mm comparison candidates if useful; 225 mm is the requested comparison centre, not an accepted size.',
            'Review barefoot dorsal/medial/lateral/plantar views and a full-body view at matching camera scale.',
            'Review the complete shoe/sock assembly and actual ankle poses, toe-box/heel/collar contact, and source-matched raw/evaluated preservation before adoption.'
        ]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    return {'report':str(output.relative_to(project_path())),'reportSha256':sha(output),
        'uniformCandidates':[{k:r[k] for k in ('id','uniformDonorScale','evaluatedTransverseSpanMeters')} for r in candidates],
        'actualObjects':inventory['total'],'appearanceAccepted':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output),ensure_ascii=False))

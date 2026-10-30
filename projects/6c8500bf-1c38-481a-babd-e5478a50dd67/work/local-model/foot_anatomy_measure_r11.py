"""Offline measured foot adaptation prerequisites from the actual R11 asset capture.

Does not launch/import Blender or alter anatomy. The unscaled donor is a shape
reference; no final adult foot length is invented from historical short shoes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

OBSERVATION = 'work/reconstruction-r02/foot-asset-observation-r11.json'
JSON_SHA = '67d6afbc595cb757c533d66f53bd494e24b29f172356bda251fb1530b98b336c'
ARRAY_SHA = '202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def bounds(a):
    lo,hi=a.min(0),a.max(0)
    return {'minimum':lo.tolist(),'maximum':hi.tolist(),'dimensions':(hi-lo).tolist()}


def to_world(a,matrix):
    m=np.asarray(matrix,np.float64)
    return np.asarray(a,np.float64)@m[:3,:3].T+m[:3,3]


def components(faces,polygon_ids):
    remaining=set(map(int,polygon_ids));groups=[]
    vertex_faces={}
    for i in remaining:
        for v in faces[i]:vertex_faces.setdefault(int(v),set()).add(i)
    while remaining:
        queue=[min(remaining)];remaining.remove(queue[0]);part=[]
        while queue:
            i=queue.pop();part.append(i)
            adjacent=set().union(*(vertex_faces[int(v)] for v in faces[i])) & remaining
            remaining-=adjacent;queue.extend(adjacent)
        groups.append(sorted(part))
    return groups


def measure(output_relative):
    observation_path=path(OBSERVATION)
    if sha(observation_path)!=JSON_SHA:raise RuntimeError('Actual foot metadata changed')
    j=json.loads(observation_path.read_text(encoding='utf-8'))
    array_path=path(j['arrays'])
    if j['arraysSha256']!=ARRAY_SHA or sha(array_path)!=ARRAY_SHA:raise RuntimeError('Actual arrays changed')
    output=path(output_relative)
    if output.exists():raise FileExistsError('Preserve previous measured foot evidence')
    with np.load(array_path,allow_pickle=False) as a:
        raw=a['asset_0_raw_positions'].astype(np.float64)
        donor=a['asset_0_evaluated_positions'].astype(np.float64)
        loops=a['asset_0_raw_loopVertex'];starts=a['asset_0_raw_polygonStart'];sizes=a['asset_0_raw_polygonSize']
        faces=[loops[s:s+n] for s,n in zip(starts,sizes)]
        attr=next(v for v in j['asset'][0]['raw']['attributes'] if v['name']=='.sculpt_face_set')
        face_sets=a[attr['data']['array']]
        nail_polygons=components(faces,np.flatnonzero(face_sets==2))
        if len(nail_polygons)!=5:raise RuntimeError('Expected actually observed five nail islands')
        tris=a['asset_0_raw_triangles'];tri_faces=a['asset_0_raw_trianglePolygon']
        nail_rows=[]
        for ids in nail_polygons:
            selected=tris[np.isin(tri_faces,ids)];t=raw[selected]
            area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)*.5
            nail_rows.append({'polygonIds':ids,'vertexIds':np.unique(selected).tolist(),
                'surfaceArea':float(area.sum()),'areaWeightedCenter':np.average(t.mean(1),axis=0,weights=area).tolist()})
        nail_rows.sort(key=lambda r:r['surfaceArea'],reverse=True)
        cap_ids=np.flatnonzero(sizes==32)
        if len(cap_ids)!=1:raise RuntimeError('Actual 32-sided shin cap changed')
        cap_id=int(cap_ids[0]);cap=raw[faces[cap_id]]
        rig=next(o for o in j['current'] if o['type']=='ARMATURE')
        bones={b['name']:b for b in rig['bones']}
        bone_rows={}
        for side in ('left','right'):
            for part in ('UpperLeg','LowerLeg','Foot','Toes'):
                name=side+part;b=bones[name]
                endpoints=to_world(np.asarray([b['headLocal'],b['tailLocal']]),rig['matrixWorld'])
                bone_rows[name]={'parent':b['parent'],'headWorld':endpoints[0].tolist(),'tailWorld':endpoints[1].tolist(),
                    'length':float(np.linalg.norm(endpoints[1]-endpoints[0]))}
        current=[]
        for i,o in enumerate(j['current']):
            if o['type']!='MESH':continue
            raw_points=to_world(a['current_'+str(i)+'_raw_positions'],o['matrixWorld'])
            evaluated=to_world(a['current_'+str(i)+'_evaluated_positions'],o['evaluatedMatrixWorld'])
            current.append({'name':o['name'],'rawWorldBounds':bounds(raw_points),
                'actualFrame1WorldBounds':bounds(evaluated),'groups':[g['name'] for g in o['vertexGroups']],
                'observationHasFullFootwearFamily':False})
        by_name={o['name']:o for o in current};donor_bounds=bounds(donor)
        old_length=by_name['Body_Complete_Shin_Foot_L']['rawWorldBounds']['dimensions'][1]
        old_front=by_name['Body_Complete_Shin_Foot_L']['rawWorldBounds']['minimum'][1]
        leg=bone_rows['leftLowerLeg']['length'];donor_length=donor_bounds['dimensions'][1]
        result={'schemaVersion':1,'mode':'offline-foot-anatomy-measure-r11',
            'source':j['source'],'sourceSha256':j['sourceSha256'],'observation':OBSERVATION,
            'observationSha256':JSON_SHA,'arraySha256':ARRAY_SHA,'helperSha256':sha(__file__),
            'lengthUnitMeaning':'Actual stored coordinate units. Historical project treats these as meters, but this observer did not capture scene.unit_settings; next current-source capture records it.',
            'donor':{'object':j['asset'][0]['name'],'bundle':j['bundle'],'bundleSha256':j['bundleSha256'],
                'nativeLocalBounds':donor_bounds,'rawVertices':len(raw),'evaluatedVertices':len(donor),
                'foremostActualVertex':{'index':int(np.argmin(donor[:,1])),'coordinate':donor[np.argmin(donor[:,1])].tolist()},
                'hindmostActualVertex':{'index':int(np.argmax(donor[:,1])),'coordinate':donor[np.argmax(donor[:,1])].tolist()},
                'nailIslands':nail_rows,'shinEndCap':{'polygonId':cap_id,'faceSet':int(face_sets[cap_id]),
                    'vertexIds':faces[cap_id].tolist(),'center':cap.mean(0).tolist(),'bounds':bounds(cap),
                    'meaning':'Closed proximal shin cut, not an ankle center or talocrural rotation axis.'},
                'ankleJointLandmarkAvailable':False,'metatarsalJointLandmarksAvailable':False,
                'existingBoneOrWeightsAvailable':bool(j['asset'][0]['vertexGroups']),
                'nativeLeftEvidence':'Largest actual nail island is on local negative X; retain that medial/hallux side before mirroring right.'},
            'historicalCurrent':current,'historicalRestBones':bone_rows,
            'measuredConflicts':{'existingToeBoneTailAheadOfOldFoot':old_front-bone_rows['leftToes']['tailWorld'][1],
                'donorLengthOverOldBareFoot':donor_length/old_length,
                'oldBareFootLengthOverLowerLegBone':old_length/leg,
                'nativeDonorLengthOverLowerLegBone':donor_length/leg,
                'meaning':'Measured model ratios, not human population standards or an accepted final adult proportion.'},
            'candidateSpecification':{
                'targetFootLength':None,'targetUniformScale':None,'targetSelectionPendingCurrentStature':True,
                'donorShapeRule':'Use one uniform anatomical scale plus a rigid frame; preserve relative arch, heel, metatarsal, toe and nail form initially. Never independently compress longitudinal anatomy to the old short sock or shoe.',
                'frameRule':'Current LowerLeg.tail/Foot.head defines the rig ankle pivot to inspect; current Foot.tail/Toes.head is only an authored toe pivot. Neither is automatically a verified donor joint landmark.',
                'requiredDonorLandmarks':['medial and lateral malleolar surface landmarks with raw/evaluated IDs','first and fifth metatarsal heads with IDs','posterior heel and plantar support patch','proximal shin connection loop and its target body cross section'],
                'footwearRule':'Adapt every actual Shoe_ and Sock_ component together around the accepted anatomical foot. Preserve the laced leather shoe design, cap seam, tongue, collar, eyelets, lace route, welt and heel construction; refit thickness and clearances after dimensional changes.',
                'skeletonRule':'Keep LowerLeg rest and ankle pivot until actual landmark fit is reviewed. If the existing Toes pivot is anatomically wrong, revise it as an explicit foot rig/animation change with new weights and contact tests; do not disguise this through mesh squash.',
                'sockRule':'Refit the lower sock over the anatomy and blend to the current upper sock/shin; avoid copying the old toe envelope. Existing skin beneath footwear remains complete.',
                'checks':['barefoot dorsal/medial/lateral/plantar close views and complete body proportion view','ankle dorsiflexion/plantarflexion and foot/heel contact','shoe/sock continuous surface crossings and complete toe/heel containment where enclosed','toe articulation with compatible forefoot shoe bending or explicit rigid-shoe motion constraints','new neutral and dynamic ground contact; no reuse of old short-shoe audit as proof'],
                'minimalCurrentCapture':'foot_proportion_capture_r11.py: current head crown/bare sole/shoe sole spans, unit scale, actual REST rig, complete shoe/sock inventory/surfaces, lower body join patch and baseline compatibility'},
            'savedBlend':False,'geometryModified':False,'anatomyOrAppearanceAccepted':False}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return {'report':str(output.relative_to(project_path())),'reportSha256':sha(output),
            'donorLength':donor_length,'oldFootLength':old_length,'toeBoneOverhang':result['measuredConflicts']['existingToeBoneTailAheadOfOldFoot'],
            'targetFootLengthSelected':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='work/reconstruction-r02/foot-anatomy-measure-r11.json')
    args=parser.parse_args()
    print(json.dumps(measure(args.output),ensure_ascii=False))


if __name__=='__main__':main()

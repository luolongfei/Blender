"""Fit the observed CC0 dental/tongue asset to actual adult oral landmarks.

Offline plan from the root's actual Blender observation. New meshes are strict
face subsets of the observed Jaw asset. Its outer oral bag is deliberately
excluded because the female head already has its original continuous bag.
No generated geometry, original-source edit, or Blender launch occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
OBS='work/reconstruction-r02/adult-oral-current-r11.json'
OBS_ARRAY_SHA='314594940150e31e641ea0ed191a2be0c444872c1fd9141b2075a3f79de9ecea'
PARTS={'upper_teeth':'R11 Adult upper dental arch','lower_teeth':'R11 Adult lower dental arch',
       'upper_gingiva':'R11 Adult upper gingiva and palate','lower_gingiva':'R11 Adult lower gingiva',
       'tongue':'R11 Adult anatomical tongue'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def closest_triangles(point,triangles):
    a=triangles[:,0];b=triangles[:,1];c=triangles[:,2];ab=b-a;ac=c-a;ap=np.asarray(point)-a
    aa=np.sum(ab*ab,1);bb=np.sum(ab*ac,1);cc=np.sum(ac*ac,1);d=np.sum(ap*ab,1);e=np.sum(ap*ac,1)
    den=aa*cc-bb*bb;u=(cc*d-bb*e)/np.maximum(den,1e-30);v=(aa*e-bb*d)/np.maximum(den,1e-30)
    inside=(den>1e-24)&(u>=0)&(v>=0)&(u+v<=1)
    points=a+u[:,None]*ab+v[:,None]*ac;dist=np.sum((points-point)**2,1);dist[~inside]=np.inf
    for q,r in ((a,b),(b,c),(c,a)):
        edge=r-q;t=np.clip(np.sum((point-q)*edge,1)/np.maximum(np.sum(edge*edge,1),1e-30),0,1)
        x=q+t[:,None]*edge;dd=np.sum((x-point)**2,1);take=dd<dist;points[take]=x[take];dist[take]=dd[take]
    i=int(np.argmin(dist));normal=np.cross(ab[i],ac[i]);normal/=max(np.linalg.norm(normal),1e-20)
    return i,points[i],float(np.dot(point-points[i],normal))


def proximity(points,head_world,head_faces,head_groups):
    face_ids=np.flatnonzero(np.isin(head_groups,[7,8,17,22]));f=head_faces[face_ids]
    triangles=np.vstack((f[:,[0,1,2]],f[:,[0,2,3]]));original_faces=np.r_[face_ids,face_ids]
    active=np.unique(triangles);coords=head_world[active];adjacency=[[] for _ in head_world]
    for i,t in enumerate(triangles):
        for v in t:adjacency[int(v)].append(i)
    signed=[];nearest=[];ids=[]
    for point in points:
        d=np.sum((coords-point)**2,axis=1);n=active[np.argpartition(d,8)[:8]]
        candidates=np.unique(np.concatenate([np.asarray(adjacency[int(i)],np.int32) for i in n]))
        index,co,sg=closest_triangles(point,head_world[triangles[candidates]])
        signed.append(sg);nearest.append(co);ids.append(original_faces[candidates[index]])
    return np.asarray(signed),np.asarray(nearest),np.asarray(ids,np.int32)


def plan():
    observed=json.loads((P/OBS).read_text(encoding='utf8'));array_path=P/observed['arrays']
    if sha(array_path)!=OBS_ARRAY_SHA or observed['arraysSha256']!=OBS_ARRAY_SHA:raise RuntimeError('Actual oral observation changed')
    with np.load(array_path) as n:a={k:n[k].copy() for k in n.files}
    record=observed['donorObjects'][0]
    if record['name']!='Jaw - Realistic' or len(observed['donorObjects'])!=1:raise RuntimeError('Expected actual single CC0 donor object')
    v=a['donor_0_raw_local'];loops=a['donor_0_raw_loops'];starts=a['donor_0_raw_face_starts'];sizes=a['donor_0_raw_face_sizes'];sets=a['donor_0_raw_face_sets']
    faces=[loops[s:s+c] for s,c in zip(starts,sizes)]
    if v.shape!=(5294,3) or len(faces)!=5058 or set(np.unique(sets))!=set([31,93,155,157,*range(158,186)]):raise RuntimeError('Observed donor topology differs')
    teeth={}
    for group in range(158,186):
        ids=np.unique(np.concatenate([faces[i] for i in np.flatnonzero(sets==group)]));center=v[ids].mean(0)
        teeth[group]={'ids':ids,'center':center,'upper':bool(center[2]>0)}
    upper=[g for g,t in teeth.items() if t['upper']];lower=[g for g,t in teeth.items() if not t['upper']]
    if len(upper)!=14 or len(lower)!=14:raise RuntimeError('Actual 14+14 tooth component classification changed')
    central=sorted(upper,key=lambda g:abs(teeth[g]['center'][0]))[:2]
    source_inc_width=float(np.mean([np.ptp(v[teeth[g]['ids'],0]) for g in central]))
    land=json.loads((P/'work/reconstruction-r02/female-face-landmarks-r11.json').read_text(encoding='utf8'))
    rim=land['mouth']['openingRimCandidate']['sourceVertexIndices'];cage=a['head_0_cage_world'];mouth_width=float(np.ptp(cage[rim,0]))
    # 20.5% of the actual mouth corner width gives a restrained central incisor
    # width; recorded as a design choice, not an ethnicity/dental measurement.
    scale=mouth_width*.205/source_inc_width
    if not .78<scale<.95:raise RuntimeError('Actual mouth requires a new dental scale review')
    contact=observed['headStates'][0]['standardLipContact']['sections'][4]
    seam=np.mean([contact['upperNearestWorld'],contact['lowerNearestWorld']],axis=0)
    central_ids=np.concatenate([teeth[g]['ids'] for g in central]);front=float(v[central_ids,1].min());edge=float(v[central_ids,2].min())
    translation=np.array([float(cage[rim,0].mean()),float(seam[1]+.0045-scale*front),float(seam[2]-.0015-scale*edge)])
    transformed=v*scale+translation
    # Fit the donor's non-dental soft-tissue skirt inside the shorter current
    # female oral floor. Tooth components and the visible gingival margins
    # remain rigid; only the deep lower skirt and posterior junction change.
    original_transformed=transformed.copy();soft_delta=np.zeros_like(transformed)
    gum_ids=np.unique(np.concatenate([faces[i] for i in np.flatnonzero(np.isin(sets,[93,155]))]))
    lower_gum_ids=np.unique(np.concatenate([faces[i] for i in np.flatnonzero(sets==155)]))
    posterior=np.clip((transformed[gum_ids,1]+.042)/.015,0.,1.);posterior=posterior*posterior*(3.-2.*posterior)
    soft_delta[gum_ids,1]=-.004*posterior
    base=np.clip((1.357-transformed[lower_gum_ids,2])/.009,0.,1.);base=base*base*(3.-2.*base)
    soft_delta[lower_gum_ids,2]=.0049*base
    transformed+=soft_delta
    clearance_adjustments=[]
    for group in (93,155):
        ids=np.unique(np.concatenate([faces[i] for i in np.flatnonzero(sets==group)]))
        sg,nearest,_=proximity(transformed[ids],a['head_0_world'],a['head_0_faces'],a['head_0_groups'])
        take=sg<.00030;selected=ids[take];delta=np.zeros((len(selected),3),float)
        if len(selected):
            direction=transformed[selected]-nearest[take];length=np.linalg.norm(direction,axis=1)
            direction*=np.where(sg[take]<0,-1.,1.)[:,None]/np.maximum(length[:,None],1e-12)
            delta=nearest[take]+direction*.00035-transformed[selected]
            if np.linalg.norm(delta,axis=1).max()>.0014:raise RuntimeError('Soft-tissue surface correction exceeds reviewed 1.4mm local allowance')
            transformed[selected]+=delta;soft_delta[selected]+=delta
        clearance_adjustments.append({'faceSet':group,'points':len(selected),'maximumDeltaMeters':float(np.linalg.norm(delta,axis=1).max()) if len(selected) else 0.})
    check_tri=[]
    for i,face in enumerate(faces):
        if sets[i]!=157:check_tri.extend((face[0],face[k],face[k+1]) for k in range(1,len(face)-1))
    check_tri=np.asarray(check_tri,np.int32);old_tri=original_transformed[check_tri];new_tri=transformed[check_tri]
    old_n=np.cross(old_tri[:,1]-old_tri[:,0],old_tri[:,2]-old_tri[:,0]);new_n=np.cross(new_tri[:,1]-new_tri[:,0],new_tri[:,2]-new_tri[:,0])
    old_area=np.linalg.norm(old_n,axis=1);new_area=np.linalg.norm(new_n,axis=1);valid=old_area>1e-12
    alignment=np.sum(old_n[valid]*new_n[valid],axis=1)/(old_area[valid]*new_area[valid])
    if (alignment<=0).any():raise RuntimeError('Fitted soft oral tissue flips original donor triangles')
    definitions={'upper_teeth':upper,'lower_teeth':lower,'upper_gingiva':[93],'lower_gingiva':[155],'tongue':[31]}
    arrays={'donor_reference_local':v,'target_all_world':transformed,'rigid_fit_all_world':original_transformed,
            'soft_tissue_fit_delta_world':soft_delta,'source_observed_cage_world':cage,
            'fit_translation_world':translation,'fit_scale':np.array([scale])};parts={}
    for name,groups in definitions.items():
        fi=np.flatnonzero(np.isin(sets,groups));ids=np.unique(np.concatenate([faces[i] for i in fi]));mapping=np.full(len(v),-1,np.int32);mapping[ids]=np.arange(len(ids))
        pf=[mapping[faces[i]] for i in fi];pl=np.concatenate(pf);ps=np.r_[0,np.cumsum([len(f) for f in pf])[:-1]].astype(np.int32);pn=np.asarray([len(f) for f in pf],np.int32)
        arrays[name+'_source_ids']=ids;arrays[name+'_source_face_ids']=fi;arrays[name+'_world']=transformed[ids]
        arrays[name+'_loops']=pl;arrays[name+'_face_starts']=ps;arrays[name+'_face_sizes']=pn;arrays[name+'_face_sets']=sets[fi]
        sg,nearest,original_faces=proximity(transformed[ids],a['head_0_world'],a['head_0_faces'],a['head_0_groups'])
        arrays[name+'_approx_head_signed']=sg;arrays[name+'_approx_head_nearest']=nearest;arrays[name+'_approx_head_face_ids']=original_faces
        parts[name]={'objectName':PARTS[name],'sourceFaceSets':groups,'sourceVertices':len(ids),'sourceFaces':len(fi),
                     'worldBounds':[transformed[ids].min(0).tolist(),transformed[ids].max(0).tolist()],
                     'parentRole':'cranial' if name.startswith('upper_') else 'mandibular',
                     'approximateSignedNearestHeadMetersP0P01P50P99P100':np.percentile(sg,[0,1,50,99,100]).tolist(),
                     'approximatePointsInsideHeadBeyondHalfMillimeter':int((sg<-.0005).sum()),
                     'approximation':'Closest triangle among 8 nearest actual evaluated vertex stars. Blender BVH must verify full actual subdivided output.'}
    report={'source':observed['source'],'sourceSha256':observed['sourceSha256'],'observation':OBS,'observationArraysSha256':OBS_ARRAY_SHA,
            'donor':observed['donor'],'donorSha256':observed['donorSha256'],'donorLicense':'CC0; exact already-downloaded Blender Human Base Meshes 1.4.1',
            'fitScale':scale,'fitTranslationWorld':translation.tolist(),'actualMouthCornerWidthMeters':mouth_width,
            'desiredCentralIncisorWidthMeters':mouth_width*.205,'desiredUpperIncisalEdgeBelowActualLipContactMeters':.0015,
            'desiredUpperCentralFrontBehindActualLipContactMeters':.0045,'upperToothGroups':upper,'lowerToothGroups':lower,
            'softTissueFit':{'lowerDeepSkirtMaximumLiftMeters':.0049,'lowerSkirtUpperFixedWorldZ':1.357,
                            'lowerSkirtSmoothHeightMeters':.009,'posteriorGumMaximumForwardMeters':.004,
                            'changedSourceVertices':int(np.count_nonzero(np.linalg.norm(soft_delta,axis=1)>1e-10)),
                            'additionalActualSurfaceClearanceAdjustments':clearance_adjustments,
                            'minimumRawTriangleNormalCosine':float(alignment.min()),'rawTriangleFlips':0,
                            'allDentalComponentsRemainUniformlyScaledAndRigid':True},
            'parts':parts,'excludedOuterCavityFaceSet':157,'originalHeadBagPreserved':True,
            'sourceMeshRawCoordinatesUnchanged':True,'blenderApplied':False,'appearanceAccepted':False,
            'limits':['A fitted dental design candidate; source mesh data and all 28 tooth components remain attributable to the actual CC0 donor.',
                      'Outer donor mouth bag is excluded, not duplicated over the existing adult head oral cavity.',
                      'Actual subdivided surfaces, neutral lip/tooth hiding, open-mouth clearance and jaw/smile/head-turn combinations remain required.',
                      'Pre-existing head lip intersections are retained in the oral observation and cannot be hidden or accepted by adding teeth.']}
    return arrays,report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=(P/args.output).resolve();npz=out.with_suffix('.npz');out.relative_to((P/'work/reconstruction-r02').resolve())
    if out.exists() or npz.exists():raise RuntimeError('Preserve earlier plan')
    arrays,report=plan();np.savez_compressed(npz,**arrays);report.update(arrays=npz.relative_to(P).as_posix(),arraysSha256=sha(npz),helperSha256=sha(__file__))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'output':args.output,'fitScale':report['fitScale'],'fitTranslationWorld':report['fitTranslationWorld'],'parts':report['parts']}))


if __name__=='__main__':main()

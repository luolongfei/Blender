"""Read-only contact interpretation for the independent crown-flow proposal."""
import hashlib,json,runpy
from pathlib import Path
import numpy as np

def main():
    root=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());p=root/'projects/6c8500bf-1c38-481a-babd-e5478a50dd67';w=p/'work/reconstruction-r02';here=Path(__file__).parent
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    out=w/'reference-crown-flow-proposal-r11-contact-proof.json'
    if out.exists():raise RuntimeError('Do not overwrite review evidence')
    a=np.load(w/'reference-crown-flow-proposal-r11.npz');base=np.load(w/'reference-adult-front-detail-v10e-r11.groom-front-detail.npz');g=np.load(w/'reference-gather-rebuild-v3-r11.npz');head=np.load(w/'reference-tie-alignment-r11-v2.npz')
    pure=runpy.run_path(str(here/'reference_crown_flow_plan_r11.py'));Surface=runpy.run_path(str(here/'collar_continuous_offline_r11.py'))['Surface'];surface=Surface(head['head_world'],head['head_triangles'])
    s=a['starts'];n=a['lengths'];source=a['source_post_tie_world'];target=a['target_world'];raw=base['reference-front-detail_world'];contact=base['head-contact_world'];rr=g['source_radius'];radius=np.array([rr[x:x+y].max() for x,y in zip(s,n)])
    if not np.array_equal(s,base['reference-front-detail_starts']) or not np.array_equal(n,base['reference-front-detail_lengths']) or not np.array_equal(source[s],raw[s]):raise RuntimeError('Measured source roots no longer correspond to actual contact observation')
    rows=[];new_signed=[];old_signed=[];points=[];ids=[];extra_correction=[]
    for index,i in enumerate(a['selected_curve_ids']):
        if index%300==0:print('CROWN_ROOT_CONTACT '+str(index),flush=True)
        start=int(s[i]);count=int(n[i]);q=target[start:start+count].astype(float).copy();before=source[start:start+count].astype(float)
        # The new layer preserves p0 bitexact. Its downstream head-contact
        # result is therefore the already observed actual p0 for this source.
        # Other changed controls already have measured >=0.4mm clearance.
        q[0]=contact[start]
        after_pts,_=pure['catmull_samples'](q,np.array([0]),count=6);before_pts,_=pure['catmull_samples'](before,np.array([0]),count=6)
        after_pts=np.r_[after_pts,((q[0]+q[1])*.5)[None]];before_pts=np.r_[before_pts,((before[0]+before[1])*.5)[None]]
        for new,old in zip(after_pts,before_pts):
            h,no,_,_=surface.find_nearest(new);ns=float(np.dot(new-h,no))-float(radius[i]);h,no,_,_=surface.find_nearest(old);bs=float(np.dot(old-h,no))-float(radius[i])
            new_signed.append(ns);old_signed.append(bs);points.append(new);ids.append(int(i))
        extra_correction.append(float(np.linalg.norm(q[0]-before[0])))
    new_signed=np.asarray(new_signed);old_signed=np.asarray(old_signed);sample=a['sample_signed_head'];curve=a['sample_curve_id'];offset=a['sample_control_or_segment_offset'];kind=a['sample_kind'];surface_signed=sample-radius[curve]
    nonroot=(offset>0)|(kind==0)
    gate=float(min(new_signed.min(),surface_signed[nonroot].min()))
    report={'status':'offline-post-contact-estimate; actual-current-native-check-required',
            'sourceArraysSha256':sha(w/'reference-crown-flow-proposal-r11.npz'),
            'actualV10eStageArraysSha256':sha(w/'reference-adult-front-detail-v10e-r11.groom-front-detail.npz'),
            'actualHeadArraysSha256':sha(w/'reference-tie-alignment-r11-v2.npz'),
            'allPostTieRootsExactToObservedPreContactRoots':True,'rootPositionReplacementOnlyInTemporaryProof':True,
            'productionPlanAndLayerRootsRemainUnmodified':True,'actualObservedRootContactCorrectionMetersP0P50P95P100':np.percentile(extra_correction,[0,50,95,100]).tolist(),
            'rootSegmentEstimateSamples':len(new_signed),'rootSegmentBeforeContactSurfaceMetersP0P1P50P100':np.percentile(old_signed,[0,1,50,100]).tolist(),
            'rootSegmentAfterExistingContactEstimateSurfaceMetersP0P1P50P100':np.percentile(new_signed,[0,1,50,100]).tolist(),
            'rootSegmentAfterExistingContactEstimateBelowMinusHalfMm':int((new_signed<-.0005).sum()),
            'otherChangedSamples':int(nonroot.sum()),'otherChangedMinimumSurfaceClearanceMeters':float(surface_signed[nonroot].min()),
            'combinedMinimumEstimatedSurfaceClearanceMeters':gate,
            'errors':[] if gate>=-.0005 else ['Conditional post-contact estimate still penetrates beyond -0.5 mm'],
            'worstRootSamples':[{'curve':int(ids[k]),'world':points[k].tolist(),'surfaceSignedMeters':float(new_signed[k])} for k in np.argsort(new_signed)[:20]],
            'limitations':['This is an offline estimate using actual v10e head-contact root positions and uniform Catmull-Rom interpolation; it is not Blender native evaluated proof.',
                           'The successful current tie candidate must be recaptured and source topology, roots, point identities and current head surface recalibrated before installation.',
                           'The layer installs before existing head contact. Its raw pre-contact root p0 may be under the surface; moving p0 in this layer would violate the intended identity contract.',
                           'Current head-contact native curves and complete actual rendering remain mandatory before appearance adoption.']}
    out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:report[k] for k in ('combinedMinimumEstimatedSurfaceClearanceMeters','rootSegmentAfterExistingContactEstimateBelowMinusHalfMm','errors')}))

if __name__=='__main__':main()

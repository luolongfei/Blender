"""Read-only actual tail-shape statistics; no proposed asset coordinates."""
import hashlib,json,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
api=runpy.run_path(str(HERE/'repair_reference_gather_native_v4_r11.py'));P=api['P'];W=P/'work/reconstruction-r02'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def percent(values):return np.percentile(values,[5,50,95]).tolist()
def arcs(q):return np.concatenate([np.zeros(q.shape[:-2]+(1,)),np.cumsum(np.linalg.norm(np.diff(q,axis=-2),axis=-1),axis=-1)],axis=-1)

def main():
    observation=W/'reference-post-tie-actual-r11.json';obs=json.loads(observation.read_text());op=P/obs['arrays'];proofpath=W/'reference-current-band-native-proof-r11.json';proof=json.loads(proofpath.read_text());pp=P/proof['arrays'];out=W/'reference-tail-rest-readonly-r11.json';plot=W/'reference-tail-rest-readonly-r11.png'
    if out.exists() or plot.exists():raise RuntimeError('Use unused analysis outputs')
    if obs['status']!='collected' or sha(op)!=obs['arraysSha256'] or sha(pp)!=proof['arraysSha256'] or proof['observationSha256']!=sha(observation):raise RuntimeError('Actual observation/native band evidence mismatch')
    a=np.load(op);caps=np.load(pp);headwidth=float(a['ear_positive_world'][:,0].max()-a['ear_negative_world'][:,0].min());data={};stats=[];band=[];sampleids=np.unique(np.linspace(0,2479,32,dtype=int));B=api['native_basis'](35,12)
    for side,sign in [('positive',1),('negative',-1)]:
        q=a['tail_'+side+'_world'].reshape(2480,35,3);n=a['native_tail_'+side+'_world'].reshape(2480,128,3);ra=arcs(q);na=arcs(n);median=np.median(q,axis=0);nm=np.median(n,axis=0);delta=n-n[:,:1];forward=-delta[:,:,1];drop=-delta[:,:,2];outward=sign*delta[:,:,0];tangent=np.diff(median,axis=0);angle=np.degrees(np.arccos(np.clip(-tangent[:,2]/np.linalg.norm(tangent,axis=1),-1,1)));nativeerrors=[float(np.linalg.norm(api['resampled_basis'](q[i],B,128)@q[i]-n[i],axis=1).max()) for i in sampleids];peak=np.argmax(sign*nm[:,0]);data[side]=(q,n,median,nm)
        stats.append({'side':side,'rawCurves':2480,'controlsPerCurve':35,'nativePointsPerCurve':128,'nativeWorldBoundsMeters':[n.min((0,1)).tolist(),n.max((0,1)).tolist()],'nativeArcLengthMetersP05P50P95':percent(na[:,-1]),'tipDropFromOwnRootMetersP05P50P95':percent(drop[:,-1]),'maximumOutwardFromOwnRootMetersP05P50P95':percent(outward.max(1)),'maximumForwardFromOwnRootMetersP05P50P95':percent(forward.max(1)),'tipForwardFromOwnRootMetersP05P50P95':percent(forward[:,-1]),'rootToFirstControlsArcMetersP05P50P95':np.percentile(ra[:,1:6],[5,50,95],axis=0).tolist(),'rootFirstTwoMedianSegmentAnglesFromGravityDegrees':angle[:2].tolist(),'nativeMedianOutwardPeakIndex':int(peak),'nativeMedianOutwardPeakArcFraction':float(peak/127),'medianControlWorld':median.tolist(),'medianNativeWorld':nm.tolist(),'medianControlSegmentAnglesFromGravityDegrees':angle.tolist(),'nativeModelCheck':{'sampleCurveIds':sampleids.tolist(),'model':'clamped Catmull-Rom resolution12 then arc128','maximumActualResidualMeters':max(nativeerrors),'support':'Point p3 affects native p1->p2 through the signed adjacent-control basis. This is confirmed against current raw/native data, not assumed from the object label.'}})
        offset=0 if side=='positive' else 2480*127
        for suffix in ('a','b'):
            pre='tie_'+side+'_'+suffix;ids=caps[pre+'_segment_ids'];gap=caps[pre+'_clearance'];near=(ids>=offset)&(ids<offset+2480*127)&(gap<=.003);local=ids[near]-offset;cid=local//127;seg=local%127;end=na[cid,seg+1]
            band.append({'ring':pre,'nativeTailSegmentsWithin3mmConservativeClearance':len(local),'latestNearNativeSegmentOffset':int(seg.max()),'latestNearNativeArcMeters':float(end.max()),'nearSegmentEndArcMetersP05P50P95':percent(end),'nearSegmentsReachingPastControlP1':int((end>ra[cid,1]).sum()),'nearSegmentsReachingPastControlP2':int((end>ra[cid,2]).sum()),'nearSegmentsReachingPastControlP3':int((end>ra[cid,3]).sum()),'minimumOwnCurveArcGapFromNearBandEndToP3Meters':float((ra[cid,3]-end).min()),'selection':'Existing exact all512-triangle current native capsule proof; retain same-side segments with conservative clearance<=3mm.'})
    both=np.concatenate([data['positive'][1].reshape(-1,3),data['negative'][1].reshape(-1,3)]);totalwidth=float(np.ptp(both[:,0]));report={'schemaVersion':1,'status':'read_only_analysis','source':obs['source'],'sourceSha256':obs['sourceSha256'],'observation':str(observation.relative_to(P)),'observationSha256':sha(observation),'observationArraysSha256':sha(op),'nativeBandProof':str(proofpath.relative_to(P)),'nativeBandProofSha256':sha(proofpath),'nativeBandProofArraysSha256':sha(pp),'helperSha256':sha(__file__),'referenceImage':'references/user-original-portrait-2026-10-28.png','referenceImageSha256':sha(P/'references/user-original-portrait-2026-10-28.png'),'actualImagesReviewed':['previews/reconstruction-r02/reference-ties-v3-r11/full-groom-front.png','previews/reconstruction-r02/reference-ties-v3-r11/full-groom-quarter.png'],'coordinateConvention':'+Y posterior, -Y anterior, +Z upward; positive/negative are actual worldX sides. All control indices are zero based.','earOuterWidthMeters':headwidth,'bothTailNativeTotalWidthMeters':totalwidth,'bothTailWidthOverEarOuterWidth':totalwidth/headwidth,'tails':stats,'bandEnvelopeInfluence':band,'findings':[
      'p0-p2 already lock about26mm of arc, about18mm forward movement and12mm drop. Median takeoff is62-64 degrees from gravity, so p3 must not abruptly turn vertical.',
      'p3-p16 continue outward while moving strongly forward; the median lateral peak is near native arc58-60%. p17-p34 then return inward in a synchronized broad hook. The two large similar arches and forward-dominant side profile explain the stiff S/hook appearance.',
      'Keeping all first3 pins is compatible with a softer descending free tail. Smooth its exit tangent through a finite transition, then give the middle mass a mainly downward direction and reserve small uneven end curls for the final15-20%.',
      'Changing p3 already changes native geometry between p1 and p2. Actual band-near samples extend beyond p1 and some beyond p2, so first3 protection alone does not preserve the fitted-band native envelope.',
      'A conservative starting design keeps p0-p4 and their radii unchanged, reshaping from p5. In the empirically matched local basis, p5 first affects p3->p4; all current <=3mm band-near segments end before p3. This reduces coupling but does not remove the requirement to rerun actual native band/collision checks.'
    ],'designProposal':{'status':'shape_guidance_only_not_asset_or_collision_accepted','priority':'Retain the reference black low twin tails, new ear-adjacent knot positions, existing root/pins and coherent hair material; change free-tail rest shape only after band/crown integration.','rootControlPolicy':'Minimum first3 exact; recommended first5 geometric support controls/radii exact where compatible with the chosen design, without silently changing which3 are dynamics pins.','totalFrontWidthInitialDesignWindowOverEarWidth':[1.50,1.65],'totalFrontWidthInitialDesignWindowMeters':[1.50*headwidth,1.65*headwidth],'medianMaximumLateralExcursionFromRootMeters':[.030,.040],'tipDropInitialDesignWindowMeters':[.135,.165],'tipForwardInitialDesignWindowMeters':[.075,.110],'staticArcLengthIfKeepingReferenceTipHeightInitialWindowMeters':[.190,.220],'windowBasis':'Illustration-consistent initial design ranges, not metric reconstruction from a single drawing. Current source numbers are measured; these target windows must be judged in the same actual front/quarter camera and against current blouse/collar/shoulder geometry.','lengthTradeoff':'Do not simultaneously force the current254mm arc, unchanged end height and a large reduction of162mm forward excursion: surplus length would be forced into new coils. Prefer the reference tip height and a shorter redesigned static free-tail arc, or explicitly accept lower tips if keeping the old arc. This is a static rest-design decision, not stretch during animation.','flow':'Keep small initial outward exit at the tie, ease into downward fall after the protected support region, allow mild distributed waves with different phases/lengths across existing strand groups, and vary the thin end hooks rather than curling the entire bundle in unison.','collisionNeed':'The one complete post-tie observation does not include current blouse/collar/shoulder surfaces. Moving hair backward/down toward the body requires current full visible cloth/head/neck collision checks before adopting any coordinates.','futureBandDependency':'Any changed raw free-control point or radius, including p3, requires native recapture and full ring capsule validation. Keeping p0-p4 can leave the current local underlying band-contact geometry intact, but resampling and future dynamics still require fresh validation.'},'savedBlend':False,'tailCoordinatesWritten':False,'pinDataChanged':False,'bandChanged':False,'candidateCoordinatesGenerated':False,'appearanceAccepted':False}
    with out.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    try:
        from PIL import Image,ImageDraw,ImageFont
        im=Image.new('RGB',(1400,850),'#f7f8fa');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);d.text((30,20),'ACTUAL v3 native tail geometry — read-only analysis, no new tail design',fill='#20242b',font=font)
        panels=[(40,90,650,730,'FRONT  X / Z',0),(750,90,610,730,'SIDE  anterior(-Y) / Z',1)]
        for x0,y0,pw,ph,title,mode in panels:
            d.text((x0,y0-33),title,fill='#20242b',font=font);xmin,xmax=(-.18,.18) if mode==0 else (-.05,.18);zmin,zmax=1.20,1.54
            def project(v):
                u=v[0] if mode==0 else -v[1]
                return (x0+(u-xmin)/(xmax-xmin)*pw,y0+(zmax-v[2])/(zmax-zmin)*ph)
            for z in np.arange(1.22,1.541,.04):
                yy=project((0,0,z))[1];d.line((x0,yy,x0+pw,yy),fill='#dce1e6');d.text((x0+3,yy-18),f'{z:.2f} m',font=small,fill='#75808c')
            for point in a['head_world'][::220]:
                xp,yp=project(point)
                if x0<=xp<=x0+pw and y0<=yp<=y0+ph:d.ellipse((xp-1,yp-1,xp+1,yp+1),fill='#c6cbd1')
            for side,color in [('positive','#a23541'),('negative','#285ea7')]:
                if mode==1 and side=='negative':continue
                q,n,med,nm=data[side]
                for idx in np.linspace(0,2479,32,dtype=int):d.line([project(point) for point in n[idx]],fill=('#cf9ea4' if side=='positive' else '#9db4d5'),width=1)
                d.line([project(point) for point in nm],fill=color,width=4)
                for idx in (0,1,2,3,4,8,12,16,20,24,28,34):
                    xp,yp=project(med[idx]);d.ellipse((xp-3,yp-3,xp+3,yp+3),fill=color);d.text((xp+5,yp-10),str(idx),font=small,fill=color)
                for suffix in ('a','b'):
                    ring=a['tie_'+side+'_'+suffix+'_raw_world'];d.line([project(point) for point in ring[::8]],fill='#22262c',width=2)
            d.line((x0+pw-32,y0+35,x0+pw-32,y0+105),fill='#58636f',width=3);d.polygon([(x0+pw-38,y0+96),(x0+pw-26,y0+96),(x0+pw-32,y0+109)],fill='#58636f');d.text((x0+pw-95,y0+12),'gravity',font=small,fill='#58636f')
        im.save(plot)
    except ImportError:pass
    print(json.dumps({'report':str(out.relative_to(P)),'reportSha256':sha(out),'plot':str(plot.relative_to(P)) if plot.exists() else None,'plotSha256':sha(plot) if plot.exists() else None,'actualWidthMm':totalwidth*1000,'actualWidthOverEarWidth':totalwidth/headwidth,'band':band,'noAssetsModified':True}),flush=True)

if __name__=='__main__':main()

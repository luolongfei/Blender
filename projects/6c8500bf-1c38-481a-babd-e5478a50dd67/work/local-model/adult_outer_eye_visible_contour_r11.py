"""Actual neutral before/preview silhouette and outer head-section evidence."""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;P=HERE.parent.parent
SOURCE=P/'previews/reconstruction-r02/outer-eye-neutral-direction-preview-r11/neutral-geometry-preview.npz'
OUT=P/'work/reconstruction-r02/adult-outer-eye-visible-contour-r11'
EPS=1e-7
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)

class Slices:
    def __init__(self,world,tri):
        self.faces=world[tri];self.lo=self.faces[:,:,0].min(1);self.hi=self.faces[:,:,0].max(1)
    def lines(self,x):
        p=self.faces[(self.lo<x)&(self.hi>x)]
        pa=p[:,[0,1,2],:];pb=p[:,[1,2,0],:]
        den=pb[:,:,0]-pa[:,:,0]
        t=np.divide(x-pa[:,:,0],den,out=np.full_like(den,np.nan),where=den!=0)
        hit=(t>0)&(t<1);valid=hit.sum(1)==2
        points=pa+t[:,:,None]*(pb-pa)
        seg=points[valid][hit[valid]].reshape(-1,2,3)
        dz=seg[:,1,2]-seg[:,0,2];seg=seg[abs(dz)>1e-12];dz=seg[:,1,2]-seg[:,0,2]
        slope=(seg[:,1,1]-seg[:,0,1])/dz;intercept=seg[:,0,1]-slope*seg[:,0,2]
        return np.column_stack([seg[:,:,2].min(1),seg[:,:,2].max(1),slope,intercept])

def depths(lines,z):
    if not len(lines):return np.full(len(z),np.inf)
    y=z[:,None]*lines[None,:,2]+lines[None,:,3]
    valid=(z[:,None]>=lines[None,:,0]-1e-12)&(z[:,None]<=lines[None,:,1]+1e-12)
    return np.where(valid,y,np.inf).min(1)

def intervals(head,globe,x):
    h,g=head.lines(x),globe.lines(x)
    if not len(g):return []
    zlo,zhi=g[:,0].min(),g[:,1].max()
    h=h[(h[:,1]>=zlo)&(h[:,0]<=zhi)]
    breaks=[g[:,:2].ravel(),np.clip(h[:,:2].ravel(),zlo,zhi)]
    if len(h):
        den=g[:,None,2]-h[None,:,2]
        z=np.divide(h[None,:,3]-g[:,None,3]-EPS,den,out=np.full_like(den,np.nan),where=abs(den)>1e-15)
        overlap=(z>=np.maximum(g[:,None,0],h[None,:,0]))&(z<=np.minimum(g[:,None,1],h[None,:,1]))
        breaks.append(z[overlap])
    b=np.unique(np.concatenate(breaks));mid=(b[:-1]+b[1:])*.5
    gy,hy=depths(g,mid),depths(h,mid)
    active=np.isfinite(gy)&(gy<hy-EPS)&((b[1:]-b[:-1])>1e-12)
    found=[]
    for i in np.flatnonzero(active):
        start,end=float(b[i]),float(b[i+1])
        if found and start-found[-1][1]<1e-9:found[-1][1]=end
        else:found.append([start,end])
    return found

def contour(head,globe,eye_world):
    lo,hi=float(eye_world[:,0].min()),float(eye_world[:,0].max())
    xs=np.linspace(lo+1e-8,hi-1e-8,int(np.ceil((hi-lo)/.00005))+1)
    rows=[]
    for x in xs:
        seg=intervals(head,globe,float(x))
        if seg:rows.append([x,seg[0][0],seg[-1][1],len(seg)])
    rows=np.asarray(rows)
    ends=[]
    for index,bound,direction in [(0,lo,-1),(-1,hi,1)]:
        inside=float(rows[index,0]);outside=bound
        for _ in range(20):
            mid=(inside+outside)*.5
            if intervals(head,globe,mid):inside=mid
            else:outside=mid
        seg=intervals(head,globe,inside)
        ends.append(dict(x=inside,lowerZ=seg[0][0],upperZ=seg[-1][1],gapMeters=seg[-1][1]-seg[0][0],
                         xBracketMeters=abs(inside-outside),globeBoundX=bound))
    return rows,ends

def differential(x,z):
    # Explicit 0.6 mm observation scale, not a fitted design target.
    output=np.full((len(x),3),np.nan)
    for i in range(len(x)):
        use=abs(x-x[i])<=.00030001
        if use.sum()<5:continue
        xx=(x[use]-x[i])*1000;zz=z[use]*1000
        a,b,c=np.polyfit(xx,zz,2)
        output[i]=[b,2*a,2*a/(1+b*b)**1.5]
    return output

def run():
    if OUT.with_suffix('.json').exists():raise RuntimeError('Preserve earlier evidence')
    meta=json.loads(SOURCE.with_suffix('.json').read_text());assert sha(SOURCE)==meta['arraysSha256']
    with np.load(SOURCE) as a:
        ids=a['originalVertexIds'];before=a['beforeFullWorld'][ids];after=a['frame_2_fullWorld'][ids]
        t0=a['triangles'];t1=a['frame_2_triangles']
        eyes={s:(a['eye_'+s+'_world'],a['eye_'+s+'_triangles']) for s in 'LR'}
    heads={'before':Slices(before,t0),'preview':Slices(after,t1)}
    report=dict(source=str(SOURCE.relative_to(P)),arraysSha256=sha(SOURCE),helperSha256=sha(__file__),
        method='Exact actual triangle-plane segments. All segment Z endpoints and head/globe front-depth equality roots partition each X section; midpoint tests select visible globe intervals. No ideal sphere, raster Z bin or old triangulation.',
        visibleDepthThresholdMeters=EPS,xGridStepMaximumMeters=.00005,derivativeScaleMeters=.0006,
        limits=['Endpoints are front-visible globe endpoints, not certified anatomical skin canthus or wet-line junctions.',
                'Outer head-only profiles explicitly continue past the globe; any depth-pocket landmark is descriptive rather than a new certified canthal endpoint.',
                'Slope and curvature use a local quadratic over 0.6 mm to distinguish macroscopic steps from individual triangle facets. Raw exact contours are retained.'],
        sides={},sourceModified=False,productionAccepted=False)
    payload={}
    for side,(ew,et) in eyes.items():
        globe=Slices(ew,et);md=meta['plan']['sides'][side];sign=md['outwardSign'];center=np.array(md['centerWorld']);sr={}
        for label,head in heads.items():
            rows,ends=contour(head,globe,ew);outward=sign*(rows[:,0]-center[0]);order=np.argsort(outward);rows=rows[order];outward=outward[order]
            outer=ends[1 if sign>0 else 0];inner=ends[0 if sign>0 else 1]
            d0=differential(outward,rows[:,1]);d1=differential(outward,rows[:,2]);payload[side+'_'+label+'_contour']=rows
            payload[side+'_'+label+'_lowerDerivatives']=d0;payload[side+'_'+label+'_upperDerivatives']=d1
            inward=sign*(outer['x']-rows[:,0]);domain=(inward>=.0012)&(inward<=.005)
            sr[label]=dict(frontVisibleWidthMeters=abs(outer['x']-inner['x']),outerVisibleGlobeEndpoint=outer,innerVisibleGlobeEndpoint=inner,
                outerEndpointZRelativeCenterMm=((outer['upperZ']+outer['lowerZ'])*.5-center[2])*1000,
                disconnectedColumns=int((rows[:,3]>1).sum()),transitionMetrics={})
            for name,dd,zi in [('lower',d0,1),('upper',d1,2)]:
                ii=np.flatnonzero(domain&np.isfinite(dd[:,2]));ext=ii[np.argsort(abs(dd[ii,2]))[-5:][::-1]]
                sr[label]['transitionMetrics'][name]=dict(slopeMin=float(dd[ii,0].min()),slopeMax=float(dd[ii,0].max()),
                    maximumAbsoluteCurvaturePerMm=float(abs(dd[ii,2]).max()),strongestCurvature=[dict(x=float(rows[k,0]),inwardFromVisibleEndpointMm=float(inward[k]*1000),
                    zRelativeEyeMm=float((rows[k,zi]-center[2])*1000),slope=float(dd[k,0]),curvaturePerMm=float(dd[k,2])) for k in ext])
        # Exact head-front continuation into the visible crease beyond the globe.
        z=center[2]+np.arange(-.006,.00601,.000025)
        absolute_x=np.array([.0425,.0428,.043,.04325,.0435,.04375,.044,.0445,.045,.0455])
        payload[side+'_headSectionZ']=z
        sectionrows=[]
        for label,head in heads.items():
            yy=np.asarray([depths(head.lines(sign*x),z) for x in absolute_x])
            payload[side+'_'+label+'_headFrontDepthSections']=yy
        payload[side+'_headSectionX']=sign*absolute_x
        sr['headContinuationXRangeMeters']=[float(sign*absolute_x[0]),float(sign*absolute_x[-1])]
        sr['transitionStartOptions']=[dict(lastOuterMillimetres=d,actualX=float(sr['preview']['outerVisibleGlobeEndpoint']['x']-sign*d*.001),
            outwardFromEyeCenterMm=float(sign*(sr['preview']['outerVisibleGlobeEndpoint']['x']-sign*d*.001-center[0])*1000)) for d in (6.,7.5,8.,9.)]
        report['sides'][side]=sr
        print('CONTOUR '+side+' '+json.dumps(sr),flush=True)
    np.savez_compressed(OUT.with_suffix('.npz'),**payload);report['contourArraysSha256']=sha(OUT.with_suffix('.npz'))
    OUT.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=js),encoding='utf8')
    print('DONE '+sha(OUT.with_suffix('.json')),flush=True)
if __name__=='__main__':run()

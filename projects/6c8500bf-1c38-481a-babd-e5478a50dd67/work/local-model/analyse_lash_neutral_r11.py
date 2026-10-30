"""Offline attribution of actual neutral eyelash growth and frontal background.

Consumes adult_lash_geometry_observe_r11 JSON/NPZ; does not import Blender,
edit images or modify models. Point tests are diagnostic, not shaft collision QA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return np.quantile(v,[0,.1,.5,.9,1]).tolist() if len(v) else None


class FrontSurface:
    """Exact projected triangle hit using a conservative XZ AABB cell index."""
    def __init__(self,vertices,triangles,cell=.001):
        self.tri=np.asarray(vertices,dtype=float)[triangles];self.cell=cell;self.grid={}
        self.a=self.tri[:,0];self.e=self.tri[:,1]-self.a;self.f=self.tri[:,2]-self.a
        self.d=self.e[:,0]*self.f[:,2]-self.e[:,2]*self.f[:,0]
        lo=np.floor(self.tri[:,:,[0,2]].min(1)/cell).astype(np.int32)
        hi=np.floor(self.tri[:,:,[0,2]].max(1)/cell).astype(np.int32)
        for i in np.flatnonzero(abs(self.d)>1e-15):
            for x in range(lo[i,0],hi[i,0]+1):
                for z in range(lo[i,1],hi[i,1]+1):self.grid.setdefault((x,z),[]).append(i)
    def depth(self,points):
        points=np.asarray(points);result=np.full(len(points),np.inf)
        for j,p in enumerate(points):
            key=tuple(np.floor(p[[0,2]]/self.cell).astype(int));ids=self.grid.get(key)
            if not ids:continue
            ids=np.asarray(ids,np.int32);q=p-self.a[ids];e=self.e[ids];f=self.f[ids];d=self.d[ids]
            u=(q[:,0]*f[:,2]-q[:,2]*f[:,0])/d;v=(e[:,0]*q[:,2]-e[:,2]*q[:,0])/d
            valid=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
            if valid.any():result[j]=(self.a[ids,1]+u*e[:,1]+v*f[:,1])[valid].min()
        return result


def analyse(observation_relative,output_relative):
    p=project_path().resolve();path=(p/observation_relative).resolve();out=(p/output_relative).resolve()
    for f in (path,out):f.relative_to(p/'work/reconstruction-r02')
    if out.exists():raise RuntimeError('Preserve prior offline analysis')
    observation=json.loads(path.read_text(encoding='utf8'));array_path=(p/observation['arrays']).resolve()
    if sha(array_path)!=observation['arraysSha256']:raise RuntimeError('Actual observation arrays changed')
    a=np.load(array_path);head=FrontSurface(a['head_world'],a['head_triangles'])
    eyes={s:FrontSurface(a['eye_'+s+'_world'],a['eye_'+s+'_triangles']) for s in ('L','R')}
    result={'observation':observation_relative,'observationSha256':sha(path),'arraysSha256':observation['arraysSha256'],
            'sourceSha256':observation['sourceSha256'],'helperSha256':sha(__file__),'lashes':{},
            'quantileOrder':[0,.1,.5,.9,1],'appearanceAccepted':False,
            'scope':'Actual control-point front-ray backgrounds, initial directions and endpoint spans. Does not classify interpolated shaft interiors or claim full collision clearance.'}
    for name,row in observation['lashes'].items():
        tag=row['arrayTag'];points=a[tag+'_world'];counts=a[tag+'_counts'];starts=np.r_[0,np.cumsum(counts)[:-1]]
        ends=np.cumsum(counts)-1;root=points[starts];tip=points[ends];initial=points[starts+1]-root
        initial/=np.linalg.norm(initial,axis=1)[:,None]
        triangles=a[tag+'_boundTrianglesWorld'];norm=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None]
        length=np.array([np.linalg.norm(np.diff(points[s:s+c],axis=0),axis=1).sum() for s,c in zip(starts,counts)])
        shaft=np.ones(len(points),bool);shaft[starts]=False
        matrix=a[tag+'_matrix'];scale=float(np.linalg.svd(matrix[:3,:3],compute_uv=False).max());radius=a[tag+'_radius']*scale
        depth=head.depth(points);gap=depth-points[:,1];surface=np.isfinite(depth)
        side=name[-1];eye_depth=eyes[side].depth(points)
        visible_globe=np.isfinite(eye_depth)&(eye_depth<depth-1e-7)
        skin_background=surface&~visible_globe
        exposed=~surface|(gap>radius);buried=surface&(gap<-radius);intersects=surface&~exposed&~buried
        delta=tip-root
        # Exact visible upper aperture at each root X from sampled real triangle
        # surfaces, restricted to the actual globe bounds, 25 micrometer Z step.
        eye_world=a['eye_'+side+'_world'];lo,hi=eye_world[:,2].min(),eye_world[:,2].max()
        margins=[]
        if ' upper ' in name:
            for r in root:
                zs=np.arange(lo,hi+.0000125,.000025);query=np.column_stack((np.full(len(zs),r[0]),np.zeros(len(zs)),zs))
                gd=eyes[side].depth(query);hd=head.depth(query);visible=np.isfinite(gd)&(gd<hd-1e-7)
                margins.append(float(r[2]-zs[visible].max()) if visible.any() else float('nan'))
        result['lashes'][name]={'curves':len(counts),'points':len(points),
            'polylineLengthMillimeterQuantiles':quantile(length*1000),
            'rootToTipForwardMillimeterQuantiles':quantile(-delta[:,1]*1000),
            'rootToTipVerticalMillimeterQuantiles':quantile(delta[:,2]*1000),
            'rootToTipScreenSpanMillimeterQuantiles':quantile(np.linalg.norm(delta[:,[0,2]],axis=1)*1000),
            'initialForwardComponentQuantiles':quantile(-initial[:,1]),'initialUpComponentQuantiles':quantile(initial[:,2]),
            'initialExitDotRootTriangleNormalQuantiles':quantile(np.sum(initial*norm,axis=1)),
            'upperRootVerticalDistanceAboveVisibleApertureMmQuantiles':quantile(np.asarray(margins)*1000),
            'upperRootApertureMeasurementGridStepMm':.025 if margins else None,
            'frontDepthGapToSkinMillimeterQuantiles':quantile(gap[surface]*1000),
            'nonRootControlPoints':int(shaft.sum()),'nonRootPointsClearlyInFrontOfSkin':int((shaft&exposed).sum()),
            'nonRootPointsClearlyBehindSkin':int((shaft&buried).sum()),'nonRootPointsWithinRadiusOfSkin':int((shaft&intersects).sum()),
            'nonRootPointsWithSkinBackground':int((shaft&skin_background).sum()),
            'nonRootPointsWithVisibleGlobeBackground':int((shaft&visible_globe).sum()),
            'rootFrontGapMillimeterQuantiles':quantile(gap[starts]*1000),
            'actualAttributes':row['actualAttributes'],'actualMaterials':row['actualMaterials']}
        print('R11_LASH_NEUTRAL_ANALYSIS '+name,flush=True)
    with out.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print('R11_LASH_NEUTRAL_ANALYSIS_COMPLETE '+str(out),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();analyse(args.observation,args.output)

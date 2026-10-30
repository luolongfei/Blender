"""Measured C2 profile fairing for one continuous lower-lip to chin surface.

The rejected v1 mixed two independently placed bumps. This candidate instead
matches position, tangent and curvature at the two ends of a complete actual
front profile. The actual inner mouth, opening halo and chin base stay fixed.
No whole-face generation, image calibration or Blender execution is performed.
"""
from pathlib import Path
import runpy
import numpy as np


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def state(data,prefix):return api('lower_face_continuity_surface_r11')['state'](data,prefix)


def quality(before,after,faces,active):return api('lower_face_continuity_surface_r11')['quality'](before,after,faces,active)


def plan(data,land):
    old=api('lower_face_continuity_surface_r11');smooth=old['smooth'];state=old['state']
    a=state(data,'state_0');b=state(data,'state_1');w=a['world'];faces=a['faces'];ids=a['originalVertexIds'];c=a['currentMixedCageWorld'];chin=c[7972]
    tri=old['triangles'](faces);n=old['normals'](w,faces)
    index=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex'](w,tri)
    mid_ids=np.asarray(land['mouth']['vermilionMidLoop']['sourceVertexIndices'],np.int32)
    if len(mid_ids)!=40 or mid_ids[20]!=7984:raise RuntimeError('Actual lower-vermilion mid-loop required')
    p=c[mid_ids[14:27]];coeff=np.polynomial.polynomial.polyfit((p[:,0]-chin[0])*1000,(p[:,2]-chin[2])*1000,4)
    xs=np.linspace(chin[0]-.014,chin[0]+.014,113);ts=np.linspace(0.,1.,161);z0=chin[2]+.0125
    top=chin[2]+np.polynomial.polynomial.polyval((xs-chin[0])*1000,coeff)/1000
    source_profiles=[];targets=[];curves=[]
    M=np.asarray([[1,1,1],[3,4,5],[6,12,20]],float)
    for x,z1 in zip(xs,top):
        if not .010<z1-z0<.026:raise RuntimeError('Observed lower-lip/chin profile span is unexpected')
        h=z1-z0;zs=z0+ts*h;points=np.c_[np.full(len(zs),x),np.zeros(len(zs)),zs]
        ys,_,_=index.front(points)
        if not np.isfinite(ys).all():raise RuntimeError('Actual anterior profile has a missing ray')
        def jet(z):
            sample=np.linspace(z-.0014,z+.0014,29);pts=np.c_[np.full(len(sample),x),np.zeros(len(sample)),sample]
            values,_,_=index.front(pts)
            if not np.isfinite(values).all():raise RuntimeError('Actual endpoint neighborhood has a missing ray')
            fitted=np.polynomial.polynomial.polyfit(sample-z,values,3)
            exact,_,_=index.front(np.array([[x,0,z]]))
            return float(exact[0]),float(fitted[1]),float(2*fitted[2])
        y0,m0,k0=jet(z0);y1,m1,k1=jet(z1)
        base=np.asarray([y0,h*m0,h*h*k0*.5])
        tail=np.linalg.solve(M,np.array([y1-base.sum(),h*m1-base[1]-2*base[2],h*h*k1-2*base[2]]))
        polynomial=np.r_[base,tail];target=np.polynomial.polynomial.polyval(ts,polynomial)
        source_profiles.append(ys);targets.append(target);curves.append(polynomial)
    source_profiles=np.asarray(source_profiles);targets=np.asarray(targets);delta_grid=targets-source_profiles
    # Gentle transverse fairing shares the correction between nearby measured
    # profiles without changing their endpoint value/tangent/curvature support.
    kernel=np.exp(-.5*(np.arange(-8,9)/3.)**2);kernel/=kernel.sum()
    padded=np.pad(delta_grid,((8,8),(0,0)),mode='edge');smoothed=np.zeros_like(delta_grid)
    for j,weight in enumerate(kernel):smoothed+=weight*padded[j:j+len(xs)]
    delta_grid=smoothed*smooth(ts[None,:]/.12)*smooth((1-ts[None,:])/.12)
    protected=np.zeros(len(w),bool);protected[np.unique(faces[np.isin(a['groups'],[2,3,4,5,7,8])])]=True
    opening_distance=old['line_distance'](w,a['openingCageWorld'])
    ztop=chin[2]+np.polynomial.polynomial.polyval((w[:,0]-chin[0])*1000,coeff)/1000
    t=(w[:,2]-z0)/(ztop-z0)
    protected|=(t<=0)|(t>=1)|(abs(w[:,0]-chin[0])>=.024)|(n[:,1]>=-.10)|(opening_distance<=.003)
    candidates=np.flatnonzero(~protected)
    front,_,_=index.front(w[candidates]);front_sheet=np.isfinite(front)&(abs(front-w[candidates,1])<=5e-6)
    protected[candidates[~front_sheet]]=True
    # The same topological points must avoid the actual opening in warm1 too.
    bmap=np.searchsorted(b['originalVertexIds'],ids);present=bmap<len(b['originalVertexIds']);present[present]&=b['originalVertexIds'][bmap[present]]==ids[present]
    protected|=~present
    bdistance=old['line_distance'](b['world'],b['openingCageWorld']);boral=np.zeros(len(b['world']),bool);boral[np.unique(b['faces'][np.isin(b['groups'],[2,3,4,5,7,8])])]=True
    protected[present]|=boral[bmap[present]]|(bdistance[bmap[present]]<=.003)
    amount=np.zeros(len(w));chosen=np.flatnonzero(~protected)
    # Bilinear sampling of a smooth, densely measured profile field; retained
    # geometry remains the existing L3 mesh with identical points and topology.
    gx=np.clip((w[chosen,0]-xs[0])/(xs[1]-xs[0]),0,len(xs)-1-1e-10);ix=np.floor(gx).astype(int);fx=gx-ix
    gt=np.clip(t[chosen]*(len(ts)-1),0,len(ts)-1-1e-10);it=np.floor(gt).astype(int);ft=gt-it
    values=(1-fx)*((1-ft)*delta_grid[ix,it]+ft*delta_grid[ix,it+1])+fx*((1-ft)*delta_grid[ix+1,it]+ft*delta_grid[ix+1,it+1])
    values*=1-smooth((abs(w[chosen,0]-chin[0])-.012)/.012)
    protection=smooth((opening_distance[chosen]-.003)/.002)*smooth((-n[chosen,1]-.10)/.55)
    protection*=smooth((bdistance[bmap[chosen]]-.003)/.002)
    amount[chosen]=.8*values*protection
    amount[abs(amount)<1e-9]=0.;offset=np.zeros_like(w);offset[:,1]=amount;active=amount!=0
    if not 500<int(active.sum())<14000 or abs(amount).max()>.0028:raise RuntimeError('C2 profile candidate support/amplitude is outside the measured scope')
    checks=[]
    for s in (a,b):
        match=np.searchsorted(ids,s['originalVertexIds']);found=match<len(ids);found[found]&=ids[match[found]]==s['originalVertexIds'][found]
        delta=np.zeros_like(s['world']);delta[found]=offset[match[found]]
        q=old['quality'](s['world'],s['world']+delta,s['faces'],np.any(delta!=0,axis=1));checks.append(q)
        if q['nonpositiveNormalCount'] or q['minimumNormalAlignment']<.75 or q['triangleAreaRatioRange'][0]<.55 or q['triangleAreaRatioRange'][1]>1.7:
            raise RuntimeError('Current two-state C2 surface candidate folds/strains: '+str(q))
    arrays={'originalVertexIds':ids,'neutralWorld':w,'faces':faces,'worldOffsets':offset.astype(np.float32),'protected':protected,
        'profileXsWorld':xs,'profileNormalizedHeights':ts,'profileBottomWorldZ':np.asarray(z0),'profileTopWorldZ':top,
        'profileCurrentWorldY':source_profiles,'profileTargetWorldY':targets,'profileAppliedWorldYDelta':delta_grid,
        'profilePolynomialCoefficients':np.asarray(curves)}
    report={'method':'Actual front profiles joined by a quintic matching endpoint position, tangent and curvature. Transverse 0.75mm fairing, with C2 zero boundary support. No independently placed filling/retraction bumps.',
        'affectedActualVertices':int(active.sum()),'minimumAnteriorOffsetMeters':float(amount.min()),'maximumPosteriorOffsetMeters':float(amount.max()),'measuredProfileBlend':.8,
        'chinBaseProtectedHeightMeters':.0125,'openingProtectedSpaceDistanceMeters':.003,'openingFadeMeters':.002,
        'xCoordinatesUnchanged':True,'zCoordinatesUnchanged':True,'actualInnerMouthFixed':True,
        'sourceStates':'Actual f501 neutral and warm1; existing c4dc layer must be disabled before this field is evaluated.',
        'offlineStateChecks':checks,'appearanceAccepted':False,
        'scope':'Central external lower-vermilion undersurface through the mental slope; existing lip opening and inner anatomy retained. A visible quarter A/B is required to judge continuity, volume and expression.'}
    return arrays,report

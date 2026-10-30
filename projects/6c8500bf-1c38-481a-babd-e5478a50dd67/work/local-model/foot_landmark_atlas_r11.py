"""Offline orthographic mesh atlas for selecting actual donor landmarks.

This is a numerical mesh plot, not a Blender render or generated image.
It preserves evaluated vertex IDs and records provisional surface landmarks.
No fitting, joint-centre certification or anatomy alteration is performed.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def unit(x):
    x=np.asarray(x,float);return x/np.linalg.norm(x)

def atlas():
    p=project_path();src=p/'work/reconstruction-r02/foot-asset-observation-r11.json'
    if sha(src)!='67d6afbc595cb757c533d66f53bd494e24b29f172356bda251fb1530b98b336c':raise RuntimeError('Donor capture changed')
    j=json.loads(src.read_text());npz=p/j['arrays']
    if sha(npz)!='202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564':raise RuntimeError('Captured arrays changed')
    a=np.load(npz);v=a['asset_0_evaluated_positions'].astype(float);t=a['asset_0_evaluated_triangles'];n=a['asset_0_evaluated_normals']
    raw=a['asset_0_raw_positions'];rt=a['asset_0_raw_triangles'];rpoly=a['asset_0_raw_trianglePolygon']
    # Candidate extrema within observed ankle levels. These are surface
    # prominences to inspect, explicitly not automatic functional pivots.
    candidates={'medial-malleolar-surface':3454,'lateral-malleolar-surface':5301,
        'heel-rear-surface':int(np.argmax(v[:,1])),'toe-tip-surface':int(np.argmin(v[:,1]))}
    regions={'heel-plantar':((-.012,.032),(-.005,.032)),
             'first-metatarsal-plantar':((-.027,-.007),(-.151,-.117)),
             'fifth-metatarsal-plantar':((.038,.055),(-.137,-.108))}
    region_records={}
    for label,((xmin,xmax),(ymin,ymax)) in regions.items():
        ids=np.flatnonzero((v[:,0]>=xmin)&(v[:,0]<=xmax)&(v[:,1]>=ymin)&(v[:,1]<=ymax)&(n[:,2]<-.5))
        if len(ids)<6:raise RuntimeError('Plantar candidate region too sparse')
        low=ids[v[ids,2]<=np.quantile(v[ids,2],.25)]
        selected=int(low[np.argmin(np.linalg.norm(v[low,:2]-v[low,:2].mean(0),axis=1))]);candidates[label]=selected
        region_records[label]={'evaluatedIds':low.tolist(),'centroidLocalMeters':v[low].mean(0).tolist(),
            'minimumZ':float(v[low,2].min()),'representativeEvaluatedId':selected}
    landmark_records={}
    for label,i in candidates.items():
        nearest=int(np.argmin(np.linalg.norm(raw-v[i],axis=1)))
        landmark_records[label]={'evaluatedId':i,'evaluatedLocalMeters':v[i].tolist(),'nearestRawVertexId':nearest,
            'nearestRawLocalMeters':raw[nearest].astype(float).tolist(),'nearestRawDistanceMeters':float(np.linalg.norm(raw[nearest]-v[i])),
            'meaning':'Provisional observed surface landmark, not joint centre or anatomical certification.'}
    size=600;canvas=Image.new('RGB',(size*3,size*2),(244,243,239));draw=ImageDraw.Draw(canvas)
    views=[('DORSAL',(0,0,1),(0,-1,0)),('MEDIAL',(-1,0,0),(0,0,1)),('LATERAL',(1,0,0),(0,0,1)),
        ('PLANTAR',(0,0,-1),(0,-1,0)),('ANKLE FRONT',(0,-1,0),(0,0,1)),('QUARTER',(-1,-1,.7),(0,0,1))]
    colors=[(192,59,47),(52,85,190),(129,65,157),(37,132,83),(170,111,28),(29,143,153),(189,54,121)]
    for panel,(name,eye,up) in enumerate(views):
        e=unit(eye);u=unit(np.asarray(up)-e*np.dot(up,e));r=unit(np.cross(u,e));basis=np.column_stack((r,u,e));q=v@basis
        lo=q[:,:2].min(0);hi=q[:,:2].max(0);center=(lo+hi)*.5;scale=min((size-100)/(hi-lo))
        xx=panel%3*size;yy=panel//3*size
        xy=(q[:,:2]-center)*scale;xy[:,1]*=-1;xy+=np.array([xx+size*.5,yy+size*.53])
        faces=v[t];normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        light=unit(e+u*.6+r*.2);intensity=np.clip(.36+.64*np.maximum(normals@light,0),0,1)
        for k in np.argsort(q[t,2].mean(1)):
            shade=intensity[k];color=tuple(int(x*shade) for x in (205,198,186))
            draw.polygon([tuple(z) for z in xy[t[k]]],fill=color)
        draw.text((xx+22,yy+15),name,fill=(20,20,20))
        for index,(label,i) in enumerate(candidates.items()):
            # Show annotations on all projections but visibly mark rear
            # landmarks with hollow circles instead of claiming visibility.
            x,y=xy[i];color=colors[index]
            front=n[i]@e>0
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color if front else None,outline=color,width=2)
            draw.text((x+6,y-5),str(i),fill=color)
        draw.text((xx+18,yy+size-23),'actual captured mesh / metres / no Blender render',fill=(75,75,75))
    output=p/'work/reconstruction-r02/foot-landmark-atlas-r11.png';report=output.with_suffix('.json')
    if output.exists() or report.exists():raise FileExistsError('Preserve existing atlas')
    canvas.save(output)
    record={'schemaVersion':1,'sourceCaptureSha256':sha(src),'sourceArraySha256':sha(npz),'helperSha256':sha(__file__),
        'landmarks':landmark_records,'plantarRegions':region_records,'atlas':str(output.relative_to(p)),'atlasSha256':sha(output),
        'landmarksVisuallyReviewed':False,'jointCentreCertified':False,'savedBlend':False,'geometryEdited':False,
        'scope':'Offline orthographic plot of actual captured donor triangles. Circle IDs are evaluated vertices; raw nearest IDs are correspondence hints, not exact subdivision inverse.'}
    report.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'atlas':str(output),'report':str(report),'sha256':sha(report)}))

if __name__=='__main__':atlas()

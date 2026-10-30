"""Offline triangle plot of one unapplied common shoe-map candidate.

Uses original captured raw shoe topology and the untouched placed donor.
It is intentionally labelled as a numerical plot, not a Blender render.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def plot(relative):
    p=project_path();path=(p/relative).resolve();path.relative_to(p.resolve());stage=p/'work/reconstruction-r02'
    record=json.loads(path.read_text());a=np.load(p/record['arrays'])
    if sha(p/record['arrays'])!=record['arraysSha256']:raise RuntimeError('Candidate array changed')
    source=stage/'foot-proportion-capture-r11.json'
    if sha(source)!=record['sourceCaptureSha256']:raise RuntimeError('Source raw topology changed')
    source_record=json.loads(source.read_text());c=np.load(p/source_record['arrays']);d=np.load(stage/'foot-asset-observation-r11.npz')
    placement_record=json.loads((stage/'foot-225-placement-plan-r11.json').read_text());placed=np.load(p/placement_record['arrays'])
    parts=[];types=[];objects={x['name']:x for x in source_record['objects']}
    for row in record['objects']:
        if row['side']!='L':continue
        pre=row['arrayPrefix'];m=np.asarray(objects[row['name']]['matrixWorld']);v=a[pre+'_targetRaw']@m[:3,:3].T+m[:3,3]
        loops=c[pre+'_raw_loops'];starts=c[pre+'_raw_polygonStarts'];sizes=c[pre+'_raw_polygonSizes'];triangles=[]
        for start,size in zip(starts,sizes):
            f=loops[start:start+size]
            for k in range(1,size-1):triangles.append((f[0],f[k],f[k+1]))
        parts.append(v[np.asarray(triangles)])
        color=(56,61,65) if any(x in row['name'] for x in ('Outsole','Heel','Welt')) else (158,156,151) if any(x in row['name'] for x in ('Eyelet','Aglet')) else (101,111,119)
        types.extend([color]*len(triangles))
    t=d['asset_0_evaluated_triangles'];parts.append(placed['donor_L_evaluatedWorld'][t]);types.extend([(206,187,161)]*len(t))
    triangles=np.concatenate(parts);types=np.asarray(types);canvas=Image.new('RGB',(1800,820),(242,240,236));draw=ImageDraw.Draw(canvas)
    for index,(title,eye,up) in enumerate([('MEDIAL',(-1,0,.15),(0,0,1)),('LATERAL',(1,0,.15),(0,0,1)),('QUARTER',(-1,-1,.7),(0,0,1))]):
        e=np.asarray(eye,float);e/=np.linalg.norm(e);u=np.asarray(up,float);u-=e*(u@e);u/=np.linalg.norm(u);r=np.cross(u,e)
        q=triangles@np.column_stack((r,u,e));center=(q[:,:,:2].max((0,1))+q[:,:,:2].min((0,1)))*.5
        scale=min(535/np.ptp(q[:,:,0]),700/np.ptp(q[:,:,1]));xy=(q[:,:,:2]-center)*scale;xy[:,:,1]*=-1;xy+=np.array([index*600+300,420])
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
        light=e+u*.6+r*.2;light/=np.linalg.norm(light);shade=.4+.6*np.maximum(normals@light,0)
        for k in np.argsort(q[:,:,2].mean(1)):draw.polygon([tuple(z) for z in xy[k]],fill=tuple((types[k]*shade[k]).astype(int)))
        draw.text((index*600+20,20),title+' / common shoe map / offline raw preview',fill='black')
        draw.text((index*600+20,790),'No native shell evaluation, sock fit or collision acceptance',fill='black')
    output=path.with_suffix('.png');report=path.with_name(path.stem+'-plot.json')
    if output.exists() or report.exists():raise FileExistsError('Preserve actual prior plot')
    canvas.save(output)
    report.write_text(json.dumps({'candidateSha256':sha(path),'candidateArraysSha256':record['arraysSha256'],
        'helperSha256':sha(__file__),'image':str(output.relative_to(p)),'imageSha256':sha(output),'executedBlender':False,
        'scope':'Offline plot of raw source shoe geometry through common map plus actual placed donor triangles; no material or modifier acceptance.'},indent=2)+'\n')
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',required=True);a=parser.parse_args();plot(a.candidate)

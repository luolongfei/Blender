"""Lightweight exact-array shape comparison, without Blender or cache changes."""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();W=P/'work/reconstruction-r02';HERE=Path(__file__).resolve().parent
obs=W/'tail-observation-r10.npz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(obs)!='9ddb5f6ffe1fcc640256782451fcd54b327785c62270cd9ea749d33f5ad2c879':raise RuntimeError('Observed arrays changed')
current=runpy.run_path(str(HERE/'tail_shape_adult_r11.py'));prior=runpy.run_path(str(HERE/'tail_shape_candidate_r10.py'))
with np.load(obs,allow_pickle=False) as data:
    old=[data[f'tail_{i}_world'].copy() for i in range(2)]
    garments=np.concatenate([data[f'garment_{i}_world'] for i in range(3)])
new=[];rows=[];r10=[]
for i,a in enumerate(old):
    points,row=current['design'](a,i);new.append(points);rows.append(row);r10.append(prior['design'](a)[0])
report={'schemaVersion':1,'observation':'work/reconstruction-r02/tail-observation-r10.npz','observationSha256':sha(obs),
    'module':'work/local-model/tail_shape_adult_r11.py','moduleSha256':sha(HERE/'tail_shape_adult_r11.py'),
    'scope':'Offline exact observed geometry only; not actual new-source Blender/GN/cloth/dynamic evaluation.',
    'changes':rows,'staticProposal':True,'actualBlenderExecution':False,'appearanceAccepted':False,
    'cacheConsequences':{
        'invalidated':['Original tail rest hashes and representative-guide map/base cache.',
            'All prior tail simulations: body performance, CMU walk, authored clips and standing warmup.',
            'Embedded native tail guide offsets, mapped playback and earlier parity/collision reports.',
            'Runtime tail card geometry/weights/bone rest placement and model exports as applicable.'],
        'retained':['Original native CURVES data with exact radii and mapping attributes, before the new copy is edited.',
            'Original embedded cache guide/action/node data as preserved history; external NPZ cache files untouched.',
            'Exact first three points of every tail strand, thus donor-to-tail joins and tie attachment reference.',
            'Body animation and cloth caches. They remain their existing data, with new combined hair/garment contact checks still required.'],
        'nextRequired':'Adopt static form only after actual images, then rebuild representative guides, collision padding and per-action simulation on the adopted head/neck/garment; re-embed and reopen-verify.'}}
(W/'tail-shape-adult-r11-offline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
image=Image.new('RGB',(1500,980),'#f5f5f2');draw=ImageDraw.Draw(image)
sets=[('Original actual arrays',old),('R10 unexecuted proposal',r10),('R11 shorter C study',new)]
for col,(title,arrays) in enumerate(sets):
    for view in range(2):
        ox=col*500;oy=view*465+45
        def project(a):
            a=np.asarray(a);h=a[...,0] if view==0 else a[...,1]
            mid=0 if view==0 else -.075
            return np.stack([ox+250+(h-mid)*1050,oy+410-(a[...,2]-1.09)*1320],axis=-1)
        draw.text((ox+18,oy-22),title+(' / front' if view==0 else ' / side'),fill='#202020')
        cloud=garments[(garments[:,2]>1.085)&(garments[:,2]<1.39)][::8]
        for x,y in project(cloud):
            if ox+5<x<ox+495 and oy<y<oy+425:draw.point((int(x),int(y)),fill='#d7dbdd')
        for side,a in enumerate(arrays):
            q=a.reshape(8,310,35,3)
            for group in range(8):
                for strand in range(0,310,31):
                    points=[tuple(p) for p in project(q[group,strand])]
                    draw.line(points,fill=('#57575b' if side==0 else '#85858a'),width=1)
            center=q.mean((0,1));draw.line([tuple(p) for p in project(center)],fill='#cf6c45',width=3)
            for x,y in project(center[:3]):draw.ellipse((x-2,y-2,x+2,y+2),fill='#28815f')
        draw.rectangle((ox+5,oy,ox+495,oy+425),outline='#d0d0ca')
draw.text((18,960),'Exact-array technical projections; gray points: observed garment. Not a Blender render or collision approval.',fill='#333333')
image.save(W/'tail-shape-adult-r11-projections.png')
print(json.dumps({'report':str(W/'tail-shape-adult-r11-offline.json'),'projection':str(W/'tail-shape-adult-r11-projections.png'),
    'moduleSha256':report['moduleSha256']},indent=2))

"""Orthographic geometry diagnostics from actual arrays, not a beauty render."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True);ap.add_argument('--output',required=True);ar=ap.parse_args()
 path=(P/ar.candidate).resolve();out=(P/ar.output).resolve();r=json.loads(path.read_text());a=np.load(P/r['arrays']);obs=json.loads((P/r['observation']).read_text());h=np.load(P/obs['arrays']);plan=json.loads((P/r['tiePlan']).read_text())
 if sha(P/r['arrays'])!=r['arraysSha256']:raise RuntimeError('Array hash mismatch')
 im=Image.new('RGB',(1800,1000),(247,246,242));d=ImageDraw.Draw(im);f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
 d.text((22,15),'ACTUAL ARRAY DIAGNOSTIC - same source before/after; not a Blender render',font=f,fill=(20,20,20))
 d.text((22,44),'Gray: head | Orange: original gather | Blue: proposed gather | Purple: preserved guide-13 face curtain | Green: new tail first 3 points',font=small,fill=(30,30,30))
 ids=a['gather_curve_ids'];s=a['source_starts'];l=a['source_lengths'];q=a['source_world'];v=a['target_world'];roles=a['source_group']%256
 # Deterministic root-space strata, plus explicit critical low-nape/ear curves.
 selected=[]
 for role in [40,41]:
  pool=ids[roles[ids]==role];root=q[s[pool]];order=np.lexsort((root[:,0],root[:,1],root[:,2]));selected.extend(pool[order[np.linspace(0,len(pool)-1,28).astype(int)]].tolist())
 selected=sorted(set(selected+[i for i in [2493,2150,17700,23528,25850,11600,24269] if i in ids]))
 panels=[('FRONT / X-Z',0,2,(-.12,.12,1.345,1.55),False),('SIDE / Y-Z',1,2,(-.095,.105,1.345,1.55),False),('TOP / X-Y',0,1,(-.12,.12,-.095,.105),True)]
 for z,(title,ax,ay,bounds,top) in enumerate(panels):
  left=z*600+30;right=(z+1)*600-25;upper=112;lower=880;x0,x1,y0,y1=bounds
  scale=min((right-left)/(x1-x0),(lower-upper)/(y1-y0));cx=(left+right)/2;cy=(upper+lower)/2
  def xy(p):return (cx+(float(p[ax])-(x0+x1)/2)*scale,cy-(float(p[ay])-(y0+y1)/2)*scale)
  d.text((left,82),title,font=f,fill=(25,25,25));d.rectangle((left,upper,right,lower),outline=(210,210,205))
  head=h['head_world'];mask=(head[:,2]>1.35)&(head[:,2]<1.55)
  for pt in head[mask][::18]:
   px,py=xy(pt)
   if left<px<right and upper<py<lower:d.point((px,py),fill=(205,205,200))
  for i in selected:
   st=int(s[i]);n=int(l[i]);d.line([xy(pt) for pt in q[st:st+n]],fill=(225,151,76),width=1);d.line([xy(pt) for pt in v[st:st+n]],fill=(25,97,170),width=2)
  for i in a['preserved_face_curtain_curve_ids'][::4]:
   st=int(s[i]);n=int(l[i]);d.line([xy(pt) for pt in v[st:st+n]],fill=(156,68,163),width=2)
  for side in ['positive','negative']:
   tr=plan['transforms'][side];tail=h['tail_'+side+'_world'].reshape(2480,35,3);tail=(tail-np.asarray(tr['originWorld']))@np.asarray(tr['rotationWorld']).T+np.asarray(tr['targetWorld'])
   for curve in tail[::70]:d.line([xy(pt) for pt in curve[:3]],fill=(35,136,73),width=1)
  d.text((left,898),f'{len(selected)} deterministic gather paths; full curves remain in NPZ',font=small,fill=(60,60,60))
 d.text((22,946),'Candidate: '+str(path.name)+' | status='+r['status']+' | visual acceptance remains pending actual Blender views',font=small,fill=(25,25,25))
 if out.exists():raise RuntimeError('Unused diagnostic output required')
 im.save(out);print(str(out))
if __name__=='__main__':main()

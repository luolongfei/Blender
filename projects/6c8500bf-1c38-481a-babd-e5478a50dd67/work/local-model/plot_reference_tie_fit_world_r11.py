"""Actual world-coordinate front/side diagnostic; no depth occlusion/render."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);ar=ap.parse_args();plan=P/ar.plan;out=P/ar.output;r=json.loads(plan.read_text());a=np.load(P/r['arrays']);oj=json.loads((P/r['observation']).read_text());ob=np.load(P/oj['arrays']);gj=json.loads((P/r['gatherPlan']).read_text());g=np.load(P/gj['arrays'])
 if out.exists():raise RuntimeError('Unused diagram output required')
 gather_world=g['target_world'];gather_starts=g['source_starts'];gather_lengths=g['source_lengths'];gather_ids=g['gather_curve_ids']
 if hashlib.sha256((P/r['arrays']).read_bytes()).hexdigest()!=r['arraysSha256']:raise RuntimeError('Frozen band arrays changed')
 im=Image.new('RGB',(1500,900),(247,246,242));d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
 d.text((22,15),'ACTUAL WORLD GEOMETRY - fixed pin/gather and proposed bands; occlusion disabled',font=font,fill=(20,20,20));d.text((22,48),'Gray: head/old ring | Orange: actual ears | Green: fixed tail pins | Blue: terminal gather | Black: candidate band',font=small,fill=(30,30,30))
 for panel,ax,xmin,xmax,name in [(0,0,-.12,.12,'FRONT / X-Z'),(1,1,-.10,.10,'SIDE / Y-Z')]:
  left=panel*750+25;right=(panel+1)*750-25;top=105;bottom=775;scale=min((right-left)/(xmax-xmin),(bottom-top)/.19);cx=(left+right)*.5;cy=(top+bottom)*.5
  def xy(p):return (cx+(float(p[ax])-(xmin+xmax)*.5)*scale,cy-(float(p[2])-1.445)*scale)
  d.text((left,82),name,font=font,fill=(20,20,20));d.rectangle((left,top,right,bottom),outline=(210,210,205))
  head=ob['head_world'];selected=head[(head[:,2]>1.35)&(head[:,2]<1.54)]
  for p in selected[::18]:
   x,y=xy(p)
   if left<x<right and top<y<bottom:d.point((x,y),fill=(205,205,200))
  for side in ['positive','negative']:
   for p in ob['ear_'+side+'_world'][::6]:
    x,y=xy(p);d.point((x,y),fill=(190,139,77))
   for curve in a['tail_'+side+'_first3_world'][::42]:d.line([xy(p) for p in curve],fill=(40,138,76),width=1)
  for curve in gather_ids[::70]:
   start=gather_starts[curve];count=gather_lengths[curve];points=gather_world[start+count-5:start+count];d.line([xy(p) for p in points],fill=(65,120,175),width=1)
  for ring in r['rings']:
   pre=ring['prefix'];old=a[pre+'_source_expected_after_transport_world'].reshape(32,8,3);new=a[pre+'_candidate_world'].reshape(32,8,3)
   for values,color,width in [(old,(165,165,165),1),(new,(15,15,15),2)]:
    for k in [0,2,4,6]:
     points=values[:,k];d.line([xy(p) for p in np.vstack([points,points[0]])],fill=color,width=width)
    for k in range(0,32,4):d.line([xy(p) for p in np.vstack([values[k],values[k,0]])],fill=color,width=1)
 d.text((22,810),'Bands: 4.4 mm axial width / 1.4 mm radial thickness; original axis planes and vertex topology retained.',font=small,fill=(30,30,30));d.text((22,838),'Diagram is a read-only array projection, not Blender appearance or native collision approval. Current saved-source reobservation remains required.',font=small,fill=(30,30,30));im.save(out);print(str(out))
if __name__=='__main__':main()

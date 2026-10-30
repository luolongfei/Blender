"""Offline technical projections and stage deltas from actual GN observations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();W=P/'work/reconstruction-r02'
STAGES=('main-raw','main-roll','main-interpolate','main-parting','main-big-clump','main-small-clump',
        'main-noise-subset','main-shrinkwrap-subset','main-main-joined','secondary-direct-evaluated','main-secondary-noise','main-full')


def main():
    record=json.loads((W/'donor-gn-stages-r11.json').read_text(encoding='utf8'));path=W/'donor-gn-stages-r11.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['arraysSha256']:raise RuntimeError('Actual stage arrays changed')
    z=np.load(path,allow_pickle=False);rows={r['prefix']:r for r in record['stages']};head=z['head_world'];hi=float(head[:,2].max())
    pixels_per_meter=1320.;panel_width=420;panel_height=420;image=Image.new('RGB',(panel_width*4,panel_height*3+55),'#f4f3ef');draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('arial.ttf',14)
    except OSError:font=ImageFont.load_default()
    for i,prefix in enumerate(STAGES):
        ox=(i%4)*panel_width;oy=(i//4)*panel_height
        def project(q):
            q=np.asarray(q);return np.stack([ox+panel_width*.5+q[...,0]*pixels_per_meter,oy+45+(hi+.02-q[...,2])*pixels_per_meter],axis=-1)
        cloud=head[(head[:,2]>hi-.25)][::18]
        for x,y in project(cloud):
            if ox+2<x<ox+panel_width-2 and oy+40<y<oy+panel_height-5:draw.point((int(x),int(y)),fill='#d4d6d6')
        q=z[prefix+'_world'];starts=z[prefix+'_starts'];lengths=z[prefix+'_lengths'];stride=max(1,int(np.ceil(len(starts)/900)))
        for k in range(0,len(starts),stride):
            st=int(starts[k]);n=int(lengths[k]);coords=project(q[st:st+n]);draw.line([tuple(v) for v in coords],fill='#292a2d',width=1)
            x,y=coords[0];draw.ellipse((x-1,y-1,x+1,y+1),fill='#49836b')
        r=rows[prefix];draw.rectangle((ox+2,oy+2,ox+panel_width-3,oy+panel_height-3),outline='#b6b6b1')
        draw.rectangle((ox+4,oy+4,ox+panel_width-5,oy+38),fill='#f4f3ef')
        draw.text((ox+12,oy+8),prefix,fill='#202020',font=font)
        draw.text((ox+12,oy+23),f'{r["curves"]} curves / {r["points"]} points; draw every {stride}',fill='#444444',font=font)
    draw.text((12,panel_height*3+10),'Actual exported GN geometry, same orthographic projection. Gray: current head; green: roots. Subset outputs labelled.',fill='#202020',font=font)
    draw.text((12,panel_height*3+29),'Technical path diagnostic only: uniform one-pixel strands and bounded sampling do not show final hair density, material or acceptance.',fill='#202020',font=font)
    image.save(W/'donor-gn-stage-projection-r11.png')
    comparisons=[]
    pairs=(('main-raw','main-roll'),('main-roll','main-profile'),('main-interpolate','main-parting'),
           ('main-trim','main-big-clump'),('main-big-clump','main-small-clump'),
           ('main-noise-subset','main-shrinkwrap-subset'),('secondary-raw','secondary-direct-evaluated'))
    for a,b in pairs:
        qa=z[a+'_world'];qb=z[b+'_world'];row={'before':a,'after':b}
        same=(qa.shape==qb.shape and np.array_equal(z[a+'_starts'],z[b+'_starts']) and np.array_equal(z[a+'_lengths'],z[b+'_lengths']))
        row['topologyMatches']=same
        if same:
            d=np.linalg.norm(qb-qa,axis=1);root=d[z[a+'_starts']]
            row.update(pointDisplacementMetersP50P95P99Max=np.percentile(d,[50,95,99,100]).tolist(),rootDisplacementMetersP50P95Max=np.percentile(root,[50,95,100]).tolist())
        comparisons.append(row)
    summary={'source':record['source'],'sourceSha256':record['sourceSha256'],'observationSha256':record['arraysSha256'],
             'stageComparisons':comparisons,'stageStatistics':record['stages'],'rendered':False,
             'scope':'Actual stage coordinates and explicitly sampled technical projections; no final appearance or physical-motion acceptance.'}
    (W/'donor-gn-stage-analysis-r11.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps({'comparison':comparisons,'projection':str(W/'donor-gn-stage-projection-r11.png')},indent=2),flush=True)
if __name__=='__main__':main()

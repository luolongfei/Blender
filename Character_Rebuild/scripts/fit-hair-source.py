"""Read-only GLB hair-envelope measurements and fitted lock specification.

Run from the workspace root with the configured stdlib Python.  No bpy and no
model writes.  The fitted cap uses the original dark-hair surface samples;
positions along corrupted fringe/tail boundaries are deliberately rebuilt.
"""
import json, struct, zlib, math, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.context import workspace_path


def read_input():
    b = workspace_path('character-hq-textured.glb').read_bytes()
    n = struct.unpack_from('<I', b, 12)[0]
    g = json.loads(b[20:20+n])
    binary = memoryview(b)[28+n:]
    def accessor(i):
        a=g['accessors'][i]; v=g['bufferViews'][a['bufferView']]
        off=v.get('byteOffset',0)+a.get('byteOffset',0)
        k={'VEC2':2,'VEC3':3,'SCALAR':1}[a['type']]
        fmt='f' if a['componentType']==5126 else 'I'
        ar=struct.unpack_from('<'+fmt*(a['count']*k),binary,off)
        return [ar[j:j+k] for j in range(0,len(ar),k)]
    p=g['meshes'][0]['primitives'][0]
    pos=accessor(p['attributes']['POSITION']); uv=accessor(p['attributes']['TEXCOORD_0'])
    vi=g['bufferViews'][g['images'][0]['bufferView']]
    png=bytes(binary[vi['byteOffset']:vi['byteOffset']+vi['byteLength']])
    w,h=struct.unpack_from('>II',png,16); ch={2:3,6:4}[png[25]]
    blocks=[]; q=8
    while q+12<=len(png):
        size=struct.unpack_from('>I',png,q)[0]; typ=png[q+4:q+8]
        if typ==b'IDAT': blocks.append(png[q+8:q+8+size])
        q+=12+size
    raw=zlib.decompress(b''.join(blocks)); stride=w*ch
    pixels=bytearray(w*h*ch)
    def paeth(a,b,c):
        p=a+b-c; pa,pb,pc=abs(p-a),abs(p-b),abs(p-c)
        return a if pa<=pb and pa<=pc else b if pb<=pc else c
    for y in range(h):
        f=raw[y*(stride+1)]; row=y*stride
        for x in range(stride):
            a=pixels[row+x-ch] if x>=ch else 0
            u=pixels[row-stride+x] if y else 0
            c=pixels[row-stride+x-ch] if y and x>=ch else 0
            v=raw[y*(stride+1)+1+x]
            if f==1:v+=a
            elif f==2:v+=u
            elif f==3:v+=(a+u)//2
            elif f==4:v+=paeth(a,u,c)
            pixels[row+x]=v&255
    points=[]
    for xyz,st in zip(pos,uv):
        x,y,z=xyz[0],-xyz[2],xyz[1]
        if z<.54 or abs(x)>.205:continue
        px=min(w-1,max(0,round(st[0]*(w-1))));py=min(h-1,max(0,round(st[1]*(h-1))))
        col=tuple(pixels[(py*w+px)*ch:(py*w+px)*ch+3])
        if max(col)<145 and sum(col)/3<105 and max(col)-min(col)<55:
            points.append((x,y,z,*col))
    return points


def main():
    pts=read_input()
    center=(0.0,.020,.855)
    crown=[p for p in pts if p[2]>.79 or (p[1]>.025 and p[2]>.735 and abs(p[0])<.12)]
    source=[]
    for p in crown:
        d=tuple(p[i]-center[i] for i in range(3)); r=math.sqrt(sum(v*v for v in d))
        if r>.225 or r<.08:continue
        source.append((tuple(v/r for v in d),r,p[:3]))
    around=64; rows=20
    cap=[]; measures=[]
    radii=(.151,.157,.139)
    for i in range(rows+1):
        row=[]
        for j in range(around):
            theta=2*math.pi*j/around
            rear=(1-math.cos(theta))/2
            # Forehead terminates above eyes; back gathers behind ears.
            phi_max=1.48+.95*rear
            phi=.025+(phi_max-.025)*i/rows
            d=(math.sin(phi)*math.sin(theta),-math.sin(phi)*math.cos(theta),math.cos(phi))
            nearest=sorted(((sum(a*b for a,b in zip(d,s[0])),s[1],s[2]) for s in source),reverse=True)[:12]
            valid=[v for v in nearest if v[0]>.988]
            prior=1/math.sqrt(sum((d[k]/radii[k])**2 for k in range(3)))
            if valid:
                r=statistics.median(v[1] for v in valid)
                # Robust envelope: at most 12% drift from measured axis bound.
                r=min(prior*1.12,max(prior*.88,r))
                # Envelope regularisation removes reconstruction ripples,
                # not arbitrary extra volume.
                r=.9*r+.1*prior
            else:r=prior
            row.append(tuple(center[k]+d[k]*r for k in range(3)))
            if j%8==0 and i%4==0:
                measures.append(dict(row=i,azimuth=j,position=row[-1],samples=len(valid),
                                     nearestSource=[list(v[2]) for v in nearest[:3]],
                                     nearestCosine=nearest[0][0],fittedRadius=r))
        cap.append(row)
    # Two mild circular smoothing passes preserve measured broad volume while
    # eliminating unrepeatable ripple at sparse angular sampling boundaries.
    for _ in range(2):
        old=cap
        cap=[[(tuple((old[i][(j-1)%around][k]+6*old[i][j][k]+old[i][(j+1)%around][k])/8 for k in range(3)))
              for j in range(around)] for i in range(rows+1)]
    result={
      'coordinateSpace':'Unscaled Blender import: x=glTF X, y=-glTF Z, z=glTF Y',
      'source':'character-hq-textured.glb',
      'sourceSha256':'105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2',
      'status':'fitted numerical candidate; root must render and inspect before installation',
      'sourceHairSampleCount':len(pts),'capSourceSampleCount':len(source),
      'cap':{'center':center,'around':around,'rows':rows+1,'surface':cap,'thickness':.003},
      'measurements':measures,
      'landmarks':{'top':[0,.020,.994],'tieLeft':[-.120,.100,.744],'tieRight':[.120,.100,.744],
                    'tailTipLeft':[-.136,.118,.565],'tailTipRight':[.136,.118,.565]},
      'locks':[],'tails':[],
      'notes':['Cap data fits filtered source hair samples, regularised to remove ripples.',
               'Hair sample classifier excludes strongly chromatic skin, but is not semantic segmentation.',
               'Fringe/temple/tip paths are manual reconstruction aligned to actual source and original 7224, not claimed scans.',
               'Original hair must be removed by integrator before use; this module never edits input mesh.',
               'Broad source envelope is preserved; exact new face clearance needs integration check.']}
    locks=result['locks']
    # Main fringe follows source crown/forehead, with separated tapered tips
    # instead of the source's jagged eye-line hem.
    bang_data=[
      ('Fringe_CentreLeft',[(0,-.005,.990),(-.025,-.078,.965),(-.037,-.133,.919),(-.032,-.148,.866),(-.015,-.141,.821)],
       [.004,.014,.022,.021,.0007],[.001,.004,.008,.007,.0006]),
      ('Fringe_Left', [(-.012,.005,.988),(-.060,-.069,.956),(-.084,-.120,.908),(-.089,-.133,.855),(-.073,-.126,.817)],
       [.004,.023,.026,.023,.0007],[.001,.005,.009,.007,.0006]),
      ('Fringe_CentreRight',[(.003,.006,.989),(.026,-.073,.964),(.031,-.125,.925),(.046,-.143,.874),(.066,-.133,.831)],
       [.004,.012,.017,.018,.0007],[.001,.004,.006,.006,.0006]),
      ('Fringe_Right',[(.006,.011,.989),(.060,-.054,.958),(.087,-.105,.917),(.101,-.120,.868),(.106,-.105,.820)],
       [.004,.022,.027,.024,.0007],[.001,.005,.009,.007,.0006]),
      ('Fringe_LeftFine',[(-.035,-.052,.968),(-.056,-.115,.925),(-.061,-.150,.871),(-.048,-.139,.826)],
       [.004,.009,.008,.0005],[.001,.003,.003,.0004]),
      ('Fringe_RightFine',[(.023,-.038,.975),(.046,-.111,.929),(.060,-.149,.885),(.083,-.133,.838)],
       [.004,.008,.008,.0005],[.001,.003,.003,.0004])]
    for name,points,widths,depths in bang_data:
        locks.append(dict(name=name,points=points,widths=widths,depths=depths,outward=[0,-1,0],rings=37,sides=12))
    for sign,label in [(-1,'Left'),(1,'Right')]:
        locks.append(dict(name=f'Temple_{label}',points=[(sign*.094,-.043,.920),(sign*.122,-.073,.860),
                     (sign*.122,-.082,.792),(sign*.113,-.069,.720),(sign*.097,-.045,.683)],
                     widths=[.008,.014,.013,.009,.0006],depths=[.003,.0045,.0045,.0035,.0005],outward=[sign*.5,-1,0],rings=41,sides=12))
        spine=[(sign*.120,.100,.744),(sign*.140,.124,.704),(sign*.151,.134,.656),
               (sign*.155,.132,.605),(sign*.139,.116,.561)]
        result['tails'].append(dict(name=f'Ponytail_{label}',spine=spine,widths=[.022,.040,.043,.035,.0012],
                                  depths=[.019,.029,.032,.027,.001],front=[0,-1,0],lock_count=5))
    out=workspace_path('Character_Rebuild','reports','hair-spec.json');out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'out':str(out),'sourceHairSamples':len(pts),'capSourceSamples':len(source),
                      'capVertices':sum(map(len,cap)),'locks':len(locks),'tails':len(result['tails'])}))

if __name__=='__main__':main()

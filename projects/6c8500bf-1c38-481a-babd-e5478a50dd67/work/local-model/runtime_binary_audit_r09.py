"""Read-only exported GLB audit: counts, tail bind/rest and posterior coverage.

Uses NumPy/Pillow on the actual container; never imports/launches Blender.
Posterior raster is a bounded numerical diagnostic, not a rendered acceptance.
"""
import hashlib,io,json,struct,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def read(path,binary=True):
    with Path(path).open('rb') as stream:
        magic,version,total=struct.unpack('<III',stream.read(12))
        if (magic,version)!=(0x46546c67,2):raise ValueError('Expected GLB2')
        size,kind=struct.unpack('<II',stream.read(8));document=json.loads(stream.read(size))
        if not binary:return document,None
        size,kind=struct.unpack('<II',stream.read(8));return document,stream.read(size)


def accessor(document,binary,index):
    a=document['accessors'][index]
    if a.get('sparse'):raise ValueError('Explicit sparse decode needed')
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dtype=np.dtype({5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
    if 'bufferView' not in a:return np.zeros((a['count'],width),dtype=dtype)
    view=document['bufferViews'][a['bufferView']];offset=view.get('byteOffset',0)+a.get('byteOffset',0)
    values=np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=offset,strides=(view.get('byteStride',width*dtype.itemsize),dtype.itemsize)).copy()
    if a.get('normalized') and dtype.kind in 'iu':values=np.maximum(values.astype(float)/np.iinfo(dtype).max,-1)
    return values


def world_matrices(document):
    nodes=document['nodes'];parents={child:i for i,n in enumerate(nodes) for child in n.get('children',[])};cache={}
    def world(index):
        if index in cache:return cache[index]
        n=nodes[index]
        if 'matrix' in n:local=np.asarray(n['matrix']).reshape(4,4).T
        else:
            x,y,z,w=n.get('rotation',[0,0,0,1]);local=np.eye(4)
            local[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@np.diag(n.get('scale',[1,1,1]))
            local[:3,3]=n.get('translation',[0,0,0])
        cache[index]=(world(parents[index])@local) if index in parents else local
        return cache[index]
    for index in range(len(nodes)):world(index)
    return cache,parents


def counts(document):
    result=[]
    for index,node in enumerate(document['nodes']):
        if 'mesh' not in node:continue
        primitives=document['meshes'][node['mesh']]['primitives'];triangles=0
        for p in primitives:
            if p.get('mode',4)!=4:raise ValueError('Expected triangles')
            triangles+=document['accessors'][p.get('indices',p['attributes']['POSITION'])]['count']//3
        result.append({'name':node['name'],'node':index,'triangles':triangles,'primitives':len(primitives)})
    return sorted(result,key=lambda r:r['triangles'],reverse=True)


def tail_rest(document,binary,world,parents):
    rows=[]
    for index,node in enumerate(document['nodes']):
        if not node.get('name','').startswith('RT06 Cards / R02 Strands / R02 Twin tail '):continue
        side=node['name'][-1];skin=document['skins'][node['skin']];joints=skin['joints']
        inverse=accessor(document,binary,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        transforms=np.asarray([np.linalg.inv(world[index])@world[j]@inverse[k] for k,j in enumerate(joints)])
        allowed={'head'}|{f'RT06 tail_{side}_{i:02d}' for i in range(6)}
        used=set();errors=[];sum_errors=[];rest_errors=[];head_weights=[]
        for p in document['meshes'][node['mesh']]['primitives']:
            pos=accessor(document,binary,p['attributes']['POSITION']);ji=accessor(document,binary,p['attributes']['JOINTS_0']).astype(int)
            weights=accessor(document,binary,p['attributes']['WEIGHTS_0']);h=np.column_stack([pos,np.ones(len(pos))]);deformed=np.zeros_like(h)
            for slot in range(4):deformed+=weights[:,slot,None]*np.einsum('nij,nj->ni',transforms[ji[:,slot]],h)
            rest_errors.extend(np.linalg.norm(deformed[:,:3]-pos,axis=1).tolist());sum_errors.extend(np.abs(weights.sum(1)-1).tolist())
            for k in np.unique(ji[weights>1e-7]):used.add(document['nodes'][joints[k]]['name'])
            head_id=next(k for k,j in enumerate(joints) if document['nodes'][j]['name']=='head')
            head_weights.extend(np.sum(np.where(ji==head_id,weights,0),axis=1).tolist())
        chain=[next(i for i,n in enumerate(document['nodes']) if n.get('name')==f'RT06 tail_{side}_{k:02d}') for k in range(6)]
        if parents[chain[0]]!=next(i for i,n in enumerate(document['nodes']) if n.get('name')=='head'):errors.append('root parent differs')
        if any(parents[b]!=a for a,b in zip(chain,chain[1:])):errors.append('chain parent differs')
        rows.append({'mesh':node['name'],'weightedBones':sorted(used),'unexpectedWeightedBones':sorted(used-allowed),
            'weightSumMaximumError':max(sum_errors),'restSkinnedPositionMaximumErrorMeters':max(rest_errors),
            'fullyHeadAnchoredExportedVertices':sum(w>.999999 for w in head_weights),
            'chainRestWorldHeads':[world[i][:3,3].tolist() for i in chain],'chainParentErrors':errors})
    return rows


def alpha_image(document,binary,primitive,cache):
    material=document['materials'][primitive['material']];info=material.get('pbrMetallicRoughness',{}).get('baseColorTexture')
    if not info:return None
    image_id=document['textures'][info['index']]['source']
    if image_id not in cache:
        image=document['images'][image_id];view=document['bufferViews'][image['bufferView']]
        raw=binary[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']]
        cache[image_id]=np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'))[:,:,3]/255.
    return cache[image_id]


def posterior_coverage(document,binary,world,grid=160):
    lookup={n.get('name'):i for i,n in enumerate(document['nodes'])};root_name='R06 attached scalp root substrate'
    root_index=lookup[root_name];root_node=document['nodes'][root_index];root_primitive=document['meshes'][root_node['mesh']]['primitives'][0]
    root_pos=accessor(document,binary,root_primitive['attributes']['POSITION']);root_pos=root_pos@world[root_index][:3,:3].T+world[root_index][:3,3]
    low=root_pos[:,:2].min(0);high=root_pos[:,:2].max(0);span=high-low;image_cache={}
    def raster(names,alpha_cutoff):
        depth=np.full((grid,grid),np.inf);stats=[]
        for name in names:
            index=lookup[name];node=document['nodes'][index];covered=0
            for p in document['meshes'][node['mesh']]['primitives']:
                pos=accessor(document,binary,p['attributes']['POSITION']);pos=pos@world[index][:3,:3].T+world[index][:3,3]
                uv=accessor(document,binary,p['attributes']['TEXCOORD_0']);alpha=alpha_image(document,binary,p,image_cache)
                triangles=accessor(document,binary,p['indices']).reshape(-1,3).astype(int)
                for tri in triangles:
                    xyz=pos[tri];q=(xyz[:,:2]-low)/span;a,b,c=q;ab=b-a;ac=c-a;den=ab[0]*ac[1]-ab[1]*ac[0]
                    if abs(den)<1e-14:continue
                    lo=np.maximum(0,np.ceil(q.min(0)*grid-.5).astype(int));hi=np.minimum(grid-1,np.floor(q.max(0)*grid-.5).astype(int))
                    if np.any(lo>hi):continue
                    x=(np.arange(lo[0],hi[0]+1)+.5)/grid-a[0];y=(np.arange(lo[1],hi[1]+1)[:,None]+.5)/grid-a[1]
                    u=(x*ac[1]-y*ac[0])/den;v=(ab[0]*y-ab[1]*x)/den;inside=(u>=0)&(v>=0)&(u+v<=1)
                    coords=uv[tri[0]]+(uv[tri[1]]-uv[tri[0]])*u[:,:,None]+(uv[tri[2]]-uv[tri[0]])*v[:,:,None]
                    if alpha is not None:
                        iy=np.clip((coords[:,:,1]*alpha.shape[0]).astype(int),0,alpha.shape[0]-1);ix=np.clip((coords[:,:,0]*alpha.shape[1]).astype(int),0,alpha.shape[1]-1)
                        inside &= alpha[iy,ix]>alpha_cutoff
                    z=xyz[0,2]+(xyz[1,2]-xyz[0,2])*u+(xyz[2,2]-xyz[0,2])*v
                    region=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];np.minimum(region,np.where(inside,z,np.inf),out=region);covered+=int(inside.sum())
            stats.append({'name':name,'coveredTrianglePixelHits':covered})
        return depth,stats
    card_names=[n for n in lookup if n and n.startswith('RT06 Cards / ') and 'Twin tail' not in n]
    shell,_=raster([root_name],.1);cards,stats=raster(card_names,.1);mask=np.isfinite(shell)
    any_card=mask&np.isfinite(cards);in_front=any_card&(cards<shell-1e-5);behind=any_card&~in_front
    return {'grid':grid,'projectedRootSamples':int(mask.sum()),'cardCoverageAnywhereInProjection':int(any_card.sum()),
        'cardSamplesInFrontOfRoot':int(in_front.sum()),'cardSamplesAtOrBehindRoot':int(behind.sum()),
        'visibleCardFractionOfProjectedRoot':float(in_front.sum()/max(1,mask.sum())),
        'cardObjects':stats,'camera':'Orthographic -Z, looking toward +Z; source rig rest pose',
        'scope':'Pixel-center triangle depth and nearest alpha >0.1. Does not reproduce filtering, translucency composition, highlights, shadows or viewport pose. At/behind includes front-head cards naturally behind the posterior shell; this figure alone cannot establish embedded posterior cards.'}


def run():
    project=project_path();file=project/'work/runtime-r06/preview-export-r08/review-preview.vrm'
    d,binary=read(file);world,parents=world_matrices(d);new=counts(d);old,_=read(project/'exports/avatar-preview.vrm',False);previous=counts(old)
    source=json.loads((file.parent/'export-report.json').read_text(encoding='utf-8'))['source'];names={r['name'] for r in source['meshes']}
    report={'file':str(file.relative_to(project)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'triangles':sum(r['triangles'] for r in new),'meshNodes':len(new),'renderPrimitives':sum(r['primitives'] for r in new),
        'top10':new[:10],'allMeshes':new,'unexpectedNodes':sorted({r['name'] for r in new}-names),
        'oldTriangles':sum(r['triangles'] for r in previous),'oldMeshNodes':len(previous),
        'tailRestSkin':tail_rest(d,binary,world,parents),'posteriorCoverage':posterior_coverage(d,binary,world)}
    path=project/'work/runtime-r06/r09-binary-audit.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'allMeshes','top10'}},ensure_ascii=False));return report


if __name__=='__main__':run()

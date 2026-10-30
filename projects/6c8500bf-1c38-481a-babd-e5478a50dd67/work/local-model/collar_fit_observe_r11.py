"""Read-only reproduction of the failed collar target before any shape key.

Exports original raw garment cages, the actual radial/projection target and
the folded triangles, plus finite actual skin meshes. No fitting is saved.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def observe(source_relative,output_relative):
    import bpy
    a=runpy.run_path(str(Path(__file__).with_name('collar_fit_r11.py')))
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();npz=output.with_suffix('.npz')
    for p in (source,output):p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if source.suffix!='.blend' or output.suffix!='.json' or output.exists() or npz.exists():raise RuntimeError('Explicit source and unused observation output required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent source required')
    targets=[bpy.data.objects[n] for n in (a['BODICE'],)+a['COLLARS']+a['SEAMS']]
    before={o.name:a['_stable_contract'](o) for o in targets};uv=a['_api']('production_uv_r08');pose=uv['_pose_contract']();source_sha=a['_sha'](source)
    rig=bpy.data.objects['Character_Rig'];old_pose=rig.data.pose_position;arrays={};report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':a['_sha'](__file__),'failedHelperSha256':a['_sha'](Path(__file__).with_name('collar_fit_r11.py')),'objects':{}}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();skin=a['_skin_trees']()
        points={o.name:a['_cage_world'](o) for o in targets};tri={o.name:a['_triangles'](o.data) for o in targets}
        bodice=np.array([a['_radial_clear'](p,skin) for p in points[a['BODICE']]])
        tree=a['_bvh'](bodice,tri[a['BODICE']]);targets_coords={a['BODICE']:bodice}
        for name,seam in zip(a['COLLARS'],a['SEAMS']):
            original=points[name];radial=np.array([a['_radial_clear'](p,skin) for p in original]);projected,details=a['_project_collar'](radial,original,tri[name],tree,skin)
            targets_coords[name]=projected;targets_coords[seam]=a['_transport'](points[seam],original,projected,tri[name])
        for index,obj in enumerate(targets):
            name=obj.name;prefix='garment_'+str(index);original=points[name];target=targets_coords[name];t=tri[name]
            n0=np.cross(original[t[:,1]]-original[t[:,0]],original[t[:,2]]-original[t[:,0]])
            n1=np.cross(target[t[:,1]]-target[t[:,0]],target[t[:,2]]-target[t[:,0]])
            denom=np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1);valid=denom>1e-20
            cos=np.ones(len(t));cos[valid]=np.sum(n0[valid]*n1[valid],axis=1)/denom[valid]
            for suffix,data in [('original',original),('failed_target',target),('triangles',t),('normal_cosine',cos)]:arrays[prefix+'_'+suffix]=data
            report['objects'][name]={'prefix':prefix,'vertices':len(original),'triangles':len(t),'foldedTriangleIds':np.flatnonzero(cos<=.02).tolist(),
                'minimumNormalCosine':float(cos.min()),'maximumDisplacementMeters':float(np.linalg.norm(target-original,axis=1).max())}
        dg=bpy.context.evaluated_depsgraph_get()
        for index,name in enumerate(a['SKIN']):
            obj=bpy.data.objects[name];ev=obj.evaluated_get(dg);mesh=ev.to_mesh()
            try:
                v=a['_world'](ev,a['_array'](mesh.vertices).astype(float));t=a['_triangles'](mesh);tv=v[t]
                keep=(tv.max(1)[:,2]>=1.18)&(tv.min(1)[:,2]<=1.325)&(np.abs(tv[:,:,0]).min(1)<.18)
                t=t[keep];ids=np.unique(t);mapping=np.full(len(v),-1,np.int32);mapping[ids]=np.arange(len(ids))
                arrays['skin_'+str(index)+'_world']=v[ids];arrays['skin_'+str(index)+'_triangles']=mapping[t]
            finally:ev.to_mesh_clear()
    finally:
        rig.data.pose_position=old_pose;bpy.context.view_layer.update()
    if {o.name:a['_stable_contract'](o) for o in targets}!=before or uv['_pose_contract']()!=pose or a['_sha'](source)!=source_sha:raise RuntimeError('Original source/garments/pose changed')
    np.savez_compressed(npz,**arrays);report.update(arrays=str(npz.relative_to(project_path())),arraysSha256=a['_sha'](npz),savedBlend=False,sourceContractsPreserved=True,
        interpretation='Exact failed target diagnosis, not an applied garment change or appearance acceptance.')
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_FAILED_COLLAR_OBSERVED '+json.dumps({'output':str(output.relative_to(project_path())),'objects':report['objects']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);observe(a.source,a.output)

"""Observe actual source and evaluated standing tails; never edit/save a blend."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def values(data,prop):
    a=np.empty(len(data)*3,np.float32);data.foreach_get(prop,a);return a.reshape(-1,3)
def world(a,m):return a.astype(float)@m[:3,:3].T+m[:3,3]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=(P/args.source).resolve();out=(P/args.output).resolve()
    source.relative_to((P/'work/reconstruction-r02').resolve());out.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent source required')
    if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('Keep previous observations')
    before=sha(source); rig,parity=api('final_detail_review_r09')['_matched_standing'](bpy)
    graph=bpy.context.evaluated_depsgraph_get();arrays={};rows={}
    for i,name in enumerate(('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')):
        o=bpy.data.objects[name];e=o.evaluated_get(graph);prefix='tail_'+str(i)
        raw=values(o.data.position_data,'vector');ev=values(e.data.position_data,'vector')
        if raw.shape!=ev.shape or raw.shape!=(86800,3):raise RuntimeError('Tail topology changed')
        mat=np.asarray(o.matrix_world,dtype=float);emat=np.asarray(e.matrix_world,dtype=float)
        arrays[prefix+'_local']=raw;arrays[prefix+'_world']=world(raw,mat)
        arrays[prefix+'_evaluated_local']=ev;arrays[prefix+'_evaluated_world']=world(ev,emat)
        arrays[prefix+'_matrix']=mat
        radii=np.empty(len(raw),np.float32);o.data.attributes['radius'].data.foreach_get('value',radii)
        arrays[prefix+'_radii']=radii
        rows[name]={'prefix':prefix,'data':o.data.name,'curves':len(o.data.curves),'points':len(raw),
            'dataSha256':api('posterior_groom_r09')['_data_hash'](o.data),
            'parent':o.parent.name,'parentBone':o.parent_bone,
            'modifiers':[{'name':m.name,'type':m.type,'viewport':m.show_viewport,'render':m.show_render} for m in o.modifiers]}
    for i,name in enumerate(('Blouse_Tucked_Bodice','Blouse_Puff_Sleeve_L','Blouse_Puff_Sleeve_R')):
        o=bpy.data.objects[name];e=o.evaluated_get(graph);mesh=e.to_mesh()
        try:
            mesh.calc_loop_triangles();arrays['garment_'+str(i)+'_world']=world(values(mesh.vertices,'co'),np.asarray(e.matrix_world))
            arrays['garment_'+str(i)+'_triangles']=np.asarray([list(t.vertices) for t in mesh.loop_triangles],np.int32)
        finally:e.to_mesh_clear()
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    result={'source':args.source,'sourceSha256':before,'objects':rows,'standingParity':parity,
        'arrayFile':out.with_suffix('.npz').name,'arraySha256':sha(out.with_suffix('.npz')),
        'scope':'Actual authored and evaluated frame-one tail geometry with current garment surfaces. No saved-source mutation or new dynamics.'}
    if sha(source)!=before:raise RuntimeError('Source disk changed during observation')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if 'R10_hbm_hand_provenance' in bpy.data.objects['Hand L · continuous anatomy']:
        hand=api('hand_asset_fit_r10')
        for side in ('L','R'):
            print('HAND_CAMERA_DIAGNOSTIC='+json.dumps(hand['hand_camera'](None,rig,side,None,diagnostics_only=True)))
    print('TAIL_OBSERVATION='+json.dumps({'path':args.output,'sha256':result['arraySha256']}))
if __name__=='__main__':main()

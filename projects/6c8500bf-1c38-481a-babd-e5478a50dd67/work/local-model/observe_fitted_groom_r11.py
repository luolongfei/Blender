"""Read-only actual REST head and fitted-guide export for the restyle repair."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=(P/a.source).resolve();out=(P/a.output).resolve();arrays=out.with_suffix('.npz')
    for path in (source,out,arrays):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or out.exists() or arrays.exists():
        raise RuntimeError('Explicit source and unused output required')
    original_sha=sha(source)
    fit=runpy.run_path(str(Path(__file__).with_name('donor_groom_adult_r11.py')))
    coords=fit['_coords'];world=fit['_world'];rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    data={};rows=[]
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        h=bpy.data.objects['R11 Female head study'];e=h.evaluated_get(bpy.context.evaluated_depsgraph_get())
        e.data.calc_loop_triangles()
        data['head_world']=world(coords(e),e.matrix_world)
        data['head_triangles']=np.array([tuple(t.vertices) for t in e.data.loop_triangles],np.int32)
        data['head_normals']=np.array([list(v.normal) for v in e.data.vertices],np.float32)
        for i,name in enumerate(('R11 Donor / long hair main','R11 Donor / long hair strands')):
            obj=bpy.data.objects[name];data[f'guide_{i}_world']=world(coords(obj),obj.matrix_world)
            data[f'guide_{i}_local']=coords(obj)
            data[f'guide_{i}_starts']=np.array([c.first_point_index for c in obj.data.curves],np.int32)
            data[f'guide_{i}_lengths']=np.array([c.points_length for c in obj.data.curves],np.int32)
            rows.append({'name':name,'curves':len(obj.data.curves),'points':len(obj.data.position_data)})
        for i,name in enumerate(('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')):
            obj=bpy.data.objects[name];q=world(coords(obj),obj.matrix_world)
            starts=np.array([c.first_point_index for c in obj.data.curves],np.int32)
            data[f'tail_{i}_pins']=q[starts];d=q[starts+1]-q[starts]
            data[f'tail_{i}_directions']=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
        for side in ('L','R'):
            obj=bpy.data.objects['R11 Adult sclera '+side];q=world(coords(obj),obj.matrix_world)
            data[f'eye_{side}_center']=(q.min(0)+q.max(0))*.5
            data[f'eye_{side}_radius']=np.array((q.max(0)-q.min(0)).max()*.5)
        keys={k.name:float(k.value) for k in h.data.shape_keys.key_blocks}
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    if sha(source)!=original_sha:raise RuntimeError('Input file changed')
    with arrays.open('xb') as f:np.savez_compressed(f,**data)
    report={'source':a.source,'sourceSha256':original_sha,'arrays':str(arrays.relative_to(P)),
            'arraysSha256':sha(arrays),'objects':rows,'headVertices':len(data['head_world']),
            'headTriangles':len(data['head_triangles']),'keys':keys,'savedBlend':False,
            'scope':'Actual REST geometry, no scene-file writes or hairstyle acceptance.'}
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_FITTED_GROOM_OBSERVED '+json.dumps({'output':a.output,'arrays':report['arrays']}),flush=True)
if __name__=='__main__':main()

"""Read actual head, raw guides and scalp coverage for the reference-led v6."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--source',required=True);pa.add_argument('--output',required=True);a=pa.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=(P/a.source).resolve();out=(P/a.output).resolve();npz=out.with_suffix('.npz')
    for p in (source,out,npz):p.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or out.exists() or npz.exists():raise RuntimeError('Explicit source and unused outputs required')
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));coords=fit['_coords'];world=fit['_world'];rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;data={};report={'source':a.source,'sourceSha256':sha(source),'helperSha256':sha(__file__)}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        head=bpy.data.objects['R11 Female head study'].evaluated_get(graph);head.data.calc_loop_triangles();data['head_world']=world(coords(head),head.matrix_world);data['head_triangles']=np.asarray([tuple(t.vertices) for t in head.data.loop_triangles],np.int32)
        for i,name in enumerate(('R11 Donor / long hair main','R11 Donor / long hair strands')):
            obj=bpy.data.objects[name];data[f'guide_{i}_world']=world(coords(obj),obj.matrix_world);data[f'guide_{i}_local']=coords(obj);data[f'guide_{i}_matrix']=np.asarray(obj.matrix_world)
            data[f'guide_{i}_starts']=np.array([c.first_point_index for c in obj.data.curves],np.int32);data[f'guide_{i}_lengths']=np.array([c.points_length for c in obj.data.curves],np.int32)
        scalp=bpy.data.objects['R11 Donor / long hair growth mesh'];scalp.data.calc_loop_triangles();data['scalp_world']=world(coords(scalp),scalp.matrix_world);data['scalp_local']=coords(scalp);data['scalp_triangles']=np.array([tuple(t.vertices) for t in scalp.data.loop_triangles],np.int32);data['scalp_edges']=np.array([tuple(e.vertices) for e in scalp.data.edges],np.int32)
        data['scalp_polygon_start']=np.array([p.loop_start for p in scalp.data.polygons],np.int32);data['scalp_polygon_size']=np.array([p.loop_total for p in scalp.data.polygons],np.int32);data['scalp_loop_vertex']=np.array([l.vertex_index for l in scalp.data.loops],np.int32)
        for side in ('L','R'):
            obj=bpy.data.objects['R11 Adult sclera '+side];q=world(coords(obj),obj.matrix_world);data['eye_'+side+'_center']=(q.min(0)+q.max(0))*.5;data['eye_'+side+'_radius']=np.array((q.max(0)-q.min(0)).max()*.5)
        report['scalpAttributes']=[{'name':v.name,'type':v.data_type,'domain':v.domain,'count':len(v.data)} for v in scalp.data.attributes]
        report['restyle']=json.loads(bpy.context.scene['R11_actual_donor_restyle'])
    finally:rig.data.pose_position=pose;bpy.context.view_layer.update()
    with npz.open('xb') as f:np.savez_compressed(f,**data)
    report.update(arrays=str(npz.relative_to(P)),arraysSha256=sha(npz),savedBlend=False)
    if sha(source)!=report['sourceSha256']:raise RuntimeError('Source changed')
    with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print('R11_FRONT_GROOM_V6_OBSERVED '+str(out),flush=True)
if __name__=='__main__':main()

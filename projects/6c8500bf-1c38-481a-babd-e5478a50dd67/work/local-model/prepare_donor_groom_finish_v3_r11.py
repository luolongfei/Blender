"""Save an independent measured donor-groom v3 repair with dense-output proof."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
EXPECTED_HELPER='ee8eae92f6163ef9c9667d681547acb99557badb2f5a89117288d27e54a9b072'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def actual_dense_proof():
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    probe=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))
    obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    original=modifier.node_group;memo={};replacements=[];arrays={};rows=[]
    try:
        tree=review['copy_render_branch'](original,memo,replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        for label,name in (('styled-before-contact','Join Geometry.001'),('head-contact','R11 full head clearance'),('final-taper','R11 strand tip taper')):
            node=tree.nodes[name];geometry=next(s for s in node.outputs if s.type=='GEOMETRY')
            for link in list(output.links):tree.links.remove(link)
            tree.links.new(geometry,output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
            rows.append(probe['capture'](obj,label,arrays))
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    a=arrays['styled-before-contact_world'];b=arrays['head-contact_world'];c=arrays['final-taper_world']
    if a.shape!=b.shape or not np.array_equal(b,c):raise RuntimeError('Contact/profile changed dense point topology or taper changed positions')
    delta=np.linalg.norm(b-a,axis=1);largest=float(delta.max())
    if largest>.015:raise RuntimeError('Dense head-contact correction exceeds 15 mm; inspect styling instead of saving '+str(largest))
    radii=arrays['final-taper_radius']
    if float(radii.max())>.000080 or float(radii.min())<0.:raise RuntimeError('Final evaluated hair radius is outside candidate range')
    return {'stages':rows,'renderBranchReplacements':replacements,
            'headContactDisplacementMetersP50P95P99Max':np.percentile(delta,[50,95,99,100]).tolist(),
            'profilePreservesDensePositionsExact':True,'rendered':False,
            'scope':'Actual dense render-branch positions/radii in Blender; no beauty or continuous-motion acceptance.'},arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    report_path=out.with_suffix('.groom-finish.json');array_path=out.with_suffix('.groom-finish.npz')
    for path in (source,out,report_path,array_path):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,report_path,array_path)):raise RuntimeError('Fresh independent source and unused outputs required')
    module=HERE/'donor_groom_finish_v3_r11.py'
    if sha(module)!=EXPECTED_HELPER:raise RuntimeError('Review changed v3 helper')
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    names=('R11 Female head study','R01 Head','R11 Donor / long hair growth mesh','R11 Donor / long hair strands',
           'R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
    before={name:contract(bpy.data.objects[name]) for name in names}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        report=runpy.run_path(str(module))['apply'](expected_source_relative=a.source)
        proof,arrays=actual_dense_proof()
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    if before!={name:contract(bpy.data.objects[name]) for name in names}:raise RuntimeError('Protected head/secondary/growth/tail contracts changed')
    report['protectedGeometryContracts']=before;report['actualDenseOutput']=proof
    report['faceFibreBinding']=runpy.run_path(str(HERE/'face_fibre_binding_r11.py'))['validate'](neutral=False)
    bpy.context.scene['R11_donor_groom_finish_v3']=json.dumps(report,ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Candidate save failed')
    with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out),wrapperSha256=sha(__file__),
                  actualDenseArrays=str(array_path.relative_to(P)),actualDenseArraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_DONOR_GROOM_FINISH_V3_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)
if __name__=='__main__':main()

"""Explicit rejected-technical v6 preview, never a production candidate."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
SOURCE_SHA='151641fa7bdc32a185d742738333d3e3868acdc5b111c986a45a06ec9aa5b71a'
HELPER_SHA='297b6b6533321f6893b4f211d992c290927b11a91444a593118eaa4844acaf2c'
ARRAY_SHA='6e9ec2168349618a78cb1c36a6872c8cca0df98613c38ad062adef0a2e63cd25'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--output',required=True);a=pa.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=(P/a.source).resolve();out=(P/a.output).resolve();report_path=out.with_suffix('.preview.json')
    for p in (source,out,report_path):p.relative_to((P/'work/reconstruction-r02').resolve())
    if 'preview' not in out.stem.lower() or out.suffix!='.blend':raise RuntimeError('Output must explicitly be a preview blend')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists() or report_path.exists():raise RuntimeError('Fresh exact failed-v6 source and unused preview output required')
    helper=HERE/'reference_front_groom_v6_r11.py';evidence_path=P/'work/reconstruction-r02/reference-adult-front-groom-v6-r11.groom-reference.json';arrays_path=evidence_path.with_suffix('.npz')
    if sha(source)!=SOURCE_SHA or sha(helper)!=HELPER_SHA or sha(arrays_path)!=ARRAY_SHA:raise RuntimeError('The exact measured failed-v6 source/helper/evidence is required')
    evidence=json.loads(evidence_path.read_text());old=np.load(arrays_path,allow_pickle=False)
    if evidence.get('savedBlend') or not evidence.get('error'):raise RuntimeError('Expected actual rejected v6 evidence')
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract'];names=('R11 Female head study','R01 Head','R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
    before={n:contract(bpy.data.objects[n]) for n in names};rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();report=runpy.run_path(str(helper))['apply'](expected_source_relative=a.source)
        obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);original=modifier.node_group
        try:
            modifier.node_group=runpy.run_path(str(HERE/'adult_full_groom_review_v6_r11.py'))['copy_render_branch'](original,{},[]);obj.update_tag();bpy.context.view_layer.update()
            arrays={};row=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture'](obj,'actual',arrays)
            if not np.array_equal(arrays['actual_world'],old['final-taper_world']) or not np.array_equal(arrays['actual_radius'],old['final-taper_radius']):raise RuntimeError('Preview does not exactly reproduce the actual rejected dense geometry')
        finally:modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    finally:rig.data.pose_position=pose;bpy.context.view_layer.update()
    if before!={n:contract(bpy.data.objects[n]) for n in names}:raise RuntimeError('Preview changed protected face/body/tails')
    report.update(technicalAcceptance=False,productionUseProhibited=True,appearanceAccepted=False,
        previewPurpose='Visual-only review of crown coverage and original-design fringe locks; explicitly contains the measured right-eye hair penetration.',
        knownActualFailures=evidence['actualDenseOutput']['errors'],knownActualDenseEvidence=evidence['actualDenseOutput'],
        actualFailedEvidence=str(evidence_path.relative_to(P)),actualFailedArraysSha256=ARRAY_SHA,reproducesFailedDensePositionsAndRadiiExact=True,wrapperSha256=sha(__file__))
    bpy.context.scene['R11_REVIEW_ONLY_FAILED_V6']=json.dumps({'technicalAcceptance':False,'productionUseProhibited':True,'actualFailures':report['knownActualFailures']})
    text=bpy.data.texts.new('READ ME - FAILED V6 PREVIEW ONLY');text.write('NOT FOR PRODUCTION\nMeasured final hair enters the R eye envelope by 2.4986 mm.\nOnly for reviewing reference hairstyle coverage.\n'+json.dumps(report['knownActualDenseEvidence'],indent=2))
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Preview save failed')
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print('R11_FAILED_V6_VISUAL_PREVIEW_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256'],'technicalAcceptance':False,'productionUseProhibited':True}),flush=True)
if __name__=='__main__':main()

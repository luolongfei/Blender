"""Read-only saved-source contract diagnosis across evaluation/REST transitions.

Does not apply a hairstyle, edit geometry or save a blend. Reports each changed
field so a later production guard can compare the same evaluated pose states.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent;MAIN='R11 Donor / long hair main'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def difference(a,b):
    return {name:{field:{'before':a[name].get(field),'after':b[name].get(field)} for field in set(a[name])|set(b[name]) if a[name].get(field)!=b[name].get(field)} for name in a if a[name]!=b[name]}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    for path in (source,out):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists():raise RuntimeError('Fresh source and unused report required')
    digest=sha(source);contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    protected=[o for o in bpy.data.objects if o.type in {'MESH','CURVES'} and o.name!=MAIN and o.get('R11_groom_role_root_source')!=MAIN]
    snapshot=lambda:{o.name:contract(o) for o in protected}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;report={'source':a.source,'sourceSha256':digest,'sourcePose':pose,'savedBlend':False}
    initial=snapshot();bpy.context.view_layer.update();settled=snapshot()
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();rest=snapshot()
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    restored=snapshot()
    report.update(initialContracts=initial,settledSourceContracts=settled,restContracts=rest,restoredSourceContracts=restored,
        initialToSettled=difference(initial,settled),settledToRest=difference(settled,rest),initialToRest=difference(initial,rest),settledToRestored=difference(settled,restored))
    if sha(source)!=digest:raise RuntimeError('Read-only source changed')
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_V9_CONTRACT_TRANSITION '+json.dumps({k:{name:list(fields) for name,fields in report[k].items()} for k in ('initialToSettled','settledToRest','initialToRest','settledToRestored')}),flush=True)
if __name__=='__main__':main()

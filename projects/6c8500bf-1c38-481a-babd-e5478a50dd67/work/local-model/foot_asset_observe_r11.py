"""Minimal actual Foot - Realistic + current feet/shoes/socks/rig observation.

The catalog classifies Foot - Realistic as an OBJECT (and its mesh), not a
collection. Append that exact object temporarily, preserving raw and evaluated
geometry, source groups/UVs and metadata. Do not fit, save, render or start an app.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

OBSERVER_SHA='b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'
CATALOG='work/reconstruction-r02/hand-asset-inventory-r10.json'
ASSET='Foot - Realistic'
SURFACES=('Body_Complete_Shin_Foot_', 'Sock_Ribbed_', 'Shoe_Leather_Upper_',
          'Shoe_Outsole_', 'Shoe_Stacked_Heel_')
OPTIONAL_PREFIXES=('Shoe_Collar_', 'Shoe_Toe_Cap_', 'Shoe_Toecap_', 'Shoe_Welt_')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def observe(source_relative,source_sha,output_relative):
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Disposable background Object Mode required')
    catalog=json.loads(_path(CATALOG).read_text(encoding='utf-8'))
    if ASSET not in catalog['libraryNames']['objects']:
        raise RuntimeError('Exact requested foot object is absent from the actual inventory')
    expected=[prefix+side for prefix in SURFACES for side in ('L','R')]
    missing=[name for name in expected if name not in bpy.data.objects]
    if missing:
        available=[o.name for o in bpy.data.objects if any(term in o.name.lower() for term in ('foot','shoe','sock'))]
        raise RuntimeError('Known foot surfaces missing; explicitly reconcile actual names before observing: '+json.dumps({'missing':missing,'actualFootCandidates':available}))
    selected=[bpy.data.objects[name] for name in expected]
    optional=[o for o in bpy.data.objects if any(o.name.startswith(prefix) for prefix in OPTIONAL_PREFIXES)]
    selected+=sorted(optional,key=lambda o:o.name)
    if any(o.type!='MESH' for o in selected) or len(selected)>24:
        raise RuntimeError('Unexpected foot surface types/count; keep observation bounded')
    rig=bpy.data.objects['Character_Rig']
    required_bones=[side+part for side in ('left','right') for part in ('LowerLeg','Foot','Toes')]
    if any(name not in rig.data.bones for name in required_bones):raise RuntimeError('Existing foot/toes hierarchy changed')
    helper=Path(__file__).with_name('hand_asset_observe_r10.py')
    if _sha(helper)!=OBSERVER_SHA:raise RuntimeError('Frozen generic asset observer changed')
    api=runpy.run_path(str(helper))
    # Keep the generic raw/evaluated reader and reversible library lifecycle.
    # Only the explicitly observed target-object list changes; old hands are
    # not loaded or edited as a side effect of foot work.
    api['observe'].__globals__['HANDS']=tuple(o.name for o in selected)
    api['observe'].__globals__['NAILS']=()
    result=api['observe'](source_relative,source_sha,(ASSET,),CATALOG,output_relative,frame=1)
    report_path=_path(output_relative);report=json.loads(report_path.read_text(encoding='utf-8'))
    report['mode']='foot-observe-r11'
    report['footObserver']={'helperSha256':_sha(__file__),'genericObserverSha256':OBSERVER_SHA,
        'libraryCategoryUsed':'objects','assetObject':ASSET,
        'assetAlsoInMeshCatalog':ASSET in catalog['libraryNames']['meshes'],
        'assetInCollectionCatalog':ASSET in catalog['libraryNames']['collections'],
        'requiredCurrentSurfaceNames':expected,'optionalCurrentSurfaceNames':[o.name for o in optional],
        'requiredExistingBones':required_bones}
    report['scope']='Actual current lower-shin/ankle/foot, sock and shoe shell raw/evaluated surfaces; full existing rig matrices; exact temporary CC0 HBM Foot - Realistic object. No fit, weight transfer, toe geometry acceptance, hidden-shoe acceptance or footwear change.'
    report['adaptationRequirements']=[
        'Identify real donor heel, sole, arch, ankle transition, toe and nail topology from the returned numeric data; do not infer them from the asset name.',
        'Fit to actual existing foot/ankle bones and measured shoe interior without changing the original shoe design or collapsing the arch/toes merely to pass an enclosure test.',
        'Preserve original shins/feet and weights until the new joined ankle and visible barefoot views have passed actual inspection.',
        'Existing rigid closed shoes constrain toe motion; define and test compatible shoe/foot motion before claiming toe articulation.',
        'Check bare foot dorsal/medial/lateral/plantar views, toe/nail closeups and actual ankle poses in addition to shoe/sock clearances.'
    ]
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    result['footScope']=report['scope'];result['helperSha256']=report['footObserver']['helperSha256']
    return result


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/foot-asset-observation-r11.json')
    args=parser.parse_args(argv)
    print('R11_FOOT_OBSERVATION '+json.dumps(observe(args.source,args.source_sha,args.output),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

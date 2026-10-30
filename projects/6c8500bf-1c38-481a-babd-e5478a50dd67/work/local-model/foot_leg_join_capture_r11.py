"""Read-only actual calf continuation and current foot-control capture.

Complements the already complete footwear capture with the two retained Leg
objects that were deliberately outside its inventory. No model or action edit.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PIN='b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(source_relative,source_sha,output_relative):
    import bpy
    p=project_path();source=(p/source_relative).resolve();output=(p/output_relative).resolve()
    source.relative_to(p.resolve());output.relative_to(p.resolve());arrays_path=output.with_suffix('.npz')
    if output.exists() or arrays_path.exists():raise FileExistsError('Preserve existing actual capture')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or sha(source)!=source_sha:
        raise RuntimeError('Exact disposable saved source required')
    scene=bpy.context.scene
    if scene.frame_current!=1 or scene.frame_subframe!=0:raise RuntimeError('Current authored frame 1 required')
    helper=Path(__file__).with_name('hand_asset_observe_r10.py')
    if sha(helper)!=PIN:raise RuntimeError('Frozen observer changed')
    api=runpy.run_path(str(helper));rig=bpy.data.objects['Character_Rig']
    legs=[bpy.data.objects['Leg '+side+' · knee calf ankle'] for side in ('L','R')]
    guarded=legs+[rig];contract=api['_source_contract'](guarded);ids=set(api['_ids'](bpy));pose=rig.data.pose_position
    arrays={};record={'schemaVersion':1,'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'genericObserverSha256':PIN,'frame':1,'oldPosePosition':pose,'rigCurrentPose':api['_object'](rig,'rig',arrays,bpy.context.evaluated_depsgraph_get()),
        'legObjects':[],'savedBlend':False,'sourceGeometryEdited':False,'appearanceAccepted':False}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        for side,obj in zip(('L','R'),legs):
            row=api['_object'](obj,'leg_'+side,arrays,graph);row['evaluatedCoordinateMeaning']='REST rig, all existing actual mesh modifiers retained; object-local points with evaluatedMatrixWorld.'
            record['legObjects'].append(row)
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        restored=api['_source_contract'](guarded)==contract and set(api['_ids'](bpy))==ids and sha(source)==source_sha
        if not restored:raise RuntimeError('Leg/rig capture restoration failed')
    record['restorationExact']=True;record['arrays']=str(arrays_path.relative_to(p));np.savez_compressed(arrays_path,**arrays)
    record['arraysSha256']=sha(arrays_path)
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return {'report':str(output.relative_to(p)),'reportSha256':sha(output),'arraySha256':sha(arrays_path),'savedBlend':False}

def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/foot-leg-join-capture-r11.json');a=parser.parse_args(argv)
    print('R11_FOOT_LEG_JOIN_CAPTURE '+json.dumps(run(a.source,a.source_sha,a.output)),flush=True)

if __name__=='__main__':main()

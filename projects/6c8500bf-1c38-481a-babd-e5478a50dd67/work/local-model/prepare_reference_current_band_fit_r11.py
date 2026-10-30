"""Save only four fitted ring mesh copies after current native-path validation.

Independent Blender background entry point. It never transports or reshapes
tail/gather hair. The frozen pure installer owns geometric checks and rollback.
"""
import argparse,json,runpy,sys
from pathlib import Path
import bpy
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
CORE='reference_current_band_fit_r11.py';CORE_SHA='6ce757acda51b72372e8be90413f78a9c98c3443c44eef12a0d5b5e4ead12308'
OBSERVER='observe_reference_tie_alignment_r11_v2.py';OBSERVER_SHA='c488fccec705d4a805f142dab3c2fa8fe902bf42d3a59fd6307d04f248cf7c5c'

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def path(relative):
    value=(P/str(relative).replace('\\','/')).resolve();value.relative_to((P/'work/reconstruction-r02').resolve());return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','plan','observation','native-proof','output'):parser.add_argument('--'+flag,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=path(args.source);out=path(args.output);reportpath=out.with_suffix('.band-fit.json');arraypath=out.with_suffix('.band-fit.npz');proofpath=path(args.native_proof)
    if out.suffix.lower()!='.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,reportpath,arraypath)):raise RuntimeError('Fresh explicit background source and unused independent blend/report outputs required')
    if sha(HERE/CORE)!=CORE_SHA or sha(HERE/OBSERVER)!=OBSERVER_SHA:raise RuntimeError('Frozen band installer/observer changed')
    core=runpy.run_path(str(HERE/CORE));observer=runpy.run_path(str(HERE/OBSERVER));inputs=core['load_inputs'](args.plan,args.observation);proof=json.loads(proofpath.read_text());pa=path(proof['arrays'])
    if proof.get('helperSha256')!=CORE_SHA or sha(pa)!=proof['arraysSha256'] or proof.get('status')!='passed_native_sample_capsules' or proof.get('errors'):raise RuntimeError('An intact successful proof from the frozen actual-native capsule validator is required')
    for key,value in inputs['identity'].items():
        if proof.get(key)!=value:raise RuntimeError('Native proof belongs to another actual source/candidate: '+key)
    if source!=path(inputs['observation']['source']) or sha(source)!=inputs['observation']['sourceSha256']:raise RuntimeError('Native proof must be installed only on its actually observed saved source')
    bpy.context.view_layer.update();rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene;pose=rig.data.pose_position;frame=(scene.frame_current,scene.frame_subframe);action=rig.animation_data.action if rig.animation_data else None
    objects=[obj for obj in bpy.data.objects if obj.type in ('MESH','CURVES')];before={obj.name:observer['raw_contract'](obj) for obj in objects};ringnames={ring['object'] for ring in inputs['plan']['rings']};inventory=set(bpy.data.objects.keys());arrays={};report={'schemaVersion':1,'source':args.source,'sourceSha256':sha(source),'output':args.output,'plan':args.plan,'observation':args.observation,'nativeProof':args.native_proof,'nativeProofSha256':sha(proofpath),**inputs['identity'],'wrapperSha256':sha(__file__),'coreSha256':CORE_SHA,'savedBlend':False,'technicalAcceptance':False,'appearanceAccepted':False,'dynamicsValid':False,'sourcePose':pose,'errors':[]}
    manifests=inputs['observation']['arrayManifest'];dependency_keys=[key for key in manifests if key.startswith(('native_gather_final_','native_tail_positive_','native_tail_negative_','tail_positive_','tail_negative_','head_'))]
    report['fitValidityDependencies']={'sourceSha256':inputs['observation']['sourceSha256'],'observationArraysSha256':inputs['observation']['arraysSha256'],'actualGeometryArrays':{key:manifests[key] for key in dependency_keys},'tailRootsAndAllPinsUnchangedByInstallation':True,'gatherPathsUnchangedByInstallation':True,'permanentFitClaimed':False,'requiresRevalidationAfter':['Any tail rest-path/root/pin/width/radius change','Any gather approach path/radius change near the bands','Any band geometry/object transform change','Any head/ear change affecting the current collision envelope'],'reason':'The fitted band is constrained by the CURRENT complete native gather/tail envelope, including all tail-root pins. Future tail-shape work can change that envelope even if the visible knot location is retained.'}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();print('R11_BAND_INSTALL_CURRENT_NATIVE_VERIFIED',flush=True)
        report['installation']=core['install_checked'](inputs,proof,arrays)
        rig.data.pose_position=pose;bpy.context.view_layer.update();after={obj.name:observer['raw_contract'](obj) for obj in objects};changes={}
        for name,old in before.items():
            difference={key:{'before':old[key],'after':after[name][key]} for key in old if old[key]!=after[name][key]}
            allowed={'data','geometryKeysSha256'} if name in ringnames else set();unexpected={key:value for key,value in difference.items() if key not in allowed}
            if unexpected:changes[name]=unexpected
        report['protectedRestoredPoseChanges']=changes
        if changes:report['errors'].append('Protected geometry/metadata changed after original-pose restoration')
        if set(bpy.data.objects.keys())!=inventory:report['errors'].append('Object inventory changed')
        if (scene.frame_current,scene.frame_subframe)!=frame or (rig.animation_data.action if rig.animation_data else None)!=action:report['errors'].append('Scene frame or rig action changed')
        if sha(source)!=report['sourceSha256']:report['errors'].append('Saved input source changed')
        if report['errors']:raise RuntimeError('; '.join(report['errors']))
        report['technicalAcceptance']=True;report['nextRequired']=['Same-camera front and quarter review of fitted black bands','Revalidate band fit against any future tail-root/pin/gather envelope changes','Rebuild and validate dynamics after adopting final static tail shape']
    except Exception as exc:
        report['error']=str(exc)
        if arrays:
            with arraypath.open('xb') as stream:np.savez_compressed(stream,**arrays)
            report.update(arrays=str(arraypath.relative_to(P)),arraysSha256=sha(arraypath))
        with reportpath.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
        raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    scene['R11_reference_band_fit']=json.dumps({'planSha256':report['planSha256'],'observationSha256':report['observationSha256'],'nativeProofSha256':report['nativeProofSha256'],'permanentFitClaimed':False,'requiresRevalidationAfterTailOrGatherChanges':True})
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Independent fitted-band source save failed')
    with arraypath.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report.update(savedBlend=True,outputSha256=sha(out),arrays=str(arraypath.relative_to(P)),arraysSha256=sha(arraypath))
    with reportpath.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
    print('R11_FITTED_BANDS_SAVED '+json.dumps({'output':args.output,'sha256':report['outputSha256'],'appearanceAccepted':False,'permanentFitClaimed':False}),flush=True)

if __name__=='__main__':main()

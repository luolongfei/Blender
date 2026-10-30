"""Integrator-only R09 stage runner. NEVER starts another Blender process.

Prepare with the FINAL production source open (no active NLA; the selected
scalp/gather/fringe paths must have no active modifiers). Existing cached tail
GN is not sampled or transferred by this regional refresh.
  -- prepare --source work/reconstruction-r02/FINAL.blend
     --runtime work/runtime-r06/r09-tail-colliders.blend --stage r09-final

Then reopen the previous saved runtime checkpoint for EACH explicit stage:
  -- face|scalp|fringe|gather|root|tail|skin-bake|root-bake|iris|export-prepare
     --bundle work/runtime-r06/r09-final-integration.json
     --runtime work/runtime-r06/INPUT.blend --output work/runtime-r06/NEW.blend

Finally, with that export-ready checkpoint open:
  -- export --bundle ... --runtime ... --export-folder work/runtime-r06/preview-r09

Each successful mutation is one NEW checkpoint with its own report. Source
and prior checkpoints remain intact. No formal exports/avatar.* are written.
Failed stage folders are preserved; choose a fresh --stage or --bakes-root.

First stage, from workspace root in PowerShell (SET $finalSourceRelative to the
actual adopted source; this docstring is a recipe, not automatic execution):
  $cfg = Get-Content -Raw config/local.win32.json | ConvertFrom-Json
  $ws = Get-Content -Raw config/workspace.json | ConvertFrom-Json
  $project = Join-Path (Join-Path (Get-Location) $ws.projectsRoot) $ws.defaultProject
  & $cfg.blenderExecutable --background (Join-Path $project $finalSourceRelative) `
    --python (Join-Path $project 'work/local-model/runtime_integrate_driver_r09.py') `
    -- prepare --source $finalSourceRelative `
    --runtime work/runtime-r06/r09-tail-colliders.blend --stage r09-final

This prepares source payloads only, not atlases, Cycles bakes or exports.
Budget 2-6 minutes and roughly 0.3-0.5 GiB new files for this first stage,
subject to actual final source size and local I/O. The previous R08 face-only
source-copy-to-manifest timestamps span about 48 s (not full process timing),
with 113 MiB source copy and 44 MiB face payload. Latest combined source measured
152 MiB. Subsequent standalone geometry/material checkpoints may total 4-6 GiB;
these are planning estimates, not measured R09 final execution results.
"""
import argparse,importlib.util,json,re,sys
from pathlib import Path
import bpy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
COMPLETE=('face','scalp','fringe','gather','root','tail','skin-bake','root-bake','iris')
MARKER='RT09_integration_bundle';PROGRESS='RT09_integration_completed'


def _api(name):
    spec=importlib.util.spec_from_file_location('integration09_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def _common():return _api('runtime_scalp_refresh_r09')


def _prepare(args):
    common=_common();source,rig=common._guard('work/reconstruction-r02')
    if not args.source or not args.runtime or not args.stage:raise ValueError('Explicit final --source, initial --runtime and fresh --stage required')
    if source!=common._path(args.source,'work/reconstruction-r02'):raise RuntimeError('Open the explicit final production source')
    if not re.fullmatch(r'r09[a-z0-9-]*',args.stage):raise ValueError('Use an explicit fresh r09 stage name')
    initial=common._path(args.runtime,'work/runtime-r06')
    if not initial.is_file():raise FileNotFoundError(initial)
    bundle_path=common._path('work/runtime-r06/'+args.stage+'-integration.json','work/runtime-r06')
    if bundle_path.exists():raise FileExistsError(bundle_path)
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):raise RuntimeError('Resolve production NLA in the disposable preparation source; do not sample an active action stack')
    objects=('R06 continuous scalp flow','R06 full gathered roots L','R06 full gathered roots R','R03 Forelocks from crown')
    for name in objects:
        obj=bpy.data.objects[name]
        if obj.hide_render or any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Restore native rest groom before preparing: '+name)
    for name in objects[:3]+('R06 attached scalp root substrate',):
        if not bpy.data.objects[name].get('R09_root_projection_original_data'):raise RuntimeError('PN pre-edit geometry record missing: '+name)
    if not bpy.data.objects[objects[-1]].get('R09_forelock_original_data'):raise RuntimeError('Fringe pre-edit geometry record missing')
    source_sha=common._sha(source);members={};reports={}
    # Face preparation intentionally runs LAST: it saves/opens a disposable
    # source copy for the 16 x 21 snapshots. Numeric/native preparation above
    # still reads the exact final source pathname and does not change it.
    for region in ('root','scalp','fringe','gather'):
        api=_api('runtime_'+region+'_refresh_r09');stage=args.stage+'-'+region
        reports[region]=api.prepare_payload(args.source,stage)
        relative='work/runtime-r06/'+stage+('-manifest.json' if region=='gather' else '/manifest.json')
        members[region]={'file':relative,'sha256':common._sha(common._path(relative,'work/runtime-r06'))}
    reports['face']=_api('runtime_face_refresh_r08').prepare_payload(args.source,args.stage)
    relative='work/runtime-r06/'+args.stage+'-face-refresh.json'
    members['face']={'file':relative,'sha256':common._sha(common._path(relative,'work/runtime-r06'))}
    for region,report in reports.items():
        actual=report['source']['sha256'] if region=='face' else report['sourceSha256']
        if actual!=source_sha:raise RuntimeError('Payload regions do not share one final source')
    if common._sha(source)!=source_sha:raise RuntimeError('Production source changed on disk')
    result={'schema':1,'stage':args.stage,'source':args.source,'sourceSha256':source_sha,
        'initialRuntime':args.runtime,'initialRuntimeSha256':common._sha(initial),'members':members,
        'requiredStages':list(COMPLETE),'sequence':list(COMPLETE)+['export-prepare','export'],
        'rootMapsMustBeRebaked':True,'sharedClothBakesRetained':True,'tailGeometryAndRigRestRetained':True,
        'automaticLaunchOrRender':False,'appearanceAccepted':False,
        'limits':['Native source caches are not VRM physics.',
            'Tail parameters are the measured improved 66 mm candidate, with residual lower-tail/bodice grazing and visible forward lift.',
            'Fresh combined source still requires actual front/back/side and motion review in the target viewer.']}
    common._json(bundle_path,result)
    print('RT09_INTEGRATION_PREPARED',json.dumps({'bundle':str(bundle_path),'sourceSha256':source_sha,'members':members}),flush=True)


def _load_bundle(args):
    common=_common()
    if not args.bundle or not args.runtime:raise ValueError('Explicit --bundle and --runtime required')
    bundle_path=common._path(args.bundle,'work/runtime-r06');bundle=json.loads(bundle_path.read_text(encoding='utf-8'))
    if bundle.get('schema')!=1 or set(bundle['members'])!={'face','scalp','fringe','gather','root'}:raise RuntimeError('Unknown integration bundle')
    source=common._path(bundle['source'],'work/reconstruction-r02')
    if common._sha(source)!=bundle['sourceSha256']:raise RuntimeError('Final production source changed since preparation')
    for row in bundle['members'].values():
        if common._sha(common._path(row['file'],'work/runtime-r06'))!=row['sha256']:raise RuntimeError('Prepared member manifest changed')
    runtime,rig=common._guard('work/runtime-r06')
    if runtime!=common._path(args.runtime,'work/runtime-r06'):raise RuntimeError('Open the explicit runtime checkpoint')
    identifier=json.dumps({'file':args.bundle,'sha256':common._sha(bundle_path)},sort_keys=True)
    if rig.get(MARKER):
        if rig[MARKER]!=identifier:raise RuntimeError('Runtime belongs to another integration bundle')
        done=json.loads(rig.get(PROGRESS,'[]'))
    else:
        if args.runtime!=bundle['initialRuntime'] or common._sha(runtime)!=bundle['initialRuntimeSha256']:
            raise RuntimeError('First stage must use the exact initial runtime recorded in this bundle')
        done=[]
    if not isinstance(done,list) or len(done)!=len(set(done)):raise RuntimeError('Invalid stage record')
    return common,bundle,runtime,rig,identifier,done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare',*COMPLETE,'export-prepare','export'))
    for name in ('source','runtime','stage','bundle','output','bakes-root','export-folder'):parser.add_argument('--'+name)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if args.phase=='prepare':return _prepare(args)
    common,bundle,runtime,rig,identifier,done=_load_bundle(args)
    if args.phase in done:raise RuntimeError('Stage already applied in this checkpoint: '+args.phase)
    if args.phase in ('export-prepare','export') and set(COMPLETE)-set(done):raise RuntimeError('Integration incomplete: '+repr(sorted(set(COMPLETE)-set(done))))
    if args.phase=='export':
        if 'export-prepare' not in done or not args.export_folder:raise RuntimeError('Run export-prepare and supply a new explicit --export-folder')
        output=common._path(args.export_folder,'work/runtime-r06')
        if output.exists():raise FileExistsError('Preserve previous preview output: '+str(output))
        result=_api('runtime_export_r06').export(output_relative=args.export_folder)
        print('RT09_PREVIEW_EXPORTED',json.dumps({'folder':str(output),'source':bundle['source'],'appearanceAccepted':False}),flush=True)
        return result
    if not args.output:raise ValueError('Every mutation requires a NEW explicit --output Blend checkpoint')
    output=common._path(args.output,'work/runtime-r06')
    if output.suffix.lower()!='.blend' or output.exists() or output.with_suffix('.json').exists():raise FileExistsError('New Blend/report required')
    before_sha=common._sha(runtime);phase=args.phase
    if phase=='face':
        result=_api('runtime_face_refresh_r08').replace(bundle['members']['face']['file'],args.runtime,recull=True)
    elif phase in ('scalp','fringe','gather','root'):
        result=_api('runtime_'+phase+'_refresh_r09').apply_payload(bundle['members'][phase]['file'])
    elif phase=='tail':
        result=_api('runtime_tail_envelope_r09').apply()
    elif phase in ('skin-bake','root-bake','iris'):
        prerequisite='root' if phase=='root-bake' else 'face'
        if prerequisite not in done:raise RuntimeError('Apply '+prerequisite+' before this material stage')
        plan=common._path('work/runtime-r06/material-stage-plan.json','work/runtime-r06')
        if not plan.is_file():raise RuntimeError('Retained material-stage plan missing; do not re-plan already baked clothes')
        channel='skin' if phase=='skin-bake' else ('root' if phase=='root-bake' else 'iris')
        bakes=args.bakes_root or 'work/runtime-r06/'+bundle['stage']+'-bakes'
        common._path(bakes,'work/runtime-r06')
        result=_api('runtime_material_stage_r06').run_stage(channel,output_root=bakes)
    elif phase=='export-prepare':
        result=_api('runtime_export_r06').prepare([bundle['source'],bundle['initialRuntime']],require_baked=True)
    else:raise ValueError(phase)
    if common._sha(runtime)!=before_sha:raise RuntimeError('Input runtime changed on disk')
    rig[MARKER]=identifier;rig[PROGRESS]=json.dumps([*done,phase])
    output.parent.mkdir(parents=True,exist_ok=True)
    status=bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if 'FINISHED' not in status or not output.is_file():raise RuntimeError('New checkpoint save failed')
    # The export preparation includes this NEW saved file's identity. Rebuild
    # the on-disk contract after save, without exporting or another save.
    if phase=='export-prepare':result=_api('runtime_export_r06').prepare([bundle['source'],bundle['initialRuntime']],require_baked=True)
    report={'phase':phase,'bundle':args.bundle,'source':bundle['source'],'sourceSha256':bundle['sourceSha256'],
        'inputRuntime':args.runtime,'inputSha256':before_sha,'output':args.output,'outputSha256':common._sha(output),
        'completedStages':[*done,phase],'result':result,'appearanceAccepted':False,'savedBlend':True}
    common._json(output.with_suffix('.json'),report)
    print('RT09_INTEGRATION_STAGE_SAVED',json.dumps({k:report[k] for k in ('phase','output','outputSha256','completedStages')}),flush=True)


if __name__=='__main__':main()

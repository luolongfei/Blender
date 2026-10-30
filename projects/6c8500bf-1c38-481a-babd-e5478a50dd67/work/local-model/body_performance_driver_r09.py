"""Serial background R09 body authoring gate; root launches Blender explicitly.

  blender --background --python body_performance_driver_r09.py --
    --source work/reconstruction-r02/reference-production-uv-r08.blend --preview

Paths are relative to the active project, or explicit absolute project work paths.
Only successful action creation AND static shoe checks permit a new Blend save.
--preview then renders four 480x640/12-sample body-only stills serially. No bake.
"""
import argparse
import hashlib
import json
import runpy
import sys
import traceback
from pathlib import Path
try:
    import bpy
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _path(value,scope):
    path=(project_path()/value).resolve();path.relative_to((project_path()/scope).resolve())
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/body-action-r09.blend')
    parser.add_argument('--report-dir',default='work/reconstruction-r02/body-action-r09-run')
    parser.add_argument('--preview-dir',default='previews/reconstruction-r02/body-poses-r09')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(argv)
    if bpy is None or not bpy.app.background:raise RuntimeError('Only explicit isolated background Blender execution is supported')
    source=_path(args.source,'work');target=_path(args.output,'work');reports=_path(args.report_dir,'work')
    previews=_path(args.preview_dir,'previews')
    main_source=(project_path()/'work/character.blend').resolve()
    if source==main_source or target==main_source or source==target:
        raise RuntimeError('Independent source and new output copies required; formal master is excluded')
    if not source.is_file() or source.suffix.lower()!='.blend':raise RuntimeError('Explicit source .blend is missing')
    if target.exists() or target.suffix.lower()!='.blend':raise RuntimeError('Output must be a new independent .blend')
    if reports.exists() and any(reports.iterdir()):raise RuntimeError('Use a new empty report directory')
    if args.preview and previews.exists() and any(previews.iterdir()):raise RuntimeError('Use a new empty preview directory')
    reports.mkdir(parents=True,exist_ok=True);source_sha=_sha(source)
    record={'sourceBlend':str(source),'sourceFileSha256':source_sha,'outputBlend':str(target),
        'status':'started','savedBlend':False,'previewRequested':args.preview,
        'scope':'Independent authored R09 body action and finite shoe gate. No cloth/hair bake or existing action/cache rewrite.'}
    runner_report=reports/'driver-report.json'
    try:
        bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
        if Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Blender did not open the exact requested source')
        body=runpy.run_path(str(Path(__file__).with_name('body_performance_r09.py')))
        relative=lambda path:path.relative_to(project_path().resolve()).as_posix()
        print('R09_BODY_DRIVER_CREATE',flush=True)
        creation=body['create_action'](report_relative=relative(reports/'action.json'))
        print('R09_BODY_DRIVER_VERIFY',flush=True)
        validation=body['validate_static_poses'](report_relative=relative(reports/'static-poses.json'))
        record.update(action=creation['action'],creationReport=str(reports/'action.json'),staticReport=str(reports/'static-poses.json'))
        if not validation['passedFiniteShoeCriteria']:
            raise RuntimeError('Static/between-key shoe check failed; no new Blend is saved')
        if _sha(source)!=source_sha:raise RuntimeError('Input source file changed externally during the run; refusing to publish this derivative')
        target.parent.mkdir(parents=True,exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        record.update(savedBlend=True,outputFileSha256=_sha(target),status='body-action-saved',
            finiteShoeGatePassed=True,actionLeftInactive=True,
            note='Saved copy retains its original active action. The new R09 action is present and inactive; select it only with matched/new dynamics or the static-preview helper.')
        runner_report.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('R09_BODY_ACTION_SAVED '+str(target),flush=True)
        if args.preview:
            preview=runpy.run_path(str(Path(__file__).with_name('body_pose_previews_r09.py')))
            preview['render_key_poses'](action_name=creation['action'],output_relative=relative(previews))
            record.update(status='body-action-saved-and-pose-previews-rendered',previewReport=str(previews/'render-report.json'))
        runner_report.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('R09_BODY_DRIVER_COMPLETE '+json.dumps({'savedBlend':record['savedBlend'],'report':str(runner_report)}),flush=True)
        return record
    except Exception as error:
        record.update(status='failed-after-save' if record['savedBlend'] else 'failed-no-save',error=str(error),traceback=traceback.format_exc())
        runner_report.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])

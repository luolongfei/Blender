"""Integrator-owned Blender --python runner; never starts another process.

With production reference-face-flow-r08.blend open:
  -- prepare --source work/reconstruction-r02/reference-face-flow-r08.blend
With current r07-material-sock.blend open:
  -- merge --runtime work/runtime-r06/r07-material-sock.blend
     --output work/runtime-r06/r08-geometry-face-refresh.blend
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','merge','restore'))
    parser.add_argument('--source',default='work/reconstruction-r02/reference-face-flow-r08.blend')
    parser.add_argument('--runtime',default='work/runtime-r06/r07-material-sock.blend')
    parser.add_argument('--stage',default='r08')
    parser.add_argument('--manifest',default='work/runtime-r06/r08-face-refresh.json')
    parser.add_argument('--merge-report',help='Optional explicit successful merge report for rollback; otherwise use the path recorded in the rig')
    parser.add_argument('--output')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    api=runpy.run_path(str(Path(__file__).with_name('runtime_face_refresh_r08.py')))
    if args.phase=='prepare':
        result=api['prepare_payload'](source_relative=args.source,stage=args.stage)
    else:
        if not args.output:raise ValueError('Supply an explicit NEW --output runtime Blend path')
        output=api['_path'](args.output)
        if output.suffix.lower()!='.blend' or output.exists():raise FileExistsError('Use a new Blend checkpoint: '+str(output))
        source=Path(bpy.data.filepath).resolve();source_sha=api['_sha'](source)
        if args.phase=='merge':result=api['replace'](manifest_relative=args.manifest,runtime_relative=args.runtime,recull=True)
        else:result=api['restore'](manifest_relative=args.manifest,merge_report_relative=args.merge_report)
        if api['_sha'](source)!=source_sha:raise RuntimeError('Input runtime file changed on disk')
        status=bpy.ops.wm.save_as_mainfile(filepath=str(output))
        if 'FINISHED' not in status or not output.is_file():raise RuntimeError('New checkpoint save failed')
        result.update(savedBlend=True,output=str(output),outputSha256=api['_sha'](output))
        output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RT08_FACE_DRIVER_DONE',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

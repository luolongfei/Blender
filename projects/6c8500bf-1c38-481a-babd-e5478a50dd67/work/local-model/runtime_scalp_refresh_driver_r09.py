"""Explicit Blender --python runner, never launches Blender or renders.

With final new production candidate open:
  -- prepare --source work/reconstruction-r02/ACTUAL-CANDIDATE.blend --stage r09-scalp
With current runtime open:
  -- apply --runtime work/runtime-r06/ACTUAL-RUNTIME.blend
     --manifest work/runtime-r06/r09-scalp/manifest.json
     --output work/runtime-r06/r09-scalp-candidate.blend
"""
import argparse,json,runpy,sys
from pathlib import Path
import bpy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','apply','restore'))
    parser.add_argument('--source');parser.add_argument('--runtime');parser.add_argument('--stage')
    parser.add_argument('--region',choices=('scalp','fringe'),default='scalp')
    parser.add_argument('--manifest');parser.add_argument('--output')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    api=runpy.run_path(str(Path(__file__).with_name('runtime_'+args.region+'_refresh_r09.py')))
    stage=args.stage or 'r09-'+args.region
    if args.phase=='prepare':
        if not args.source:raise ValueError('Explicit --source required')
        result=api['prepare_payload'](args.source,stage)
        print('RT09_REGION_PREPARED',json.dumps({'manifest':f'work/runtime-r06/{stage}/manifest.json','region':args.region,
            'nativeCurves':result['nativeCurves'],'groups':len(result['spec']['groups']),'savedBlend':False}),flush=True)
        return
    if not args.runtime or not args.output:raise ValueError('Explicit --runtime and new --output required')
    runtime=api['_path'](args.runtime,'work/runtime-r06');output=api['_path'](args.output,'work/runtime-r06')
    if runtime!=Path(bpy.data.filepath).resolve():raise RuntimeError('Opened runtime differs from explicit --runtime')
    if output.exists() or output.suffix.lower()!='.blend' or output.with_suffix('.json').exists():raise FileExistsError('New Blend/report checkpoint paths required')
    source_sha=api['_sha'](runtime)
    if args.phase=='apply':
        if not args.manifest:raise ValueError('Explicit --manifest required')
        result=api['apply_payload'](args.manifest)
    else:result={'restored':api['restore']()}
    if api['_sha'](runtime)!=source_sha:raise RuntimeError('Opened runtime source changed on disk')
    output.parent.mkdir(parents=True,exist_ok=True)
    status=bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if 'FINISHED' not in status or not output.is_file():raise RuntimeError('New runtime checkpoint save failed')
    result.update(savedBlend=True,output=str(output),outputSha256=api['_sha'](output),inputSha256=source_sha)
    api['_json'](output.with_suffix('.json'),result)
    print('RT09_SCALP_SAVED',json.dumps({'output':str(output),'sha256':result['outputSha256'],
        'triangles':result.get('triangles'),'appearanceAccepted':False}),flush=True)


if __name__=='__main__':main()

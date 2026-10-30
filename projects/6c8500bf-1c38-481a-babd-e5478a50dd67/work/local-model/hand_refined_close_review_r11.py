"""Three explicit close views of the actually integrated R11 hands.

Run one view per disposable Blender process. This adapter adds exact input
identity, the integrated-hand marker and evaluated camera metadata to the
frozen R11 review. It does not alter anatomy, lighting, pose guards or save.
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

ADAPTER_SHA='faa3d4768181aa67ec59d06594853cf18f8afa6449638487f19161b7d6bbb370'
INTEGRATOR_SHA='e95ad2f0dc4b103d71b9e4e4e2844ed519d3dce37d810803c89f24e5e6cb94d1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(source,source_sha,side,view,label,samples):
    import bpy
    project=project_path();path=(project/source).resolve();path.relative_to(project.resolve())
    if sha(path)!=source_sha or Path(bpy.data.filepath).resolve()!=path:
        raise RuntimeError('Exact current saved source SHA required')
    marker=json.loads(bpy.context.scene.get('R11_refined_hands_integrated','{}'))
    if marker.get('helperSha256')!=INTEGRATOR_SHA:
        raise RuntimeError('Expected actual successful refined-hand v6 integration')
    helper=Path(__file__).with_name('hand_asset_pose_review_r11.py')
    if sha(helper)!=ADAPTER_SHA:raise RuntimeError('Frozen R11 review changed')
    wrapper=runpy.run_path(str(helper));api=wrapper['_api']();camera_fn=api['_camera']
    def camera(*args,**kwargs):
        camera_object,record=camera_fn(*args,**kwargs)
        bpy.context.view_layer.update()
        evaluated=camera_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
        record['evaluatedCameraMatrixWorld']=[list(row) for row in evaluated.matrix_world]
        record['cameraType']=camera_object.data.type
        record['resolution']=[bpy.context.scene.render.resolution_x,bpy.context.scene.render.resolution_y]
        return camera_object,record
    api['run'].__globals__['_camera']=camera
    result=api['run'](source,side,'relaxed',view,label,samples)
    report_path=(project/result['report']).resolve();report_path.relative_to(project.resolve())
    report=json.loads(report_path.read_text(encoding='utf-8'))
    report['refinedHandsReview']={'helperSha256':sha(__file__),'frozenReviewSha256':ADAPTER_SHA,
        'integrationHelperSha256':INTEGRATOR_SHA,'scope':'Visible rest hand, nails and cuff. The separately recorded three-pose wrist core check remains a separate finite test.',
        'appearanceAccepted':False,'savedBlend':False}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--side',choices=('L','R'),default='R')
    parser.add_argument('--view',choices=('dorsal','palm','cuff'),required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--samples',type=int,choices=(24,32),default=32)
    a=parser.parse_args(argv)
    print('R11_REFINED_HAND_CLOSE '+json.dumps(run(a.source,a.source_sha,a.side,a.view,a.label,a.samples)),flush=True)


if __name__=='__main__':main()

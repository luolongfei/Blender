"""Render explicit saved-head expressions at genuine evaluated strengths.

Plan contains view (front/quarter/eyes/mouth), resolution, samples, fps and
frames [{label, keys: {exact shape key name: value}}]. Every frame is rendered;
the reusable encoder can consume render-report.json. Source stays unchanged.
"""
import argparse,hashlib,json,re,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();plan_path=(P/a.plan).resolve();out=(P/a.output).resolve()
    source.relative_to((P/'work/reconstruction-r02').resolve());plan_path.relative_to(P.resolve());out.relative_to((P/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh saved independent candidate required')
    plan=json.loads(plan_path.read_text(encoding='utf8'));frames=plan['frames'];view=plan['view']
    if view not in ('front','quarter','eyes','mouth') or not 1<=len(frames)<=180:raise ValueError('Explicit bounded review plan required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve earlier review files; use a fresh directory')
    out.mkdir(parents=True,exist_ok=True);head=bpy.data.objects['R11 Female head study'];keys=head.data.shape_keys.key_blocks
    baseline={k.name:float(k.value) for k in keys};requested=set().union(*(row['keys'] for row in frames))
    for row in frames:
        for name,value in row['keys'].items():
            if name not in keys or not name.startswith('R11 ') or not 0<=float(value)<=1:raise ValueError('Unsupported explicit expression '+name)
    scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
    if view=='eyes':
        centers=[]
        for side in ('L','R'):
            ev=bpy.data.objects['R11 Adult sclera '+side].evaluated_get(dg);co=np.asarray([ev.matrix_world@v.co for v in ev.data.vertices])
            centers.append((co.min(0)+co.max(0))*.5)
        target=np.mean(centers,axis=0);target[2]+=.003;scale=.115
    elif view=='mouth':
        target=np.array(head.matrix_world@head.data.vertices[8005].co,copy=True);target[2]-=.002;scale=.095
    else:
        ev=head.evaluated_get(dg);co=np.asarray([ev.matrix_world@v.co for v in ev.data.vertices]);target=(co.min(0)+co.max(0))*.5;target[2]=co[:,2].max()-.145;scale=.365
    if plan.get('hideGroom',view in ('front','quarter')):
        for obj in scene.objects:
            name=obj.name.lower()
            if obj.type not in ('MESH','CURVES','CURVE') or obj.hide_render:continue
            if obj.name.startswith(('R11 Recombed ','R11 Donor / ')):obj.hide_render=True
            elif not obj.name.startswith('R11 ') and (any(t in name for t in ('hair','forelock','scalp','gather','cheek wisps','tail','tie ring')) or name.startswith('r02 tie ')):obj.hide_render=True
    direction=np.asarray((.6,-.8,0.) if view=='quarter' else (0.,-1.,0.))
    camera=bpy.data.objects.new('R11 expression actual review camera',bpy.data.cameras.new('R11 expression review'))
    scene.collection.objects.link(camera);camera.location=Vector(target+direction*2.)
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=scale;camera.data.dof.use_dof=False;scene.camera=camera
    width,height=map(int,plan.get('resolution',[960,1080]));samples=int(plan.get('samples',24));fps=float(plan.get('fps',30))
    if not 320<=width<=1600 or not 240<=height<=1600 or not 8<=samples<=64 or not 1<=fps<=60:raise ValueError('Explicit bounded image settings required')
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
    scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    source_sha=sha(source);detail=api('final_detail_review_r09');binding=api('face_fibre_binding_r11')
    report={'source':a.source,'sourceSha256':source_sha,'sha256':source_sha,'plan':a.plan,'planSha256':sha(plan_path),'helperSha256':sha(__file__),
        'view':view,'resolution':[width,height],'samples':samples,'fps':fps,'framePattern':'frame-%04d.png','frames':[],
        'sourceSceneFrame':1,'savedKeyValues':baseline,'lights':detail['_lights'](scene),'world':detail['_world'](scene),
        'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
        'camera':{'matrixWorld':[list(r) for r in camera.matrix_world],'orthographicScale':scale,'depthOfField':False},
        'appearanceAccepted':False,'sourceSaved':False,
        'scope':'Actual independently evaluated explicit facial strengths, body/time fixed at saved scene frame 1. Each image is rendered without interpolation. Binding checks do not alone prove lid/lip contact, natural expression or physical collision.'}
    for number,row in enumerate(frames,1):
        for name in requested:keys[name].value=float(row['keys'].get(name,baseline[name]))
        bpy.context.view_layer.update()
        actual={name:float(keys[name].value) for name in requested}
        if any(abs(actual[name]-float(row['keys'].get(name,baseline[name])))>1e-6 for name in requested):raise RuntimeError('Driver or animation overrides requested expression')
        check=binding['validate'](neutral=False);png=out/('frame-%04d.png'%number);scene.render.filepath=str(png)
        bpy.ops.render.render(write_still=True)
        if not png.exists():raise RuntimeError('Rendered frame missing')
        item={'frame':number,'label':row.get('label',str(number)),'keys':actual,'imageSha256':sha(png),'fibreBinding':check}
        report['frames'].append(item)
        with png.with_suffix('.json').open('x',encoding='utf8') as f:json.dump({'sourceSha256':source_sha,**item},f,ensure_ascii=False,indent=2)
        print('R11_EXPRESSION_FRAME '+json.dumps({'frame':number,'keys':actual,'image':str(png)}),flush=True)
    if sha(source)!=source_sha:raise RuntimeError('Source changed while rendering')
    with (out/'render-report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()

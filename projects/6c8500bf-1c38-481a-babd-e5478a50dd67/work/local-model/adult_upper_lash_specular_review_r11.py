"""One actual no-specular upper-lash image paired with an existing source image.

Requires the identical saved source and verified prior coverage report/image.
Same 800x480 camera, lights, 32 samples and seed; no radius/geometry changes.
Zero specular is an isolation experiment only, never a production material.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def run(source_relative,baseline_relative,output_relative):
    import bpy
    import numpy as np
    from mathutils import Vector
    p=project_path().resolve();source=(p/source_relative).resolve();baseline_dir=(p/baseline_relative).resolve();out=(p/output_relative).resolve()
    source.relative_to(p/'work/reconstruction-r02')
    for path in (baseline_dir,out):path.relative_to(p/'previews/reconstruction-r02')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists():
        raise RuntimeError('Fresh saved source and unused output folder required')
    baseline=json.loads((baseline_dir/'report.json').read_text(encoding='utf8'));source_sha=sha(source)
    row=next(row for row in baseline['cases'] if row['case']=='source');baseline_image=(p/row['image']).resolve()
    baseline_image.relative_to(p/'previews/reconstruction-r02')
    if baseline['sourceSha256']!=source_sha or sha(baseline_image)!=row['imageSha256']:
        raise RuntimeError('Use the exact source of the already rendered baseline')
    if baseline['resolution']!=[800,480] or baseline['samples']!=32 or baseline['seed']!=29 or baseline['camera']['horizontalScaleMeters']!=.140:
        raise RuntimeError('Unexpected baseline camera/render settings')
    binding=api('face_fibre_binding_r11');coverage=api('adult_fibre_visibility_diagnostic_r11');detail=api('final_detail_review_r09')
    scene=bpy.context.scene;head=bpy.data.objects[binding['HEAD']];protected=binding['_protected'](head)
    frame=(scene.frame_current,scene.frame_subframe);scene.frame_set(1);bpy.context.view_layer.update()
    keys={k.name:float(k.value) for k in head.data.shape_keys.key_blocks} if head.data.shape_keys else {}
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    if keys!=baseline['headKeyValues'] or detail['_lights'](scene)!=baseline['lights'] or detail['_world'](scene)!=baseline['world'] or colour!=baseline['colourManagement']:
        raise RuntimeError('Baseline key/light/world/colour conditions differ')
    before=binding['validate'](neutral=False);objects=[bpy.data.objects[name] for name in coverage['NAMES']]
    contracts={o.name:binding['_curve_contract'](o.data) for o in objects};materials={o.name:coverage['_materials'](o) for o in objects}
    def points(obj):
        e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return (binding['_array'](e.data.position_data,'vector'),binding['_array'](e.data.attributes['radius'].data,'value',1))
    actual={o.name:points(o) for o in objects};originals={o.name:o.data for o in objects}
    target=np.asarray(baseline['camera']['target']);camera=bpy.data.objects.new('TEMP R11 upper lash specular camera',bpy.data.cameras.new('TEMP R11 upper lash specular camera'))
    scene.collection.objects.link(camera);camera.location=Vector(target+np.asarray([0.,-2.,0.]));camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.sensor_fit='HORIZONTAL';camera.data.ortho_scale=.140;camera.data.dof.use_dof=False
    prior_camera=scene.camera;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32;scene.cycles.seed=29;scene.cycles.use_denoising=True
    scene.render.resolution_x=800;scene.render.resolution_y=480;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    hidden={}
    for name in baseline['hiddenGroomOnly']:
        obj=bpy.data.objects[name];hidden[name]=obj.hide_render;obj.hide_render=True
    copied_data=[];copied_materials={};edits=[];out.mkdir(parents=True)
    try:
        for side in ('L','R'):
            obj=bpy.data.objects['R11 Adult lashes upper '+side]
            if len(obj.data.materials)!=1 or obj.data.materials[0] is None:raise RuntimeError('Expected one upper-lash material')
            old=obj.data.materials[0]
            if old.name not in copied_materials:
                material=old.copy();copied_materials[old.name]=material;changed=[]
                shaders=[n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
                if len(shaders)!=1:raise RuntimeError('Expected exact existing opaque Principled lash material')
                shader=shaders[0];socket=shader.inputs['Specular IOR Level']
                if socket.is_linked or shader.inputs['Coat Weight'].is_linked or float(shader.inputs['Coat Weight'].default_value)!=0.:
                    raise RuntimeError('Inspect linked or coated upper-lash shader before isolation')
                prior=float(socket.default_value);socket.default_value=0.
                if prior<=0.:raise RuntimeError('Source has no specular response to isolate')
                edits.append({'material':old.name,'input':'Specular IOR Level','before':prior,'temporaryAfter':0.})
            data=obj.data.copy();copied_data.append(data);obj.data=data;data.materials[0]=copied_materials[old.name]
        bpy.context.view_layer.update()
        for obj in objects:
            if any(not np.array_equal(a,b) for a,b in zip(actual[obj.name],points(obj))):raise RuntimeError('Upper-lash material isolation changed geometry or radii')
        png=out/'upper-no-specular.png';scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
        if not png.is_file():raise RuntimeError('Actual no-specular image missing')
        report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
                'baselineReport':str((baseline_dir/'report.json').relative_to(p)),'baselineReportSha256':sha(baseline_dir/'report.json'),
                'baselineImage':str(baseline_image.relative_to(p)),'baselineImageSha256':row['imageSha256'],
                'image':str(png.relative_to(p)),'imageSha256':sha(png),'edits':edits,'sourceSaved':False,
                'resolution':[800,480],'samples':32,'seed':29,'denoising':True,'camera':baseline['camera'],
                'actualSixFibreCenterlinesAndRadiiExact':True,'bindingBefore':before,'diagnosticOnly':True,'appearanceAccepted':False,
                'scope':'Only the two upper-lash specular levels are zero. Existing verified source render supplies the baseline. A response difference isolates specular contribution; does not by itself prove growth-direction anatomy or authorize a zero-specular production shader.'}
    finally:
        for obj in objects:obj.data=originals[obj.name]
        for name,value in hidden.items():bpy.data.objects[name].hide_render=value
        scene.camera=prior_camera;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
        for data in copied_data:
            if data.users==0:bpy.data.hair_curves.remove(data)
        for material in copied_materials.values():
            if material.users==0:bpy.data.materials.remove(material)
    if any(binding['_curve_contract'](o.data)!=contracts[o.name] or coverage['_materials'](o)!=materials[o.name] for o in objects):
        raise RuntimeError('Original fibre data/materials not restored')
    if binding['_protected'](head)!=protected or sha(source)!=source_sha:raise RuntimeError('Protected source changed')
    report['bindingAfterRestore']=binding['validate'](neutral=False)
    with (out/'report.json').open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_UPPER_LASH_SPECULAR_DIAGNOSTIC '+json.dumps({'output':output_relative,'sourceSha256':source_sha}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);run(args.source,args.baseline,args.output)

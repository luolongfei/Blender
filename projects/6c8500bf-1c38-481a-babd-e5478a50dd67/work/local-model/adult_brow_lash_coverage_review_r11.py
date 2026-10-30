"""Two temporary actual renders of brow/lash coverage, never a saved asset.

Source and restrained cosmetic-coating cases share camera, light, materials,
curve centers, binding, geometry, expression and seed. Only physical radius
changes in copied Curves data: brows 1.50, upper lashes 1.35, lower lashes 1.15.
The image is diagnostic; this does not establish anatomy, makeup or acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

FACTORS = {**{'R11 Adult natural brow '+s: 1.50 for s in ('L', 'R')},
           **{'R11 Adult lashes upper '+s: 1.35 for s in ('L', 'R')},
           **{'R11 Adult lashes lower '+s: 1.15 for s in ('L', 'R')}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def run(source_relative, output_relative):
    import bpy
    import numpy as np
    from mathutils import Vector
    p = project_path().resolve(); source = (p/source_relative).resolve(); out = (p/output_relative).resolve()
    source.relative_to(p/'work/reconstruction-r02'); out.relative_to(p/'previews/reconstruction-r02')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists():
        raise RuntimeError('Fresh saved independent source and unused output directory required')
    source_sha = sha(source); scene = bpy.context.scene
    binding = api('face_fibre_binding_r11'); optics = api('adult_fibre_visibility_diagnostic_r11')
    head=bpy.data.objects['R11 Female head study'];head_before=binding['_protected'](head)
    frame = (scene.frame_current, scene.frame_subframe)
    scene.frame_set(1); bpy.context.view_layer.update()
    detail = api('final_detail_review_r09')
    before_binding = binding['validate'](neutral=False)
    objects = [bpy.data.objects[name] for name in FACTORS]
    originals = {o.name: o.data for o in objects}
    contracts = {o.name: binding['_curve_contract'](o.data) for o in objects}
    def actual(o):
        e = o.evaluated_get(bpy.context.evaluated_depsgraph_get()); data=e.data
        return {'points': binding['_array'](data.position_data,'vector'),
                'counts': np.asarray([c.points_length for c in data.curves],np.int32),
                'radii': binding['_array'](data.attributes['radius'].data,'value',1).ravel(),
                'matrix': np.asarray(e.matrix_world)}
    original_actual = {o.name: actual(o) for o in objects}
    eyes=[]
    for side in ('L','R'):
        e=bpy.data.objects['R11 Adult sclera '+side].evaluated_get(bpy.context.evaluated_depsgraph_get())
        co=binding['_transform'](binding['_array'](e.data.vertices,'co'),e.matrix_world)
        eyes.append((co.min(0)+co.max(0))*.5)
    target=np.mean(eyes,axis=0); target[2]+=.010
    camera=bpy.data.objects.new('TEMP R11 brow-lash coverage camera',bpy.data.cameras.new('TEMP R11 coverage camera'))
    scene.collection.objects.link(camera); camera.location=Vector(target+np.asarray([0.,-2.,0.]))
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.sensor_fit='HORIZONTAL';camera.data.ortho_scale=.140;camera.data.dof.use_dof=False
    original_camera=scene.camera;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32;scene.cycles.seed=29;scene.cycles.use_denoising=True
    scene.render.resolution_x=800;scene.render.resolution_y=480;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    # Match structural face reviews: remove only existing groom occlusion, for
    # both cases identically. Actual brow and lash objects stay visible.
    hidden={}
    for obj in scene.objects:
        name=obj.name.lower()
        if obj.type not in ('MESH','CURVES','CURVE') or obj.hide_render:continue
        if obj.name.startswith(('R11 Recombed ','R11 Donor / ')) or (not obj.name.startswith('R11 ') and
                (any(t in name for t in ('hair','forelock','scalp','gather','cheek wisps','tail','tie ring')) or name.startswith('r02 tie '))):
            hidden[obj.name]=obj.hide_render;obj.hide_render=True
    keys={k.name:float(k.value) for k in head.data.shape_keys.key_blocks} if head.data.shape_keys else {}
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
            'resolution':[800,480],'samples':32,'seed':29,'denoising':True,'sourceSaved':False,'diagnosticOnly':True,
            'appearanceAccepted':False,'temporaryRadiusFactors':FACTORS,'headKeyValues':keys,
            'camera':{'target':target.tolist(),'horizontalScaleMeters':.140,'depthOfField':False},
            'bindingBefore':before_binding,'lights':detail['_lights'](scene),'world':detail['_world'](scene),
            'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
            'hiddenGroomOnly':list(hidden),'cases':[],
            'scope':'Same geometry, exact strand centerlines/binding/materials and lighting. Radius-only diagnostic; no saved makeup or relaxed binding contracts.'}
    out.mkdir(parents=True)
    copied=[]
    try:
        for case in ('source','restrained-coating'):
            if case=='restrained-coating':
                for obj in objects:
                    data=originals[obj.name].copy();copied.append(data);obj.data=data
                    attr=data.attributes['radius'];radius=binding['_array'](attr.data,'value',1).ravel()
                    attr.data.foreach_set('value',radius*np.float32(FACTORS[obj.name]))
            bpy.context.view_layer.update();evidence={}
            for obj in objects:
                measured=actual(obj);base=original_actual[obj.name]
                if any(not np.array_equal(measured[k],base[k]) for k in ('points','counts','matrix')):
                    raise RuntimeError('Radius diagnostic changed actual centerlines, counts or transform '+obj.name)
                expected=base['radii']*(np.float32(FACTORS[obj.name]) if case!='source' else np.float32(1.))
                if not np.array_equal(measured['radii'],expected):raise RuntimeError('Actual radius does not match requested coating '+obj.name)
                evidence[obj.name]=optics['_measure'](obj,camera,scene)
            png=out/(case+'.png');scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
            if not png.is_file():raise RuntimeError('Actual diagnostic image missing')
            report['cases'].append({'case':case,'image':str(png.relative_to(p)),'imageSha256':sha(png),'fibres':evidence,
                                    'centerlinesCountsAndTransformsExact':True,'evaluatedRadiusFormulaExact':True})
            # Persist each real image's provenance even if a later case fails.
            with png.with_suffix('.json').open('x',encoding='utf8') as stream:
                json.dump({'sourceSha256':source_sha,**report['cases'][-1]},stream,ensure_ascii=False,indent=2)
            print('R11_BROW_LASH_COVERAGE '+json.dumps({'case':case,'image':str(png)}),flush=True)
    finally:
        for obj in objects:obj.data=originals[obj.name]
        for name,value in hidden.items():bpy.data.objects[name].hide_render=value
        scene.camera=original_camera;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
        for data in copied:
            if data.users==0:bpy.data.hair_curves.remove(data)
    if any(binding['_curve_contract'](o.data)!=contracts[o.name] for o in objects):raise RuntimeError('Original fibre data was not restored')
    if binding['_protected'](head)!=head_before:raise RuntimeError('Protected head/key/material data changed')
    if sha(source)!=source_sha:raise RuntimeError('Saved source changed')
    report['bindingAfterRestore']=binding['validate'](neutral=False)
    report['originalFibreDataAndHeadRestored']=True
    report['temporaryRenderSettingsNotSaved']=True
    with (out/'report.json').open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_BROW_LASH_COVERAGE_COMPLETE '+json.dumps({'output':output_relative,'sourceSha256':source_sha}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);run(args.source,args.output)

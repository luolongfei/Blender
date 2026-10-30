"""Two in-memory lip-normal diagnostics on the current natural-rest source.

Compare original / nominal 5 um only. Preserve every static sculpt, current
seal, face maps, optical constants, roughness, light and camera. Explicit public
expressions alone are set to rest. No Blend is saved and no appearance decision
is encoded. Current material/normal and pixel-scale evidence is captured here,
so a separate material-observer process is not necessary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HELPER = 'lip_microrelief_candidate_r11_v2.py'
HELPER_SHA = 'e289a7bb912497b7d4efa638ac976148c99c3968b26340edeb1d51032acbba84'
PUBLIC = ('R11 blink L', 'R11 blink R', 'R11 gentle smile',
          'R11 natural soft smile', 'R11 warm closed smile', 'R11 jaw open')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plain(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, 'name_full'):
        return {'idType': value.bl_rna.identifier, 'name': value.name_full}
    try:
        return [plain(x) for x in value]
    except TypeError:
        raise RuntimeError('Unsupported actual material value: '+type(value).__name__)


def graph(tree):
    if tree is None:
        return None
    nodes=[]
    for n in tree.nodes:
        row={'name':n.name, 'type':n.bl_idname, 'inputs':[]}
        for prop in ('attribute_name','layer_name','operation','noise_dimensions',
                     'normalize','subsurface_method','distribution','is_active_output'):
            if hasattr(n,prop): row[prop]=plain(getattr(n,prop))
        for socket in n.inputs:
            item={'name':socket.name,'identifier':socket.identifier,'type':socket.bl_idname,
                  'linked':socket.is_linked}
            for prop in ('enabled','is_unavailable'):
                if hasattr(socket,prop): item[prop]=bool(getattr(socket,prop))
            if hasattr(socket,'default_value'): item['default']=plain(socket.default_value)
            row['inputs'].append(item)
        if n.type=='TEX_IMAGE' and n.image:
            row['image']={'name':n.image.name,'size':list(n.image.size),
                          'colourSpace':n.image.colorspace_settings.name,
                          'packed':bool(n.image.packed_file)}
        nodes.append(row)
    return {'nodes':nodes,'links':sorted([[l.from_node.name,l.from_socket.identifier,
             l.to_node.name,l.to_socket.identifier] for l in tree.links])}


def material_observation(head, evaluated, camera, scene, module):
    import bpy
    import numpy as np
    candidates=[(i,s.material) for i,s in enumerate(head.material_slots) if s.material
        and s.material.use_nodes and any(n.type=='VERTEX_COLOR' and n.layer_name==module['COLOR']
        for n in s.material.node_tree.nodes)]
    if len(candidates)!=1: raise RuntimeError('Actual current refined skin material ambiguous')
    slot,material=candidates[0]
    shader=[n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
    if len(shader)!=1: raise RuntimeError('Expected one actual current skin shader')
    def ancestors(socket):
        seen=set();stack=[l.from_node for l in socket.links]
        while stack:
            n=stack.pop()
            if n.name in seen: continue
            seen.add(n.name);stack.extend(l.from_node for s in n.inputs for l in s.links)
        return sorted(seen)
    chains={name:ancestors(shader[0].inputs[name]) for name in ('Base Color','Roughness','Normal')}
    rows={}
    for name in (module['REST'], module['MASK'], module['COLOR'], 'R11 refined natural skin roughness'):
        attr=head.data.attributes.get(name)
        if attr is None: raise RuntimeError('Missing current material input: '+name)
        field,width={'FLOAT_VECTOR':('vector',3),'FLOAT_COLOR':('color',4),'FLOAT':('value',1)}[attr.data_type]
        values=module['_array'](attr.data,field,width)
        rows[name]={'domain':attr.domain,'type':attr.data_type,'count':len(values),
                    'sha256':module['_ah'](values),'min':values.min(0).tolist(),
                    'median':np.median(values,axis=0).tolist(),'max':values.max(0).tolist()}
    normal_noises=[n for n in material.node_tree.nodes if n.name in chains['Normal'] and n.type=='TEX_NOISE']
    projections=np.asarray(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
        x=scene.render.resolution_x,y=scene.render.resolution_y,scale_x=1.,scale_y=1.),float)
    pixel_world=[2/(abs(projections[0,0])*scene.render.resolution_x),
                 2/(abs(projections[1,1])*scene.render.resolution_y)]
    # The rest field is transported with the surface. Compare actual L3 edge
    # lengths inside the visible lip mask; this is not a measured pore size.
    mesh=evaluated.data
    rest=module['_array'](mesh.attributes[module['REST']].data,'vector').astype(float)
    mask=module['_array'](mesh.attributes[module['MASK']].data,'value',1).ravel()
    edges=module['_array'](mesh.edges,'vertices',2,np.int32)
    selected=edges[np.all(mask[edges]>.1,axis=1)]
    if len(selected)<40: raise RuntimeError('Too few actual lip edges for scale observation')
    co=module['_array'](mesh.vertices,'co').astype(float)
    matrix=np.asarray(evaluated.matrix_world,float)
    world=co@matrix[:3,:3].T+matrix[:3,3]
    a=np.linalg.norm(rest[selected[:,1]]-rest[selected[:,0]],axis=1)
    b=np.linalg.norm(world[selected[:,1]]-world[selected[:,0]],axis=1)
    valid=(a>1e-12)&np.isfinite(a)&np.isfinite(b)
    ratios=np.percentile(b[valid]/a[valid],[5,50,95]).tolist()
    return {'slot':slot,'material':material.name,'activeInputAncestors':chains,
        'materialGraph':graph(material.node_tree),'worldGraph':graph(scene.world.node_tree) if scene.world else None,
        'attributes':rows,'pixelWorldUnitsXY':pixel_world,'actualProjectionMatrix':projections.tolist(),
        'actualLipEdgeWorldToRestLengthRatiosP5P50P95':ratios,
        'noiseScaleInputs':[{'name':n.name,'linked':n.inputs['Scale'].is_linked,
                            'value':float(n.inputs['Scale'].default_value)} for n in normal_noises],
        'sceneUnitScaleLength':scene.unit_settings.scale_length,
        'scaleMeaning':'1/NoiseScale is a nominal rest-coordinate scale, not measured pore diameter. Edge ratios capture current lip deformation; directional distortion and sampling/denoising remain separate.'}


def review(source_relative, output_relative, expected_sha, samples=64):
    import bpy
    import numpy as np
    from mathutils import Vector
    p=project_path().resolve();source=(p/source_relative).resolve();output=(p/output_relative).resolve()
    source.relative_to(p/'work/reconstruction-r02');output.relative_to(p/'previews/reconstruction-r02')
    if source.suffix!='.blend' or Path(bpy.data.filepath).resolve()!=source or not bpy.app.background or bpy.data.is_dirty:
        raise RuntimeError('Fresh saved independent source required')
    if sha(source)!=expected_sha or not 32<=samples<=128: raise RuntimeError('Source hash or samples differ')
    if output.exists() and any(output.iterdir()): raise RuntimeError('Preserve existing diagnostic images')
    helper=Path(__file__).with_name(HELPER)
    if sha(helper)!=HELPER_SHA: raise RuntimeError('Pinned current material helper changed')
    module=runpy.run_path(str(helper))
    if module['MARKER'] in bpy.context.scene: raise RuntimeError('Use the unchanged original material')
    scene=bpy.context.scene;head=bpy.data.objects[module['HEAD']];keys=head.data.shape_keys.key_blocks
    if 'R11 relaxed lip seal' not in keys or float(keys['R11 relaxed lip seal'].value)!=1.:
        raise RuntimeError('Review the actually saved strength-one rest seal')
    if float(head.get('R11 lower face continuity',0.))!=0.: raise RuntimeError('Rejected lower-face v1 remains active')
    original_frame=(scene.frame_current,scene.frame_subframe);old_camera=scene.camera
    original_public={n:float(keys[n].value) for n in PUBLIC if n in keys}
    saved_values={k.name:float(k.value) for k in keys}
    camera=None
    try:
        scene.frame_set(1);bpy.context.view_layer.update()
        driver_paths=set() if not head.data.shape_keys.animation_data else {
            f.data_path for f in head.data.shape_keys.animation_data.drivers}
        if any(keys[n].path_from_id('value') in driver_paths for n in original_public):
            raise RuntimeError('Public expression is driven; inspect actual control before reset')
        for name in original_public: keys[name].value=0.
        head.update_tag();bpy.context.view_layer.update()
        fixed_values={k.name:float(k.value) for k in keys}
        for name in ('R11 relaxed lip seal','R11 refined upper lid roll','R11 lower face contour v2'):
            if name not in keys or float(keys[name].value)!=1.: raise RuntimeError('Required selected sculpt changed: '+name)
        if 'R11 refined outer eye corners' not in keys or float(keys['R11 refined outer eye corners'].value) != 0.:
            raise RuntimeError('Current rejected radial field must remain zero')
        ev=head.evaluated_get(bpy.context.evaluated_depsgraph_get())
        attr=ev.data.attributes.get(module['MASK'])
        if attr is None or attr.domain!='POINT' or attr.data_type!='FLOAT': raise RuntimeError('Actual lip support missing')
        mask=module['_array'](attr.data,'value',1).ravel();ids=np.flatnonzero(mask>.1)
        if len(ids)<40: raise RuntimeError('Actual mouth support insufficient')
        points=np.asarray([ev.matrix_world@ev.data.vertices[int(i)].co for i in ids],float)
        lo,hi=points.min(0),points.max(0);target=(lo+hi)*.5;width=float(hi[0]-lo[0])
        if not .020<width<.080: raise RuntimeError('Actual evaluated mouth bounds differ')
        camera=bpy.data.objects.new('R11 refined lip microrelief diagnostic',bpy.data.cameras.new('R11 refined lip diagnostic'))
        scene.collection.objects.link(camera);camera.location=Vector(target+np.array([0.,-2.,0.]))
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.type='ORTHO';camera.data.ortho_scale=max(.085,width*1.7);camera.data.dof.use_dof=False;scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples
        scene.cycles.seed=41;scene.cycles.use_animated_seed=False;scene.cycles.use_denoising=False
        scene.render.resolution_x=960;scene.render.resolution_y=600;scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG';output.mkdir(parents=True,exist_ok=True)
        detail=runpy.run_path(str(Path(__file__).with_name('final_detail_review_r09.py')))
        bpy.context.view_layer.update()
        def setting_contract():
            return {'frame':[scene.frame_current,scene.frame_subframe],
                'cameraMatrix':[list(r) for r in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world],
                'orthographicScale':camera.data.ortho_scale,'depthOfField':camera.data.dof.use_dof,
                'resolution':[scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage],
                'samples':scene.cycles.samples,'seed':scene.cycles.seed,'animatedSeed':scene.cycles.use_animated_seed,
                'denoising':scene.cycles.use_denoising,'lights':detail['_lights'](scene),'world':detail['_world'](scene),
                'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
                                    'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
                'allKeyValues':{k.name:float(k.value) for k in keys},
                'headScalars':{n:float(head[n]) for n in ('R11 lower face continuity','R11 upper lid relief') if n in head}}
        fixed=setting_contract()
        report={'source':source_relative,'sourceSha256':expected_sha,'reviewHelperSha256':sha(__file__),
            'materialHelper':HELPER,'materialHelperSha256':HELPER_SHA,'savedKeyValues':saved_values,
            'explicitPublicRestValues':{n:0. for n in original_public},'fixedSettings':fixed,
            'materialObservation':material_observation(head,head.evaluated_get(bpy.context.evaluated_depsgraph_get()),camera,scene,module),
            'evaluatedMouthBounds':[lo.tolist(),hi.tolist()],'cases':[],
            'scope':'Two actual same-source normal-only diagnostics; no source saved, no appearance/contact acceptance.',
            'blendSaved':False,'appearanceAccepted':False}
        for name,microns in [('original',None),('5um',5.)]:
            installed=False;row={'case':name,'nominalNewHeightMicrons':microns}
            try:
                if microns is not None: row['candidate']=module['apply'](microns);installed=True
                bpy.context.view_layer.update()
                if setting_contract()!=fixed or {k.name:float(k.value) for k in keys}!=fixed_values:
                    raise RuntimeError('Current static shape, expression or render setup changed')
                png=output/(name+'.png');scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
                if not png.is_file() or sha(source)!=expected_sha: raise RuntimeError('Image or source identity check failed')
                row.update(image=png.relative_to(p).as_posix(),imageSha256=sha(png))
            finally:
                if installed: row['restore']=module['restore']()
            report['cases'].append(row)
            print('R11_REFINED_LIP_MICRORELIEF_FRAME '+json.dumps({'case':name,'image':row['image']}),flush=True)
        report['sameSettingsAllCases']=setting_contract()==fixed
        if not report['sameSettingsAllCases']: raise RuntimeError('Final diagnostic settings differ')
        with (output/'comparison.json').open('x',encoding='utf8') as stream: json.dump(report,stream,ensure_ascii=False,indent=2)
        print('R11_REFINED_LIP_MICRORELIEF_REVIEW_DONE '+json.dumps({'output':output_relative}),flush=True)
        return report
    finally:
        for name,value in original_public.items(): keys[name].value=value
        scene.camera=old_camera;scene.frame_set(original_frame[0],subframe=original_frame[1]);bpy.context.view_layer.update()
        if camera:
            data=camera.data;bpy.data.objects.remove(camera,do_unlink=True)
            if data.users==0: bpy.data.cameras.remove(data)
        if sha(source)!=expected_sha: raise RuntimeError('Saved source changed during diagnostics')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-source-sha256',required=True);parser.add_argument('--samples',type=int,default=64)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source,args.output,args.expected_source_sha256,args.samples)

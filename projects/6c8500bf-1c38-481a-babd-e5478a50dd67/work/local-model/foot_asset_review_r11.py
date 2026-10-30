"""Bounded bare-foot anatomy views before fitting the actual CC0 foot asset.

Uses an explicit current source or a temporarily appended Foot - Realistic
OBJECT. Both use the same neutral clay inspection setup; shoes/socks are only
temporarily hidden for inspection. No source, rig, footwear or geometry save.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def review(source_relative,source_sha,case,view,label,side='L'):
    import bpy
    from mathutils import Vector
    source=_path(source_relative)
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or _sha(source)!=source_sha:
        raise RuntimeError('Exact source SHA in disposable background Object Mode required')
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label):raise ValueError('Use unique safe label')
    folder=_path('previews/reconstruction-r02/'+label);folder.mkdir(parents=True,exist_ok=True)
    image=folder/(case+'-'+side.lower()+'-'+view+'.png');report=image.with_suffix('.json')
    if image.exists() or report.exists():raise FileExistsError('Preserve prior foot inspection images')
    observer=runpy.run_path(str(Path(__file__).with_name('hand_asset_observe_r10.py')))
    original_ids=observer['_ids'](bpy);scene=bpy.context.scene;old_camera=scene.camera
    original_visibility={o.as_pointer():(o,o.hide_render) for o in scene.objects}
    old_override=bpy.context.view_layer.material_override;subject=None;success=False;result={}
    try:
        if case=='donor':
            catalog=json.loads(_path('work/reconstruction-r02/hand-asset-inventory-r10.json').read_text(encoding='utf-8'))
            bundle=_path(catalog['bundle'])
            if 'Foot - Realistic' not in catalog['libraryNames']['objects'] or _sha(bundle)!=catalog['bundleSha256']:
                raise RuntimeError('Exact foot object inventory/bundle mismatch')
            with bpy.data.libraries.load(str(bundle),link=False) as (available,requested):
                if 'Foot - Realistic' not in available.objects:raise RuntimeError('Foot is not an available object')
                requested.objects=['Foot - Realistic']
            subject=requested.objects[0]
            if subject is None or subject.type!='MESH':raise RuntimeError('Actual foot object is not a mesh')
            scene.collection.objects.link(subject)
            # Actual donor nail islands establish this as the left-oriented
            # foot: the broad hallux nail is on local -X. No mirroring is done.
            rotation=subject.matrix_world.to_3x3()
            medial=(rotation@Vector((-1,0,0))).normalized();up=(rotation@Vector((0,0,1))).normalized();forward=(rotation@Vector((0,-1,0))).normalized()
            evidence={'bundle':catalog['bundle'],'bundleSha256':catalog['bundleSha256'],
                      'asset':'Foot - Realistic','category':'objects','mirrored':False,
                      'donorSideEvidence':'Observed five nail islands: largest hallux plate centered at local X=-0.01491 m; original donor coordinates retained.'}
            if side!='L':raise ValueError('Native donor inspection is left-oriented; use --side L')
        else:
            if scene.frame_current!=1 or scene.frame_subframe!=0:raise RuntimeError('Keep actual current frame 1')
            subject=bpy.data.objects['Body_Complete_Shin_Foot_'+side]
            rig=bpy.data.objects['Character_Rig'];prefix='left' if side=='L' else 'right'
            bone=rig.data.bones[prefix+'Foot'];pose=rig.pose.bones[prefix+'Foot']
            transform=rig.matrix_world@pose.matrix@bone.matrix_local.inverted()
            medial=(transform.to_3x3()@Vector((-1 if side=='L' else 1,0,0))).normalized()
            up=(transform.to_3x3()@Vector((0,0,1))).normalized();forward=(transform.to_3x3()@Vector((0,-1,0))).normalized()
            evidence={'object':subject.name,'frame':1,'posePosition':rig.data.pose_position,
                      'footBone':prefix+'Foot','footPoseMatrix':np.asarray(pose.matrix).tolist(),
                      'footRestMatrix':np.asarray(bone.matrix_local).tolist(),'footwearGeometryEdited':False}
        graph=bpy.context.evaluated_depsgraph_get();ev=subject.evaluated_get(graph);mesh=ev.to_mesh()
        try:
            points=np.asarray([list(ev.matrix_world@v.co) for v in mesh.vertices],np.float64)
            vertex_count=len(mesh.vertices);face_count=len(mesh.polygons)
        finally:ev.to_mesh_clear()
        target=Vector((points.min(0)+points.max(0))*.5)
        directions={'dorsal':up+forward*.25,'medial':medial,'lateral':-medial,'sole':-up,'quarter':up*.55+medial+forward*.55}
        direction=directions[view].normalized()
        camera=bpy.data.objects.new('R11 foot anatomy camera',bpy.data.cameras.new('R11 foot anatomy camera'));scene.collection.objects.link(camera)
        camera.data.type='ORTHO';camera.data.dof.use_dof=False;camera.data.clip_start=.01;camera.data.clip_end=100
        utility=runpy.run_path(str(Path(__file__).with_name('reference_final_stills_r08.py')))
        scene.render.resolution_x=768;scene.render.resolution_y=640;scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=1.;scene.render.pixel_aspect_y=1.
        camera_up=forward if view in ('dorsal','sole') else up
        utility['_aim'](camera,target+direction*.65,target,camera_up)
        camera.data.ortho_scale=utility['_fit_scale'](camera,[Vector(p) for p in points],1.12)
        scene.camera=camera
        clay=bpy.data.materials.new('R11 temporary neutral foot clay');clay.use_nodes=True
        shader=next(n for n in clay.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=(.34,.34,.34,1);shader.inputs['Roughness'].default_value=.58
        bpy.context.view_layer.material_override=clay
        for obj in scene.objects:
            if obj not in (subject,camera) and obj.type not in ('ARMATURE','EMPTY','CAMERA'):obj.hide_render=True
        subject.hide_render=False
        for name,position,energy,size in (
            ('key',target+direction*.45+up*.25-medial*.18,35.,.28),
            ('fill',target+direction*.3-up*.12+medial*.25,12.,.3)):
            light=bpy.data.lights.new('R11 foot '+name,'AREA');light.energy=energy;light.shape='DISK';light.size=size
            obj=bpy.data.objects.new(light.name,light);scene.collection.objects.link(obj);obj.location=position
            obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32;scene.cycles.use_denoising=True
        scene.render.resolution_x=768;scene.render.resolution_y=640;scene.render.resolution_percentage=100
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_sequencer=False
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(image)
        bpy.ops.render.render(write_still=True)
        result={'source':source_relative,'sourceSha256':source_sha,'case':case,'view':view,'side':side,'evidence':evidence,
                'actualEvaluatedVertices':vertex_count,'actualEvaluatedFaces':face_count,
                'worldBounds':[points.min(0).tolist(),points.max(0).tolist()],
                'orthographicScale':camera.data.ortho_scale,'samples':32,'neutralClayInspection':True,
                'shoesAndSocksTemporarilyHidden':True,'image':str(image.relative_to(project_path())),'imageSha256':_sha(image),
                'helperSha256':_sha(__file__),'savedBlend':False,'appearanceAccepted':False,
                'scope':'Visible actual foot/ankle form only. No fit, rig acceptance, shoe/sock clearance or final skin/nail material claim.'}
        success=True
    finally:
        bpy.context.view_layer.material_override=old_override;scene.camera=old_camera
        for obj,hidden in original_visibility.values():obj.hide_render=hidden
        added=[item for pointer,item in observer['_ids'](bpy).items() if pointer not in original_ids]
        if added:bpy.data.batch_remove(ids=added)
        bpy.context.view_layer.update()
        if set(observer['_ids'](bpy))!=set(original_ids):raise RuntimeError('Foot inspection failed temporary-data cleanup')
        if _sha(source)!=source_sha:raise RuntimeError('Foot inspection source changed on disk')
    if success:
        with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--case',choices=('current','donor'),required=True);parser.add_argument('--view',choices=('dorsal','medial','lateral','sole','quarter'),required=True)
    parser.add_argument('--side',choices=('L','R'),default='L');parser.add_argument('--label',required=True);args=parser.parse_args(argv)
    print('R11_FOOT_REVIEW '+json.dumps(review(args.source,args.source_sha,args.case,args.view,args.label,args.side),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

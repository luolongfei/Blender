import bpy, sys, json, hashlib, shutil, math
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
OUT=workspace_path('Character_Rebuild')
for part in ['originals','stages','checks/before','reports','textures']:
    (OUT/part).mkdir(parents=True,exist_ok=True)
src=workspace_path('character-hq-textured.glb')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
backup=OUT/'originals/character-hq-textured.glb'
if not backup.exists(): shutil.copy2(src,backup)
assert digest(src)==digest(backup)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(backup))
objs=[o for o in bpy.context.scene.objects if o.type=='MESH']
for i,o in enumerate(objs):
    o.name='Original_Character' if i==0 else 'Original_Character_'+str(i)
    bpy.context.view_layer.objects.active=o
    o.select_set(True)
    bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
    o.select_set(False)
    o.data.calc_loop_triangles()
report={'input':str(src),'sha256':digest(src),'backupVerified':True,'Blender':bpy.app.version_string,'objects':[]}
for o in objs:
    report['objects'].append({'name':o.name,'vertices':len(o.data.vertices),'triangles':len(o.data.loop_triangles),'polygons':len(o.data.polygons),'uvLayers':list(o.data.uv_layers.keys()),'bbox':[[min((o.matrix_world@v.co)[a] for v in o.data.vertices),max((o.matrix_world@v.co)[a] for v in o.data.vertices)] for a in range(3)]})
report['images']=[{'name':im.name,'size':list(im.size),'packed':bool(im.packed_file)} for im in bpy.data.images]
scene=bpy.context.scene
scene.render.engine='BLENDER_EEVEE'
scene.render.resolution_x=1000;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.film_transparent=False
scene.world=bpy.data.worlds.new('Neutral_Review_World');scene.world.use_nodes=True
bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(.72,.72,.72,1);bg.inputs['Strength'].default_value=.6
scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast' if False else 'None'
scene.view_settings.exposure=0;scene.view_settings.gamma=1
def light(name,loc,power,size):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size
    ob=bpy.data.objects.new(name,d);scene.collection.objects.link(ob);ob.location=loc;ob.rotation_euler=(Vector((0,0,.25))-ob.location).to_track_quat('-Z','Y').to_euler()
light('Review_Key',(-3,-4,5),350,5)
light('Review_Fill',(3,-2,2),200,4)
light('Review_Back',(0,3,3),250,4)
d=bpy.data.cameras.new('Review_Camera');cam=bpy.data.objects.new('Review_Camera',d);scene.collection.objects.link(cam);scene.camera=cam;d.type='ORTHO';d.lens=50
views=[('front',(0,-5,0),(0,0,0),2.2),('right',(5,0,0),(0,0,0),2.2),('back',(0,5,0),(0,0,0),2.2),('quarter',(3,-5,0),(0,0,0),2.2),('face-front',(0,-4,.78),(0,0,.78),.64),('face-side',(4,0,.78),(0,0,.78),.64),('hand-top',(.86,0,3),(.86,0,.545),.34),('hand-palm',(.86,0,-3),(.86,0,.545),.34)]
report['reviewViews']=[{'name':n,'eye':eye,'target':target,'ortho':ortho} for n,eye,target,ortho in views]
(OUT/'reports/stage-a.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'stages/A-imported-original.blend'))
for n,eye,target,ortho in views:
    cam.location=eye;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();d.ortho_scale=ortho
    scene.render.filepath=str(OUT/'checks/before'/f'{n}.png')
    bpy.ops.render.render(write_still=True)
print('STAGE_A_DONE',json.dumps(report['objects']))

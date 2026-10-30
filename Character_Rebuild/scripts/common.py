import bpy,sys,json,math
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
OUT=workspace_path('Character_Rebuild')

def neutral_review():
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=1000;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.film_transparent=False
    scene.view_settings.view_transform='Standard'
    scene.view_settings.look='None';scene.view_settings.exposure=-.35
    scene.view_settings.gamma=1
    background=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    background.inputs['Strength'].default_value=.35
    for name,power in [('Review_Key',150),('Review_Fill',100),('Review_Back',100)]:
        bpy.data.objects[name].data.energy=power
    fill=bpy.data.objects.get('Review_FrontFill')
    if fill is None:
        data=bpy.data.lights.new('Review_FrontFill','AREA');data.shape='DISK';data.size=5
        fill=bpy.data.objects.new('Review_FrontFill',data);scene.collection.objects.link(fill)
    fill.data.energy=40;fill.location=(0,-3,1.5)
    fill.rotation_euler=(Vector((0,0,.6))-fill.location).to_track_quat('-Z','Y').to_euler()
    # Sample the real area-light dimensions rather than a hard shadow from
    # each light's centre. Identical settings are used for before/after/import.
    for ob in scene.objects:
        if ob.type=='LIGHT' and ob.name.startswith('Review_'):
            ob.data.use_shadow_jitter=True
    scene.eevee.taa_render_samples=96
    scene.eevee.shadow_ray_count=3
    scene.camera.data.dof.use_dof=False
    return scene

def render_views(destination,names=None):
    dest=OUT/destination;dest.mkdir(parents=True,exist_ok=True)
    scene=neutral_review();cam=scene.camera
    views=json.loads((OUT/'reports/stage-a.json').read_text())['reviewViews']
    views += [dict(name='face-quarter',eye=(3,-5,.78),target=(0,0,.78),ortho=.64),dict(name='face-other-side',eye=(-4,0,.78),target=(0,0,.78),ortho=.64),dict(name='hand-other-top',eye=(-.86,0,3),target=(-.86,0,.545),ortho=.34),dict(name='hand-other-palm',eye=(-.86,0,-3),target=(-.86,0,.545),ortho=.34),dict(name='hand-oblique',eye=(1.2,-1.6,1.8),target=(.86,0,.545),ortho=.34)]
    views += [dict(name='face-back',eye=(0,4,.78),target=(0,0,.78),ortho=.64),dict(name='face-back-quarter',eye=(1.5,4,.86),target=(0,.025,.79),ortho=.59)]
    metadata={'sourceBlend':bpy.data.filepath,'engine':scene.render.engine,'viewTransform':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma,'resolution':[scene.render.resolution_x,scene.render.resolution_y],'depthOfField':False,'areaShadowJitter':True,'samples':scene.eevee.taa_render_samples,'lights':[{'name':o.name,'position':list(o.location),'rotation':list(o.rotation_euler),'power':o.data.energy,'size':o.data.size} for o in scene.objects if o.type=='LIGHT' and o.name.startswith('Review_')],'views':[v for v in views if not names or v['name'] in names]}
    (dest/'review-settings.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for v in views:
        if names and v['name'] not in names:continue
        cam.location=v['eye'];cam.rotation_euler=(Vector(v['target'])-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=v['ortho']
        scene.render.filepath=str(dest/(v['name']+'.png'))
        bpy.ops.render.render(write_still=True)

def collection(name):
    c=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(c);return c

def save(stage):
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'stages'/stage))

def add_references():
    c=collection('References_Original_Four_Views_NOT_EXPORT')
    image_path=OUT/'reports/reference-audit-images/IMG_7224-original.png'
    image=bpy.data.images.load(str(image_path),check_existing=True);image.pack()
    # Four columns of one real supplied board, no synthetic view substitution.
    for i,(name,location,rot) in enumerate([('Front',(0,.55,0),(math.pi/2,0,0)),('Nose_Screen_Left',(-.55,0,0),(math.pi/2,0,math.pi/2)),('Back',(0,-.55,0),(math.pi/2,0,math.pi)),('Nose_Screen_Right',(.55,0,0),(math.pi/2,0,-math.pi/2))]):
        h=2.19;w=h*image.size[0]/image.size[1]/4
        me=bpy.data.meshes.new('Reference_'+name);me.from_pydata([(-w/2,-h/2,0),(w/2,-h/2,0),(w/2,h/2,0),(-w/2,h/2,0)],[],[(0,1,2,3)])
        uv=me.uv_layers.new()
        for d,t in zip(uv.data,[(i/4,0),((i+1)/4,0),((i+1)/4,1),(i/4,1)]):d.uv=t
        ob=bpy.data.objects.new('Reference_'+name,me);c.objects.link(ob);ob.location=location;ob.rotation_euler=rot;ob.hide_render=True;ob.hide_set(True)
        mat=bpy.data.materials.new('Reference_'+name);mat.use_nodes=True
        n=mat.node_tree.nodes;n.clear();tex=n.new('ShaderNodeTexImage');tex.image=image;emit=n.new('ShaderNodeEmission');out=n.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(tex.outputs['Color'],emit.inputs['Color']);mat.node_tree.links.new(emit.outputs[0],out.inputs['Surface']);me.materials.append(mat)
        ob['purpose']='Original A-pose identity and proportion reference; input GLB T-pose preserved'
    c.hide_render=True
    landmarks={'head_top':.9947,'chin':.680,'shoulder':.603,'waist':.285,'skirt_hem':-.222,'foot_sole':-.9991}
    for name,z in landmarks.items():
        e=bpy.data.objects.new('Guide_'+name,None);c.objects.link(e);e.location=(0,0,z);e.empty_display_type='CIRCLE';e.empty_display_size=.25;e.hide_render=True;e.hide_set(True)
    return landmarks

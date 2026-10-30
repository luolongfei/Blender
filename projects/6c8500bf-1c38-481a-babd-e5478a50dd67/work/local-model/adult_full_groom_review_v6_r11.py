"""Actual reference-v6 whole hairstyle with new face wisps and retired donor curls.

Measures the evaluated head, full dense donor hair, visible native tails and
collars. Temporary copied GN trees evaluate Is Viewport=False for the bounds;
the original trees are restored before rendering. No source-file saves.
"""
import argparse,hashlib,json,math,runpy,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
DIRECTIONS={'front':(0.,-1.,0.),'quarter':(.6,-.8,0.),'back':(0.,1.,0.)}
REQUIRED=('R11 Female head study','R11 Donor / long hair main','R11 Donor / reference face wisps',
          'Blouse_PeterPan_Collar_L','Blouse_PeterPan_Collar_R')
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def visible_objects(scene):
    found=set()
    def walk(collection):
        if collection.hide_render:return
        found.update(o for o in collection.objects if not o.hide_render)
        for child in collection.children:walk(child)
    walk(scene.collection)
    return found


def copy_render_branch(tree,memo,replacements):
    if tree in memo:return memo[tree]
    copy=tree.copy();copy.name='R11 review-only render branch / '+tree.name;memo[tree]=copy
    for node in list(copy.nodes):
        if node.type=='GROUP' and node.node_tree:
            node.node_tree=copy_render_branch(node.node_tree,memo,replacements)
        elif node.bl_idname=='GeometryNodeIsViewport':
            constant=copy.nodes.new('FunctionNodeInputBool');constant.boolean=False
            constant.label='Review bounds: actual render branch (Is Viewport=False)'
            links=list(node.outputs[0].links)
            for link in links:
                target=link.to_socket;copy.links.remove(link);copy.links.new(constant.outputs[0],target)
            replacements.append({'originalTree':tree.name,'node':node.name,'replacedLinks':len(links)})
    return copy


def points_and_radius(obj,graph):
    e=obj.evaluated_get(graph)
    if e.type=='CURVES':
        items=e.data.position_data;key='vector';curve_count=len(e.data.curves)
    elif e.type=='MESH':items=e.data.vertices;key='co';curve_count=None
    else:raise RuntimeError('No evaluated mesh/curves for framing: '+obj.name)
    if not len(items):raise RuntimeError('Empty evaluated framing geometry: '+obj.name)
    a=np.empty(len(items)*3,np.float32);items.foreach_get(key,a);a=a.reshape(-1,3)
    m=np.asarray(e.matrix_world,dtype=float);world=a@m[:3,:3].T+m[:3,3]
    radius=0.
    if e.type=='CURVES':
        attr=e.data.attributes.get('radius')
        if attr and len(attr.data):
            values=np.empty(len(attr.data),np.float32);attr.data.foreach_get('value',values)
            radius=float(values.max()*np.linalg.svd(m[:3,:3],compute_uv=False).max())
    if not np.isfinite(world).all() or not 0.<=radius<.01:raise RuntimeError('Invalid actual framing geometry/radius: '+obj.name)
    lo=world.min(0)-radius;hi=world.max(0)+radius
    return {'object':obj.name,'evaluatedType':e.type,'evaluatedPoints':len(a),'evaluatedCurves':curve_count,
            'maximumWorldRadiusMeters':radius,'worldBounds':[lo.tolist(),hi.tolist()]}


def measure(scene):
    visible=visible_objects(scene)
    for name in REQUIRED:
        if bpy.data.objects.get(name) not in visible:raise RuntimeError('Required visible framing object missing: '+name)
    objects=[bpy.data.objects[name] for name in REQUIRED]
    objects.extend(bpy.data.objects[name] for name in TAILS if bpy.data.objects.get(name) in visible)
    for obj in visible:
        if obj.type=='MESH' and (obj.name.startswith('R02 Tie ') or 'tie ring' in obj.name.lower()) and obj not in objects:objects.append(obj)
    memo={};replacements=[];modified=[];hidden=[]
    try:
        for obj in objects:
            hidden.append((obj,obj.hide_viewport,obj.hide_get()))
            obj.hide_viewport=False;obj.hide_set(False)
            for modifier in obj.modifiers:
                # Framing uses the render-enabled stack, including GN output.
                modified.append((modifier,modifier.show_viewport,modifier.node_group if modifier.type=='NODES' else None))
                modifier.show_viewport=modifier.show_render
                if modifier.type=='NODES' and modifier.show_render and modifier.node_group:
                    modifier.node_group=copy_render_branch(modifier.node_group,memo,replacements)
            obj.update_tag()
        bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        rows=[points_and_radius(obj,graph) for obj in objects]
    finally:
        for modifier,viewport,tree in reversed(modified):
            modifier.show_viewport=viewport
            if modifier.type=='NODES':modifier.node_group=tree
        for obj,viewport,hidden_state in hidden:
            obj.hide_viewport=viewport;obj.hide_set(hidden_state);obj.update_tag()
        bpy.context.view_layer.update()
    return rows,{'renderBranchMeasurement':True,'copiedNodeGroups':len(memo),'isViewportFalseReplacements':replacements,
                 'originalNodeGroupsRestoredBeforeRender':True,
                 'scope':'Actual evaluated point positions of copied original GN with render branch selected; full dense hair plus native tails, no synthetic silhouette.'}


def corners(rows):
    result=[]
    for row in rows:
        lo,hi=np.asarray(row['worldBounds'])
        result.extend([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    return np.asarray(result,dtype=float)


def framing(rows):
    cloud=corners(rows);head=np.asarray(rows[0]['worldBounds'])
    target=head.mean(0);target[2]=(cloud[:,2].min()+cloud[:,2].max())*.5
    aspect=1200/1400;vertical=float(cloud[:,2].max()-cloud[:,2].min())+.04
    width=0.
    for direction in DIRECTIONS.values():
        right=np.array([-direction[1],direction[0],0.])
        width=max(width,float(np.abs((cloud-target)@right).max())*2+.04)
    scale=math.ceil(max(.48,vertical,width/aspect)/.005)*.005
    return {'target':target.tolist(),'orthographicScale':scale,'sensorFit':'VERTICAL',
            'resolution':[1200,1400],'pixelAspect':[1.,1.],'distance':2.,'depthOfField':False,
            'method':'One scale and target enclosing all three view projections of actual radius-expanded bounds, minimum 20 mm border and 0.48 m portrait height.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--case',choices=tuple(DIRECTIONS),required=True)
    parser.add_argument('--camera-spec',help='Project-relative earlier review JSON; reuse its camera target/scale/resolution exactly')
    parser.add_argument('--samples',type=int,default=32)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not 16<=a.samples<=128:raise ValueError('Use 16..128 samples for this static review')
    source=(P/a.source).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    out=(P/a.output).resolve();out.relative_to((P/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh explicitly opened independent source required')
    out.mkdir(parents=True,exist_ok=True);image_path=out/('full-groom-'+a.case+'.png');report_path=image_path.with_suffix('.json')
    if image_path.exists() or report_path.exists():raise RuntimeError('Keep earlier images; use a new output folder')
    source_sha=sha(source);scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update()
    rows,geometry=measure(scene);spec=framing(rows);reuse=None
    if a.camera_spec:
        path=(P/a.camera_spec).resolve();path.relative_to((P/'previews/reconstruction-r02').resolve())
        spec=json.loads(path.read_text(encoding='utf8'))['camera']
        if spec['sensorFit']!='VERTICAL' or spec['depthOfField'] is not False:raise RuntimeError('Incompatible camera specification')
        reuse={'path':a.camera_spec,'sha256':sha(path)}
    target=np.asarray(spec['target'],dtype=float);direction=np.asarray(DIRECTIONS[a.case]);right=np.array([-direction[1],direction[0],0.])
    cloud=corners(rows);scale=float(spec['orthographicScale']);width=scale*spec['resolution'][0]/spec['resolution'][1]
    x=np.abs((cloud-target)@right);y=np.abs(cloud[:,2]-target[2])
    if x.max()>width*.5-.002 or y.max()>scale*.5-.002:
        raise RuntimeError('Actual full hairstyle would be cropped by reused camera; create a shared wider framing specification')
    camera=bpy.data.objects.new('R11 full groom review camera',bpy.data.cameras.new('R11 full groom review camera'))
    scene.collection.objects.link(camera);camera.location=Vector(target+direction*float(spec['distance']))
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.sensor_fit='VERTICAL';camera.data.ortho_scale=scale;camera.data.dof.use_dof=False
    scene.camera=camera;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
    scene.render.resolution_x=spec['resolution'][0];scene.render.resolution_y=spec['resolution'][1];scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=1.;scene.render.pixel_aspect_y=1.
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.filepath=str(image_path)
    bpy.context.view_layer.update();frame=np.asarray([list(v) for v in camera.data.view_frame(scene=scene)])
    if abs(float(np.ptp(frame[:,1]))-scale)>1e-5 or abs(float(np.ptp(frame[:,0]))-width)>1e-5:
        raise RuntimeError('Actual Blender camera frame differs from computed portrait framing')
    detail=runpy.run_path(str(HERE/'final_detail_review_r09.py'))
    report={'source':a.source,'sourceSha256':source_sha,'helperSha256':sha(__file__),'case':a.case,'frame':1,
            'camera':spec,'cameraSpecReused':reuse,'actualCameraMatrix':[list(row) for row in camera.matrix_world],
            'actualCameraLocalFrame':frame.tolist(),'actualFramingObjects':rows,'denseGeometryMeasurement':geometry,
            'minimumBoundsBorderMeters':{'horizontal':float(width*.5-x.max()),'vertical':float(scale*.5-y.max())},
            'lights':detail['_lights'](scene),'world':detail['_world'](scene),
            'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
                                'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
            'engine':'Cycles CPU','samples':a.samples,'denoising':True,'savedBlend':False,
            'scope':'Actual complete hairstyle/collar framing at saved-source frame one. Original lighting, visibility, pose and hair nodes retained for render; no physical-motion or appearance acceptance.'}
    bpy.ops.render.render(write_still=True)
    if not image_path.exists() or sha(source)!=source_sha:raise RuntimeError('Render or unchanged source check failed')
    report['imageSha256']=sha(image_path)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_FULL_GROOM_REVIEW '+json.dumps({'image':str(image_path),'sourceSha256':source_sha,'camera':spec}),flush=True)
if __name__=='__main__':main()

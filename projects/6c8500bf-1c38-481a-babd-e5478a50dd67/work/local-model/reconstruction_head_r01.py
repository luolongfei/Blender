"""CC0 animation head fitting, static direction study, not an animation transfer."""
import bpy, bmesh, math
from mathutils import Matrix, Vector

def run():
    if not bpy.data.filepath.endswith('reconstruction-r01\\candidate.blend') and not bpy.data.filepath.endswith('reconstruction-r01/candidate.blend'):
        raise RuntimeError('Candidate only')
    if bpy.data.objects.get('R01 Head'): raise RuntimeError('Already built')
    old_prefix=('Face ·','Ear ','Neck','Eye sclera','Iris brown','Upper lashes','Lower lid','Outer lash','Upper eyelid','Expressive brow','Eyebrow individual','Mouth interior','Upper dental','Lower dental','Tongue surface')
    for o in bpy.data.objects:
        if o.name.startswith(old_prefix):
            o.hide_render=True; o.hide_set(True); o['R01_superseded']=True
    rig=bpy.data.objects['Character_Rig']; rig.data.pose_position='REST'
    skin=bpy.data.materials.new('R01 Skin - form study'); skin.use_nodes=True
    n=next(n for n in skin.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    n.inputs['Base Color'].default_value=(.48,.29,.205,1)
    n.inputs['Roughness'].default_value=.5
    n.inputs['Subsurface Weight'].default_value=.08
    n.inputs['Subsurface Radius'].default_value=(1,.4,.2)
    n.inputs['Subsurface Scale'].default_value=.012
    sclera=bpy.data.materials.new('R01 Warm sclera'); sclera.use_nodes=True
    sn=next(n for n in sclera.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    sn.inputs['Base Color'].default_value=(.64,.59,.49,1); sn.inputs['Roughness'].default_value=.19
    iris=bpy.data.materials.new('R01 Brown iris'); iris.use_nodes=True
    inn=next(n for n in iris.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    inn.inputs['Base Color'].default_value=(.095,.035,.012,1); inn.inputs['Roughness'].default_value=.22
    tr=Matrix.Translation((0,.0485,1.553)) @ Matrix.Diagonal((.95,.95,.95,1)) @ Matrix.Translation((-1.4625719785690308,0,-.8803073763847351))
    imported=[o for o in bpy.data.objects if o.name.startswith('GEO-head_animation_realistic')]
    # Bake original transforms including the eye parent matrices before detaching.
    mats={o.name:tr@o.matrix_world for o in imported}
    for o in imported:
        name=o.name; o.parent=None
        o.data.transform(mats[name]); o.matrix_world=Matrix.Identity(4)
        o.hide_render=False; o.hide_set(False)
        o.data.materials.clear()
        if '.sclera.' in name: o.name='R01 Sclera '+name[-1]; o.data.materials.append(sclera)
        elif '.iris.' in name: o.name='R01 Iris '+name[-1]; o.data.materials.append(iris)
        else:
            o.name='R01 Head'; o.modifiers.clear()
            bm=bmesh.new(); bm.from_mesh(o.data)
            bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.co.z<1.245],context='VERTS')
            bm.to_mesh(o.data); bm.free()
            # Adapt broad stock jaw, nose and neck to this young character's proportions.
            for v in o.data.vertices:
                x,y,z=v.co
                jaw=math.exp(-((z-1.365)/.042)**2)
                v.co.x*=1-.09*jaw
                neck=max(0,min(1,(1.335-z)/.07))
                if neck:
                    v.co.x*=1-.12*neck
                nose=math.exp(-(x/.021)**2-((z-1.415)/.025)**2)*max(0,min(1,(-y-.065)/.025))
                v.co.y+=.003*nose
            o.data.materials.append(skin)
            sub=o.modifiers.new('R01 editable subdivision','SUBSURF'); sub.levels=2; sub.render_levels=2
            o['source']='Blender Human Base Meshes v1.4.1 / CC0 / animation head'
            o['scope']='Static fitted study; new facial rig and neck integration pending'
        for f in o.data.polygons: f.use_smooth=True
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    print('R01_HEAD_BUILT')

if __name__=='__main__': run()

"""Stable rest-space micro surface and anatomical roughness; no painted blush."""
import bpy,math

def run():
    head=bpy.data.objects['R01 Head'];mesh=head.data;ks=mesh.shape_keys.key_blocks
    if mesh.attributes.get('R04 skin rest position'):raise RuntimeError('Already surfaced')
    pos=mesh.attributes.new('R04 skin rest position','FLOAT_VECTOR','POINT')
    rough=mesh.attributes.new('R04 skin roughness','FLOAT','POINT')
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        pos.data[i].vector=q;x,y,z=q
        front=max(0,min(1,(-y-.04)/.035))
        nose=math.exp(-(x/.020)**4-((z-1.402)/.021)**4)*front
        forehead=math.exp(-((z-1.466)/.028)**4-(x/.055)**6)*front
        lip=math.exp(-(x/.026)**6-((z-1.371)/.0045)**4)*front
        rough.data[i].value=max(.34,.53-.08*nose-.04*forehead-.13*lip)
    material=head.data.materials[0].copy();material.name='R04 physical skin';head.data.materials[0]=material
    nt=material.node_tree;bs=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED')
    attr=nt.nodes.new('ShaderNodeAttribute');attr.attribute_name='R04 skin rest position'
    noise=nt.nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=5000;noise.inputs['Detail'].default_value=2
    nt.links.new(attr.outputs['Vector'],noise.inputs['Vector'])
    bump=nt.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00009
    nt.links.new(noise.outputs['Fac'],bump.inputs['Height']);nt.links.new(bump.outputs['Normal'],bs.inputs['Normal'])
    region=nt.nodes.new('ShaderNodeAttribute');region.attribute_name='R04 skin roughness';nt.links.new(region.outputs['Fac'],bs.inputs['Roughness'])
    bs.inputs['Subsurface Weight'].default_value=.14;bs.inputs['Subsurface Scale'].default_value=.0018
    head['R04_skin_note']='Procedural rest-space surface and point-domain roughness. UV retained. Texture baking/realtime adaptation still separate.'
    print('R04_SKIN_SURFACE_CREATED')

if __name__=='__main__':run()

"""Conservative glTF-safe colour-only crown flow, baked from HairUV.

There are no extra geometry strips, no normal map and no displacement.  Use
only after installing hair-spec-v4; this does not hide bad surface topology.
"""
from pathlib import Path
from math import pi


def _flow_nodes(mat):
    import materials
    n,l=mat.node_tree.nodes,mat.node_tree.links;n.clear()
    uv=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(uv.outputs['UV'],sep.inputs[0])
    def scalar(op,a,b=None):
        nd=n.new('ShaderNodeMath');nd.operation=op
        for i,x in enumerate((a,b)):
            if x is None:continue
            if isinstance(x,(int,float)):nd.inputs[i].default_value=x
            else:l.new(x,nd.inputs[i])
        return nd.outputs[0]
    u,v=sep.outputs['X'],sep.outputs['Y']
    # Mirror at central front/back parts. Downstream phase converges gently
    # toward ear-back gathering directions; no white streak or baked light.
    q=scalar('MINIMUM',u,scalar('SUBTRACT',1,u))
    denominator=scalar('SUBTRACT',1,scalar('MULTIPLY',.32,scalar('POWER',v,1.7)))
    flow=scalar('ADD',.33,scalar('DIVIDE',scalar('SUBTRACT',q,.33),denominator))
    broad=scalar('COSINE',scalar('MULTIPLY',2*pi*14,flow))
    fine_phase=scalar('ADD',scalar('MULTIPLY',2*pi*60,flow),scalar('MULTIPLY',.3,scalar('SINE',scalar('MULTIPLY',2*pi,v))))
    fine=scalar('COSINE',fine_phase)
    value=scalar('ADD',1,scalar('ADD',scalar('MULTIPLY',.055,broad),scalar('MULTIPLY',.012,fine)))
    mul=n.new('ShaderNodeMixRGB');mul.blend_type='MULTIPLY';mul.inputs[0].default_value=1
    mul.inputs[1].default_value=materials.linear_rgba((.185,.176,.160));l.new(value,mul.inputs[2])
    emission=n.new('ShaderNodeEmission');l.new(mul.outputs[0],emission.inputs['Color'])
    emission.inputs['Strength'].default_value=1
    out=n.new('ShaderNodeOutputMaterial');l.new(emission.outputs[0],out.inputs['Surface'])


def apply_crown_flow(cap_object,output_dir,size=2048):
    """Bake once in an isolated temporary scene, then apply direct image PBR.

    Existing bake is reused deliberately; revision output names are unique.
    Source model and other hair materials are untouched. Returns a report.
    """
    import bpy,materials
    if cap_object.type!='MESH' or 'HairUV' not in cap_object.data.uv_layers:
        raise ValueError('Expected rebuilt crown mesh with HairUV')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'hair-crown-flow-v4-2048.png'
    if not path.exists():
        image=materials._bake_uv_image(path,size,_flow_nodes,'Hair_Crown_Flow_V4_BaseColor')
    else:
        image=bpy.data.images.load(str(path),check_existing=True)
    material=materials.principled_material('Rebuilt_Hair_Crown_BakedFlow_V4',(.185,.176,.160),.56)
    materials.set_basecolor_image(material,path,'HairUV',pack=True)
    cap_object.data.materials.clear();cap_object.data.materials.append(material)
    material['detail_scope']='Low-contrast UV colour flow only. No bump, normal, displacement, added geometry, or baked directional light.'
    return {'object':cap_object.name,'baseColor':str(path),'imageSize':list(image.size),
            'roughness':.56,'normalMap':False,'extraGeometry':False,'imagePacked':True,
            'status':'baked technical output; appearance requires actual neutral render'}

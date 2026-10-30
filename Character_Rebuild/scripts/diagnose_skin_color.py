import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D-integrated-geometry.blend'))
report={'images':[], 'materials':[], 'expectedLinear':materials.linear_rgba(materials.PALETTE['skin'][0])}
for mat in bpy.data.materials:
    if mat.name.startswith('Rebuilt_skin'):
        bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        report['materials'].append({'name':mat.name,'color':list(bs.inputs['Base Color'].default_value),'metallic':bs.inputs['Metallic'].default_value,'roughness':bs.inputs['Roughness'].default_value,'baseLink':[(lk.from_node.name,lk.from_socket.name) for lk in bs.inputs['Base Color'].links]})
for im in bpy.data.images:
    if 'Skin' in im.name or 'skin' in im.name:
        report['images'].append({'name':im.name,'source':im.source,'filepath':im.filepath,'colorspace':im.colorspace_settings.name,'pixel0':list(im.pixels[:4]),'packed':bool(im.packed_file)})
fresh=bpy.data.images.load(str(OUT/'textures/face-skin-2048.png'),check_existing=False)
report['freshLoad']={'name':fresh.name,'source':fresh.source,'colorspace':fresh.colorspace_settings.name,'pixel0':list(fresh.pixels[:4])}
(OUT/'reports/skin-colour-diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

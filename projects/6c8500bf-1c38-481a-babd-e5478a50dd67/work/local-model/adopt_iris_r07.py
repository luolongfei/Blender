"""Use the locally authored iris texture, preserving legacy source materials."""
import bpy,json,hashlib,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
def run():
    p=project_path();texture=p/'textures/procedural-r07/iris-brown-r07-v3.png'
    if Path(bpy.data.filepath).name=='character.blend':raise RuntimeError('Use saved independent candidate')
    if bpy.data.materials.get('R07 documented brown iris'):raise RuntimeError('Already adopted')
    if hashlib.sha256(texture.read_bytes()).hexdigest()!='6bddc86582fa1b4541376082816c4d734deabbbcbe38e65710415823f0566784':raise RuntimeError('Texture identity differs')
    left=bpy.data.objects['R01 Textured iris L'];old=left.data.materials[0]
    material=old.copy();material.name='R07 documented brown iris'
    image=bpy.data.images.load(str(texture),check_existing=True);image.colorspace_settings.name='sRGB';image.pack()
    node=material.node_tree.nodes.new('ShaderNodeTexImage');node.image=image;node.label='R07 deterministic workspace iris, no external pixels'
    bs=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    material.node_tree.links.new(node.outputs['Color'],bs.inputs['Base Color'])
    # The new texture already carries the intended dark brown. The historical
    # photograph's multiply node is retained disconnected, not applied twice.
    for side in ['L','R']:bpy.data.objects['R01 Textured iris '+side].data.materials[0]=material
    material['source_record']='textures/procedural-r07/iris-source-r07.json'
    report={'texture':str(texture.relative_to(p)),'sha256':hashlib.sha256(texture.read_bytes()).hexdigest(),
            'material':material.name,'legacy_material_preserved':old.name,'newBaseColor':'Direct sRGB texture',
            'status':'Applied to both current iris patches; eye close-up render pending'}
    (p/'work/reconstruction-r02/iris-adoption-r07.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    bpy.context.view_layer.update();print(json.dumps(report));return report

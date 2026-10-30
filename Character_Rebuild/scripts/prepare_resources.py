import sys,json,bpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,add_references,render_views
import materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/A-imported-original.blend'))
for mat in bpy.data.materials:
    if mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.type=='BSDF_PRINCIPLED':n.inputs['Metallic'].default_value=0;n.inputs['Roughness'].default_value=.8
landmarks=add_references()
if not (OUT/'textures/iris-anime-1024.png').exists():materials.bake_iris_image(OUT/'textures/iris-anime-1024.png')
save('A-neutral-reference.blend')
render_views('checks/before-neutral',['front','face-front','face-side','hand-top','back'])
(OUT/'reports/reference-setup.json').write_text(json.dumps({'landmarks':landmarks,'reference':'Original IMG_7224 four-view board; A-pose only','modelPose':'Original T-pose','materialNormalization':'Both before-neutral and after use non-metallic material, neutral fixed lights and fixed cameras'},indent=2),encoding='utf-8')

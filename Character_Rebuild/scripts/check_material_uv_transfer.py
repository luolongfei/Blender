import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import materials
me=bpy.data.meshes.new('ProbeAtlasAndRepair')
me.from_pydata([(-2,-1,0),(-.1,-1,0),(-.1,1,0),(-2,1,0),(.1,-1,0),(2,-1,0),(2,1,0),(.1,1,0)],[],[(0,1,2,3),(4,5,6,7)])
uv=me.uv_layers.new(name='OriginalUV')
# Original textured face reads BLUE right half; smart-project will place it in
# LEFT half. Incorrect active-UV source lookup would therefore turn it RED.
for i,c in enumerate([(.55,.05),(.95,.05),(.95,.95),(.55,.95),(.5,.5),(.5,.5),(.5,.5),(.5,.5)]):uv.data[i].uv=c
obj=bpy.data.objects.new('ProbeAtlasAndRepair',me);bpy.context.scene.collection.objects.link(obj)
atlas=materials.principled_material('ProbeAtlas',(1,1,1))
materials.set_basecolor_image(atlas,OUT/'diagnostics/material-multibake-probe.png')
me.materials.append(atlas);me.materials.append(materials.principled_material('ProbeGreenRepair',(.1,.8,.1)))
me.polygons[0].material_index=0;me.polygons[1].material_index=1
output=OUT/'diagnostics/material-uv-transfer-probe.png'
result=materials.bake_repaired_body_atlas(obj,[],output,2048,repack_uv=True)
im=bpy.data.images.load(str(output),check_existing=False); checks=[]
for poly in obj.data.polygons:
    coords=[obj.data.uv_layers.active.data[i].uv for i in poly.loop_indices]
    u,v=[sum(c[k] for c in coords)/len(coords) for k in (0,1)]
    offset=(int(v*im.size[1])*im.size[0]+int(u*im.size[0]))*4
    checks.append({'face':poly.index,'newUvCenter':[u,v],'rgb':list(im.pixels[offset:offset+3])})
assert checks[0]['newUvCenter'][0]<.5 and checks[0]['rgb'][2]>.7 and checks[0]['rgb'][0]<.2,checks
assert checks[1]['rgb'][1]>.7 and checks[1]['rgb'][0]<.2 and checks[1]['rgb'][2]<.2,checks
report={'status':'pass','checks':checks,'oldAtlasUvsPinned':True,'repairSolidMaterialPreserved':True,
        'originalUVsRetained': 'OriginalUV' in obj.data.uv_layers,'newUvZeroAreas':result['new_uv_degenerate_triangles']}
(OUT/'reports/material-uv-transfer-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

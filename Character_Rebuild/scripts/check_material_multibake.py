import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import materials
me=bpy.data.meshes.new('ProbeTwoFaces')
me.from_pydata([(-2,-1,0),(-.1,-1,0),(-.1,1,0),(-2,1,0),(.1,-1,0),(2,-1,0),(2,1,0),(.1,1,0)],[],[(0,1,2,3),(4,5,6,7)])
uv=me.uv_layers.new(name='OriginalUV')
for i,c in enumerate([(0,0),(1,0),(1,1),(0,1),(.5,.5),(.5,.5),(.5,.5),(.5,.5)]):uv.data[i].uv=c
obj=bpy.data.objects.new('ProbeTwoFaces',me);bpy.context.scene.collection.objects.link(obj)
for name,col in [('ProbeRed',(.8,.1,.1)),('ProbeBlue',(.1,.1,.8))]:me.materials.append(materials.principled_material(name,col))
me.polygons[0].material_index=0;me.polygons[1].material_index=1
output=OUT/'diagnostics/material-multibake-probe.png'
if output.exists(): raise FileExistsError(output)
result=materials.bake_repaired_body_atlas(obj,[],output,2048,repack_uv=True)
im=bpy.data.images.load(str(output),check_existing=False)
pixels=im.pixels
checks=[]
for poly in me.polygons:
    coords=[obj.data.uv_layers.active.data[i].uv for i in poly.loop_indices]
    u,v=[sum(c[k] for c in coords)/len(coords) for k in (0,1)]
    x,y=int(u*im.size[0]),int(v*im.size[1]);offset=(y*im.size[0]+x)*4
    rgb=list(pixels[offset:offset+3]); checks.append({'face':poly.index,'uvCenter':[u,v],'rgb':rgb})
assert checks[0]['rgb'][0]>.7 and checks[0]['rgb'][2]<.2,checks
assert checks[1]['rgb'][2]>.7 and checks[1]['rgb'][0]<.2,checks
assert 'OriginalUV' in obj.data.uv_layers
report={'status':'pass','checks':checks,'uvLayers':[l.name for l in obj.data.uv_layers],
        'sourceGeometryUnchanged':len(obj.data.vertices)==8 and len(obj.data.polygons)==2,
        'repairedMaterialPlaceholderUvsDoNotReadAtlas':True,'newUvZeroAreas':result['new_uv_degenerate_triangles']}
(OUT/'reports/material-multibake-probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

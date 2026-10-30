import bpy,runpy,json
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('runtime_material_bake_r06.py')))
obj=bpy.data.objects['RT06 Face / R01 Head'];ids=[p.index for p in obj.data.polygons if obj.data.materials[p.material_index].name=='R04 physical skin']
print('R08_SOURCE_UVS',json.dumps([api['_uv_probe'](obj.data,u,ids) for u in obj.data.uv_layers]))
print('R08_SKIN_SEAMS',sum(e.use_seam for e in obj.data.edges))

import bpy,bmesh,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
from clothing_contact_e7 import sat_object
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E7-uv-normals-corrected.blend'))
o=bpy.data.objects['Rebuilt_FrontCollar_R']
bm=bmesh.new();bm.from_mesh(o.data);bm.faces.ensure_lookup_table();bm.verts.ensure_lookup_table()
f=bm.faces[639];layer=bm.loops.layers.uv.active;uv={l.vert.index:l[layer].uv.copy() for l in f.loops};mat=f.material_index
bm.faces.remove(f)
for ids in [(705,719,718),(705,706,719)]:
    ff=bm.faces.new([bm.verts[i] for i in ids]);ff.material_index=mat;ff.smooth=True
    for l,i in zip(ff.loops,ids):l[layer].uv=uv[i]
bm.to_mesh(o.data);bm.free();o.data.update()
r=sat_object(o)
print('DIAGONAL_PROBE',json.dumps(r))

import bpy, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from pipeline.context import workspace_path
out = workspace_path('Character_Rebuild')
bpy.ops.wm.open_mainfile(filepath=str(out/'stages/D-integrated-geometry.blend'))
obj = bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
mesh = obj.data
indices = [v.index for v in mesh.vertices if abs(v.co.x)<.24 and v.co.z>.50]
chosen = set(indices)
data = {'object':obj.name,'matrixWorld':[list(r) for r in obj.matrix_world],
        'vertices':[[i,*mesh.vertices[i].co] for i in indices],
        'polygons':[[p.index,*p.vertices] for p in mesh.polygons if all(i in chosen for i in p.vertices)],
        'source':'D-integrated-geometry.blend'}
(out/'reports/d-yoke-local.json').write_text(json.dumps(data),encoding='utf-8')
print(json.dumps({'capture':str(out/'reports/d-yoke-local.json'),'vertices':len(data['vertices']),'faces':len(data['polygons'])}))

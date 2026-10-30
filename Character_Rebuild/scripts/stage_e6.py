import bpy,sys,json,hashlib,bmesh
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,neutral_review
assert hashlib.sha256((OUT/'scripts/hair_normal_delta_v5.py').read_bytes()).hexdigest()=='be79f9295b832686286187dd516ebf0001634d1fd4605a7ceb9e7b3cc1c406ae'
texture=OUT/'textures/hair-crown-flow-normal-v5-2048.png'
assert hashlib.sha256(texture.read_bytes()).hexdigest()=='f6865585f534ed757db8b33b6202b0dd1fee2c6f966ba0b3fd09098f6e8e2816'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E5-integrated-refinements.blend'))
from hair_normal_delta_v5 import apply_crown_normal
report=json.loads((OUT/'reports/hair-normal-delta-v5.json').read_text())
cap=bpy.data.objects['Hair_Crown_SourceEnvelope']
installed=apply_crown_normal(cap,texture,report['sourceCapHashes'],strength=.85)
# Hide the residual rear scalp pinpoints by fitting only the hair-covered rear
# head to the actual inner cap. The cap and its baked normal map are unchanged.
head=bpy.data.objects['Head_Skin_Quad_Retopo']
bvh=BVHTree.FromObject(cap,bpy.context.evaluated_depsgraph_get())
centre=Vector((0,.02,.855));moves=[]
for v in head.data.vertices:
    if not (v.co.z>.775 and v.co.y>.055):continue
    delta=v.co-centre;direction=delta.normalized()
    pos,normal,index,distance=bvh.ray_cast(centre,direction,1)
    if pos is not None and delta.length>distance-.004:
        target=centre+direction*max(.001,distance-.004)
        amount=(target-v.co).length
        assert amount<.025,'Rear scalp requires unexpectedly large correction'
        moves.append(amount);v.co=target
head.data.update()
bm=bmesh.new();bm.from_mesh(head.data);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(head.data);bm.free()
neutral_review();save('E6-export-source.blend')
(OUT/'reports/stage-e6.json').write_text(json.dumps({'normalInstall':installed,'rearHiddenScalpFit':{'movedVertices':len(moves),'maxDisplacement':max(moves,default=0),'clearanceSourceUnits':.004,'visibleFaceEyesMouthUnchanged':True,'capGeometryUVUnchanged':True},'pending':'Final combined render and clean GLB reimport'},indent=2),encoding='utf-8')
print('E6_EXPORT_SOURCE_SAVED')

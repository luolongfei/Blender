import bpy,json
p=bpy.ops.export_scene.gltf.get_rna_type().properties
print(json.dumps({k:{'default':getattr(v,'default',None),'enum':[(x.identifier,x.name) for x in v.enum_items] if v.type=='ENUM' else None} for k,v in p.items() if any(x in k for x in ['anim','skin','apply','def_bone'])},indent=2))

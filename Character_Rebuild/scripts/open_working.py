import bpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from pipeline.context import workspace_path
bpy.ops.wm.open_mainfile(filepath=str(workspace_path('Character_Rebuild/stages/A-imported-original.blend')))
try:
    bpy.context.scene.blendermcp_auto_start_server=True
    bpy.ops.blendermcp.start_server()
except Exception as exc: print('MCP_START',repr(exc))
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.shading.type='MATERIAL'
            area.spaces.active.region_3d.view_distance=3.2
            area.spaces.active.region_3d.view_location=(0,0,0)
print('NEW_INPUT_OPENED',bpy.data.filepath)

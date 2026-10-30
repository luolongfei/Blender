"""Raise only the hidden central root-shell edge beneath the thick forelocks."""
import bpy,numpy as np,json

def run():
    obj=bpy.data.objects['R06 attached scalp root substrate'];rig=bpy.data.objects['Character_Rig']
    if obj.get('R06_central_hairline'):raise RuntimeError('Already applied')
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    try:
        a=np.empty(len(obj.data.vertices)*3,np.float32);obj.data.vertices.foreach_get('co',a);local=a.reshape(-1,3)
        m=np.array(obj.matrix_world);world=local@m[:3,:3].T+m[:3,3]
        top=world[:,2].max();center=(world.min(axis=0)+world.max(axis=0))*.5
        x=world[:,0]-center[0];y=world[:,1]-center[1];z=world[:,2]
        def smooth(t):t=np.clip(t,0,1);return t*t*(3-2*t)
        front=smooth((-y-.024)/.038);central=1-smooth((np.abs(x)-.045)/.04)
        threshold=top-.048-.021*np.minimum(1,(np.abs(x)/.074)**2)
        fade=smooth((z-threshold)/.007)
        alpha=obj.data.attributes['R06_root_edge_alpha'];old=np.empty(len(world),np.float32);alpha.data.foreach_get('value',old)
        backup=obj.data.attributes.new('R06_root_edge_before_central','FLOAT','POINT');backup.data.foreach_set('value',old)
        new=old*(1-front*central*(1-fade));alpha.data.foreach_set('value',new)
        obj['R06_central_hairline']=json.dumps({'centerTopDrop':.048,'sideDrop':.021,'fade':.007,'changedSamples':int((abs(new-old)>.001).sum())})
        obj.data.update_tag();print(obj['R06_central_hairline'])
    finally:rig.data.pose_position=pose;bpy.context.view_layer.update()

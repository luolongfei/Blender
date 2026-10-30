"""Correct the neck join's original overbroad Z-only weight mask, preserving its neck."""
import bpy,json
from mathutils import kdtree

def run():
    h=bpy.data.objects['R01 Head'];source=bpy.data.objects['Body_Complete_Trunk_Arms_Pelvis']
    if h.get('R04_restored_arm_weights'):raise RuntimeError('Already restored')
    tree=kdtree.KDTree(len(source.data.vertices))
    for v in source.data.vertices:tree.insert(v.co,v.index)
    tree.balance();changed=0;maximum=0.;old=[]
    for v in h.data.vertices:
        if abs(v.co.x)<.085 or not 1.250<v.co.z<1.315:continue
        _,idx,d=tree.find(v.co)
        if d>1e-6:continue
        weights={source.vertex_groups[g.group].name:g.weight for g in source.data.vertices[idx].groups}
        previous={h.vertex_groups[g.group].name:g.weight for g in v.groups}
        error=max(abs(previous.get(n,0)-weights.get(n,0)) for n in set(previous)|set(weights))
        if error<1e-7:continue
        old.append([v.index,previous]);maximum=max(maximum,error)
        for g in list(v.groups):h.vertex_groups[g.group].remove([v.index])
        for name,value in weights.items():h.vertex_groups[name].add([v.index],value,'REPLACE')
        changed+=1
    h['R04_restored_arm_weights']=json.dumps({'vertices':changed,'maximum_weight_error_removed':maximum,'source':source.name,'scope':'Restore original body weights outside central neck; source and every facial shape unchanged'})
    h['R04_arm_weight_backup']=json.dumps(old)
    bpy.context.view_layer.update();print('R04_ARM_WEIGHTS_RESTORED',changed,maximum)

if __name__=='__main__':run()

"""Keep measured role-aware interpolation; bypass unsafe legacy styling maps."""
import hashlib,runpy
from pathlib import Path
import bpy
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bypass(tree,name,label):
    node=tree.nodes[name];inputs=[s for s in node.inputs if s.type=='GEOMETRY'];outputs=[s for s in node.outputs if s.type=='GEOMETRY']
    if len(inputs)!=1 or len(outputs)!=1 or len(inputs[0].links)!=1:raise RuntimeError('Unexpected geometry routing at '+name)
    source=inputs[0].links[0].from_socket;links=list(outputs[0].links)
    reroute=tree.nodes.new('NodeReroute');reroute.name=label;tree.links.new(source,reroute.inputs[0])
    destinations=[]
    for link in links:
        destination=link.to_socket;destinations.append(destination.node.name+'/'+destination.name)
        tree.links.remove(link);tree.links.new(reroute.outputs[0],destination)
    node.label='Retained legacy node — bypassed after actual v4 failure'
    return {'retainedNode':name,'retainedGroup':node.node_tree.name,'bypassNode':reroute.name,
            'geometrySource':source.node.name+'/'+source.name,'reconnectedDestinations':destinations,'active':False}


def apply(*,expected_source_relative):
    helper=HERE/'donor_groom_groups_v4_r11.py'
    if sha(helper)!='28c542dc9eff960c2043766388dbdc05b37617e0b9467b5a4c38a01bca75a537':raise RuntimeError('Measured v4 base changed')
    report=runpy.run_path(str(helper))['apply'](expected_source_relative=expected_source_relative)
    obj=bpy.data.objects['R11 Donor / long hair main'];tree=next(m.node_group for m in obj.modifiers if m.type=='NODES' and m.show_render)
    report['bypassedLegacyStyling']=[bypass(tree,name,label) for name,label in (
        ('Group.014','R11 v5 after parting bypass'),('Group','R11 v5 after neutral trim bypass'),
        ('Group.008','R11 v5 after big clump bypass'),('Group.006','R11 v5 after small clump bypass'))]
    report['roleGrouping']['clumpGroupsActive']=False
    report['roleGrouping']['partingSnapActive']=False
    report['unclumpedHelperSha256']=sha(__file__)
    report['stage']='Role/island-aware interpolation with explicit geometry bypasses for legacy parting snap and both main clumps; old nodes retained, actual image judgment pending.'
    obj['R11_donor_gn_finish_v5']='Native grouped interpolation; legacy parting snap and big/small clumps bypassed after measured crossing/root drift'
    tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
    return report

if __name__=='__main__':print('Explicit independent candidate apply only.')

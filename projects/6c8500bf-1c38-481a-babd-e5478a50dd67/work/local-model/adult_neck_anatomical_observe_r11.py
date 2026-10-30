"""Supplementary read-only anatomical Mask boundary observation.

Only a temporary copy of the new female head and its mesh receives the exact
author head/neck vertex selection. The saved/current source is never edited.
This yields the REAL Multires+Mask boundary; raw54*8 is not assumed.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def observe(source_relative,output_relative='work/reconstruction-r02/adult-neck-anatomical-observation-r11.json'):
    import bpy
    A=runpy.run_path(str(Path(__file__).with_name('adult_neck_fit_r11.py')))
    uv=A['_api']('production_uv_r08');source=A['_path'](source_relative,'.blend')
    output=A['_path'](output_relative,'.json');payload=output.with_suffix('.npz')
    if not bpy.app.background or bpy.data.is_dirty or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved independent R11 candidate required')
    if output.exists() or payload.exists():raise RuntimeError('Retain observation history; new output names required')
    head=bpy.data.objects['R11 Female head study'];old=bpy.data.objects['R01 Head']
    if len(head.data.vertices)!=10582:raise RuntimeError('Expected unchanged full female raw topology')
    bpy.context.view_layer.update();source_sha=A['_sha'](source);pose_before=uv['_pose_contract']()
    before={o.name:A['_object_contract'](o,uv) for o in (head,old)}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    scene=bpy.context.scene;frame=scene.frame_current;subframe=scene.frame_subframe
    subs=[(m,m.levels) for m in old.modifiers if m.type=='SUBSURF']
    original_objects=set(o.as_pointer() for o in bpy.data.objects)
    temporary=None;data=None;arrays={};report={}
    try:
        scene.frame_set(1);rig.data.pose_position='REST'
        for modifier,_ in subs:modifier.levels=modifier.render_levels
        bpy.context.view_layer.update()
        selection=json.loads((project_path()/'work/reconstruction-r02/female-head-selection-r11.json').read_text(encoding='utf-8'))
        temporary=head.copy();data=head.data.copy();temporary.data=data
        temporary.name='R11 temporary anatomical-boundary observation';scene.collection.objects.link(temporary)
        temporary.hide_render=True;temporary.hide_viewport=False;temporary.hide_set(False)
        vg=temporary.vertex_groups.new(name='R11 observation exact author head neck')
        vg.add(list(map(int,selection['original_vertex_indices'])),1.,'REPLACE')
        masks=[m for m in temporary.modifiers if m.type=='MASK']
        if len(masks)!=1:raise RuntimeError('Expected exactly one independent new-head display Mask')
        masks[0].vertex_group=vg.name;masks[0].threshold=.5
        bpy.context.view_layer.update()
        head_record=A['_evaluated'](temporary,'anatomical_new_head',arrays,bpy)
        old_record=A['_evaluated'](old,'old_displayed_body',arrays,bpy)
        head_record['materials']=[s.material.name if s.material else None for s in head.material_slots]
        old_record['materials']=[s.material.name if s.material else None for s in old.material_slots]
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'helperSha256':A['_sha'](Path(__file__)),'dependencyHelperSha256':A['_sha'](Path(__file__).with_name('adult_neck_fit_r11.py')),
            'femaleToTargetRestWorld':json.loads(head['R11_female_to_target']),
            'newHeadAnatomical':head_record,'oldDisplayedBody':old_record,
            'sourceRawNeckRing':selection['neck_ring_old_indices'],'sourceRawHeadVertices':len(selection['original_vertex_indices']),
            'method':'Temporary object+mesh copy with author head/neck selection, original L3 and actual post-Multires Mask. No assumption about evaluated vertex order or 54x8 boundary count.',
            'knownOldDisplayScope':'Existing R11 old-face deletion keeps old anatomical-body material AND old R02 oral mucosa. Distinguish the torso boundary from the small orphan oral component in fitting.',
            'savedBlend':False,'rendered':False,'originalSourceGeometryEdited':False}
    finally:
        if temporary is not None:bpy.data.objects.remove(temporary,do_unlink=True)
        if data is not None and data.users==0:bpy.data.meshes.remove(data)
        for modifier,levels in subs:modifier.levels=levels
        rig.data.pose_position=pose;scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before:raise RuntimeError('Original pose not restored')
    if any(A['_object_contract'](o,uv)!=before[o.name] for o in (head,old)):
        raise RuntimeError('Original head/body changed during temporary-copy observation')
    if set(o.as_pointer() for o in bpy.data.objects)!=original_objects:raise RuntimeError('Temporary observation object leaked')
    if A['_sha'](source)!=source_sha:raise RuntimeError('Saved source changed')
    with payload.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report['arrays']={'path':str(payload.relative_to(project_path())),'sha256':A['_sha'](payload)}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_ANATOMICAL_NECK_OBSERVED '+json.dumps({'report':str(output),'sourceSha256':source_sha,'arraysSha256':report['arrays']['sha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/adult-neck-anatomical-observation-r11.json')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source,args.output)

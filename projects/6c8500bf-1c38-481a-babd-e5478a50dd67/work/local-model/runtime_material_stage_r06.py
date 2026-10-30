"""Explicit bounded material stages from the actual 169-mesh R06 inventory.

No auto-run. Call plan(), then run_stage('skin'|'root'|'iris'|'blouse'|'skirt'|
'sock') one at a time in a disposable runtime Blender process. Save/review each
successful stage separately. Constants and existing direct image PBR stay intact.
This file does not launch Blender, change production sources or save a Blend.
"""
import hashlib,importlib.util,json
from pathlib import Path

def _api():
    p=Path(__file__).with_name('runtime_material_bake_r06.py');s=importlib.util.spec_from_file_location('runtime_channels_r06',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def _eye_api():
    p=Path(__file__).with_name('runtime_eye_material_r09.py');s=importlib.util.spec_from_file_location('runtime_current_eye09',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def _iris_stage(objects,api):return _eye_api().stage(objects,api)

def plan():
    api=_api();api._guard();bpy=api.bpy
    assembly_path=Path(__file__).with_name('runtime_assembly_r06.py');s=importlib.util.spec_from_file_location('runtime_stage_assembly',assembly_path)
    assembly=importlib.util.module_from_spec(s);s.loader.exec_module(assembly);_,objects=assembly.current_meshes()
    head=bpy.data.objects['RT06 Face / R01 Head']
    slots=[i for i,m in enumerate(head.data.materials) if m and m.name=='R04 physical skin']
    if len(slots)!=1:raise RuntimeError('Current face must have one explicit R04 physical skin slot')
    shared={label:{'material':material,'objects':[o.name for o in objects if len(o.data.materials)==1 and o.data.materials[0].name==material],
                   'channel':channel,'resolution':resolution}
            for label,material,channel,resolution in (
                ('blouse','Character / blouse','normal',1024),('skirt','Character / skirt','normal',1024),
                ('sock','Character / sock','normal',1024))}
    shared['iris']=_iris_stage(objects,api)
    expected={'blouse':5,'skirt':2,'sock':6,'iris':2}
    for label,row in shared.items():
        if len(row['objects'])!=expected[label]:raise RuntimeError('Material ownership changed; inspect before baking '+label)
    report={'schema':1,'source':bpy.data.filepath,'explicitMeshCount':len(objects),'explicitMeshes':[o.name for o in objects],
        'skin':{'object':head.name,'slots':slots,'resolution':2048,'channels':['basecolor','roughness','normal','alpha'],
                'unlistedSlots':'Preserved constants; no images generated for oral/body materials'},
        'root':{'object':'R06 attached scalp root substrate','resolution':1024,'channels':['basecolor','roughness','normal','alpha']},
        'shared':shared,'newImagesExpected':11+int(shared['iris']['action']=='bake'),
        'preserved':['clothing-basecolor portable material and normal image atlas',
          'Constant PBR metal, button, leather clearcoat, trims, laces, nails, mouth and body materials',
          'Protected hair alpha atlases, including the five native-path projected region maps',
          'Current eye source: direct R07 image or actual baked adopted R09 HSV/Multiply; sclera and optical constants remain current'],
        'losses':['SSS cannot be retained by this preview PBR route; source physical materials remain in production file',
          'Hair atlas preservation is not appearance acceptance; the earlier shared analytic atlas failed review, and new regional maps require their own review'],
        'scope':'Plan only; no UV changes, image bake or material assignment'}
    (api.project_path()/'work/runtime-r06/material-stage-plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def run_stage(stage,output_root='work/runtime-r06/channel-bakes-v1'):
    api=_api();api._guard();manifest=api.project_path()/'work/runtime-r06/material-stage-plan.json'
    report=json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else plan()
    if any(api.bpy.data.objects.get(name) is None for name in report['explicitMeshes']):raise RuntimeError('Prepared runtime mesh inventory no longer exists')
    if stage=='iris':
        # Re-read current assignments so an older prototype plan cannot overwrite
        # a newly adopted documented iris texture with a historical bake.
        objects=[api.bpy.data.objects[n] for n in report['explicitMeshes']]
        row=_iris_stage(objects,api)
        report['shared']['iris']=row;report['newImagesExpected']=11+int(row['action']=='bake')
        manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if row['action']=='preserve':
            source,root=api._guard();dest=api._within_output(output_root+'/iris',root)
            skipped={'stage':'iris','status':'skipped; preserved current direct image colour','source':str(source),**row,
                     'imagesBaked':0,'uvChanged':False,'materialChanged':False,'blendSaved':False}
            (dest/'iris-skip-report.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf-8')
            return skipped
        result=_eye_api().bake_current(row,objects,api,output_relative=output_root+'/iris')
        result['sourceEyeToneCandidate']=report.get('faceRefreshEyeToneCandidate')
        result['sourceIrisOpticsCandidate']=report.get('faceRefreshIrisOpticsCandidate')
        dest=api.project_path()/output_root/'iris'
        with (dest/'iris-adopted-source-change-record.json').open('x',encoding='utf-8') as stream:
            json.dump({'sourceEyeToneCandidate':result['sourceEyeToneCandidate'],
                'sourceIrisOpticsCandidate':result['sourceIrisOpticsCandidate'],'actualCurrentEyeStage':row,
                'actualBakeImage':result['image'],'opticsAndScleraPreservedExact':True,
                'revertedToUnadjustedR07Image':False,'appearanceAccepted':False},stream,ensure_ascii=False,indent=2)
        return result
    if stage=='skin':
        row=report['skin']
        return api.bake([row['object']],material_domains={row['object']:[{'name':'physical-skin','slots':row['slots'],'resolution':row['resolution']}]},
            preserve_unlisted=True,repack_uv='SMART_PROJECT_BOUNDED',output_relative=output_root+'/skin')
    if stage=='root':
        row=report['root'];result=api.bake([row['object']],resolution=row['resolution'],output_relative=output_root+'/root')
        obj=api.bpy.data.objects[row['object']]
        if obj.get('RT09_root_bake_required'):
            if not obj.data.materials or not all(m.get('r06_baked_material') for m in obj.data.materials):
                raise RuntimeError('Refreshed root did not acquire actual baked PBR materials')
            obj['RT09_root_bake_required']=False
            obj['RT09_root_baked_geometry_source_sha256']=obj['RT09_root_geometry_source_sha256']
        return result
    if stage in report['shared']:
        row=report['shared'][stage]
        return api.bake_shared_channel(row['objects'],row['material'],channel=row['channel'],resolution=row['resolution'],output_relative=output_root+'/'+stage)
    raise ValueError('Unknown material stage: '+stage)

if __name__=='__main__':print('Loaded only. plan(), then run_stage(name) in sequential disposable stages.')

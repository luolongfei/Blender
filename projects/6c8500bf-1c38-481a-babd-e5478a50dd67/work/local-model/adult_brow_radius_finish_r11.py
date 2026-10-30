"""Author the reviewed 1.50x brow radius on two new, fully recorded data copies.

No lash, centerline, root, material, binding map, head or light is changed.
Both the previous unbound reference and bound data remain retained. The new
unbound reference is an explicit authored-radius revision, not a rewrite of
history. The unchanged binding validator checks the new radius exactly.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

NAMES=('R11 Adult natural brow L','R11 Adult natural brow R')
MARKER='R11_authored_brow_radius_finish'
FACTOR=1.50


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def apply():
    import bpy
    import numpy as np
    scene=bpy.context.scene;source=Path(bpy.data.filepath).resolve()
    source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or MARKER in scene:
        raise RuntimeError('Fresh independent source, no stacked radius treatment')
    binding=api('face_fibre_binding_r11');before_validation=binding['validate'](neutral=False)
    original_record=json.loads(scene[binding['MARKER']]);record=copy.deepcopy(original_record)
    head=bpy.data.objects[binding['HEAD']];protected=binding['_protected'](head);external=binding['_external'](bpy)
    original_data={};preserved=[];created=[];changes=[]
    def snapshot(obj):
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.data
        return {'points':binding['_array'](data.position_data,'vector'),
                'radii':binding['_array'](data.attributes['radius'].data,'value',1),
                'counts':np.asarray([c.points_length for c in data.curves],np.int32),
                'matrix':np.asarray(evaluated.matrix_world)}
    objects=[bpy.data.objects[name] for name in NAMES]
    before_actual={obj.name:snapshot(obj) for obj in objects}
    try:
        for obj in objects:
            item=record['objects'][obj.name];prior_item=copy.deepcopy(item)
            reference=bpy.data.hair_curves[item['originalData']];bound=obj.data
            if obj.type!='CURVES' or item.get('radiusChange'):
                raise RuntimeError('Inspect existing fibre revision '+obj.name)
            original_data[obj.name]=bound
            reference_before=binding['_curve_contract'](reference);bound_before=binding['_curve_contract'](bound)
            for data in (reference,bound):
                if not any(row['data']==data for row in preserved):
                    preserved.append({'data':data,'fakeUser':data.use_fake_user,'contract':binding['_curve_contract'](data)})
                    data.use_fake_user=True
            new_reference=reference.copy();created.append(new_reference)
            new_reference.name=reference.name+' - R11 authored radius 150 reference';new_reference.use_fake_user=True
            new_bound=bound.copy();created.append(new_bound)
            new_bound.name=bound.name+' - R11 authored radius 150 bound'
            for data,prior in ((new_reference,reference_before),(new_bound,bound_before)):
                radius=binding['_array'](data.attributes['radius'].data,'value',1).ravel()
                data.attributes['radius'].data.foreach_set('value',radius*np.float32(FACTOR))
                current=binding['_curve_contract'](data);comparison=copy.deepcopy(current)
                comparison['attributes']['radius']=prior['attributes']['radius']
                if comparison!=prior:raise RuntimeError('Brow authoring changed more than radius '+obj.name)
            if binding['_mapping_hash'](binding['_mapping'](new_bound))!=item['mappingSha256']:
                raise RuntimeError('Brow surface binding map changed')
            obj.data=new_bound
            item['originalData']=new_reference.name;item['candidateData']=new_bound.name
            item['originalFakeUser']=True
            item['originalContractSha256']=binding['_hash'](binding['_curve_contract'](new_reference))
            current=binding['_curve_contract'](new_bound)
            if 'currentMaterialContractSha256' in item or current['materials']!=binding['_curve_contract'](new_reference)['materials']:
                item['currentMaterialContractSha256']=binding['_hash'](current)
            change={'object':obj.name,'factor':FACTOR,'priorBindingItem':prior_item,
                    'priorReferenceData':reference.name,'priorBoundData':bound.name,
                    'newReferenceData':new_reference.name,'newBoundData':new_bound.name,
                    'priorReferenceContract':reference_before,'priorBoundContract':bound_before,
                    'authoredReferenceContract':binding['_curve_contract'](new_reference),'authoredBoundContract':current,
                    'referenceMeaning':'Explicit newly authored radius revision; prior unmodified reference/data retained with full prior binding item.',
                    'reason':'Reviewed actual source/coating image: retain 1.50x brow radius only; lash radii stay unchanged.',
                    'helperSha256':sha(__file__)}
            item['radiusChange']={key:change[key] for key in ('factor','priorReferenceData','priorBoundData','newReferenceData','newBoundData','referenceMeaning','reason','helperSha256')}
            change['authoredBindingItem']=copy.deepcopy(item)
            changes.append(change)
        record.setdefault('authoredRadiusRevisionHistory',[]).append({'helperSha256':sha(__file__),'source':str(source.relative_to(project_path())),
                    'sourceSha256':sha(source),'objects':list(NAMES),'factor':FACTOR,'previousReferenceRetained':True})
        scene[binding['MARKER']]=json.dumps(record,ensure_ascii=False)
        bpy.context.view_layer.update();actual_proof={}
        for obj in objects:
            before=before_actual[obj.name];after=snapshot(obj)
            if any(not np.array_equal(before[k],after[k]) for k in ('points','counts','matrix')):
                raise RuntimeError('Actual brow centerline, count or transform changed '+obj.name)
            if not np.array_equal(after['radii'],before['radii']*np.float32(FACTOR)):
                raise RuntimeError('Actual authored radius differs from exact requested formula '+obj.name)
            scale=max(abs(float(v)) for v in obj.matrix_world.to_scale())
            actual_proof[obj.name]={'actualCenterlineCountAndTransformExact':True,'radiusFormulaExact':True,
                    'physicalRadiusMicrometersBefore':np.quantile(before['radii']*scale*1e6,[0,.5,1]).tolist(),
                    'physicalRadiusMicrometersAfter':np.quantile(after['radii']*scale*1e6,[0,.5,1]).tolist()}
        validation=binding['validate'](neutral=False)
        if binding['_protected'](head)!=protected or binding['_external'](bpy)!=external:
            raise RuntimeError('Brow radius revision changed protected head, other objects, materials, pose or lights')
        for row in preserved:
            if binding['_curve_contract'](row['data'])!=row['contract']:raise RuntimeError('Retained prior fibre data changed')
        for name,item in original_record['objects'].items():
            if name not in NAMES and item!=record['objects'][name]:raise RuntimeError('Other fibre binding entry changed')
        report={'source':str(source.relative_to(project_path())),'sourceSha256':sha(source),'helperSha256':sha(__file__),
                'changes':changes,'factor':FACTOR,'bindingBefore':before_validation,'bindingAfter':validation,
                'actualRadiusProof':actual_proof,'originalBindingRecord':original_record,
                'retainedDataFakeUserBefore':{row['data'].name:row['fakeUser'] for row in preserved},
                'appearanceAccepted':False,'sourceSaved':False,
                'scope':'Only two brow radii authored on retained copies. Existing binding validator and tolerances are unchanged. No lash change.',
                'restoreOrder':'Restore this radius revision before restoring an older brow-material candidate; do not use old material restore across a later geometry revision.'}
        scene[MARKER]=json.dumps(report,ensure_ascii=False);return report
    except Exception:
        for name,data in original_data.items():bpy.data.objects[name].data=data
        scene[binding['MARKER']]=json.dumps(original_record,ensure_ascii=False)
        for row in preserved:row['data'].use_fake_user=row['fakeUser']
        for data in created:
            data.use_fake_user=False
            if data.users==0:bpy.data.hair_curves.remove(data)
        bpy.context.view_layer.update();raise


def restore():
    """Restore only this revision; refuse later edits to these brow records/data."""
    import bpy
    scene=bpy.context.scene;binding=api('face_fibre_binding_r11')
    report=json.loads(scene[MARKER]);record=json.loads(scene[binding['MARKER']]);binding['validate'](neutral=False)
    for change in report['changes']:
        obj=bpy.data.objects[change['object']]
        if record['objects'][change['object']]!=change['authoredBindingItem']:
            raise RuntimeError('Later brow binding revision exists; do not overwrite it during restore')
        if obj.data.name!=change['newBoundData'] or binding['_curve_contract'](obj.data)!=change['authoredBoundContract']:
            raise RuntimeError('Later brow change exists; do not overwrite it during restore')
        if binding['_curve_contract'](bpy.data.hair_curves[change['priorBoundData']])!=change['priorBoundContract']:
            raise RuntimeError('Retained pre-radius data no longer exact')
        if binding['_curve_contract'](bpy.data.hair_curves[change['priorReferenceData']])!=change['priorReferenceContract']:
            raise RuntimeError('Retained pre-radius reference no longer exact')
    for change in report['changes']:
        bpy.data.objects[change['object']].data=bpy.data.hair_curves[change['priorBoundData']]
        record['objects'][change['object']]=change['priorBindingItem']
    for name,value in report['retainedDataFakeUserBefore'].items():bpy.data.hair_curves[name].use_fake_user=value
    record.setdefault('authoredRadiusRevisionHistory',[]).append({'restoredHelperSha256':report['helperSha256'],'objects':list(NAMES),'restored':True})
    scene[binding['MARKER']]=json.dumps(record,ensure_ascii=False);del scene[MARKER]
    bpy.context.view_layer.update();return binding['validate'](neutral=False)


def prepare(source_relative,output_relative):
    import bpy
    p=project_path().resolve();source=(p/source_relative).resolve();out=(p/output_relative).resolve()
    for path in (source,out):path.relative_to(p/'work/reconstruction-r02')
    report_path=out.with_suffix('.brow-radius.json')
    if source.suffix!='.blend' or out.suffix!='.blend' or out.exists() or report_path.exists():raise RuntimeError('Fresh separate blend/report outputs required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh exact saved independent source required')
    source_sha=sha(source);report=apply()
    if sha(source)!=source_sha:raise RuntimeError('Saved source changed')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Brow radius candidate save failed')
    report.update(output=output_relative,outputSha256=sha(out),savedBlend=True)
    with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_BROW_RADIUS_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(args.source,args.output)

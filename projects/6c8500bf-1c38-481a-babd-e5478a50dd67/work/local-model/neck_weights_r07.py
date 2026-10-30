"""Bounded neck skin correction after R06 head-pivot calibration.

Explicit run() changes only R01 Head's head/neck weights. No rest bones,
coordinates, shape keys, modifiers, poses, actions, hair or cloth caches change.
The existing upperChest/neck collar blend is retained; excess early head weight
is transferred to neck. apply_and_save() saves only a new independent candidate.
No Blender launch, rendering, or change occurs on import.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
REPORT='work/reconstruction-r02/neck-weights-r07.json'
MARKER='R07_neck_weight_correction'
PARAMETERS={'headReleaseStartZ':1.290,'headReleaseEndZ':1.370,
            'regionStartZ':1.265,'regionFullZ':1.275,
            'frontProtectedY':-.060,'frontFullY':-.025,
            'sideFullAbsX':.055,'sideProtectedAbsX':.075}


def _smooth(a,b,x):
    t=np.clip((np.asarray(x)-a)/(b-a),0.,1.)
    return t*t*(3.-2.*t)


def plan_from_data(data):
    """Pure offline plan from current pre-skin shape combination, never raw co.

    The AP mask preserves front chin/lips/face and softly joins the throat.
    The lateral mask protects shoulder/side-face regions. Current upperChest,
    shoulder, arm and all other weights are preserved point-for-point.
    """
    p=PARAMETERS;vertices=data['vertices'];matrix=np.asarray(data['matrixWorld'])
    coords=np.asarray([v['currentMix'] for v in vertices],dtype=float)
    world=coords@matrix[:3,:3].T+matrix[:3,3]
    x,y,z=world.T
    mask=(_smooth(p['regionStartZ'],p['regionFullZ'],z)*
          _smooth(p['frontProtectedY'],p['frontFullY'],y)*
          (1.-_smooth(p['sideFullAbsX'],p['sideProtectedAbsX'],np.abs(x))))
    release=_smooth(p['headReleaseStartZ'],p['headReleaseEndZ'],z)
    changes=[]
    for i,v in enumerate(vertices):
        old=v['weights'];head=float(old.get('head',0.));neck=float(old.get('neck',0.))
        transfer=head*float(mask[i])*(1.-float(release[i]))
        if transfer<=1.e-8: continue
        changes.append({'id':int(v['id']),'worldCurrentMix':world[i].tolist(),
            'mask':float(mask[i]),'headRelease':float(release[i]),
            'before':{k:old[k] for k in ('head','neck') if k in old},
            'after':{'head':head-transfer,'neck':neck+transfer},
            'transferredHeadToNeck':transfer})
    if not 20<=len(changes)<=1200:
        raise RuntimeError(f'Unexpected bounded correction size: {len(changes)} vertices')
    changed_ids={r['id'] for r in changes}
    bounds=np.asarray([r['worldCurrentMix'] for r in changes])
    protected=[v['id'] for i,v in enumerate(vertices) if y[i]<=p['frontProtectedY'] or
               abs(x[i])>=p['sideProtectedAbsX'] or z[i]>=p['headReleaseEndZ'] or z[i]<=p['regionStartZ']]
    if changed_ids.intersection(protected): raise RuntimeError('Protected vertex selected')
    return {'parameters':dict(p),'coordinateSelection':'Current pre-skin relative shape-key combination in world space; mesh.vertices coordinates are not used for selection.',
            'changes':changes,'changedVertexCount':len(changes),
            'changedRegionMinimum':bounds.min(axis=0).tolist(),'changedRegionMaximum':bounds.max(axis=0).tolist(),
            'maximumTransferredHeadWeight':max(r['transferredHeadToNeck'] for r in changes),
            'upperChestWeightsUnchangedByDesign':True,'protectedVertexCount':len(protected)}


def _coords(points):
    a=np.empty(len(points)*3,dtype=np.float64);points.foreach_get('co',a)
    return a.reshape((-1,3))


def _weights(obj):
    names={g.index:g.name for g in obj.vertex_groups}
    return [{names[g.group]:float(g.weight) for g in v.groups} for v in obj.data.vertices]


def _read(obj):
    keys=obj.data.shape_keys
    if not keys or not keys.use_relative: raise RuntimeError('Existing relative shape keys required')
    basis=_coords(keys.reference_key.data);mix=basis.copy();weights=_weights(obj);metadata=[]
    for key in keys.key_blocks:
        metadata.append({'name':key.name,'value':float(key.value),'mute':bool(key.mute),
                         'relativeKey':key.relative_key.name,'vertexGroup':key.vertex_group})
        if key==keys.reference_key or key.mute or not key.value: continue
        diff=_coords(key.data)-_coords(key.relative_key.data)
        if key.vertex_group:
            diff*=np.asarray([w.get(key.vertex_group,0.) for w in weights])[:,None]
        mix+=float(key.value)*diff
    return {'matrixWorld':[list(r) for r in obj.matrix_world],
        'vertices':[{'id':i,'currentMix':mix[i].tolist(),'weights':w} for i,w in enumerate(weights)],
        'shapeKeys':metadata,'maxBasisCurrentMixDifferenceMeters':float(np.max(np.linalg.norm(mix-basis,axis=1)))}


def _geometry_hash(obj):
    h=hashlib.sha256();h.update(_coords(obj.data.vertices).tobytes())
    for key in obj.data.shape_keys.key_blocks:
        h.update(key.name.encode());h.update(_coords(key.data).tobytes())
    return h.hexdigest()


def _file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _rest(rig):
    return {b.name:{'head':list(b.head_local),'tail':list(b.tail_local),
                   'matrix':[list(r) for r in b.matrix_local]} for b in rig.data.bones}


def _guard():
    if bpy is None:raise RuntimeError('Explicit execution in the existing Blender candidate required')
    source=Path(bpy.data.filepath).resolve();project=project_path().resolve()
    if not bpy.data.filepath or source==(project/'work/character.blend').resolve():
        raise RuntimeError('Use a saved independent candidate, never the formal main source')
    source.relative_to((project/'work').resolve())
    if bpy.context.mode not in {'OBJECT','POSE'}:raise RuntimeError('Leave edit/sculpt mode first')
    rig=bpy.data.objects.get('Character_Rig');obj=bpy.data.objects.get('R01 Head')
    if rig is None or obj is None or obj.type!='MESH' or obj.library or obj.data.library:
        raise RuntimeError('Existing local rig and R01 Head mesh required')
    if 'R06_head_pivot_calibration' not in rig:raise RuntimeError('Calibrated head pivot required')
    if max(abs(rig.data.bones['head'].head_local[i]-v) for i,v in enumerate((0.,.020,1.370)))>2e-6:
        raise RuntimeError('Unexpected head pivot; do not apply this bounded correction')
    if any(obj.vertex_groups.get(n) is None for n in ('head','neck','upperChest')):
        raise RuntimeError('Existing head, neck and upperChest groups required')
    if obj.data.shape_keys and any(k.vertex_group in {'head','neck'} for k in obj.data.shape_keys.key_blocks):
        raise RuntimeError('A shape-key mask uses the edited skin groups; separate its mask before correction')
    if not any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):
        raise RuntimeError('R01 Head must retain its existing skin binding')
    if max(abs(rig.matrix_world[i][j]-(1. if i==j else 0.)) for i in range(4) for j in range(4))>1e-6:
        raise RuntimeError('Expected unit rig world transform')
    return source,rig,obj


def _set_pair(obj,ident,weights):
    for name in ('head','neck'):
        group=obj.vertex_groups[name]
        if name in weights:group.add([ident],float(weights[name]),'REPLACE')
        else:group.remove([ident])


def run(report_relative=REPORT):
    """Apply once, record exact prior weights for rollback, do not save/render.

    Follow with the existing finite neck pose review, using a new output folder.
    This does not claim those visible poses pass until the renders are inspected.
    """
    source,rig,obj=_guard()
    if MARKER in obj:raise RuntimeError('Correction already applied; retain history or use explicit rollback')
    report_path=(project_path()/report_relative).resolve()
    report_path.relative_to((project_path()/'work').resolve())
    if report_path.exists():raise RuntimeError('Use a new report path; previous weight evidence is retained')
    data=_read(obj);plan=plan_from_data(data);before=_weights(obj)
    geometry_before=_geometry_hash(obj);rest_before=_rest(rig);source_sha=_file_hash(source)
    report={'sourceBlend':str(source),'sourceFileSha256':source_sha,'sourceHadUnsavedChanges':bool(bpy.data.is_dirty),
            'blenderVersion':bpy.app.version_string,'object':obj.name,'shapeKeyValuesAtSelection':data['shapeKeys'],
            'maxBasisCurrentMixDifferenceMeters':data['maxBasisCurrentMixDifferenceMeters'],
            'geometrySha256':geometry_before,'plan':plan,'applied':False,'savedBlend':False,
            'reviewRequired':'Compare the existing +/-15 degree side head poses and +/-30 degree turn poses. No render or animation acceptance is implied by the weight checks.'}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for change in plan['changes']:_set_pair(obj,change['id'],change['after'])
        obj.data.update();obj.update_tag();bpy.context.view_layer.update()
        after=_weights(obj);changed={c['id'] for c in plan['changes']};total_error=0.
        for ident,(a,b) in enumerate(zip(before,after)):
            if ident not in changed and a!=b:raise RuntimeError(f'Unscoped vertex weights changed: {ident}')
            if {k:v for k,v in a.items() if k not in ('head','neck')}!={k:v for k,v in b.items() if k not in ('head','neck')}:
                raise RuntimeError(f'Non-head/neck weights changed: {ident}')
            total_error=max(total_error,abs(sum(a.values())-sum(b.values())))
        if total_error>2e-7:raise RuntimeError(f'Total weight changed beyond float storage: {total_error}')
        if _geometry_hash(obj)!=geometry_before:raise RuntimeError('Raw mesh or shape-key coordinates changed')
        if _rest(rig)!=rest_before:raise RuntimeError('A bone rest transform changed')
        for change in plan['changes']:change['actualAfter']={k:after[change['id']][k] for k in ('head','neck')}
        report.update(applied=True,totalWeightMaximumError=total_error,allOtherWeightsExactlyUnchanged=True,
                      allBoneRestExactlyUnchanged=True,rawMeshAndAllShapeKeyCoordinatesExactlyUnchanged=True,
                      poseActionsModifiersCachesUnchangedByScope=True)
        obj[MARKER]=str(report_path)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as error:
        for change in plan['changes']:_set_pair(obj,change['id'],change['before'])
        if MARKER in obj:del obj[MARKER]
        obj.data.update();obj.update_tag();bpy.context.view_layer.update()
        report.update(applied=False,rolledBack=True,error=str(error))
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    return report


def apply_and_save(output_relative='work/reconstruction-r02/reference-neck-weights-r07.blend',report_relative=REPORT):
    """Save only a new candidate; never replace the input or an existing file."""
    _guard();target=(project_path()/output_relative).resolve();target.relative_to((project_path()/'work').resolve())
    if target.exists() or target.suffix.lower()!='.blend':raise RuntimeError('A new .blend candidate path is required')
    report=run(report_relative);bpy.ops.wm.save_as_mainfile(filepath=str(target))
    report.update(savedBlend=True,outputBlend=str(target))
    (project_path()/report_relative).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def rollback(report_relative=REPORT):
    """Restore precisely recorded prior head/neck weights, without saving."""
    _,_,obj=_guard();report=json.loads((project_path()/report_relative).read_text(encoding='utf-8'))
    if not report.get('applied') or _geometry_hash(obj)!=report['geometrySha256']:
        raise RuntimeError('Rollback requires this applied correction and unchanged raw/key geometry')
    weights=_weights(obj)
    for change in report['plan']['changes']:
        actual={k:weights[change['id']].get(k) for k in ('head','neck')}
        if actual!=change['actualAfter']:raise RuntimeError('Weights have changed since this correction; refusing stale rollback')
    for change in report['plan']['changes']:_set_pair(obj,change['id'],change['before'])
    if MARKER in obj:del obj[MARKER]
    obj.data.update();obj.update_tag();bpy.context.view_layer.update()
    return {'restoredVertexCount':len(report['plan']['changes']),'savedBlend':False}


if __name__=='__main__':
    print('Loaded only. Explicit run() applies bounded head-to-neck weight transfer; apply_and_save() saves a new candidate.')

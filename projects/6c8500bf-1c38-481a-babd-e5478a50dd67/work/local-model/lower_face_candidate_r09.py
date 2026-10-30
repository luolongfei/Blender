"""Independent R09 lower-face proportion candidate; no automatic execution.

run() uses the explicit current R08 ProductionUV source in a fresh background
process, validates an 11% lower-face Z compression, and saves only a new R09
candidate. Head key IDs, UVs, rig rest and the body are preserved. Oral objects
receive the same per-point affine map; moving lower oral parts receive sampled
driver corrections for the existing jaw parent's non-commuting rotation.
No render, lighting/groom edit, export, runtime refresh or acceptance claim.
"""
import argparse, base64, contextlib, hashlib, io, json, math, runpy, sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix
except ImportError:
    bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SOURCE='work/reconstruction-r02/reference-production-uv-r08.blend'
TARGET='work/reconstruction-r02/reference-lower-face-candidate-r09.blend'
MARKER='R09_lower_face_candidate'
PIVOT_Z=1.401
ORAL=('R02 Upper teeth','R02 Lower teeth','R02 Tongue','R03 Upper gingiva','R03 Lower gingiva')
KNOTS=np.linspace(0.,1.,9)
PROBES=(0.,.1,.2,.35,.5,.65,.75,.85,1.)
CONTROLS=('R02_blink_L','R02_blink_R','R02_jaw_open','R02_mouth_wide','R02_pucker','R02_lip_press',
          'R02_frown','R02_brow_raise','R02_brow_furrow','R03_squint_L','R03_squint_R')


def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3.-2.*t)


def spatial_weights(points):
    x,y,z=np.asarray(points,dtype=float).T
    return (smooth((z-1.270)/.040)*(1-smooth((z-1.385)/.018))
        *(1-smooth((np.abs(x)-.068)/.028))*(1-smooth((y-.015)/.055)))


def affine_map(points,weights,compression=.11):
    """Weights come from the actual neutral mix and remain fixed across keys."""
    result=np.asarray(points,dtype=float).copy()
    result[:,2]+=compression*np.asarray(weights)*(PIVOT_Z-result[:,2])
    return result


def head_weights(current,ids,edges,ocular):
    w=spatial_weights(current);lookup={int(v):i for i,v in enumerate(ids)}
    adjacent=[set() for _ in ids]
    for a,b in edges:
        if a in lookup and b in lookup:
            a,b=lookup[a],lookup[b];adjacent[a].add(b);adjacent[b].add(a)
    w[ocular]=0.;seen=set(np.flatnonzero(ocular));frontier=seen.copy()
    for factor in (.20,.55,.85):
        frontier=set().union(*(adjacent[i] for i in frontier))-seen if frontier else set()
        seen|=frontier;w[list(frontier)]*=factor
    return w


def ocular_mask(current,ids,arrays,basis):
    fixed=np.zeros(len(ids),dtype=bool)
    for name in ('R02_globe_contact','R05_aperture_L','R05_aperture_R'):
        fixed|=np.linalg.norm(arrays[name][ids]-basis[ids],axis=1)>.00005
    x,y,z=np.asarray(current).T
    orbit=(((np.abs(x)-.036)/.033)**2+((z-1.413)/.028)**2<=1)&(y<-.030)
    fixed|=orbit
    for name,co in arrays.items():
        if any(part in name.lower() for part in ('blink','squint','gaze')):
            fixed|=(np.linalg.norm(co[ids]-basis[ids],axis=1)>1e-7)&(z>1.378)&(np.abs(x)<.080)&(y<-.030)
    return fixed


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _coords(data):
    a=np.empty(len(data)*3,dtype=np.float32);data.foreach_get('co',a)
    return a.reshape(-1,3)


def _world(co,matrix):
    m=np.asarray(matrix,dtype=float)
    return np.asarray(co)@m[:3,:3].T+m[:3,3]


def _local(co,matrix):return _world(co,np.linalg.inv(np.asarray(matrix)))


def _mix(obj):
    keys=obj.data.shape_keys
    if not keys:return _coords(obj.data.vertices)
    blocks=keys.key_blocks;arrays={k.name:_coords(k.data) for k in blocks}
    result=arrays[blocks[0].name].astype(float)
    for k in list(blocks)[1:]:
        if k.vertex_group:raise RuntimeError('Vertex-group shape key needs explicit evaluation: '+k.name)
        if not k.mute:result+=(arrays[k.name]-arrays[k.relative_key.name])*k.value
    return result


def _structure(obj):
    h=hashlib.sha256()
    if obj.type=='MESH':
        for p in obj.data.polygons:h.update(np.asarray(p.vertices,dtype=np.int32).tobytes())
        for uv in obj.data.uv_layers:
            a=np.empty(len(uv.data)*2,np.float32);uv.data.foreach_get('uv',a)
            h.update(uv.name.encode());h.update(a.tobytes())
        for v in obj.data.vertices:
            for g in v.groups:h.update(np.asarray([g.group,g.weight],dtype=np.float64).tobytes())
    return h.hexdigest()


def _rest(rig):
    h=hashlib.sha256()
    for b in rig.data.bones:h.update(b.name.encode());h.update(np.asarray(b.matrix_local,dtype=np.float64).tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def _neutral_scope(rig,head):
    scene=bpy.context.scene;blocks=head.data.shape_keys.key_blocks
    owners=[rig,head.data.shape_keys]
    animation=[(o,o.animation_data.action if o.animation_data else None,
        o.animation_data.action_slot if o.animation_data else None,
        [(t,t.mute) for t in o.animation_data.nla_tracks] if o.animation_data else []) for o in owners]
    values={k.name:float(k.value) for k in blocks};pose=rig.data.pose_position
    bases={b.name:b.matrix_basis.copy() for b in rig.pose.bones};custom={}
    for key in ('R03_look_yaw','R03_look_pitch','R04_native_hair_enabled'):
        if key in rig:custom[key]=rig[key]
    master=rig.pose.bones.get('CTRL_R02_Master');switches={}
    if master:switches={k:master[k] for k in ('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R') if k in master}
    frame=(scene.frame_current,scene.frame_subframe)
    try:
        for o,_,_,tracks in animation:
            if o.animation_data:o.animation_data.action=None
            for t,_ in tracks:t.mute=True
        for b in rig.pose.bones:b.matrix_basis=Matrix.Identity(4)
        for k in custom:rig[k]=0.
        for k in switches:master[k]=0.
        for k in CONTROLS:blocks[k].value=0.
        blocks['R02_smile'].value=.45;rig.data.pose_position='POSE'
        scene.frame_set(1);rig.update_tag();bpy.context.view_layer.update()
        yield
    finally:
        for name,value in values.items():blocks[name].value=value
        for name,matrix in bases.items():rig.pose.bones[name].matrix_basis=matrix
        for k,v in custom.items():rig[k]=v
        for k,v in switches.items():master[k]=v
        for o,action,slot,tracks in animation:
            if o.animation_data:
                o.animation_data.action=action
                if action and slot:o.animation_data.action_slot=slot
            for t,mute in tracks:t.mute=mute
        rig.data.pose_position=pose;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()


def _jaw(head,value):
    head.data.shape_keys.key_blocks['R02_jaw_open'].value=float(value)
    head.data.update_tag();bpy.context.view_layer.update()


def _read_oral(obj):
    if obj.data.shape_keys:raise RuntimeError('Existing oral shape keys need explicit composition: '+obj.name)
    if obj.type=='MESH':return _coords(obj.data.vertices),None
    if obj.type!='CURVE':raise RuntimeError('Unexpected oral object type: '+obj.name)
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    try:
        return _coords(mesh.vertices),{'faces':[tuple(p.vertices) for p in mesh.polygons],
            'smooth':[bool(p.use_smooth) for p in mesh.polygons],
            'material_indices':[int(p.material_index) for p in mesh.polygons]}
    finally:ev.to_mesh_clear()


def _backup(head,ids,oral):
    arrays={'head_raw':_coords(head.data.vertices)[ids],'ids':np.asarray(ids,dtype=np.int32)}
    names=[]
    for i,key in enumerate(head.data.shape_keys.key_blocks):
        names.append(key.name);arrays['key_'+str(i)]=_coords(key.data)[ids]
    meshes=[]
    for i,r in enumerate(oral):
        if r['curve'] is None:arrays['oral_'+str(i)]=r['original'];meshes.append({'index':i,'name':r['name']})
    memory=io.BytesIO();np.savez_compressed(memory,**arrays)
    text=bpy.data.texts.new('R09 lower face numeric rollback');text.use_fake_user=True
    text.write(json.dumps({'headKeys':names,'oralMeshes':meshes,'arrays':base64.b64encode(memory.getvalue()).decode('ascii')}))
    return text.name


def _write_key(obj,name,co):
    key=obj.shape_key_add(name=name,from_mix=False)
    key.relative_key=obj.data.shape_keys.key_blocks[0]
    key.data.foreach_set('co',np.asarray(co,dtype=np.float32).ravel());key.value=0.
    return key


def _convert_gum(record):
    old=record['object'];name=old.name
    visible={'hide_render':old.hide_render,'hide_viewport':old.hide_viewport,'hide_set':old.hide_get()}
    old['R09_retained_curve_name']=name;old['R09_retained_curve_visibility']=json.dumps(visible)
    old.name='R09 retained original / '+name
    mesh=bpy.data.meshes.new(name+' mapped surface');mesh.from_pydata(record['original'].tolist(),[],record['curve']['faces'])
    for m in old.data.materials:mesh.materials.append(m)
    for p,s,mi in zip(mesh.polygons,record['curve']['smooth'],record['curve']['material_indices']):p.use_smooth=s;p.material_index=mi
    obj=bpy.data.objects.new(name,mesh)
    for col in old.users_collection:col.objects.link(obj)
    obj.parent=old.parent;obj.parent_type=old.parent_type;obj.parent_bone=old.parent_bone
    obj.matrix_parent_inverse=old.matrix_parent_inverse.copy();obj.matrix_basis=old.matrix_basis.copy()
    obj['R09_original_curve']=old.name;obj['R09_original_visibility']=json.dumps(visible)
    old.hide_render=True;old.hide_viewport=True;old.hide_set(True)
    record['object']=obj;return obj


def run(compression=.11,source_relative=SOURCE,target_relative=TARGET,diagnostics_only=False):
    if bpy is None or not bpy.app.background:raise RuntimeError('Fresh independent background process required; never run on live/main')
    if not .10<=compression<=.12:raise ValueError('This candidate is bounded to 10–12% compression')
    p=project_path();source=(p/source_relative).resolve();target=(p/target_relative).resolve()
    for path in (source,target):path.relative_to((p/'work/reconstruction-r02').resolve())
    if source!=Path(bpy.data.filepath).resolve() or bpy.data.is_dirty:raise RuntimeError('Exact fresh saved source_relative required')
    if source==target or 'candidate-r09' not in target.stem or target.exists() or target.with_suffix('.json').exists():
        raise RuntimeError('New independent candidate-r09 filename/report required')
    head=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig']
    if head.library or head.data.library or head.data.users!=1 or not head.data.shape_keys.use_relative:
        raise RuntimeError('Local single-user head with relative shape keys required')
    if head.get(MARKER):raise RuntimeError('Candidate already applied')
    if 'R08 ProductionUV' not in head.data.uv_layers:raise RuntimeError('Use the current ProductionUV source')
    if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6:raise RuntimeError('Expected current identity rig object transform')
    saved_sha=_sha(source);rest_before=_rest(rig);structure_before=_structure(head)
    reference=json.loads((p/'work/reconstruction-r02/face-contour-input-r06.json').read_text(encoding='utf-8'))
    ids=np.asarray(reference['ids'],dtype=np.int32);basis=_coords(head.data.shape_keys.key_blocks[0].data)
    if np.abs(basis[ids]-np.asarray(reference['keys']['Basis']['co'])).max()>3e-6:
        raise RuntimeError('Head index/Basis correspondence changed; refresh the explicit region export')
    report={'source':source_relative,'sourceSha256':saved_sha,'target':target_relative,'compression':compression,
        'mapping':'Per-neutral-point affine Z map, identical for Basis and every relative key; current mix reconstructed explicitly',
        'headPivotZ':PIVOT_Z,'fixedBelowZ':1.270,'neckFullAboveZ':1.310,'fixedAboveZ':1.403,
        'visualStatus':'Unrendered independent candidate; require front/quarter/side, AA/EE/OU and jaw sweep review'}
    originals={k.name:_coords(k.data) for k in head.data.shape_keys.key_blocks}
    raw=_coords(head.data.vertices);oral=[];backup=None;converted=[]
    try:
        with _neutral_scope(rig,head):
            current=_mix(head);matrix=np.asarray(head.matrix_world);world=_world(current[ids],matrix)
            ocular=ocular_mask(world,ids,originals,basis)
            weights=head_weights(world,ids,reference['edges'],ocular)
            mapped_current=_local(affine_map(world,weights,compression),matrix)
            targets={}
            for name,a in originals.items():
                out=a.copy();out[ids]=_local(affine_map(_world(a[ids],matrix),weights,compression),matrix)
                out[ids[weights==0]]=a[ids[weights==0]];targets[name]=out
            new_raw=raw.copy();new_raw[ids]=_local(affine_map(_world(raw[ids],matrix),weights,compression),matrix)
            new_raw[ids[weights==0]]=raw[ids[weights==0]]
            for name in ORAL:
                obj=bpy.data.objects[name]
                if obj.library or obj.data.library or obj.data.users!=1:raise RuntimeError('Local single-user oral geometry required: '+name)
                co,curve=_read_oral(obj);m0=np.asarray(obj.matrix_world).copy();ow=_world(co,m0);w=spatial_weights(ow)
                oral.append({'name':name,'object':obj,'original':co,'curve':curve,'matrix0':m0,'weights':w,'samples':{},'matrices':{}})
            for alpha in sorted(set(KNOTS.tolist()+list(PROBES))):
                _jaw(head,alpha)
                for r in oral:
                    m=np.asarray(r['object'].matrix_world).copy();r['matrices'][alpha]=m
                    desired=affine_map(_world(r['original'],m),r['weights'],compression)
                    if alpha in KNOTS:r['samples'][alpha]=_local(desired,m)
            _jaw(head,0.)
            report['head']={'headRegionVertices':len(ids),'changedVertices':int(np.count_nonzero(weights>1e-7)),
                'ocularVerticesFixed':int(ocular.sum()),'maximumNeutralLiftMm':float((mapped_current-current[ids])[:,2].max()*1000),
                'allKeys':len(targets),'keyNames':list(targets),'otherBodyVerticesChanged':0}
            report['oral']=[{'name':r['name'],'points':len(r['original']),
                'parentBone':r['object'].parent_bone,'sampledJawCorrection':not np.allclose(r['samples'][1.],r['samples'][0.],atol=1e-7),
                'retainedOriginalCurveAndMappedEvaluatedSurface':r['curve'] is not None} for r in oral]
            # Before writes, numerically check the corrective interpolation against
            # the exact same affine map at all requested non-knot jaw openings.
            numerical=[]
            for alpha in PROBES:
                lo=min(7,int(math.floor(alpha*8)));f=alpha*8-lo
                for r in oral:
                    local=(1-f)*r['samples'][float(KNOTS[lo])]+f*r['samples'][float(KNOTS[lo+1])]
                    m=r['matrices'][alpha];expected=affine_map(_world(r['original'],m),r['weights'],compression)
                    err=float(np.linalg.norm(_world(local,m)-expected,axis=1).max())
                    numerical.append({'object':r['name'],'opening':alpha,'maximumErrorMm':err*1000})
            if max(v['maximumErrorMm'] for v in numerical)>.05:raise RuntimeError('Jaw interpolation exceeds 0.05 mm; no writes')
            report['jawMapNumericalChecks']=numerical
            if diagnostics_only:report['savedBlend']=False;print(json.dumps(report));return report
            backup=_backup(head,ids,oral);head[MARKER]=backup
            head.data.vertices.foreach_set('co',new_raw.ravel())
            for key in head.data.shape_keys.key_blocks:key.data.foreach_set('co',targets[key.name].ravel())
            for r in oral:
                obj=_convert_gum(r) if r['curve'] else r['object']
                if r['curve']:converted.append(obj.name)
                obj.data.vertices.foreach_set('co',np.asarray(r['samples'][0.],dtype=np.float32).ravel());obj.data.update()
                moving=not np.allclose(r['samples'][1.],r['samples'][0.],atol=1e-7)
                if moving:
                    _write_key(obj,'Basis',r['samples'][0.])
                    for alpha in KNOTS[1:]:
                        key=_write_key(obj,f'R09_jaw_map_{alpha:.3f}',r['samples'][float(alpha)])
                        driver=key.driver_add('value').driver;driver.expression=f'max(0, 1-abs(opening-{alpha:.9f})*8)'
                        var=driver.variables.new();var.name='opening';var.type='SINGLE_PROP'
                        var.targets[0].id_type='KEY';var.targets[0].id=head.data.shape_keys
                        var.targets[0].data_path='key_blocks["R02_jaw_open"].value'
                obj[MARKER]=True;obj.update_tag()
            head.data.update();head.update_tag();bpy.context.view_layer.update()
            actual=_mix(head)[ids]
            if np.abs(actual-mapped_current).max()>3e-6:raise RuntimeError('Mapped shape-key stack does not reconstruct its current neutral target')
            if any(np.any(_coords(head.data.shape_keys.key_blocks[n].data)[ids[ocular]]!=a[ids[ocular]]) for n,a in originals.items()):
                raise RuntimeError('Ocular shape-key coordinates changed')
            evaluated=[]
            for alpha in PROBES:
                _jaw(head,alpha)
                for r in oral:
                    obj=r['object'];local=_mix(obj);m=np.asarray(obj.matrix_world)
                    expected=affine_map(_world(r['original'],r['matrices'][alpha]),r['weights'],compression)
                    err=float(np.linalg.norm(_world(local,m)-expected,axis=1).max())
                    evaluated.append({'object':obj.name,'opening':alpha,'maximumErrorMm':err*1000})
            if max(v['maximumErrorMm'] for v in evaluated)>.05:raise RuntimeError('Actual driven oral keys exceed 0.05 mm mapped-motion error')
            report['drivenJawMapChecks']=evaluated;_jaw(head,0.)
        if _structure(head)!=structure_before or _rest(rig)!=rest_before:raise RuntimeError('UV/topology/weights or rig rest changed')
        if _sha(source)!=saved_sha:raise RuntimeError('Original R08 source changed on disk')
        report.update({'productionUvTopologyWeightsUnchanged':True,'rigRestUnchanged':True,'savedBlend':True,
            'backupText':backup,'convertedGums':converted,'sourceUnchangedOnDisk':True,
            'knownLimits':['This proportion candidate does not remove the pre-existing box-shaped mouth action masks or redesign individual teeth',
                'Raising the chin stretches the preserved neck transition; inspect side view and high/low head poses',
                'Gingiva bevel surfaces are retained as hidden original curves and mapped mesh copies so their complete surfaces receive the same compression',
                'Existing runtime face payload and its validation must be resampled if this candidate is adopted']})
        head['R09_candidate_report']=json.dumps(report)
        bpy.ops.wm.save_as_mainfile(filepath=str(target));report['candidateSha256']=_sha(target)
        rp=target.with_suffix('.json')
        with rp.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        print('R09_LOWER_FACE_CANDIDATE '+json.dumps({'candidate':target_relative,'sha256':report['candidateSha256']}),flush=True)
        return report
    except Exception:
        if backup:restore()
        raise


def restore():
    """Restore original numeric data/curve objects in memory; never save."""
    head=bpy.data.objects['R01 Head'];marker=head.get(MARKER)
    if not marker:return
    text=bpy.data.texts.get(marker)
    if text is None:raise RuntimeError('R09 numeric rollback is missing')
    info=json.loads(text.as_string());arrays=np.load(io.BytesIO(base64.b64decode(info['arrays'])),allow_pickle=False)
    ids=arrays['ids'];raw=_coords(head.data.vertices);raw[ids]=arrays['head_raw'];head.data.vertices.foreach_set('co',raw.ravel())
    for i,name in enumerate(info['headKeys']):
        key=head.data.shape_keys.key_blocks[name];a=_coords(key.data);a[ids]=arrays['key_'+str(i)];key.data.foreach_set('co',a.ravel())
    for row in info['oralMeshes']:
        obj=bpy.data.objects[row['name']]
        if obj.get(MARKER) and obj.data.shape_keys:obj.shape_key_clear()
        obj.data.vertices.foreach_set('co',arrays['oral_'+str(row['index'])].ravel())
        if MARKER in obj:del obj[MARKER]
        obj.data.update()
    for old in list(bpy.data.objects):
        if not old.get('R09_retained_curve_name'):continue
        name=old['R09_retained_curve_name'];flags=json.loads(old['R09_retained_curve_visibility'])
        obj=bpy.data.objects.get(name)
        if obj and obj!=old:
            if obj.get('R09_original_curve')!=old.name:raise RuntimeError('Original gum name reused; explicit restoration needed')
            bpy.data.objects.remove(obj,do_unlink=True)
        old.name=name;old.hide_render=flags['hide_render'];old.hide_viewport=flags['hide_viewport'];old.hide_set(flags['hide_set'])
        del old['R09_retained_curve_name'];del old['R09_retained_curve_visibility']
    del head[MARKER]
    if 'R09_candidate_report' in head:del head['R09_candidate_report']
    head.data.update();bpy.context.view_layer.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=SOURCE);parser.add_argument('--target',default=TARGET)
    parser.add_argument('--compression',type=float,default=.11);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    run(args.compression,args.source,args.target,args.inspect)

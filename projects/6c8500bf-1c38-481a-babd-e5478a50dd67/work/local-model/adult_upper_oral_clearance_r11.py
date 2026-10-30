"""Retract only the true upper inner oral rows from intersecting lower lip.

Original outer vermilion and opening-ring cage controls are pinned exactly.
One reversible static key; existing smile controls, materials and teeth stay
unchanged. Actual L3 external influence and lip section lenses are measured.
"""
import argparse,collections,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HEAD='R11 Female head study';KEY='R11 upper oral clearance';MARKER='R11_upper_oral_clearance'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(items):
    a=np.empty((len(items),3),np.float32);items.foreach_get('co',a.ravel());return a
def world(v,m):
    m=np.asarray(m,float);return np.asarray(v)@m[:3,:3].T+m[:3,3]


def plan(current_world):
    skin=api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']();v=np.asarray(current_world,float)
    oral=skin['_vertices_of_groups'](donor,{7});outer=skin['_vertices_of_groups'](donor,{17});lower=skin['_vertices_of_groups'](donor,{8,22})
    boundary=oral&outer;adj=[[] for _ in v]
    for a,b in donor['edges']:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
    steps=np.full(len(v),999,np.int32);steps[boundary]=0;queue=collections.deque(map(int,np.flatnonzero(boundary)))
    while queue:
        i=queue.popleft()
        if steps[i]>=3:continue
        for j in adj[i]:
            if oral[j] and steps[j]>steps[i]+1:steps[j]=steps[i]+1;queue.append(j)
    inner=oral&~outer&~lower
    row_weight=np.zeros(len(v));row_weight[steps==1]=1.;row_weight[steps==2]=.33;row_weight[steps==3]=.08
    corner=np.exp(-((abs(v[:,0])-.0155)/.0045)**2)
    # Restrict to the three internal support rows at the front of the mouth.
    posterior=np.clip((-.086-v[:,1])/.012,0.,1.);posterior=posterior*posterior*(3-2*posterior)
    delta=np.zeros_like(v);w=row_weight*posterior*inner
    delta[:,2]=w*(.000075+.00065*corner);delta[:,1]=w*.00018*corner
    changed=np.linalg.norm(delta,axis=1)>1e-10
    if not 20<int(changed.sum())<150 or np.linalg.norm(delta,axis=1).max()>.0008:raise RuntimeError('Inspect new inner oral correction domain')
    if np.any(delta[outer|lower|~oral]):raise RuntimeError('Inner oral key touched protected outer or lower cage')
    tri=[]
    for s,n in zip(donor['face_starts'],donor['face_lengths']):
        f=donor['corner_vertices'][s:s+n]
        if changed[f].any():tri.extend((f[0],f[i],f[i+1]) for i in range(1,len(f)-1))
    t=np.asarray(tri,np.int32);a=v[t];b=(v+delta)[t];na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    area=np.linalg.norm(na,axis=1);area2=np.linalg.norm(nb,axis=1);valid=area>1e-12;alignment=np.sum(na[valid]*nb[valid],axis=1)/(area[valid]*area2[valid])
    if alignment.min()<=0:raise RuntimeError('Upper oral correction flips current raw faces')
    return delta,{'provenance':provenance,'changedOriginalCageIds':np.flatnonzero(changed).tolist(),'changedCagePoints':int(changed.sum()),
        'maximumDisplacementMeters':float(np.linalg.norm(delta,axis=1).max()),'minimumRawNormalCosine':float(alignment.min()),
        'originalUpperOuterLipAndLowerLipCagePinnedExact':True,
        'method':'Only upper oral set7 internal three graph rows, 0.075mm central support and up to0.725mm lateral lift plus0.18mm posterior retreat; first external/shared row fixed.',
        'actualL3ExternalCoordinatesUnchanged':None,'appearanceAccepted':False}


def lenses(rows):
    return [{'control':r['control'],'strength':r['strength'],
             'sections':[{'x':s['bodySectionX'],'crossings':len(s['crossings']),
                          'maximumDepthMeters':max([q.get('maximumLowerAboveUpperMeters') or 0. for q in s['localLenses']]+[0.]),
                          'overlapAreaSquareMeters':sum(q.get('positiveOverlapAreaSquareMeters') or 0. for q in s['localLenses'])} for s in r['nineLipSections']]} for r in rows]


def measure(head,arrays,prefix,controls):
    import bpy
    observe=api('adult_oral_observe_r11');capture=observe['capture_head'];rows=[];keys=head.data.shape_keys.key_blocks
    for control in controls:
        for other in controls:keys[other].value=0.
        capture.__globals__['SMILE']=control
        for index,value in enumerate((0.,.5,1.)):
            row=capture(value,prefix+'_'+str(len(rows)),arrays);row['control']=control;rows.append(row)
        keys[control].value=0.;bpy.context.view_layer.update()
    return rows


def prepare(source_relative,output_relative):
    import bpy
    source=(P/source_relative).resolve();out=(P/output_relative).resolve();rp=out.with_suffix('.upper-oral.json');npz=out.with_suffix('.upper-oral.npz')
    for p in (source,out,rp,npz):p.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent source required')
    if any(p.exists() for p in (out,rp,npz)) or MARKER in bpy.context.scene:raise RuntimeError('Preserve earlier candidate')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys;uv=api('production_uv_r08');qa=api('face_contact_audit_r11');binding=api('face_fibre_binding_r11')
    if KEY in keys.key_blocks or len(head.data.vertices)!=10582:raise RuntimeError('Fresh original full adult cage required')
    controls=[name for name in ('R11 natural soft smile','R11 warm closed smile') if name in keys.key_blocks]
    if not controls or any(abs(keys.key_blocks[n].value)>1e-7 for n in controls):raise RuntimeError('Neutral adult smile controls required')
    if 'R11 jaw open' in keys.key_blocks and abs(keys.key_blocks['R11 jaw open'].value)>1e-8:raise RuntimeError('Neutral jaw required')
    original=api('adult_head_bind_observe_r11')['data_contract'](head);pose=uv['_pose_contract']();active=head.active_shape_key_index;values={k.name:float(k.value) for k in keys.key_blocks};key=None;arrays={};saved=False;report={}
    try:
        with qa['rest_evaluation']():
            mix,_=uv['current_mix'](head);T=np.asarray(head.matrix_world,float);delta,report=plan(world(mix,T));local=delta@np.linalg.inv(T[:3,:3]).T
            baseline=qa['_evaluated_mesh'](head);before=measure(head,arrays,'before',controls)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.value=1.;key.data.foreach_set('co',(array(keys.reference_key.data)+local).astype(np.float32).ravel())
            bpy.context.view_layer.update();current=qa['_evaluated_mesh'](head)
            if baseline['faces']!=current['faces'] or not np.array_equal(baseline['groups'],current['groups']):raise RuntimeError('Upper oral correction changed evaluated topology')
            f=np.asarray(current['faces'],np.int32);outside_ids=np.unique(f[current['groups']!=7]);movement=np.linalg.norm(current['world']-baseline['world'],axis=1)
            outside_max=float(movement[outside_ids].max())
            report['actualL3ExternalMaximumDisplacementMeters']=outside_max
            if outside_max>.00015:raise RuntimeError('Internal correction changes external L3 by more than0.15mm: '+str(outside_max))
            after=measure(head,arrays,'after',controls);report['beforeLensSummary']=lenses(before);report['afterLensSummary']=lenses(after)
            report['actualFullBeforeSections']=before;report['actualFullAfterSections']=after
            oldmax=max(s['maximumDepthMeters'] for r in report['beforeLensSummary'] for s in r['sections'])
            newmax=max(s['maximumDepthMeters'] for r in report['afterLensSummary'] for s in r['sections'])
            if newmax>oldmax+1e-5:raise RuntimeError('Upper oral correction worsens measured overlap')
            report.update(actualL3ExternalMaximumDisplacementMeters=outside_max,maximumMeasuredOverlapBeforeMeters=oldmax,
                maximumMeasuredOverlapAfterMeters=newmax,allSampledCrossingLensesRemoved=all(s['crossings']==0 for r in report['afterLensSummary'] for s in r['sections']),
                checkedSmileControls=controls,combinedJawOrUnlistedExpressionAccepted=False,
                fibreBinding=binding['validate'](neutral=False))
            head.active_shape_key_index=active
        now=api('adult_head_bind_observe_r11')['data_contract'](head);now['keys']=[k for k in now['keys'] if k['name']!=KEY]
        if now!=original or uv['_pose_contract']()!=pose:raise RuntimeError('Upper oral correction changed original head/key/pose contract')
        report.update(source=source_relative,sourceSha256=sha(source),helperSha256=sha(__file__),newKey=KEY,newKeyDefault=1.,originalHeadContract=original,
            scope='Inner upper oral clearance candidate. True section-lens intersections are measured, not inferred from minimum distance zero. Teeth and new warm-smile/jaw combinations require fresh checks when present.')
        bpy.context.scene[MARKER]=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Upper oral candidate save failed')
        np.savez_compressed(npz,**arrays);report.update(output=output_relative,outputSha256=sha(out),arrays=npz.relative_to(P).as_posix(),arraysSha256=sha(npz),savedBlend=True)
        with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        saved=True;print('R11_UPPER_ORAL_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256'],'maximumOverlapMeters':report['maximumMeasuredOverlapAfterMeters']}),flush=True)
    except Exception as exc:
        report.update(source=source_relative,sourceSha256=sha(source),helperSha256=sha(__file__),error=str(exc),savedBlend=out.exists())
        if arrays and not npz.exists():
            np.savez_compressed(npz,**arrays);report.update(arrays=npz.relative_to(P).as_posix(),arraysSha256=sha(npz))
        if not rp.exists():
            with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        raise
    finally:
        if not saved:
            if key is not None and KEY in keys.key_blocks:head.shape_key_remove(key)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            for name,value in values.items():keys.key_blocks[name].value=value
            head.active_shape_key_index=active;bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(a.source,a.output)

"""Measured, reversible tooth-crown refinement; original topology/keys retained.

Offline analyze_file() measures actual connected crowns and shared mid-height
sections. run(source_relative=...) applies one neutral crown edit to raw mesh,
then transports it through each measured R09 jaw-map affine transform, preserving
EVERY existing key, its original residual and driver. A fixed tooth rotation must
not rotate the independent jaw-map compensation translation about a neutral root.
Only crowns are changed; no gum, rig, material, texture, source save or render.
"""
import base64
import hashlib
import io
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
NAMES = ('R02 Upper teeth', 'R02 Lower teeth')
MARKER = 'R09_dental_contact_candidate'
BACKUP = 'R09_dental_numeric_backup'
INPUT = 'work/reconstruction-r02/dental-observation-r09.json'


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _smooth(x):
    x = np.clip(x, 0., 1.); return x*x*(3.-2.*x)


def _coords(collection):
    a = np.empty(len(collection)*3, np.float64); collection.foreach_get('co', a)
    return a.reshape(-1, 3)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def components(count, edges):
    parent = np.arange(count)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return int(a)
    for a,b in edges: parent[find(a)] = find(b)
    groups = {}
    for i in range(count): groups.setdefault(find(i), []).append(i)
    return [np.array(v, dtype=np.int32) for v in groups.values()]


def hull(points):
    points = sorted(set(map(tuple, np.asarray(points)))); lower = []; upper = []
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    for point in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],point)<=0: lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],point)<=0: upper.pop()
        upper.append(point)
    return np.array(lower[:-1]+upper[:-1])


def signed_separation(a, b):
    """Convex polygon SAT gap; negative means overlapping 2D sections."""
    a = hull(a); b = hull(b)
    if len(a)<3 or len(b)<3: raise RuntimeError('Degenerate crown section')
    edges = np.r_[np.roll(a,-1,0)-a, np.roll(b,-1,0)-b]
    axes = np.c_[-edges[:,1],edges[:,0]]
    axes /= np.linalg.norm(axes,axis=1)[:,None]
    return float(max(max(np.min(b@axis)-np.max(a@axis), np.min(a@axis)-np.max(b@axis)) for axis in axes))


def section(vertices, edges, z):
    points = []
    for ia,ib in edges:
        a = vertices[ia]; b = vertices[ib]
        if abs(a[2]-z)<1e-10: points.append(a[:2])
        if (a[2]-z)*(b[2]-z)<0:
            points.append((a+(b-a)*((z-a[2])/(b[2]-a[2])))[:2])
    return hull(points)


def analyze(vertices, edges, polygons):
    vertices = np.asarray(vertices, float); groups = components(len(vertices), edges)
    if len(groups)!=10 or any(len(ids)!=96 for ids in groups):
        raise RuntimeError('Expected ten connected 96-vertex existing crowns')
    groups.sort(key=lambda ids: float(vertices[ids,0].mean()))
    centers = np.array([vertices[ids].mean(axis=0) for ids in groups])
    arch = np.gradient(centers[:,:2],axis=0)
    rows = []
    for rank, ids in enumerate(groups):
        chosen = set(ids.tolist()); faces = []
        for face in polygons:
            if face[0] not in chosen: continue
            q = vertices[np.asarray(face,dtype=np.int32)]
            normal = sum((np.cross(q[j]-q[0],q[j+1]-q[0]) for j in range(1,len(q)-1)), start=np.zeros(3))
            area = np.linalg.norm(normal)*.5
            if area<1e-12: continue
            normal /= area*2.
            if abs(normal[2])<.2 and normal[1]<0: faces.append((area,normal))
        area,normal = max(faces,key=lambda item:item[0])
        width_axis = np.array([-normal[1],normal[0],0.]); width_axis /= np.linalg.norm(width_axis)
        angle = math.atan2(width_axis[1],width_axis[0]); target = math.atan2(arch[rank,1],arch[rank,0])
        q = vertices[ids]-centers[rank]
        rows.append({'rank':rank, 'kind':min(rank,9-rank), 'ids':ids,
                     'center':centers[rank], 'widthAxis':width_axis, 'angle':angle, 'archAngle':target,
                     'widthMm':float(np.ptp(q@width_axis)*1000.), 'heightMm':float(np.ptp(q[:,2])*1000.),
                     'frontFaceAreaMm2':float(area*1e6)})
    pairs = []
    for i in range(9):
        a,b = rows[i],rows[i+1]
        lo = max(vertices[a['ids'],2].min(),vertices[b['ids'],2].min())
        hi = min(vertices[a['ids'],2].max(),vertices[b['ids'],2].max())
        if hi-lo<.001: raise RuntimeError('Adjacent crowns do not share a meaningful mid-height section')
        z = (lo+hi)/2
        sections=[]
        for row in (a,b):
            ids = set(row['ids'].tolist()); selected=[e for e in edges if e[0] in ids and e[1] in ids]
            sections.append(section(vertices,selected,z))
        gap=signed_separation(*sections)
        pairs.append({'leftRank':i,'rightRank':i+1,'sharedSectionZ':float(z),
                      'signedSectionSeparationMm':gap*1000., 'overlappingSection':gap<0})
    return rows,pairs


def _mapped(vertices, rows, widths, lower=False):
    result=vertices.copy(); matrices=np.tile(np.eye(3),(len(vertices),1,1)); offsets=np.zeros_like(vertices)
    records=[]
    for row in rows:
        rank=row['rank']; typ=min(abs(rank-4),abs(rank-5)); ids=row['ids']; center=row['center']; axis=row['widthAxis']
        # Correct only part of the opposite tooth/arch directions. No crown is
        # fully reoriented. Transform centers are not translated; separate gums
        # are untouched (the crown surfaces themselves do move near those gums).
        angle=float(np.clip((row['archAngle']-row['angle'])*.45,-math.radians(18),math.radians(18)))
        c=math.cos(angle);s=math.sin(angle);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        scale=np.eye(3)+(widths[typ]-1.)*np.outer(axis,axis)
        matrix=rotation@scale
        delta=(vertices[ids]-center)@matrix.T
        # Small corner retreat at the cutting edge uses existing bevel loops.
        # This rounds exposed corners without creating new broad face planes or
        # changing the gingival half of a crown. Fixed weights apply to all keys.
        new_axis=rotation@axis; u=delta@new_axis
        half=max(np.max(np.abs(u)),1e-8); z=vertices[ids,2]
        edge=z.max() if lower else z.min(); depth=(edge-z) if lower else (z-edge)
        corner=_smooth((np.abs(u)/half-.62)/.38)*(1.-_smooth(depth/.0011))
        amount=.00012 if typ in (0,1) else (.00006 if typ==2 else 0.)
        retreat=-np.sign(u)*corner*amount
        soften=(np.outer(retreat,new_axis)
                +np.outer(corner*amount*(-1. if lower else 1.),[0.,0.,1.]))
        # Gentle central/lateral height rhythm, restricted to the exposed edge.
        trim=(.00007 if typ==0 else .00012 if typ==1 else 0.)
        edge_weight=1.-_smooth(depth/.0014)
        soften[:,2]+=trim*edge_weight*(-1. if lower else 1.)
        result[ids]=center+delta+soften
        matrices[ids]=matrix; offsets[ids]=center-center@matrix.T+soften
        records.append({'rank':rank,'rotationTowardArchDegrees':math.degrees(angle),
                        'widthFactor':float(widths[typ]),'cuttingCornerRetreatMm':amount*1000.,
                        'edgeTrimMm':trim*1000.,'centerUntranslated':True})
    return result,matrices,offsets,records


def _gaps(vertices, rows):
    return np.array([signed_separation(vertices[rows[i]['ids'],:2],vertices[rows[i+1]['ids'],:2]) for i in range(9)])


def propose(vertices, edges, polygons, lower=False):
    vertices=np.asarray(vertices,float);rows,before=analyze(vertices,edges,polygons)
    # Five symmetric width factors fitted to measured gaps, not uniform growth.
    # Bounds retain the central/lateral/canine size rhythm already in the source.
    widths=np.ones(5);target=np.array([.00040,.00023,.00013,.00013,.00010,.00013,.00013,.00023,.00040])
    for unused in range(8):
        current,_,_,_=_mapped(vertices,rows,widths,lower);gaps=_gaps(current,rows)
        jac=[]
        for j in range(5):
            trial=widths.copy();trial[j]+=.002
            q,_,_,_=_mapped(vertices,rows,trial,lower);jac.append((_gaps(q,rows)-gaps)/.002)
        jac=np.array(jac).T
        # Small regularization prevents unnecessary width changes when geometry
        # already has a narrow contact. All proposals are checked after solving.
        lhs=np.vstack((jac,np.eye(5)*.00018));rhs=np.r_[target-gaps,(1.-widths)*.00018]
        step=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
        updated=np.clip(widths+step*.75,.88,1.10)
        if np.max(np.abs(updated-widths))<1e-5: break
        widths=updated
    candidate,matrices,offsets,changes=_mapped(vertices,rows,widths,lower)
    _,after=analyze(candidate,edges,polygons)
    displacement=np.linalg.norm(candidate-vertices,axis=1)
    if displacement.max()>.00135: raise RuntimeError('Dental refinement exceeds 1.35 mm')
    if any(p['signedSectionSeparationMm']<-.015 for p in after):
        raise RuntimeError('Candidate leaves overlapping mid-crown sections: '+str(after))
    report={'connectedCrowns':10,'verticesPerCrown':96,
        'measurement':'Convex polygon sections from actual mesh edges at each adjacent pair shared mid-height; signed SAT separation. Negative means actual section overlap, not an open gap. Not a full 3D contact/occlusion certificate.',
        'crowns':[{'rank':r['rank'],'vertexIds':r['ids'].tolist(),'center':r['center'].tolist(),
                   'widthMm':r['widthMm'],'heightMm':r['heightMm'],
                   'frontWidthAxisDegrees':math.degrees(r['angle']),
                   'actualArchTangentDegrees':math.degrees(r['archAngle'])} for r in rows],
        'beforePairs':before,'afterPairs':after,'changes':changes,
        'maximumVertexDisplacementMm':float(displacement.max()*1000.),
        'newMaximumSectionGapMm':max(p['signedSectionSeparationMm'] for p in after),
        'widthFactorsCentralToPosterior':widths.tolist(),
        'scope':'Bounded partial orientation correction, individually fitted contact widths and 0.06–0.12 mm cutting-corner retreat; original crowns/centers/rig/materials remain.'}
    return candidate,matrices,offsets,report


def analyze_file():
    p=project_path();source=p/INPUT;data=json.loads(source.read_text(encoding='utf-8'))
    output={'sourceObservation':INPUT,'sourceSha256':data['sourceSha256'],'objects':{},'actualBlenderRun':False}
    for name,obj in data['objects'].items():
        _,_,_,report=propose(obj['vertices'],obj['edges'],obj['polygons'],name==NAMES[1])
        output['objects'][name]=report
    return output


def _map(points,matrices,offsets):
    return np.einsum('nij,nj->ni',matrices,np.asarray(points))+offsets


def transport_keys(original_basis,new_basis,arrays):
    """Transport the same crown edit through existing local jaw compensation.

    Keys are actual R09 affine maps with float32 residuals. Fit/review each map,
    transform only the new neutral displacement, and add it to the untouched old
    key. Never fit away old residuals or rebuild the drivers/shape-key datablock.
    """
    center=np.asarray(original_basis).mean(axis=0)
    design=np.c_[np.asarray(original_basis)-center,np.ones(len(original_basis))]
    delta=np.asarray(new_basis)-np.asarray(original_basis);targets={};reports=[]
    for name,values in arrays.items():
        values=np.asarray(values)
        mapping=np.linalg.lstsq(design,values-center,rcond=None)[0]
        error=float(np.linalg.norm(design@mapping+center-values,axis=1).max())
        if error>5e-7:
            raise RuntimeError('Existing jaw key is not the expected affine compensation: '+name+' / '+str(error))
        determinant=float(np.linalg.det(mapping[:3]))
        if not .8<determinant<1.2:
            raise RuntimeError('Unexpected existing jaw-map scale: '+name)
        targets[name]=values+delta@mapping[:3]
        reports.append({'key':name,'maximumExistingMapFitResidualMicrometres':error*1e6,
                        'linearMap':mapping[:3].tolist(),'determinant':determinant,
                        'originalFloatResidualPreserved':True})
    return targets,reports


def _signature(obj,util):
    return {'mesh':util['_data_hash'](obj.data),
            'keys':{k.name:hashlib.sha256(_coords(k.data).tobytes()).hexdigest()
                    for k in obj.data.shape_keys.key_blocks} if obj.data.shape_keys else {}}


def _protected(bpy,util,uv):
    rows=util['_protected'](bpy,uv)
    for name in ('R02 Tongue','R03 Upper gingiva','R03 Lower gingiva'):
        obj=bpy.data.objects[name]
        rows[name]={'geometry':_signature(obj,util),'world':uv['_plain'](obj.matrix_world),
                    'parent':uv['_plain'](obj.parent),'parentBone':obj.parent_bone}
    return rows


def run(source_relative,diagnostics_only=False):
    import bpy
    p=project_path();source=(p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Exact fresh saved independent current source required')
    if bpy.context.scene.get(MARKER): raise RuntimeError('Dental candidate already applied; restore or retain it before retrying')
    if not bpy.data.objects['R01 Head'].get('R09_lower_face_candidate'):
        raise RuntimeError('Current R09 mapped oral geometry required')
    observed=json.loads((p/INPUT).read_text(encoding='utf-8'))
    util=_api('posterior_groom_r09');uv=_api('production_uv_r08')
    source_sha=_sha(source);protected_before=_protected(bpy,util,uv)
    prepared=[];backup=None;committed=False;written=False
    try:
        for name in NAMES:
            obj=bpy.data.objects[name];reference=observed['objects'][name]
            if obj.type!='MESH' or obj.library or obj.data.library or obj.data.users!=1:
                raise RuntimeError('Local single-user original tooth mesh required: '+name)
            if getattr(obj.data,'has_custom_normals',False):
                raise RuntimeError('Tooth custom normals need a reviewed update path: '+name)
            raw=_coords(obj.data.vertices)
            edges=[tuple(e.vertices) for e in obj.data.edges]
            polygons=[tuple(face.vertices) for face in obj.data.polygons]
            if edges!=[tuple(v) for v in reference['edges']] or polygons!=[tuple(v) for v in reference['polygons']]:
                raise RuntimeError('Tooth connectivity differs from measured source: '+name)
            if np.max(np.abs(raw-np.asarray(reference['vertices'])))>2e-7:
                raise RuntimeError('Tooth coordinates changed since explicit observation: '+name)
            arrays={k.name:_coords(k.data) for k in obj.data.shape_keys.key_blocks} if obj.data.shape_keys else {}
            if set(arrays)!=set(reference['keys']):
                raise RuntimeError('Existing oral key identity changed: '+name)
            for key,a in arrays.items():
                if np.max(np.abs(a-np.asarray(reference['keys'][key])))>2e-7:
                    raise RuntimeError('Oral key changed since observation: '+name+' / '+key)
            candidate,matrices,offsets,report=propose(raw,edges,polygons,name==NAMES[1])
            if arrays:
                basis_name=obj.data.shape_keys.key_blocks[0].name
                if np.max(np.abs(arrays[basis_name]-raw))>2e-7:
                    raise RuntimeError('Expected matching tooth raw/Basis coordinates')
                if any(k.relative_key.name!=basis_name for k in obj.data.shape_keys.key_blocks):
                    raise RuntimeError('Unexpected non-Basis relative jaw compensation key')
                key_targets,transport=transport_keys(raw,candidate,arrays)
            else:key_targets={};transport=[]
            rows,_=analyze(raw,edges,polygons)
            probes=[]
            for key,target in key_targets.items():
                gap=_gaps(target,rows)
                if gap.min()<-.000015:
                    raise RuntimeError('Existing jaw-map key gains tooth overlap: '+key)
                movement=np.linalg.norm(target-arrays[key],axis=1)
                if movement.max()>.00135:
                    raise RuntimeError('Mapped dental key exceeds local displacement bound: '+key)
                probes.append({'key':key,'minimumProjectedSectionGapMm':float(gap.min()*1000.),
                               'maximumDentalEditMm':float(movement.max()*1000.)})
            attachment={'parent':uv['_plain'](obj.parent),'parentType':obj.parent_type,'parentBone':obj.parent_bone,
                'basis':uv['_plain'](obj.matrix_basis),'parentInverse':uv['_plain'](obj.matrix_parent_inverse),
                'objectAnimation':uv['_animation'](obj),'keyAnimation':uv['_animation'](obj.data.shape_keys) if arrays else None,
                'keyIdentities':[(k.name,k.relative_key.name) for k in obj.data.shape_keys.key_blocks] if arrays else []}
            prepared.append({'object':obj,'raw':raw,'keys':arrays,'target':candidate,'keyTargets':key_targets,
                             'attachment':attachment,'beforeSignature':_signature(obj,util),
                             'attributes':util['_attributes'](obj.data),
                             'report':dict(report,keyChecks=probes,jawEditTransport=transport)})
        result={'source':str(source.relative_to(p)),'sourceSha256':source_sha,
                'observation':INPUT,'observedSourceSha256':observed['sourceSha256'],
                'objects':{r['object'].name:r['report'] for r in prepared},
                'savedBlend':False,'writtenInMemory':not diagnostics_only,'appearanceAccepted':False,
                'materialGumRigChanges':False,'existingKeysAndDriversReused':True,
                'requiredEvidence':'One current EE63 and OU98 close-up pair: tooth contacts, visible size rhythm/corners, no gum gaps or rigid square dental grid. Numerical spacing does not prove natural appearance.'}
        if diagnostics_only:return result
        arrays={};meta=[]
        for i,r in enumerate(prepared):
            arrays[f'{i}_raw']=r['raw'];keys=list(r['keys'])
            for k,key in enumerate(keys):arrays[f'{i}_key_{k}']=r['keys'][key]
            meta.append({'name':r['object'].name,'keys':keys})
        memory=io.BytesIO();np.savez_compressed(memory,**arrays)
        backup=bpy.data.texts.new('R09 dental numeric rollback');backup.use_fake_user=True
        backup.write(json.dumps({'objects':meta,'arrays':base64.b64encode(memory.getvalue()).decode('ascii')}))
        written=True
        for r in prepared:
            obj=r['object'];obj.data.vertices.foreach_set('co',r['target'].astype(np.float32).ravel())
            if r['keys']:
                for key in obj.data.shape_keys.key_blocks:
                    key.data.foreach_set('co',r['keyTargets'][key.name].astype(np.float32).ravel())
            obj.data.update();obj.data.update_tag();obj.update_tag()
        bpy.context.view_layer.update()
        for r in prepared:
            obj=r['object'];a=r['attachment'];keys=obj.data.shape_keys
            current={'parent':uv['_plain'](obj.parent),'parentType':obj.parent_type,'parentBone':obj.parent_bone,
                'basis':uv['_plain'](obj.matrix_basis),'parentInverse':uv['_plain'](obj.matrix_parent_inverse),
                'objectAnimation':uv['_animation'](obj),'keyAnimation':uv['_animation'](keys) if keys else None,
                'keyIdentities':[(k.name,k.relative_key.name) for k in keys.key_blocks] if keys else []}
            if current!=a:raise RuntimeError('Tooth attachment/key/driver identity changed: '+obj.name)
            attributes=util['_attributes'](obj.data)
            for name,old in r['attributes'].items():
                if name=='position':continue
                if name not in attributes or not np.array_equal(old['values'],attributes[name]['values']):
                    raise RuntimeError('Non-position tooth attribute changed: '+name)
        if _protected(bpy,util,uv)!=protected_before:raise RuntimeError('Protected gums/tongue/head/hair/rig changed')
        if _sha(source)!=source_sha:raise RuntimeError('Saved source changed')
        result['backupText']=backup.name
        result['candidateSignatures']={r['object'].name:_signature(r['object'],util) for r in prepared}
        result['protectedDataUnchanged']=True
        bpy.context.scene[MARKER]=json.dumps(result);committed=True
        return result
    finally:
        if not committed:
            for r in prepared if written else []:
                obj=r['object'];obj.data.vertices.foreach_set('co',r['raw'].ravel())
                if r['keys']:
                    for key in obj.data.shape_keys.key_blocks:key.data.foreach_set('co',r['keys'][key.name].ravel())
                obj.data.update()
            if backup is not None: bpy.data.texts.remove(backup)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


def restore():
    import bpy
    if not bpy.context.scene.get(MARKER):return False
    report=json.loads(bpy.context.scene[MARKER]);util=_api('posterior_groom_r09')
    for name,signature in report['candidateSignatures'].items():
        if _signature(bpy.data.objects[name],util)!=signature:
            raise RuntimeError('Dental data changed after this candidate; refuse blind restore: '+name)
    text=bpy.data.texts.get(report['backupText'])
    if text is None:raise RuntimeError('Retained dental rollback arrays missing')
    meta=json.loads(text.as_string());data=np.load(io.BytesIO(base64.b64decode(meta['arrays'])))
    for i,row in enumerate(meta['objects']):
        obj=bpy.data.objects[row['name']];obj.data.vertices.foreach_set('co',data[f'{i}_raw'].ravel())
        for k,name in enumerate(row['keys']):obj.data.shape_keys.key_blocks[name].data.foreach_set('co',data[f'{i}_key_{k}'].ravel())
        obj.data.update();obj.data.update_tag()
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'restored':list(NAMES),'retainedBackup':text.name,'savedBlend':False}


if __name__=='__main__':
    print('Loaded only. Call run(source_relative=...) explicitly; no automatic geometry write/save/render.')

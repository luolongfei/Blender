"""One reversible natural warm skin / light-makeup study for the new R11 head.

Composable API: apply(head_name='R11 Female head study') returns a report and
does not save or render. It reads the CURRENT new-head maps and preserves all
geometry, the source maps/material, UVs, optics and rest-space microdetail.
prepare(source, output) is the optional fresh-file wrapper. Never use on master.
No old-head indices, fixed ethnic swatch, new noise, or lighting adjustment.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
MARKER='R11_natural_adult_skin_candidate'
COLOR='R11 natural adult skin colour'
ROUGH='R11 natural adult skin roughness'
COVERAGE='R11 light makeup coverage'
LIP='R11 anatomical lip colour support'
NEW_ATTRIBUTES=(COLOR,ROUGH,COVERAGE,LIP)
MATERIAL='R11 natural warm skin and light makeup study'
DONOR_SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
PARAMETERS={
    'foundationRelativeLinearGain':[.055,.075,.085],
    'cheekRelativeBloodColour':[.018,-.003,-.008],
    'noseRelativeBloodColour':[.014,-.002,-.005],
    'earRelativeBloodColour':[.028,-.004,-.010],
    'lowerEyeRelativeWarmth':[.006,0.,-.004],
    'lipRelativeColourGain':[.035,0.,.040],
    'powderRoughnessIncrement':.008,'tZoneRoughnessReduction':.014,
    'lipRoughnessReduction':.014,
}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _array(items,field,width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype=dtype);items.foreach_get(field,a.ravel());return a


def _ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def _path(relative,suffix):
    P=project_path().resolve();p=(P/relative).resolve();p.relative_to(P/'work/reconstruction-r02')
    if p.suffix!=suffix:raise ValueError('Explicit independent reconstruction artifact required')
    return p


def _donor():
    P=project_path();folder=P/'work/reconstruction-r02'
    meta_path=folder/'female-basemesh-actual-r11.json';payload=folder/'female-basemesh-actual-r11.npz'
    landmark_path=folder/'female-face-landmarks-r11.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'));land=json.loads(landmark_path.read_text(encoding='utf-8'))
    if meta['source_sha256']!=DONOR_SHA or _sha(payload)!=meta['npz_sha256']:
        raise RuntimeError('Actual female donor array identity changed')
    if land['sourceArraysSha256']!=meta['npz_sha256'] or land['originalVertexCount']!=10582:
        raise RuntimeError('Female landmark index space differs')
    with np.load(payload) as data:
        arrays={key.removeprefix('object_00_'):data[key].copy() for key in data.files if key.startswith('object_00_')}
    return arrays,land,{'sourceArraysSha256':meta['npz_sha256'],'landmarksSha256':_sha(landmark_path),
                        'donorSha256':DONOR_SHA,'coordinateScope':'Original female body-space topology, never the replaced R01 face coordinates.'}


def _vertices_of_groups(data,wanted):
    mask=np.zeros(len(data['local']),bool)
    for start,count,group in zip(data['face_starts'],data['face_lengths'],data['face_sets']):
        if int(group) in wanted:mask[data['corner_vertices'][start:start+count]]=True
    return mask


def _graph(points,edges,allowed):
    graph=[[] for _ in points]
    for a,b in edges:
        a=int(a);b=int(b)
        if allowed[a] and allowed[b]:
            length=float(np.linalg.norm(points[a]-points[b]));graph[a].append((b,length));graph[b].append((a,length))
    return graph


def _distance(graph,seeds,limit):
    distance=np.full(len(graph),np.inf);queue=[]
    for i in seeds:distance[int(i)]=0.;heapq.heappush(queue,(0.,int(i)))
    while queue:
        d,i=heapq.heappop(queue)
        if d!=distance[i] or d>limit:continue
        for j,length in graph[i]:
            nd=d+length
            if nd<distance[j] and nd<=limit:distance[j]=nd;heapq.heappush(queue,(nd,j))
    return distance


def _diffuse(field,graph,steps=3):
    result=np.asarray(field,float).copy()
    for _ in range(steps):
        other=result.copy()
        for i,neighbors in enumerate(graph):
            if neighbors:other[i]=.5*result[i]+.5*np.mean([result[j] for j,_ in neighbors])
        result=other
    return result


def build_maps(donor,landmarks,source_colour,source_roughness,skin_vertices):
    """Pure arrays; smooth region fields follow current-source mapped topology."""
    points=np.asarray(donor['local'],float);old=np.asarray(source_colour,np.float32)
    oldrough=np.asarray(source_roughness,np.float32).reshape(-1);skin=np.asarray(skin_vertices,bool)
    if old.shape!=(10582,4) or oldrough.shape!=(10582,) or skin.shape!=(10582,):
        raise RuntimeError('Complete actual 10582-point female map required')
    x,y,z=points.T
    facial=_vertices_of_groups(donor,{17,22});ear=_vertices_of_groups(donor,{4,5})
    sockets=_vertices_of_groups(donor,{2,3});oral=_vertices_of_groups(donor,{7,8})
    external=skin&(facial|ear|_vertices_of_groups(donor,{1}))
    graph=_graph(points,donor['edges'],external)
    mouth=landmarks['mouth'];lip_ids=np.asarray(mouth['vermilionStripVertexIndices'],np.int32)
    lip_field=np.zeros(len(points));lip_field[np.asarray(mouth['vermilionMidLoop']['sourceVertexIndices'])]=1.
    lip_field[np.asarray(mouth['openingRimCandidate']['sourceVertexIndices'])]=.8
    # Outer vermilion ring fades to ordinary skin; no dark or red painted rim.
    lip_field[np.asarray(mouth['outerVermilionLoopCandidate']['sourceVertexIndices'])]=0.
    if len(lip_ids)!=120 or len(np.flatnonzero(lip_field))!=80:raise RuntimeError('Actual two-strip lip topology changed')
    mouth_center=points[np.asarray(mouth['openingRimCandidate']['sourceVertexIndices'])].mean(0)
    eye_centers=np.asarray([landmarks['eyes'][s]['boundsEllipsoidCenter'] for s in ('L','R')])
    eye=eye_centers.mean(0);ipd=abs(eye_centers[0,0]-eye_centers[1,0]);height=eye[2]-mouth_center[2]
    if not .055<ipd<.075 or not .045<height<.085:raise RuntimeError('Observed adult facial landmarks changed scale')
    front=facial&(y<eye[1]+.025)
    def seed_at(tx,tz,eligible):
        ids=np.flatnonzero(eligible)
        if not len(ids):raise RuntimeError('Actual anatomical seed domain is empty')
        score=((x[ids]-tx)/(.2*ipd))**2+((z[ids]-tz)/(.2*ipd))**2
        return int(ids[np.argmin(score)])
    cheek_seeds=[seed_at(sign*.57*ipd,eye[2]-.42*height,front&(z<eye[2])&(z>mouth_center[2]+.2*height)) for sign in (-1,1)]
    nasal=front&(abs(x)<.24*ipd)&(z>mouth_center[2]+.30*height)&(z<eye[2]-.15*height)
    nasal_ids=np.flatnonzero(nasal)
    nose_seed=int(nasal_ids[np.argmin(y[nasal_ids])]) if len(nasal_ids) else None
    if nose_seed is None:raise RuntimeError('Actual nose-tip domain missing')
    forehead_seed=seed_at(0.,eye[2]+.70*height,front&(z>eye[2]+.30*height))
    cheek=np.exp(-(_distance(graph,cheek_seeds,.08)/(.36*ipd))**2)*front
    nose=np.exp(-(_distance(graph,[nose_seed],.05)/(.20*ipd))**2)*front
    forehead=np.exp(-(_distance(graph,[forehead_seed],.07)/(.33*ipd))**2)*front
    lower_eye_ids=[i for side in ('L','R') for i in landmarks['eyes'][side]['orderedLowerInnerToOuter']]
    eye_warm=np.exp(-(_distance(graph,lower_eye_ids,.035)/(.13*ipd))**2)*front*_smooth((eye[2]-z)/(.15*height))
    ear_field=_diffuse(ear.astype(float),graph,3)*external
    chin_floor=float(points[_vertices_of_groups(donor,{22}),2].min())
    neck_fade=_smooth((z-(chin_floor-.080))/.080)
    coverage=neck_fade*external*(1.-.25*ear_field)*(1.-lip_field)
    # Interior eye socket and oral pocket receive no makeup. Shared external
    # borders retain smooth field interpolation, not a new hard mask edge.
    exclusive_internal=(sockets|oral)&(~facial)&(~ear)
    coverage[exclusive_internal]=0.
    for field in (cheek,nose,forehead,eye_warm,ear_field):field*=1.-lip_field
    base=np.median(old[external&(z>chin_floor)&(lip_field==0),:3],axis=0).astype(float)
    rgb=old[:,:3].astype(float).copy()
    rgb*=1.+coverage[:,None]*np.asarray(PARAMETERS['foundationRelativeLinearGain'])
    for field,key in ((cheek,'cheekRelativeBloodColour'),(nose,'noseRelativeBloodColour'),
                      (ear_field,'earRelativeBloodColour'),(eye_warm,'lowerEyeRelativeWarmth')):
        rgb*=1.+field[:,None]*np.asarray(PARAMETERS[key])
    rgb*=1.+lip_field[:,None]*np.asarray(PARAMETERS['lipRelativeColourGain'])
    result=old.copy();result[:,:3]=rgb.astype(np.float32)
    rough=oldrough.astype(float).copy();base_rough=float(np.median(oldrough[external&(lip_field==0)&(~np.isin(np.arange(len(points)),lip_ids))]))
    # Remove the source's abrupt constant-roughness edge on the outer lip ring.
    # This is only the derived map; the entire original source map is retained.
    rough[lip_ids]=base_rough+(oldrough[lip_ids]-base_rough)*lip_field[lip_ids]
    rough+=PARAMETERS['powderRoughnessIncrement']*coverage
    rough-=PARAMETERS['tZoneRoughnessReduction']*np.maximum(nose,.6*forehead)
    rough-=PARAMETERS['lipRoughnessReduction']*lip_field
    affected=(coverage+cheek+nose+ear_field+eye_warm+lip_field)>0
    if not np.isfinite(result).all() or result[:,:3].min()<0 or result[:,:3].max()>1 or rough.min()<.25 or rough.max()>.70:
        raise RuntimeError('Candidate maps exceed bounded natural surface range')
    if not np.array_equal(result[:,3],old[:,3]) or not np.array_equal(result[~skin],old[~skin]):
        raise RuntimeError('Alpha or non-skin colour changed')
    if not np.array_equal(rough[~skin].astype(np.float32),oldrough[~skin]):raise RuntimeError('Non-skin roughness changed')
    delta=result[:,:3]-old[:,:3]
    if np.max(abs(delta))>.065:raise RuntimeError('Colour candidate exceeds the bounded relative change')
    plan={'parameters':PARAMETERS,'actualSourceBaselineLinearRGB':base.tolist(),
        'actualSourceBaselineSRGB':np.where(base<=.0031308,12.92*base,1.055*base**(1/2.4)-.055).tolist(),
        'faceSeeds':{'cheek':cheek_seeds,'noseTip':nose_seed,'forehead':forehead_seed},
        'seedCoordinatesBodySpace':{str(i):points[i].tolist() for i in cheek_seeds+[nose_seed,forehead_seed]},
        'ipdBodySpaceMeters':float(ipd),'eyeToMouthBodySpaceMeters':float(height),
        'neckFadeBodySpaceZ':[chin_floor-.080,chin_floor],
        'changedColourVertices':int(np.count_nonzero(np.any(delta!=0,axis=1))),
        'skinVertices':int(skin.sum()),'lipSupportVertices':len(lip_ids),'lipMaximumSupportVertices':int((lip_field==1).sum()),
        'colourDeltaMin':delta.min(0).tolist(),'colourDeltaMax':delta.max(0).tolist(),
        'sourceRoughnessRange':[float(oldrough.min()),float(oldrough.max())],
        'candidateRoughnessRange':[float(rough.min()),float(rough.max())],
        'topologyDomainNotes':'Female face sets17/22, ears4/5, socket interfaces2/3, oral7/8; actual two-quad-strip vermilion. Region distances follow connected surface edges.',
        'makeupInterpretation':'Subtle relative art-direction colour/roughness candidate, not a measured ethnicity swatch or a physical spectral foundation model.',
        'lipInterpretation':'Enhance the actual lip body with a soft outer fade; do not draw a mouth line, expand pigment onto perioral skin, or change lip shape.',
        'unaffectedMapRowsExact':bool(np.array_equal(result[~affected],old[~affected]))}
    return {COLOR:result,ROUGH:rough.astype(np.float32),COVERAGE:coverage.astype(np.float32),LIP:lip_field.astype(np.float32)},plan


def _head_state(head,slot,uv,face):
    mesh=head.data;attributes=[];attr_api=_api('runtime_torso_cull_r07')
    for attr in mesh.attributes:
        if attr.name in NEW_ATTRIBUTES or attr.name in {'position','.edge_verts','.corner_vert','.corner_edge'}:continue
        attributes.append((attr.name,attr.domain,attr.data_type,_ah(attr_api['_attribute'](attr))))
    keys=mesh.shape_keys
    return {'coordinates':_ah(_array(mesh.vertices,'co')),'edges':_ah(_array(mesh.edges,'vertices',2,np.int32)),
        'loops':_ah(_array(mesh.loops,'vertex_index',1,np.int32)),
        'starts':_ah(_array(mesh.polygons,'loop_start',1,np.int32)),'sizes':_ah(_array(mesh.polygons,'loop_total',1,np.int32)),
        'polygonMaterials':_ah(_array(mesh.polygons,'material_index',1,np.int32)),
        'smooth':_ah(_array(mesh.polygons,'use_smooth',1,np.bool_)),'seams':_ah(_array(mesh.edges,'use_seam',1,np.bool_)),
        'uv':[(x.name,_ah(_array(x.data,'uv',2))) for x in mesh.uv_layers],'attributes':attributes,
        'keys':None if keys is None else {'animation':uv['_animation'](keys),'custom':uv['_custom'](keys),
            'blocks':[(k.name,_ah(_array(k.data,'co')),uv['_scalar_rna'](k),k.relative_key.name) for k in keys.key_blocks]},
        'weights':{'groups':[(g.name,g.index,g.lock_weight) for g in head.vertex_groups],
            'values':[[(g.group,float(g.weight)) for g in v.groups] for v in mesh.vertices]},
        'materials':[('candidateSkinSlot',slot) if i==slot else (s.link,s.material.name,face['_material_contract'](s.material)) for i,s in enumerate(head.material_slots)],
        'object':uv['_scalar_rna'](head),'custom':uv['_custom'](head),'meshCustom':uv['_custom'](mesh),
        'matrices':[uv['_plain'](m) for m in (head.matrix_world,head.matrix_basis,head.matrix_parent_inverse)],
        'parent':uv['_plain'](head.parent),'animation':uv['_animation'](head),'meshAnimation':uv['_animation'](mesh),
        'modifiers':[uv['_scalar_rna'](m) for m in head.modifiers],
        'uvSettings':uv['_uv_settings'](mesh)}


def _differences(before,after,path=''):
    if isinstance(before,dict) and isinstance(after,dict):
        result=[]
        for key in sorted(set(before)|set(after)):
            if key not in before or key not in after:result.append({'path':path+'/'+key,'before':before.get(key),'after':after.get(key)})
            else:result.extend(_differences(before[key],after[key],path+'/'+key))
            if len(result)>=16:break
        return result[:16]
    if before==after:return []
    return [{'path':path,'before':str(before)[:800],'after':str(after)[:800]}]


def _scene_contract(bpy,head,slot,uv,groom):
    """Stream original geometry/key coordinates; never evaluate caches or render."""
    rows=[]
    for obj in bpy.context.scene.objects:
        row={'name':obj.name,'pointer':obj.as_pointer(),'type':obj.type,'settings':uv['_scalar_rna'](obj),
             'matrix':uv['_plain'](obj.matrix_world),'animation':uv['_animation'](obj),
             'materials':[(i,s.link,s.material.as_pointer() if s.material else None) for i,s in enumerate(obj.material_slots) if not(obj==head and i==slot)]}
        data=getattr(obj,'data',None)
        if data is not None:
            row['dataPointer']=data.as_pointer()
            if obj!=head and obj.type=='MESH':
                h=hashlib.sha256();h.update(_array(data.vertices,'co').tobytes());h.update(_array(data.loops,'vertex_index',1,np.int32).tobytes())
                keys=data.shape_keys
                if keys:
                    for key in keys.key_blocks:h.update(key.name.encode());h.update(_array(key.data,'co').tobytes())
                    row['keyAnimation']=uv['_animation'](keys)
                row['meshAndKeyCoordinates']=h.hexdigest()
            elif obj!=head and obj.type=='CURVES':row['curveData']=groom['_data_hash'](data)
        rows.append(row)
    return _hash({'objects':rows,'pose':uv['_pose_contract']()})


def apply(head_name=HEAD):
    import bpy
    current=Path(bpy.data.filepath).resolve();current.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Explicit independent background candidate required')
    head=bpy.data.objects[head_name];mesh=head.data
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582:
        raise RuntimeError('Local single-user complete R11 female head required')
    if head.get('R11_original_donor_sha256')!=DONOR_SHA:raise RuntimeError('Explicit new female-head provenance required')
    if bpy.context.scene.get(MARKER) or bpy.data.materials.get(MATERIAL) or any(mesh.attributes.get(n) for n in NEW_ATTRIBUTES):
        raise RuntimeError('Preserve or restore the earlier skin candidate first')
    donor,landmarks,provenance=_donor()
    for field,actual,expected in (
        ('corners',_array(mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']),
        ('face starts',_array(mesh.polygons,'loop_start',1,np.int32).ravel(),donor['face_starts']),
        ('face sizes',_array(mesh.polygons,'loop_total',1,np.int32).ravel(),donor['face_lengths'])):
        if not np.array_equal(actual,expected):raise RuntimeError('New-head donor topology mismatch: '+field)
    groups=mesh.attributes.get('.sculpt_face_set')
    if not groups or groups.domain!='FACE' or groups.data_type!='INT' or not np.array_equal(_array(groups.data,'value',1,np.int32).ravel(),donor['face_sets']):
        raise RuntimeError('Actual new-head anatomical face sets changed')
    slots=[i for i,s in enumerate(head.material_slots) if s.material and s.material.name=='R11 Female head neutral skin study']
    if len(slots)!=1:raise RuntimeError('Expected explicit original R11 neutral skin material')
    slot=slots[0];original=head.material_slots[slot].material
    color=mesh.color_attributes.get('R01_region_color');rough=mesh.attributes.get('R04 skin roughness')
    if not color or color.domain!='POINT' or color.data_type!='FLOAT_COLOR' or not rough or rough.domain!='POINT' or rough.data_type!='FLOAT':
        raise RuntimeError('Actual new-head colour/roughness mapping differs')
    skin=np.zeros(len(mesh.vertices),bool)
    for poly in mesh.polygons:
        if poly.material_index==slot:skin[list(poly.vertices)]=True
    maps,plan=build_maps(donor,landmarks,_array(color.data,'color',4),_array(rough.data,'value',1).ravel(),skin)
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08');groom=_api('posterior_groom_r09');old=_api('natural_foundation_candidate_r10')
    nodes=old['_source_nodes'](original);graph_hash=old['_normalized_graph'](original,face,nodes)
    original_hash=face['_material_contract'](original);source_sha=_sha(current);settings=uv['_uv_settings'](mesh)
    bpy.context.view_layer.update();head_state=_head_state(head,slot,uv,face);head_before=_hash(head_state);scene_before=_scene_contract(bpy,head,slot,uv,groom)
    old_fake=original.use_fake_user;candidate=None;created=[];success=False
    report={'source':str(current.relative_to(project_path())),'sourceSha256':source_sha,'head':head_name,'materialSlot':slot,
        'sourceMayContainEarlierInMemoryModules':bool(bpy.data.is_dirty),'helperSha256':_sha(Path(__file__)),
        'provenance':provenance,'plan':plan,'originalMaterial':original.name,'originalMaterialSha256':original_hash,
        'candidateMaterial':MATERIAL,'appearanceAccepted':False,'savedBlend':False,'writtenInMemory':False,
        'scope':'New anatomical colour/roughness attributes and a copied material only. All optics, rest-space microdetail, alpha, lighting, geometry, UV, keys, weights, actions and existing source maps are retained.',
        'requiredReview':'Actual same-light structure-front/quarter and groom-quarter with the new brows and current neck fit. Check subtle warmth, face-ear-neck blend, lip volume and absence of cheek patches. No skin or ethnic-appearance acceptance is automatic.'}
    try:
        for name,values in maps.items():
            attr=mesh.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT') if name==COLOR else mesh.attributes.new(name=name,type='FLOAT',domain='POINT')
            created.append(name);attr.data.foreach_set('color' if name==COLOR else 'value',np.asarray(values,np.float32).ravel())
        uv['_restore_uv_settings'](mesh,settings)
        candidate=original.copy();candidate.name=MATERIAL
        candidate.node_tree.nodes[nodes[0][0]].layer_name=COLOR
        candidate.node_tree.nodes[nodes[1][0]].attribute_name=ROUGH
        if old['_normalized_graph'](candidate,face,nodes)!=graph_hash:raise RuntimeError('Skin graph changed beyond explicit colour and roughness attribute references')
        candidate['R11_source_material']=original.name;candidate['R11_source_material_sha256']=original_hash
        candidate['R11_skin_status']='Single unaccepted natural-light-makeup candidate; actual same-light review required'
        original.use_fake_user=True;head.material_slots[slot].material=candidate;bpy.context.view_layer.update()
        head_after=_head_state(head,slot,uv,face)
        if _hash(head_after)!=head_before:
            raise RuntimeError('Original new-head contract changed: '+json.dumps(_differences(head_state,head_after),ensure_ascii=False))
        if _scene_contract(bpy,head,slot,uv,groom)!=scene_before:raise RuntimeError('Other geometry/key coordinates, rig pose/actions or material assignments changed')
        if face['_material_contract'](original)!=original_hash or _sha(current)!=source_sha:raise RuntimeError('Original material or source file changed')
        report.update(writtenInMemory=True,originalFakeUser=old_fake,originalColourSettings=settings,headContractSha256=head_before,
            candidateMaterialSha256=face['_material_contract'](candidate),originalGeometryAndSourceMapsRetained=True,
            candidateAttributeHashes={name:_ah(values) for name,values in maps.items()})
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            head.material_slots[slot].material=original
            for name in reversed(created):
                attr=mesh.attributes.get(name)
                if attr:mesh.attributes.remove(attr)
            uv['_restore_uv_settings'](mesh,settings);original.use_fake_user=old_fake
            if candidate is not None and candidate.users==0:bpy.data.materials.remove(candidate)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];mesh=head.data
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08');original=bpy.data.materials[report['originalMaterial']]
    candidate=bpy.data.materials[report['candidateMaterial']];slot=report['materialSlot']
    if face['_material_contract'](original)!=report['originalMaterialSha256'] or face['_material_contract'](candidate)!=report['candidateMaterialSha256']:
        raise RuntimeError('Skin material changed after this candidate; preserve edits before rollback')
    if head.material_slots[slot].material!=candidate:raise RuntimeError('Current head skin assignment changed')
    for name,digest in report['candidateAttributeHashes'].items():
        attr=mesh.attributes.get(name)
        if attr is None:raise RuntimeError('Candidate map missing: '+name)
        values=_array(attr.data,'color',4) if name==COLOR else _array(attr.data,'value',1).ravel()
        if _ah(values)!=digest:raise RuntimeError('Candidate map changed: '+name)
    head.material_slots[slot].material=original
    for name in reversed(NEW_ATTRIBUTES):mesh.attributes.remove(mesh.attributes[name])
    uv['_restore_uv_settings'](mesh,report['originalColourSettings']);original.use_fake_user=report['originalFakeUser']
    candidate.use_fake_user=True;del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'restoredMaterial':original.name,'retainedCandidateMaterial':candidate.name,'savedBlend':False,
            'geometryAndOtherModulesUntouched':True}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Preserve prior candidates; use new output paths')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit source required by prepare wrapper')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NATURAL_SKIN_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source,args.output)

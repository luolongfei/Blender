"""Minimal read-only current L3 lower-lip/chin observation at neutral/warm1.

No donor Blender loading, render, source save or shape installation. Captures
the actual front lower-face ROI, current mixed cage, lip-support attribute,
oral surfaces and true open-head boundary for one connected contour candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';WARM='R11 warm closed smile';LIP='R11 anatomical lip colour support'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ah(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def array(items,field,width,dtype=np.float32):
    result=np.empty((len(items),width),dtype);items.foreach_get(field,result.ravel());return result


def capture(head,prefix,arrays,land):
    import bpy
    uv=api('production_uv_r08');mix,mix_meta=uv['current_mix'](head);mix=np.asarray(mix,np.float32)
    dg=bpy.context.evaluated_depsgraph_get();obj=head.evaluated_get(dg);mesh=obj.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        T=np.asarray(obj.matrix_world,float);world=array(mesh.vertices,'co',3)@T[:3,:3].T+T[:3,3]
        cage_world=mix@T[:3,:3].T+T[:3,3];chin=cage_world[7972];mouth=cage_world[8005]
        sizes=array(mesh.polygons,'loop_total',1,np.int32).ravel()
        if not np.all(sizes==4):raise RuntimeError('Actual fixed L3 quad head expected')
        faces=array(mesh.loops,'vertex_index',1,np.int32).ravel().reshape(-1,4)
        group=mesh.attributes.get('.sculpt_face_set');lip=mesh.attributes.get(LIP)
        if group is None or group.domain!='FACE' or lip is None or lip.domain!='POINT' or lip.data_type!='FLOAT':
            raise RuntimeError('Current actual sculpt regions and lip-support attribute required')
        groups=array(group.data,'value',1,np.int32).ravel();lip_support=array(lip.data,'value',1).ravel()
        roi=(abs(world[:,0]-chin[0])<.055)&(world[:,1]<-.045)&(world[:,2]>chin[2]-.010)&(world[:,2]<mouth[2]+.010)
        face_ids=np.flatnonzero(roi[faces].any(1));ids=np.unique(faces[face_ids]);mapping=np.full(len(world),-1,np.int32);mapping[ids]=np.arange(len(ids))
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[count==1])
        oral=np.unique(faces[np.isin(groups,[7,8])]);eyes=np.unique(faces[np.isin(groups,[2,3])])
        arrays.update({prefix+'_world':world[ids],prefix+'_originalVertexIds':ids,prefix+'_faces':mapping[faces[face_ids]],
            prefix+'_originalFaceIds':face_ids,prefix+'_groups':groups[face_ids],prefix+'_lipSupport':lip_support[ids],
            prefix+'_currentMixedCageLocal':mix,prefix+'_currentMixedCageWorld':cage_world,
            prefix+'_oralOriginalVertexIds':oral,prefix+'_oralWorld':world[oral],
            prefix+'_openBoundaryOriginalVertexIds':boundary,prefix+'_openBoundaryWorld':world[boundary]})
        lip_ids=np.asarray(land['mouth']['vermilionStripVertexIndices'],np.int32)
        opening=np.asarray(land['mouth']['openingRimCandidate']['sourceVertexIndices'],np.int32)
        arrays[prefix+'_vermilionCageIds']=lip_ids;arrays[prefix+'_vermilionCageWorld']=cage_world[lip_ids]
        arrays[prefix+'_openingCageIds']=opening;arrays[prefix+'_openingCageWorld']=cage_world[opening]
        return {'prefix':prefix,'fullEvaluatedVertices':len(world),'fullEvaluatedFaces':len(faces),'roiVertexCount':len(ids),'roiFaceCount':len(face_ids),
            'headMatrixWorld':T.tolist(),'chinAnchorWorld':chin.tolist(),'mouthAnchorWorld':mouth.tolist(),
            'roiBoundsWorld':[world[ids].min(0).tolist(),world[ids].max(0).tolist()],
            'fullTopologySha256':ah(faces),'actualOralSurfaceSha256':ah(world[oral]),'actualEyeSurfaceSha256':ah(world[eyes]),
            'actualOpenBoundarySha256':ah(world[boundary]),'currentMixMetadata':mix_meta,
            'lipAttribute':LIP,'lipAttributeMeaning':'Interpolated colour-support field on the actual surface. Zero-valued outer vermilion boundaries remain part of the lip body; this attribute alone is not the complete protection domain.'}
    finally:obj.to_mesh_clear()


def observe(source_relative,output_relative):
    import bpy
    root=project_path();source=(root/source_relative).resolve();output=(root/output_relative).resolve();npz=output.with_suffix('.npz')
    for p in (source,output):p.relative_to((root/'work/reconstruction-r02').resolve())
    if source.suffix!='.blend' or output.suffix!='.json' or output.exists() or npz.exists() or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved source and fresh independent observation output required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if WARM not in keys or any(abs(k.value)>1e-7 for k in keys if 'blink' in k.name.lower() or 'smile' in k.name.lower()):
        raise RuntimeError('Current neutral source with the installed warm smile required')
    uv=api('production_uv_r08');natural=api('natural_smile_adult_r11');qa=api('face_contact_audit_r11')
    before=natural['_key_records'](head.data,uv);pose=uv['_pose_contract']();source_sha=sha(source)
    _,land,_=api('natural_skin_adult_r11')['_donor']();arrays={};states=[]
    try:
        with qa['rest_evaluation']():
            for index,strength in enumerate((0.,1.)):
                keys[WARM].value=strength;bpy.context.view_layer.update()
                print('R11_LOWER_FACE_OBSERVE_STATE '+str(strength),flush=True)
                state=capture(head,'state_'+str(index),arrays,land);state['warmSmile']=strength;states.append(state)
    finally:
        keys[WARM].value=0.;bpy.context.view_layer.update()
    if natural['_key_records'](head.data,uv)!=before or uv['_pose_contract']()!=pose or sha(source)!=source_sha:
        raise RuntimeError('Read-only observation changed original expression/pose/source')
    np.savez_compressed(npz,**arrays)
    report={'source':str(source.relative_to(root)),'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'arrays':str(npz.relative_to(root)),'arraysSha256':sha(npz),'states':states,
        'activeReliefValue':float(head['R11 upper lid relief']) if 'R11 upper lid relief' in head else None,
        'sourceSaved':False,'sourceUnchanged':True,'appearanceAccepted':False,
        'scope':'Actual current front lower-face surface and lip/inner-mouth boundaries for one lower-lip-to-chin continuity candidate. No whole-face replacement or geometric edit.'}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_LOWER_FACE_OBSERVED '+json.dumps({'output':str(output.relative_to(root)),'arraysSha256':report['arraysSha256']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);observe(a.source,a.output)

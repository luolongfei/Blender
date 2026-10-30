"""Local eye-depth and nasal-profile refinement of the saved E4 source.

Pure delta application: never reconstructs or replaces the fitted scalp, UVs,
materials, topology, hair, body, hands or garment. No file is saved by apply().
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import struct

EXPECTED_SOURCE_SHA256='7a406fbbcf337e4098ebc98697766cdb3d88d8966286779c92a5e58e3e6928aa'
EXPECTED_FACE_MODULE_SHA256='e04464ee152bb445b6c3bc8a06921a49c3cc209c6fd5ca65bc58822bbafe81de'
EXPECTED_SPEC_SHA256='c833d7f4cd2f714034c99526b4e0be9bb7d7fdb4dbd00e7a92421aef64c243d2'
TARGET_NAMES={'Head_Skin_Quad_Retopo','Eye_Sclera_L','Eye_Iris_L','Eye_UpperLash_L',
              'Eye_Sclera_R','Eye_Iris_R','Eye_UpperLash_R','Mouth_Closed_Line','Brow_L','Brow_R'}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positions(obj,indices=None):
    h=hashlib.sha256()
    for i in (range(len(obj.data.vertices)) if indices is None else indices):
        h.update(struct.pack('<I3f',i,*obj.data.vertices[i].co))
    return h.hexdigest()


def _uv_digest(obj):
    h=hashlib.sha256()
    for layer in obj.data.uv_layers:
        h.update(layer.name.encode())
        for loop in layer.data:h.update(struct.pack('<2f',*loop.uv))
    return h.hexdigest()


def _topology(obj):
    h=hashlib.sha256()
    for p in obj.data.polygons:
        h.update(struct.pack('<I',len(p.vertices)))
        for i in p.vertices:h.update(struct.pack('<I',i))
    return h.hexdigest()


def _smoothstep(a,b,v):
    t=max(0,min(1,(v-a)/(b-a)))
    return t*t*(3-2*t)


def _orbital_alignment(surface,eyes,x,z):
    """Rotate the complete socket's local slope toward neutral forward gaze.

    The existing face-wrapped iris inherits about 27 degrees of lateral yaw.
    Correct 80% of that slope across the eye. A broad vertical transition avoids
    the eye-bag ridge caused by confining this rotation to the small aperture.
    Skin, white, iris, lashes and brows consume the same depth field.
    """
    delta=0.
    for eye in eyes:
        dx=(x-eye['x'])/eye['half_width']
        if abs(dx)>=2. or z>=.845:continue
        t=max(0,min(1,(abs(dx)-1.05)/.95))
        horizontal=1-(6*t**5-15*t**4+10*t**3)
        dz=(z-eye['z'])/.045
        vertical=math.exp(-dz**4)
        top=max(0,min(1,(z-.820)/.025))
        vertical*=1-(6*top**5-15*top**4+10*top**3)
        falloff=horizontal*vertical
        slope=(surface.front(eye['x']+.0001,eye['z'])-surface.front(eye['x']-.0001,eye['z']))/.0002
        delta-=.8*slope*(x-eye['x'])*falloff
    return delta


def apply_e4_face_delta():
    import bpy
    import face_v3
    from common import OUT
    source=OUT/'stages/E4-final-refined.blend'
    module=OUT/'scripts/face_v3.py'
    spec_path=OUT/'reports/measured-face-spec-v3.json'
    if _digest(source)!=EXPECTED_SOURCE_SHA256:raise RuntimeError('E4 source changed')
    if _digest(module)!=EXPECTED_FACE_MODULE_SHA256:raise RuntimeError('V3 basis module changed')
    if _digest(spec_path)!=EXPECTED_SPEC_SHA256:raise RuntimeError('V3 basis specification changed')
    if bpy.context.scene.get('e4_face_local_refine_applied'):
        raise RuntimeError('Local delta already applied; use a fresh E4 copy')
    spec=json.loads(spec_path.read_text())
    baseline,baseline_surface=face_v3.geometry(spec)
    candidate=copy.deepcopy(spec)
    # Keep all aperture/iris XZ landmarks. Move the curved globe behind the skin
    # opening and adapt the existing socket edge rather than sinking a loose disk.
    for eye in candidate['eyes']:
        eye.update(rim_recess=.0018,bulge=.0015,socket_depth=.0014)
    # Retain the source nose position; reduce tip projection and soften its lower
    # slope into the unchanged closed mouth instead of a pointed triangular ledge.
    candidate['nose']['lobes']=[[0,.758,.014,.020,.0125],[0,.780,.011,.024,.0028]]
    rebuilt,surface=face_v3.geometry(candidate)
    base={p['name']:p for p in baseline};new={p['name']:p for p in rebuilt}
    missing=TARGET_NAMES-set(bpy.data.objects.keys())
    if missing:raise RuntimeError(f'Missing E4 objects: {sorted(missing)}')
    objects={name:bpy.data.objects[name] for name in TARGET_NAMES}
    protected={o.name:_positions(o) for o in bpy.context.scene.objects
               if o.type=='MESH' and o.name not in TARGET_NAMES}
    uv_before={n:_uv_digest(o) for n,o in objects.items()}
    topology_before={n:_topology(o) for n,o in objects.items()}
    head=objects['Head_Skin_Quad_Retopo']
    scalp_indices=[i for i,v in enumerate(base[head.name]['vertices']) if v[2]>.845]
    scalp_before=_positions(head,scalp_indices)
    changes=[];unaffected=[]
    for name,obj in objects.items():
        if len(obj.data.vertices)!=len(base[name]['vertices']):raise RuntimeError(f'Topology differs: {name}')
        if obj.data.has_custom_normals:raise RuntimeError(f'Unexpected authored split normals: {name}')
        maximum_basis_error=0.
        deltas=[]
        for i,v in enumerate(obj.data.vertices):
            old=base[name]['vertices'][i];target=new[name]['vertices'][i]
            if name==head.name and old[2]>.845:
                # E3 modified these actual vertices radially against the crown.
                # Never replace them with a reconstructed parametric scalp.
                unaffected.append((i,tuple(v.co)));continue
            basis_error=math.sqrt(sum((v.co[k]-old[k])**2 for k in range(3)))
            maximum_basis_error=max(maximum_basis_error,basis_error)
            if basis_error>2e-6:raise RuntimeError(f'Current E4 lower face differs from protected V3 basis: {name}[{i}] {basis_error}')
            delta=[target[k]-old[k] for k in range(3)]
            if name==head.name:
                mask=(_smoothstep(.715,.725,old[2])*(1-_smoothstep(.820,.835,old[2]))
                      *(1-_smoothstep(.095,.110,abs(old[0]))))
                delta=[d*mask for d in delta]
                if old[1]<baseline_surface.section(old[2])[1]:
                    delta[1]+=_orbital_alignment(surface,candidate['eyes'],old[0],old[2])
            elif name.startswith(('Eye_Sclera_','Eye_Iris_')):
                # Rearward 0.6mm puts the full eye cap within the socket wall.
                # Iris keeps its original 0.05mm separation from the eye white.
                delta[1]+=.0006
                delta[1]+=_orbital_alignment(surface,candidate['eyes'],old[0],old[2])
            elif name.startswith(('Eye_UpperLash_','Brow_')) or name=='Mouth_Closed_Line':
                delta[1]+=_orbital_alignment(surface,candidate['eyes'],old[0],old[2])
            if max(abs(d) for d in delta)<1e-11:
                if name==head.name:unaffected.append((i,tuple(v.co)))
                continue
            v.co.x+=delta[0];v.co.y+=delta[1];v.co.z+=delta[2]
            deltas.append(math.sqrt(sum(d*d for d in delta)))
        obj.data.update()
        changes.append({'object':name,'movedVertices':len(deltas),'maximumMove':max(deltas,default=0),
                        'basisMaxError':maximum_basis_error})
    assert _positions(head,scalp_indices)==scalp_before,'Fitted scalp changed'
    assert all(tuple(head.data.vertices[i].co)==co for i,co in unaffected),'Unrelated head coordinates changed'
    assert all(_positions(bpy.data.objects[n])==sha for n,sha in protected.items()),'Unrelated object changed'
    assert all(_uv_digest(objects[n])==sha for n,sha in uv_before.items()),'UV changed'
    assert all(_topology(objects[n])==sha for n,sha in topology_before.items()),'Topology changed'
    eye_depth=[]
    for name,obj in objects.items():
        if name.startswith(('Eye_Sclera_','Eye_Iris_')):
            offsets=[v.co.y-surface.front(v.co.x,v.co.z)-_orbital_alignment(surface,candidate['eyes'],v.co.x,v.co.z)
                     for v in obj.data.vertices]
            eye_depth.append({'object':name,'minBehindUncutSkinSurface':min(offsets),
                              'maxBehindUncutSkinSurface':max(offsets)})
            assert min(offsets)>.0005,'Eye remains in front of the skin envelope'
    assert _digest(source)==EXPECTED_SOURCE_SHA256,'Source .blend bytes changed'
    bpy.context.scene['e4_face_local_refine_applied']=True
    return {'source':'stages/E4-final-refined.blend','sourceSha256':EXPECTED_SOURCE_SHA256,
            'operation':'Local existing-vertex delta; no topology/UV/material replacement',
            'changes':changes,'scalpProtectedVertexCount':len(scalp_indices),'scalpExact':True,
            'unaffectedHeadExact':True,'otherMeshObjectsExact':True,'uvExact':True,'topologyExact':True,
            'eyeDepth':eye_depth,'noseTipAnalyticYBefore':face_v3.geometry(spec)[1].front(0,.758),
            'noseTipAnalyticYAfter':surface.front(0,.758),'rigAdded':False,
            'eyeSocketAlignment':{'field':'80 percent compensation of measured horizontal face slope; C2 horizontal falloff, broad 45mm vertical quartic-Gaussian transition, protected scalp cutoff',
                                  'approxIrisCentreYawBeforeDegrees':27.3,'approxIrisCentreYawAfterDegrees':5.9,
                                  'appliedTo':'Skin socket, white, iris, upper lid ribbon and brows together; XZ and UV unchanged'},
            'appearanceStatus':'isolated candidate; requires actual same-camera comparison'}

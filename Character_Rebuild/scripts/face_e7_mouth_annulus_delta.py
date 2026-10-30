"""Apply the verified E7 mouth surround correction without replacing any mesh.

No scene or file is opened/saved by this module. The caller remains the source
integrator. The frozen coordinate data contains only 217 existing skin vertices.
"""
import hashlib
import json
import struct

SOURCE_SHA256='fb285cbe33d0015f185cdf6a83e28e059e5043726b4640e347c2000117923cdf'
DELTA_SHA256='0227e1d56ff4bce8e8846052b446177c9009e3e86799657bc5c312f6048f36a0'


def _positions(obj):
    h=hashlib.sha256()
    for v in obj.data.vertices:h.update(struct.pack('<I3f',v.index,*v.co))
    return h.hexdigest()


def _uv(obj):
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


def apply_e7_mouth_annulus_delta():
    import bpy
    from common import OUT
    path=OUT/'reports/e7-mouth-annulus-delta.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==DELTA_SHA256,'Frozen delta changed'
    assert hashlib.sha256((OUT/'stages/E7-uv-normals-corrected.blend').read_bytes()).hexdigest()==SOURCE_SHA256,'E7 source changed'
    assert not bpy.context.scene.get('e7_mouth_annulus_delta_applied'),'Mouth patch already applied'
    data=json.loads(path.read_text(encoding='utf-8'))
    head=bpy.data.objects[data['sourceObject']]
    assert _uv(head)==data['uvDigest'],'E7 UV data differs'
    assert _topology(head)==data['topologyDigest'],'E7 topology differs'
    others={o.name:_positions(o) for o in bpy.context.scene.objects if o.type=='MESH' and o!=head}
    changed={c['vertex'] for c in data['changes']}
    protected={v.index:tuple(v.co) for v in head.data.vertices if v.index not in changed}
    for c in data['changes']:
        assert tuple(head.data.vertices[c['vertex']].co)==tuple(c['before']),f"E7 vertex {c['vertex']} differs"
    for c in data['changes']:head.data.vertices[c['vertex']].co=c['after']
    head.data.update()
    assert all(tuple(head.data.vertices[i].co)==co for i,co in protected.items())
    assert all(_positions(bpy.data.objects[n])==sha for n,sha in others.items())
    assert _uv(head)==data['uvDigest'] and _topology(head)==data['topologyDigest']
    bpy.context.scene['e7_mouth_annulus_delta_applied']=True
    return {'operation':data['operation'],'sourceSha256':SOURCE_SHA256,'deltaSha256':DELTA_SHA256,
            'changedVertices':len(data['changes']),'maximumMove':data['maximumMove'],
            'topologyExact':True,'uvExact':True,'allOtherHeadVerticesExact':True,'otherObjectsExact':True,
            'verifiedSourceAndHeadOnlyReimportSelfContacts':0,'testedTolerances':[1e-8,1e-10,0],
            'note':'Root must still export and verify the final combined GLB. No global/inter-object intersection certification.'}

"""Derive exact low-cage head/neck selection without Blender or mesh edits.

The selection is intended for a mask AFTER the author's Multires modifier.
Raw index maps are supplied for landmark correspondence, not as a reason to
delete Multires base topology or to replace the author's sculpt with the cage.
"""
from pathlib import Path
import ast
import collections
import hashlib
import json
import math
import struct
import sys
import zipfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
WORK=P/'work/reconstruction-r02'
NPZ=WORK/'female-basemesh-actual-r11.npz'
META=WORK/'female-basemesh-actual-r11.json'


def read_array(archive,name):
    b=archive.read(name+'.npy')
    assert b[:6]==b'\x93NUMPY' and b[6:8]==b'\x01\x00'
    header_len=struct.unpack_from('<H',b,8)[0]
    h=ast.literal_eval(b[10:10+header_len].decode())
    assert not h['fortran_order']
    fmt={'<f4':'f','<f8':'d','<i4':'i','<i8':'q'}[h['descr']]
    values=struct.unpack('<'+fmt*math.prod(h['shape']),b[10+header_len:])
    return list(zip(*[iter(values)]*h['shape'][1])) if len(h['shape'])==2 else values


def main():
    meta=json.loads(META.read_text(encoding='utf8'))
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest()==meta['npz_sha256']
    with zipfile.ZipFile(NPZ) as z:
        v=read_array(z,'object_00_local')
        cv=read_array(z,'object_00_corner_vertices')
        st=read_array(z,'object_00_face_starts')
        ln=read_array(z,'object_00_face_lengths')
        fs=read_array(z,'object_00_face_sets')
        uv=read_array(z,'object_00_uv_00')
    faces=[list(cv[s:s+n]) for s,n in zip(st,ln)]
    head_sets={2,3,4,5,7,8,17,22}
    selected={i for i,s in enumerate(fs) if s in head_sets}
    # Default face set 1 contains both the torso and six small head/ear/oral
    # patches. Include only components fully above Z 1.40 m, not the torso.
    edge_faces=collections.defaultdict(list)
    for i,f in enumerate(faces):
        for a,b in zip(f,f[1:]+f[:1]):
            edge_faces[tuple(sorted((a,b)))].append(i)
    other=set(range(len(faces)))-selected
    added=[]
    while other:
        seed=min(other);component={seed};todo=[seed];other.remove(seed)
        while todo:
            i=todo.pop();f=faces[i]
            for a,b in zip(f,f[1:]+f[:1]):
                for j in edge_faces[tuple(sorted((a,b)))]:
                    if j in other:
                        other.remove(j);component.add(j);todo.append(j)
        vertices={j for i in component for j in faces[i]}
        if min(v[j][2] for j in vertices)>1.40:
            selected.update(component);added.extend(component)
    face_ids=sorted(selected)
    vertex_ids=sorted({j for i in face_ids for j in faces[i]})
    old_to_new=[-1]*len(v)
    for j,i in enumerate(vertex_ids):old_to_new[i]=j
    corners=[c for i in face_ids for c in range(st[i],st[i]+ln[i])]
    edges=collections.Counter(tuple(sorted((a,b))) for i in face_ids for a,b in zip(faces[i],faces[i][1:]+faces[i][:1]))
    boundary=sorted(e for e,n in edges.items() if n==1)
    adj=collections.defaultdict(list)
    for a,b in boundary:adj[a].append(b);adj[b].append(a)
    assert all(len(n)==2 for n in adj.values()),'Expected clean boundary cycles'
    ring=[min(adj)];prev=None;current=ring[0]
    while True:
        nxt=min(n for n in adj[current] if n!=prev)
        if nxt==ring[0]:break
        assert nxt not in ring
        ring.append(nxt);prev,current=current,nxt
    assert len(ring)==len(boundary),'Expected one neck boundary, no unintended facial holes'
    extras=[i for i in range(len(faces)) if i not in selected and all(old_to_new[j]>=0 for j in faces[i])]
    assert not extras, 'Vertex mask would retain unexpected low cage faces'
    mapped_faces=[[old_to_new[j] for j in faces[i]] for i in face_ids]
    report={
        'source_npz':str(NPZ.relative_to(P)), 'source_npz_sha256':meta['npz_sha256'],
        'source_blend_sha256':meta['source_sha256'],
        'purpose':'Exact raw-cage correspondence and AFTER-MULTIRES mask selection. Keep original full body base topology and authored Multires displacement intact.',
        'status':'Computed from actual official donor arrays; no source object modified and no evaluated shape replaced.',
        'counts':{'selected_faces':len(face_ids),'selected_vertices':len(vertex_ids),'selected_corners':len(corners),'neck_boundary_vertices':len(ring),'added_default_set_patches':len(added)},
        'selection':{'whole_face_sets':sorted(head_sets),'added_faces_from_other_components':sorted(added),'extra_faces_retained_by_vertex_mask':extras},
        'original_face_indices':face_ids,'original_vertex_indices':vertex_ids,
        'original_corner_indices':corners,'old_to_new_vertex_index':old_to_new,'new_to_old_vertex_index':vertex_ids,
        'new_faces':mapped_faces,'new_face_sets':[fs[i] for i in face_ids],
        'new_corner_uv':[uv[i] for i in corners],
        'neck_ring_old_indices':ring,'neck_ring_new_indices':[old_to_new[i] for i in ring],
        'neck_ring_raw_local_coordinates':[v[i] for i in ring],
        'uv_note':'Preserve original UDIM UV values and per-corner ordering. Do not normalize with fract(): the original head halves occupy separate tiles and the oral interior has its own islands.',
        'eye_rings_from_native_cache_agent':{'L':list(range(7946,7966)),'R':list(range(2812,2832)),'note':'Reported independently by native_cache from face set boundaries; mapped indices here are exact if those original landmarks are used.'},
        'eye_rings_mapped':{'L':[old_to_new[i] for i in range(7946,7966)],'R':[old_to_new[i] for i in range(2812,2832)]},
        'method':['Selected full head/neck face sets 17,22,2,3,4,5,7,8.','Filled all six small remaining components above Z1.40m; these default-set faces include ear/oral/posterior head patches.','Verified one 54-point neck cycle, all boundary degrees 2, no facial holes, no unselected face whose vertices would all be kept.'],
        'implementation_guidance':['Append complete original female body and its two eyes; preserve Multires L3.','Create a vertex group from original_vertex_indices with weight 1.0, use a Mask modifier AFTER Multires for a reversible first candidate.','If freezing a separate dense head from evaluated L3, preserve the author shape and transfer original IDs/UV through a controlled final cut; do not delete low cage faces before retaining the displacement.','Use matrix_world to align the eyes, not their object.location; original parenting includes a parent inverse.','Create new facial controls from this neutral adult shape rather than applying old failed additive eye/nose deformations.']
    }
    out=WORK/'female-head-selection-r11.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'counts':report['counts'],'added_faces':sorted(added),'neck_ring':ring,'eye_rings_mapped':report['eye_rings_mapped']},indent=2))


if __name__=='__main__':main()

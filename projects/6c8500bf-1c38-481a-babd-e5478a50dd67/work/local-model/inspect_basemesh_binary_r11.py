"""Read the official bundle's serialized DNA without starting Blender.

Read-only input; emits independent metadata, never rewrites a blend file.
Uses the Linux system libzstd for Blender's concatenated zstd frames.
"""
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

PROJECT = project_path()
SOURCE = PROJECT / 'assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
OUTPUT = PROJECT / 'work/reconstruction-r02/female-base-inventory-r11.json'


def expand(raw):
    z = ctypes.CDLL(ctypes.util.find_library('zstd'))
    for fn, args, ret in [
        ('ZSTD_findFrameCompressedSize', [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
        ('ZSTD_getFrameContentSize', [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ulonglong),
        ('ZSTD_decompress', [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
    ]:
        f = getattr(z, fn)
        f.argtypes, f.restype = args, ret
    out, offset = [], 0
    while offset < len(raw):
        src = ctypes.create_string_buffer(raw[offset:])
        size = z.ZSTD_findFrameCompressedSize(src, len(raw) - offset)
        target = z.ZSTD_getFrameContentSize(src, size)
        if target > 250_000_000 or size > len(raw) - offset:
            raise ValueError('Invalid or unsupported compressed frame')
        dest = ctypes.create_string_buffer(target)
        actual = z.ZSTD_decompress(dest, target, src, size)
        if actual != target:
            raise ValueError('Decompression size mismatch')
        out.append(dest.raw)
        offset += size
    return b''.join(out)


class Blend:
    def __init__(self, raw):
        self.b = expand(raw) if raw[:4] == b'\x28\xb5\x2f\xfd' else raw
        if self.b[:12] != b'BLENDER-v402':
            raise ValueError('This reader was verified only against the installed Blender 4.2 asset bundle')
        self.blocks, self.ptrs = [], {}
        offset = 12
        while offset + 24 <= len(self.b):
            code, size, ptr, dna, count = struct.unpack_from('<4sIQII', self.b, offset)
            block = {'code': code, 'size': size, 'ptr': ptr, 'dna': dna, 'count': count, 'data': self.b[offset+24:offset+24+size]}
            self.blocks.append(block)
            if ptr:
                self.ptrs[ptr] = block
            offset += 24 + size
        dna = next(v['data'] for v in self.blocks if v['code'] == b'DNA1')
        at = 8

        def names(tag):
            nonlocal at
            if tag is not None:
                assert dna[at:at+4] == tag
                at += 4
            n = struct.unpack_from('<I', dna, at)[0]
            at += 4
            out = []
            for _ in range(n):
                end = dna.index(b'\0', at)
                out.append(dna[at:end].decode())
                at = end + 1
            at = (at + 3) & ~3
            return out

        self.names = names(None)
        self.types = names(b'TYPE')
        assert dna[at:at+4] == b'TLEN'
        at += 4
        self.lens = struct.unpack_from('<' + 'H'*len(self.types), dna, at)
        at = (at + 2*len(self.types) + 3) & ~3
        assert dna[at:at+4] == b'STRC'
        at += 4
        n = struct.unpack_from('<I', dna, at)[0]
        at += 4
        self.structs, self.dna_types = {}, []
        for _ in range(n):
            type_id, fields = struct.unpack_from('<HH', dna, at)
            at += 4
            entries, offset = [], 0
            for _ in range(fields):
                ti, ni = struct.unpack_from('<HH', dna, at)
                at += 4
                typ, name = self.types[ti], self.names[ni]
                count = 1
                for value in re.findall(r'\[(\d+)\]', name):
                    count *= int(value)
                pointer = '*' in name
                size = (8 if pointer else self.lens[ti]) * count
                entries.append({'type': typ, 'name': name, 'offset': offset, 'size': size, 'pointer': pointer, 'count': count})
                offset += size
            self.structs[self.types[type_id]] = {'fields': entries, 'size': self.lens[type_id], 'computed': offset}
            self.dna_types.append(self.types[type_id])

    def fields(self, typ):
        s = self.structs[typ]
        if s['size'] != s['computed']:
            raise ValueError(('Unsupported implicit struct alignment', typ, s['size'], s['computed']))
        return s['fields']

    def field(self, data, typ, name):
        f = next(f for f in self.fields(typ) if f['name'].lstrip('*').split('[')[0] == name)
        b = data[f['offset']:f['offset']+f['size']]
        if f['pointer']:
            return struct.unpack('<'+'Q'*f['count'], b)[0]
        fmt = {'int':'i', 'short':'h', 'float':'f', 'double':'d', 'uint64_t':'Q', 'char':'b'}.get(f['type'])
        if fmt:
            if f['type'] == 'char' and f['count'] > 1:
                return b.split(b'\0')[0].decode('utf8', 'replace')
            vals = struct.unpack('<'+fmt*f['count'], b)
            return vals[0] if len(vals) == 1 else list(vals)
        return b

    def name(self, block):
        iddata = self.field(block['data'], self.dna_types[block['dna']], 'id')
        return self.field(iddata, 'ID', 'name')[2:]


def main():
    raw = SOURCE.read_bytes()
    expected = '3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == expected
    blend = Blend(raw)
    report = {'source': str(SOURCE.relative_to(PROJECT)), 'sha256': sha,
              'method': 'Read-only zstd decompression and serialized SDNA parsing; no Blender execution.',
              'selected_objects': [], 'all_object_names': [], 'collection_names': [],
              'dna_fields': {t: blend.fields(t) for t in ['ID', 'Object', 'Mesh', 'CustomData', 'CustomDataLayer']}}
    for block in blend.blocks:
        if block['code'] == b'GR\0\0':
            report['collection_names'].append(blend.name(block))
        if block['code'] != b'OB\0\0':
            continue
        name = blend.name(block)
        report['all_object_names'].append(name)
        if not (name.startswith('GEO-body_female_realistic') or name.startswith('GEO-head_') or ('female_realistic' in name and any(t in name for t in ['eye', 'nose', 'mouth', 'neck', 'ear']))):
            continue
        d = block['data']
        item = {'object': name}
        for k in ['type', 'loc', 'rot', 'scale', 'parent', 'data']:
            try:
                item[k] = blend.field(d, 'Object', k)
            except StopIteration:
                pass
        mesh = blend.ptrs.get(item.get('data'))
        if mesh and mesh['code'] == b'ME\0\0':
            item['mesh'] = blend.name(mesh)
            item['mesh_fields'] = {}
            for k in ['verts_num', 'edges_num', 'faces_num', 'corners_num', 'totvert', 'totpoly', 'totloop', 'totedge', 'face_offset_indices']:
                try:
                    item['mesh_fields'][k] = blend.field(mesh['data'], 'Mesh', k)
                except StopIteration:
                    pass
            item['customdata_layers'] = {}
            for domain in ['vdata', 'edata', 'fdata', 'ldata', 'pdata', 'vert_data', 'edge_data', 'face_data', 'corner_data']:
                try:
                    cd = blend.field(mesh['data'], 'Mesh', domain)
                except StopIteration:
                    continue
                lp = blend.field(cd, 'CustomData', 'layers')
                count = blend.field(cd, 'CustomData', 'totlayer')
                layers = []
                if lp and count:
                    ldata = blend.ptrs[lp]['data']
                    step = blend.structs['CustomDataLayer']['size']
                    for i in range(count):
                        ld = ldata[i*step:(i+1)*step]
                        layer = {k: blend.field(ld, 'CustomDataLayer', k) for k in ['type', 'name', 'data']}
                        layers.append(layer)
                        if layer['name'] == 'position':
                            coords = blend.ptrs[layer['data']]['data']
                            vcount = item['mesh_fields'].get('verts_num') or item['mesh_fields'].get('totvert')
                            xyz = list(struct.iter_unpack('<fff', coords[:vcount*12]))
                            item['local_bounds'] = [[min(v[i] for v in xyz) for i in range(3)], [max(v[i] for v in xyz) for i in range(3)]]
                item['customdata_layers'][domain] = layers
        report['selected_objects'].append(item)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'output':str(OUTPUT),'objects':len(report['all_object_names']), 'selected':[{'object':v['object'],'mesh':v.get('mesh'),'counts':v.get('mesh_fields'),'bounds':v.get('local_bounds')} for v in report['selected_objects']]}, indent=2))


if __name__ == '__main__':
    main()

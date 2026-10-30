"""Read existing 301-frame R04 BPHYSICS; optionally write a new decoded reference.

Use Python 3.14+ (compression.zstd). No Blender import, simulation or source edits.
inspect() returns inventory/bounds; extract() writes only NEW derived .f32/JSON.
"""
import array
import hashlib
import json
import re
import runpy
import struct
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

BASE = project_path()/'work/reconstruction-r02'
SOURCE = BASE/'blendcache_reference-cloth-body-r04'
REFERENCE = BASE/'cloth-fullbody-r07-payload.json'
PAYLOAD = BASE/'cloth-fullbody-r07-positions.f32'
FRAMES = tuple(range(1, 302))
PATTERN = re.compile(r'R02_Skirt_Cotton_(\d{6})_(\d+)\.bphys')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def inventory():
    rows=[]
    for path in sorted(SOURCE.glob('R02_Skirt_Cotton_*.bphys')):
        match=PATTERN.fullmatch(path.name); data=path.read_bytes()
        if not match or data[:8]!=b'BPHYSICS' or len(data)<25:
            raise RuntimeError('Invalid source cache header: '+path.name)
        if struct.unpack('<III',data[8:20])!=(65538,2176,22) or data[20]!=3:
            raise RuntimeError('Unreviewed point-cache layout: '+path.name)
        rows.append({'file':path.name,'frame':int(match[1]),'index':int(match[2]),
                     'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    if [r['frame'] for r in rows]!=list(FRAMES) or {r['index'] for r in rows}!={0}:
        raise RuntimeError('Exactly 301 unique frames 1..301 at stack index 0 required')
    return rows


def _decode():
    decoder=runpy.run_path(str(Path(__file__).with_name('walk_cloth_cache_recovery_r07.py')))['_locations']
    for row in inventory():
        yield row, decoder(SOURCE/row['file'])


def inspect():
    rows=[]; low=[float('inf')]*3; high=[float('-inf')]*3
    for row, points in _decode():
        rows.append(row)
        for axis in range(3):
            low[axis]=min(low[axis],min(v[axis] for v in points))
            high[axis]=max(high[axis],max(v[axis] for v in points))
    return {'framesDecoded':len(rows),'pointsPerFrame':2176,'worldBounds':[low,high],
            'float32PositionBytes':301*2176*12,'manifest':rows,
            'scope':'Existing native simulation positions, not resimulation or a collision acceptance test'}


def extract():
    if REFERENCE.exists() or PAYLOAD.exists():
        if REFERENCE.is_file() and PAYLOAD.is_file():
            report=json.loads(REFERENCE.read_text(encoding='utf-8'))
            if report['manifest']==inventory() and report['payloadSha256']==sha(PAYLOAD):
                return report
        raise RuntimeError('Existing decoded reference differs or is incomplete; preserve it')
    rows=[]
    # Exclusive output avoids replacing a preserved reference or existing payload.
    with PAYLOAD.open('xb') as stream:
        for row, points in _decode():
            values=array.array('f',(v for point in points for v in point))
            if sys.byteorder!='little': values.byteswap()
            stream.write(values.tobytes()); rows.append(row)
    if PAYLOAD.stat().st_size!=301*2176*12:
        raise RuntimeError('Derived reference has an unexpected size; retained for inspection')
    report={'sourceDirectory':str(SOURCE.relative_to(project_path())), 'manifest':rows,
            'payload':str(PAYLOAD.relative_to(project_path())), 'payloadSha256':sha(PAYLOAD),
            'shape':[301,2176,3], 'dtype':'<f4', 'space':'world', 'frames':list(FRAMES),
            'scope':'Exact existing native cloth location payload; no new solve, interpolation or source edits',
            'decoderSource':'walk_cloth_cache_recovery_r07.py, inspected Blender e13b728c6880 pointcache layout'}
    with REFERENCE.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({'reference':str(REFERENCE),'frames':301,'bytes':PAYLOAD.stat().st_size}))
    return report


if __name__=='__main__':
    print('Loaded only. inspect() is read-only; extract() creates a new derived payload reference.')

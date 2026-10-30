"""Offline triage of the preserved R09 solver samples, never R11 acceptance.

Verifies the payload hash, captured row/column ordering and topology hash before
measuring frame-to-frame motion and edge-length changes. These measurements
identify frames worth rendering; no threshold here approves/rejects physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(output):
    p = project_path()
    directory = p / 'work/reconstruction-r02/cloth-body-r09'
    capture_path = directory / 'capture-report.json'
    payload_path = directory / 'cloth-world-1-301.npz'
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    if _sha(payload_path) != capture['payloadSha256']:
        raise RuntimeError('Captured solver payload identity changed')
    rows, columns = (capture['configuration']['sample_rows'], capture['configuration']['sample_columns'])
    nr, nc = len(rows), len(columns)
    faces = [[j*nc+i, (j+1)*nc+i, (j+1)*nc+(i+1) % nc, j*nc+(i+1) % nc]
             for j in range(nr-1) for i in range(nc)]
    topology = {'vertices': nr*nc, 'faces': faces, 'materials': [0]*len(faces)}
    digest = hashlib.sha256(json.dumps(topology, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != capture['controlTopologySha256']:
        raise RuntimeError('Recorded row/column grid does not reproduce captured topology')
    with np.load(payload_path, allow_pickle=False) as payload:
        frames = payload['frames'].copy()
        points = payload['positions_world'].astype(float)
    if not np.array_equal(frames, np.arange(1, 302)) or points.shape != (301, nr*nc, 3):
        raise RuntimeError('Unexpected actual sample range/count')
    q = points.reshape(len(frames), nr, nc, 3)
    circum = np.linalg.norm(q - np.roll(q, -1, axis=2), axis=3)
    vertical = np.linalg.norm(np.diff(q, axis=1), axis=3)
    if circum[0].min() <= 1e-9 or vertical[0].min() <= 1e-9:
        raise RuntimeError('Degenerate reference edges')
    cr, vr = circum / circum[0], vertical / vertical[0]
    ratios = np.concatenate((cr.reshape(len(frames), -1), vr.reshape(len(frames), -1)), axis=1)
    fps = 30.
    speed = np.linalg.norm(np.diff(q, axis=0), axis=3) * fps
    acceleration = np.linalg.norm(np.diff(q, n=2, axis=0), axis=3) * fps**2
    cf, row, column = map(int, np.unravel_index(np.argmax(cr), cr.shape))
    stretch_frames = (np.flatnonzero(ratios.max(1) > 1.10) + 1).tolist()
    accel_index = tuple(map(int, np.unravel_index(np.argmax(acceleration), acceleration.shape)))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'schemaVersion': 1, 'helperSha256': _sha(Path(__file__)),
        'captureReport': capture_path.relative_to(p).as_posix(), 'captureReportSha256': _sha(capture_path),
        'sourceBlend': capture['sourceBlend'], 'sourceSha256': capture['sourceSha256'],
        'payload': payload_path.relative_to(p).as_posix(), 'payloadSha256': _sha(payload_path),
        'actualFrames': [1, 301], 'fpsFromCapturedActionConvention': fps,
        'action': capture['action'], 'actionSha256': capture['actionSha256'],
        'reconstructedGridMatchesCapturedTopology': True, 'topologySha256': digest,
        'edgeLengthRatioQuantilesToFrameOne': np.quantile(ratios, (0, .01, .5, .99, 1)).tolist(),
        'framesWithAnyEdgeLongerThanFrameOneByTenPercent': stretch_frames,
        'largestCircumferenceEdgeChange': {'frame': cf+1, 'controlRow': row, 'controlColumn': column,
            'sourceRow': rows[row], 'sourceColumns': [columns[column], columns[(column+1) % nc]],
            'sourceVertexIds': [row*nc+column, row*nc+(column+1) % nc],
            'referenceLengthMeters': float(circum[0, row, column]),
            'currentLengthMeters': float(circum[cf, row, column]),
            'ratio': float(cr[cf, row, column]),
            'endpointsWorldMeters': q[cf, row, [column, (column+1) % nc]].tolist()},
        'largestVerticalEdgeRatio': float(vr.max()),
        'pointSpeedMetersPerSecondQuantiles': np.quantile(speed, (.5, .99, 1)).tolist(),
        'pointAccelerationMetersPerSecondSquaredQuantiles': np.quantile(acceleration, (.5, .99, 1)).tolist(),
        'largestDiscreteAcceleration': {'centerFrame': accel_index[0]+2,
            'controlRow': accel_index[1], 'controlColumn': accel_index[2],
            'valueMetersPerSecondSquared': float(acceleration[accel_index])},
        'recommendedAdditionalReviewIntervals': [[30, 35], [130, 136], [175, 184]],
        'knownSparseImages': [1, 55, 151, 247, 301],
        'limitations': ['Preserved R09 solver payload only; no read of the current R11 evaluated source.',
            'Frame 1 is simulated and already under load; edge ratios are not a calibrated material strain.',
            'World accelerations include body transport and intentional motion; a peak alone does not imply a defect.',
            'Only skirt simulation is captured. Blouse/collar/sleeves are not certified by these samples.',
            'No collision, continuous image or customer appearance acceptance.'],
        'savedBlend': False, 'changedCache': False, 'acceptance': False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(output), 'stretchFrames': stretch_frames,
                      'accelerationCenterFrame': accel_index[0]+2}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='work/reconstruction-r02/cloth-motion-triage-r11.json')
    args = parser.parse_args()
    output = (project_path() / args.output).resolve()
    output.relative_to((project_path() / 'work').resolve())
    analyze(output)

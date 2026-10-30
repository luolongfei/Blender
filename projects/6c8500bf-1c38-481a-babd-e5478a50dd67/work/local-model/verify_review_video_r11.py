"""Independent PyAV decode of a review MP4; no Blender, image edit or re-encode.

Uses isolated project work/verification-r11/python-deps packages only in this
process. Compares every decoded RGB frame to its actual source PNG, including
pixels that moved since frame 1. Exports three unmodified decoded frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _path(value):
    if Path(value).is_absolute() or '..' in str(value).replace('\\', '/').split('/'):
        raise ValueError('Project-relative path required')
    result = (project_path() / value).resolve()
    result.relative_to(project_path().resolve())
    return result


def verify(folder_relative, video_name, encode_report_name, output_relative):
    sys.path.insert(0, str(_path('work/verification-r11/python-deps')))
    import av
    import numpy as np
    from PIL import Image
    folder = _path(folder_relative)
    if Path(video_name).name != video_name or Path(encode_report_name).name != encode_report_name:
        raise ValueError('Use plain filenames inside the exact review folder')
    video = folder / video_name
    old_report = folder / encode_report_name
    original = json.loads(old_report.read_text(encoding='utf-8-sig'))
    plan = original['input']
    expected = plan['sourceFrames']
    out = _path(output_relative)
    out.relative_to((project_path() / 'work').resolve())
    if out.exists():
        raise RuntimeError('Retain prior verification evidence')
    out.mkdir(parents=True)
    report_path = out / 'video-decode-report.json'
    video_sha = _sha(video)
    if video_sha != original['video']['sha256']:
        raise RuntimeError('MP4 no longer matches original encode report')
    rows = []
    previous = None
    first_source = None
    samples = {0, (len(expected) - 1) // 2, len(expected) - 1}
    with av.open(str(video), mode='r') as container:
        stream = container.streams.video[0]
        metadata = {'codec': stream.codec_context.name, 'averageRate': str(stream.average_rate),
                    'timeBase': str(stream.time_base), 'streamFrames': stream.frames,
                    'width': stream.width, 'height': stream.height,
                    'containerDurationSeconds': container.duration / av.time_base if container.duration else None}
        for i, frame in enumerate(container.decode(stream)):
            if i >= len(expected):
                raise RuntimeError('Video has more actual decoded frames than source PNGs')
            record = expected[i]
            png = folder / record['file']
            if _sha(png) != record['sha256']:
                raise RuntimeError('Source PNG changed: ' + record['file'])
            current = frame.to_ndarray(format='rgb24')
            with Image.open(png) as img:
                source = np.asarray(img.convert('RGB')).copy()
            if source.shape != current.shape:
                raise RuntimeError('Actual decoded/source dimensions differ')
            if first_source is None:
                first_source = source
            error = np.abs(current.astype(np.int16) - source.astype(np.int16))
            moved = np.max(np.abs(source.astype(np.int16) - first_source.astype(np.int16)), axis=2) > 16
            row = {'decodedIndex': i + 1, 'sourceFrame': record['frame'], 'pts': frame.pts,
                   'timeBase': str(frame.time_base), 'seconds': float(frame.pts * frame.time_base) if frame.pts is not None else None,
                   'decodedRgbSha256': hashlib.sha256(current.tobytes()).hexdigest(),
                   'rgbMeanAbsoluteErrorToOwnSource': float(error.mean()),
                   'rgb99thPercentileAbsoluteErrorToOwnSource': float(np.percentile(error, 99)),
                   'changedSinceFirstSourcePixelCount': int(moved.sum()),
                   'changedRegionMeanAbsoluteErrorToOwnSource': float(error[moved].mean()) if moved.any() else None,
                   'changedRegionMeanAbsoluteErrorToFirstSource': float(np.abs(current.astype(np.int16) - first_source.astype(np.int16))[moved].mean()) if moved.any() else None,
                   'meanAbsoluteRgbDifferenceFromPreviousDecoded': float(np.abs(current.astype(np.int16) - previous.astype(np.int16)).mean()) if previous is not None else None}
            if i in samples:
                path = out / f'decoded-{i+1:04d}.png'
                frame.to_image().save(path)
                row['decodedImage'] = {'path': path.relative_to(project_path()).as_posix(), 'sha256': _sha(path)}
            rows.append(row)
            previous = current
    timestamps = [r['seconds'] for r in rows]
    monotonic = all(a is not None and b is not None and b > a for a, b in zip(timestamps, timestamps[1:]))
    target_fps = float(plan['fps'])
    timing_error = max((abs((timestamps[i] - timestamps[0]) - i / target_fps) for i in range(len(rows))), default=0.)
    moving = [r for r in rows if r['changedSinceFirstSourcePixelCount'] > 100]
    result = {'status': 'decoded-await-visible-review', 'video': {'path': video.relative_to(project_path()).as_posix(), 'sha256': video_sha},
              'encoderReport': {'path': old_report.relative_to(project_path()).as_posix(), 'sha256': _sha(old_report)},
              'helperSha256': _sha(__file__), 'pyavVersion': av.__version__, 'numpyVersion': np.__version__,
              'avLibraryVersions': {k: list(v) for k, v in av.library_versions.items()},
              'metadata': metadata, 'decodedFrameCount': len(rows), 'expectedFrameCount': len(expected),
              'strictlyIncreasingPts': monotonic, 'maximumTimebaseErrorSeconds': timing_error,
              'uniqueDecodedRgbFrames': len({r['decodedRgbSha256'] for r in rows}),
              'maximumOwnSourceMeanAbsoluteRgbError': max(r['rgbMeanAbsoluteErrorToOwnSource'] for r in rows),
              'movingFramesCloserToOwnSourceThanFirstSource': all(r['changedRegionMeanAbsoluteErrorToOwnSource'] < r['changedRegionMeanAbsoluteErrorToFirstSource'] for r in moving),
              'movingFramesExamined': len(moving), 'frames': rows,
              'sourcePNGsAndMP4Unchanged': True,
              'scope': 'Independent complete MP4 decode and actual pixel/PTS correspondence only. This does not establish collision, natural motion, current R11 quality or appearance acceptance.'}
    if len(rows) != len(expected) or not monotonic or timing_error > .001:
        result['status'] = 'failed-frame-count-or-timing'
    if _sha(video) != video_sha:
        raise RuntimeError('MP4 changed during read-only verification')
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--encode-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.folder, args.video, args.encode_report, args.output)

"""Encode a complete rendered sequence; preserve earlier videos and verify output.

Blender CLI, after the R07 continuous render has completed:
  --python encode_review_r07.py -- \
    --folder previews/reconstruction-r02/combined-walk-continuous-r07

The folder's render-report.json supplies fps and exact source frame numbers.
Without a report, explicitly supply --no-summary --fps 30 --pattern frame-%04d.png
--start 1 --end 80. Default expected-count is 80; no frame-gap interpolation.
--preflight also works in ordinary Python and never starts Blender or encodes.
This module does not open or save a blend file. It uses only a transient VSE scene.
"""
import argparse
import datetime
from fractions import Fraction
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _within(base, relative):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Expected a relative path without ..')
    target = (base / relative).resolve()
    target.relative_to(base.resolve())
    return target


def _frame_numbers(values):
    numbers = []
    for value in values:
        value = value['frame'] if isinstance(value, dict) else value
        if isinstance(value, bool) or int(value) != float(value):
            raise ValueError('Source frames must be integers')
        numbers.append(int(value))
    if not numbers or len(set(numbers)) != len(numbers):
        raise ValueError('Source frame list is empty or contains duplicates')
    if numbers != sorted(numbers):
        raise ValueError('Source frame list is not in ascending order')
    if numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise ValueError('Source sequence is sparse or has gaps; render missing frames first')
    return numbers


def _pattern(pattern):
    if Path(pattern).name != pattern or '/' in pattern or '\\' in pattern:
        raise ValueError('Frame pattern must be a filename, not a path')
    tokens = list(re.finditer(r'%0?\d*d', pattern))
    if len(tokens) != 1 or pattern.count('%') != 1 or not pattern.lower().endswith('.png'):
        raise ValueError('Use one integer placeholder in a PNG pattern, e.g. frame-%04d.png')
    token = tokens[0]
    regex = re.compile('^' + re.escape(pattern[:token.start()]) + r'(\d+)' +
                       re.escape(pattern[token.end():]) + '$')
    return regex


def _png_dimensions(path):
    with path.open('rb') as stream:
        header = stream.read(24)
        stream.seek(-12, 2)
        end = stream.read(12)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise RuntimeError('Not a PNG with an IHDR header: ' + str(path))
    if end[:8] != b'\0\0\0\0IEND':
        raise RuntimeError('PNG lacks its final IEND chunk; render may be incomplete: ' + str(path))
    size = struct.unpack('>II', header[16:24])
    if not all(size) or any(value % 2 for value in size):
        raise RuntimeError('H.264 review requires positive even dimensions; no implicit resize: ' + str(size))
    return size


def _output_paths(folder, requested_name):
    if Path(requested_name).name != requested_name or Path(requested_name).suffix.lower() != '.mp4':
        raise ValueError('Output must be an MP4 filename inside the frame folder')
    stem = Path(requested_name).stem
    for version in range(1, 10000):
        chosen = stem if version == 1 else f'{stem}-{version:03d}'
        video = folder / (chosen + '.mp4')
        report = folder / (chosen + '-report.json')
        if not video.exists() and not report.exists() and not list(folder.glob(chosen + '-decoded-*.png')):
            return video, report
    raise RuntimeError('No free output version available')


def prepare(folder_relative, summary='render-report.json', fps=None, pattern=None,
            start=None, end=None, expected_count=80, output='walk-preview-r07.mp4'):
    """Read-only preflight. Supports render-report rows {'frame': n, ...}."""
    project = project_path().resolve(); folder = _within(project, folder_relative)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    summary_path = _within(folder, summary) if summary is not None else None
    data = {}
    if summary_path is not None:
        if not summary_path.is_file():
            raise RuntimeError('Completed render summary missing. Wait for rendering or use --no-summary with explicit fps/range.')
        data = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    summary_fps = (Fraction(str(data['fps'])) / Fraction(str(data.get('fpsBase', 1)))) if 'fps' in data else None
    rate = Fraction(str(fps)) if fps is not None else summary_fps
    if rate is None or not 1 <= float(rate) <= 240:
        raise ValueError('A frame rate from summary or --fps in [1, 240] is required')
    if summary_fps is not None and fps is not None and abs(float(rate - summary_fps)) > 1e-6:
        raise ValueError('CLI fps conflicts with the render summary')
    summary_pattern = data.get('framePattern')
    if pattern and summary_pattern and pattern != summary_pattern:
        raise ValueError('CLI frame pattern conflicts with the render summary')
    pattern = pattern or summary_pattern or 'frame-%04d.png'
    matcher = _pattern(pattern)
    summary_frames = _frame_numbers(data['frames']) if 'frames' in data else None
    if (start is None) != (end is None):
        raise ValueError('Supply both --start and --end, or neither')
    if start is not None:
        if end < start:
            raise ValueError('End precedes start')
        frames = list(range(start, end + 1))
        if summary_frames is not None and frames != summary_frames:
            raise ValueError('CLI range conflicts with the render summary')
    elif summary_frames is not None:
        frames = summary_frames
    else:
        raise ValueError('Without summary frames, supply an explicit complete --start/--end range')
    if len(frames) != expected_count:
        raise RuntimeError(f'Expected {expected_count} contiguous frames, found {len(frames)} in the declared range')
    discovered = {}
    for path in folder.iterdir():
        match = matcher.match(path.name)
        if match and path.is_file():
            number = int(match.group(1))
            if path.name != pattern % number:
                raise RuntimeError('Filename does not match canonical padding: ' + path.name)
            if number in discovered:
                raise RuntimeError('Duplicate source frame number: ' + str(number))
            discovered[number] = path
    if sorted(discovered) != frames:
        raise RuntimeError('Frame inventory differs from complete range: ' + json.dumps({
            'missing': sorted(set(frames) - set(discovered)),
            'unexpected': sorted(set(discovered) - set(frames))}))
    manifest = []; size = None
    for number in frames:
        path = discovered[number]; current_size = _png_dimensions(path)
        if size is None: size = current_size
        if size != current_size:
            raise RuntimeError('Mixed image dimensions: ' + path.name)
        manifest.append({'frame': number, 'file': path.name, 'bytes': path.stat().st_size,
                         'sha256': _sha256(path)})
    video, report = _output_paths(folder, output)
    return {'folder': str(folder), 'folderRelative': folder.relative_to(project).as_posix(),
            'summary': summary_path.relative_to(project).as_posix() if summary_path else None,
            'summarySha256': _sha256(summary_path) if summary_path else None,
            'renderSource': data.get('source'), 'renderSourceSha256': data.get('sha256'),
            'renderScope': data.get('scope'), 'sourceFramePattern': pattern,
            'frameNumbers': frames, 'sourceFrames': manifest,
            'fps': float(rate), 'fpsFraction': f'{rate.numerator}/{rate.denominator}',
            'fpsSource': 'CLI' if fps is not None else 'render summary',
            'frameCount': len(frames), 'expectedDurationSeconds': float(Fraction(len(frames), 1) / rate),
            'resolution': list(size), 'video': str(video), 'report': str(report),
            'scope': 'Continuous existing PNG frames only; no interpolation, audio, retiming or claim of visual acceptance'}


def encode(**options):
    plan = prepare(**options)
    if bpy is None:
        raise RuntimeError('Encoding requires Blender with its bundled FFmpeg; use prepare() for offline preflight')
    folder = Path(plan['folder']); video = Path(plan['video']); report_path = Path(plan['report'])
    result = {'status': 'encoding', 'recordedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'blenderVersion': bpy.app.version_string, 'input': plan,
              'savedBlend': False, 'historicalOutputsOverwritten': False}
    # Reserve the new report name before encoding. Another run will select a new
    # suffix instead of overwriting this run or an earlier video.
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    scene = None; clip = None
    try:
        if video.exists():
            raise RuntimeError('Chosen output appeared after preflight; retain it and rerun')
        scene = bpy.data.scenes.new('R07 temporary sequence encoding')
        editor = scene.sequence_editor_create()
        strips = editor.strips if hasattr(editor, 'strips') else editor.sequences
        first = folder / plan['sourceFrames'][0]['file']
        strip = strips.new_image('Rendered continuous review', str(first), channel=1, frame_start=1)
        for source in plan['sourceFrames'][1:]:
            strip.elements.append(source['file'])
        scene.render.resolution_x, scene.render.resolution_y = plan['resolution']
        scene.render.resolution_percentage = 100
        scene.render.fps = round(plan['fps']); scene.render.fps_base = scene.render.fps / plan['fps']
        scene.frame_start = 1; scene.frame_end = plan['frameCount']
        scene.render.use_sequencer = True; scene.render.use_file_extension = True
        scene.render.use_overwrite = False
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0.; scene.view_settings.gamma = 1.
        scene.sequencer_colorspace_settings.name = 'sRGB'
        if hasattr(scene.render.image_settings, 'media_type'):
            scene.render.image_settings.media_type = 'VIDEO'
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'; scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'; scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
        scene.render.ffmpeg.audio_codec = 'NONE'
        scene.render.filepath = str(video)
        bpy.ops.render.render(animation=True, scene=scene.name)
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError('Encoder did not produce the requested nonempty MP4')
        clip = bpy.data.movieclips.load(str(video), check_existing=False)
        decoded_fps = float(clip.fps); decoded_count = int(clip.frame_duration)
        decoded_size = list(clip.size)
        metadata = {'decodedFrames': decoded_count, 'decodedFps': decoded_fps,
                    'decodedDurationSeconds': decoded_count / decoded_fps if decoded_fps else None,
                    'decodedResolution': decoded_size,
                    'durationSource': 'Decoded frame count divided by detected movie fps'}
        result['validation'] = metadata
        if decoded_count != plan['frameCount'] or abs(decoded_fps - plan['fps']) > 1e-4 or decoded_size != plan['resolution']:
            raise RuntimeError('Movie metadata differs from source sequence: ' + json.dumps(metadata))
        if abs(metadata['decodedDurationSeconds'] - plan['expectedDurationSeconds']) > 1e-4:
            raise RuntimeError('Decoded duration differs from expected source duration')
        strips.remove(strip)
        movie = strips.new_movie('Encoded movie decoding check', str(video), channel=1, frame_start=1)
        metadata['vseDecodedFrames'] = int(movie.frame_final_duration)
        if metadata['vseDecodedFrames'] != plan['frameCount']:
            raise RuntimeError('VSE decoder frame count differs from source sequence')
        if hasattr(scene.render.image_settings, 'media_type'):
            scene.render.image_settings.media_type = 'IMAGE'
        scene.render.image_settings.file_format = 'PNG'
        decoded_samples = []
        for number in sorted({1, (plan['frameCount'] + 1) // 2, plan['frameCount']}):
            destination = folder / f'{video.stem}-decoded-{number:04d}.png'
            if destination.exists():
                raise RuntimeError('Retain existing decoded review image: ' + str(destination))
            scene.frame_set(number); scene.render.filepath = str(destination)
            bpy.ops.render.render(write_still=True, scene=scene.name)
            if list(_png_dimensions(destination)) != plan['resolution']:
                raise RuntimeError('Decoded review frame dimensions differ')
            decoded_samples.append({'movieFrame': number,
                                    'sourceFrame': plan['frameNumbers'][number - 1],
                                    'image': destination.name, 'sha256': _sha256(destination)})
        # Detect a render process replacing input PNGs during the encode.
        for source in plan['sourceFrames']:
            if _sha256(folder / source['file']) != source['sha256']:
                raise RuntimeError('Source image changed during encoding: ' + source['file'])
        if plan['summary'] and _sha256(project_path() / plan['summary']) != plan['summarySha256']:
            raise RuntimeError('Render summary changed during encoding')
        metadata['decodedSampleFrames'] = decoded_samples
        result.update(status='encoded-and-decoder-checked',
                      video={'path': video.relative_to(project_path()).as_posix(),
                             'bytes': video.stat().st_size, 'sha256': _sha256(video)},
                      scope='Technical sequence/decoder checks only; no new collision, likeness, visual or client-acceptance claim')
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('R07_VIDEO_ENCODED ' + json.dumps({'video': result['video']['path'],
              'frames': decoded_count, 'fps': decoded_fps, 'seconds': metadata['decodedDurationSeconds'],
              'report': report_path.relative_to(project_path()).as_posix()}), flush=True)
        return result
    except Exception as exc:
        result.update(status='failed; produced files retained for inspection', error=repr(exc))
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    finally:
        if scene is not None:
            bpy.data.scenes.remove(scene)
        if clip is not None and clip.users == 0:
            bpy.data.movieclips.remove(clip)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True, help='Project-relative folder of completed PNG frames')
    parser.add_argument('--summary', default='render-report.json', help='Folder-relative completed render report')
    parser.add_argument('--no-summary', action='store_true', help='Use explicit CLI fps and complete frame range')
    parser.add_argument('--fps', help='Frame rate or fraction, e.g. 30 or 30000/1001')
    parser.add_argument('--pattern', help='PNG filename pattern, normally frame-%%04d.png')
    parser.add_argument('--start', type=int); parser.add_argument('--end', type=int)
    parser.add_argument('--expected-count', type=int, default=80)
    parser.add_argument('--output', default='walk-preview-r07.mp4', help='New MP4 basename; existing names receive a numbered suffix')
    parser.add_argument('--preflight', action='store_true', help='Only read/validate the complete input sequence')
    args = parser.parse_args(argv)
    options = {'folder_relative': args.folder, 'summary': None if args.no_summary else args.summary,
               'fps': args.fps, 'pattern': args.pattern, 'start': args.start, 'end': args.end,
               'expected_count': args.expected_count, 'output': args.output}
    if args.preflight:
        print(json.dumps(prepare(**options), ensure_ascii=False, indent=2))
    else:
        encode(**options)


if __name__ == '__main__':
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    main(arguments)

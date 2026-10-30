"""Encode the actual rendered PNG sequence with Blender's bundled FFmpeg."""
import bpy,sys,json
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();label=sys.argv[sys.argv.index('--')+1]
folder=p/'previews/reconstruction-r02'/label
frames=sorted(folder.glob('frame-*.png'))
if not frames:raise RuntimeError('No rendered frames')
scene=bpy.data.scenes.new('R03 sequence encoding');bpy.context.window.scene=scene
editor=scene.sequence_editor_create()
strips=editor.strips if hasattr(editor,'strips') else editor.sequences
strip=strips.new_image('Rendered character review',str(frames[0]),channel=1,frame_start=1)
for frame in frames[1:]:strip.elements.append(frame.name)
image=bpy.data.images.load(str(frames[0]),check_existing=False);w,h=image.size;bpy.data.images.remove(image)
scene.render.resolution_x=w;scene.render.resolution_y=h;scene.render.resolution_percentage=100
scene.render.fps=15;scene.frame_start=1;scene.frame_end=len(frames)
scene.view_settings.view_transform='Standard'
if hasattr(scene.render.image_settings,'media_type'):scene.render.image_settings.media_type='VIDEO'
scene.render.image_settings.file_format='FFMPEG'
scene.render.ffmpeg.format='MPEG4';scene.render.ffmpeg.codec='H264'
scene.render.ffmpeg.constant_rate_factor='MEDIUM';scene.render.ffmpeg.ffmpeg_preset='GOOD'
scene.render.filepath=str(folder/'performance-preview.mp4')
bpy.ops.render.render(animation=True)
(folder/'video-report.json').write_text(json.dumps({'images':len(frames),'fps':15,'seconds':len(frames)/15,'resolution':[w,h],'video':scene.render.filepath,'note':'Encoded from actual sequential rendered frames; diagnostic preview, no audio'},indent=2),encoding='utf-8')
print('R03_VIDEO_ENCODED',scene.render.filepath)
# Decode the produced movie, so a successful container write alone is not QA.
movie_path=scene.render.filepath
strips.remove(strip)
movie=strips.new_movie('Encoded movie verification',movie_path,channel=1,frame_start=1)
print('R03_VIDEO_DECODED_FRAMES',movie.frame_final_duration)
if hasattr(scene.render.image_settings,'media_type'):scene.render.image_settings.media_type='IMAGE'
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(folder/'encoded-first-frame.png')
scene.frame_set(1);bpy.ops.render.render(write_still=True)

"""Assemble unretouched Blender render comparisons and bind them to saved sources."""
from pathlib import Path
import sys,json,hashlib,os,datetime
from PIL import Image,ImageDraw,ImageFont
root=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(root))
from pipeline.context import project_path
p=project_path(); out=p/'previews/reconstruction-r01'; work=p/'work/reconstruction-r01'
fontpath=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
font=ImageFont.truetype(str(fontpath),27) if fontpath.exists() else ImageFont.load_default()
small=ImageFont.truetype(str(fontpath),21) if fontpath.exists() else font
def sheet(name,items,w,h):
    sheet=Image.new('RGB',(w*len(items),h+100),(22,25,29)); d=ImageDraw.Draw(sheet)
    for i,(label,sub,file) in enumerate(items):
        im=Image.open(out/file).convert('RGB')
        assert im.size==(w,h),(file,im.size)
        sheet.paste(im,(i*w,100)); d.text((i*w+22,14),label,font=font,fill='white'); d.text((i*w+22,55),sub,font=small,fill=(177,185,191))
    sheet.save(out/name)
sheet('face-comparison.png',[
 ('改造前','迁移主源 · 原始头脸', 'before/face-front.png'),
 ('方向 A','椭圆脸 · 下颌更明确', 'review-A/face-front.png'),
 ('方向 B','下半脸缩短 · 轮廓更柔和', 'review-B/face-front.png')],640,720)
sheet('silhouette-comparison.png',[
 ('改造前','原头脸、片状发束与外撑裙摆','before/full.png'),
 ('首轮方向 A','曲线双马尾与收拢裙摆 · 静态预览','review-A/full.png')],640,800)
sheet('profile-comparison.png',[
 ('改造前','同光照、同机位','before/face-side.png'),
 ('方向 A','颅面、鼻唇与头发侧面','review-A/face-side.png'),
 ('方向 B','较柔和的下半脸','review-B/face-side.png')],640,720)
sheet('back-comparison.png',[
 ('改造前','后脑、双马尾与裙摆','before/back.png'),
 ('首轮方向 A','后脑覆盖与下垂轮廓 · 未验证动态','review-A/back.png')],640,800)
hashfile=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
sources=[p/'work/character.blend',work/'candidate-review-A.blend',work/'candidate-review-B.blend']
reports=[json.loads((out/n/'render-report.json').read_text()) for n in ['before','review-A','review-B']]
for f,r in zip(sources,reports): assert hashfile(f)==r['sha256'],f
assert reports[0]['lights']==reports[1]['lights']==reports[2]['lights']
assert reports[0]['views'][:3]==reports[1]['views'][:3]==reports[2]['views']
summary={
 'stage':'首轮静态造型提案，等待人物方向选择',
 'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'masterUnchanged':hashfile(sources[0])=='6e7361e7f5c306325d3f33c4c9f6f56b1a489d5f4579bb4be951d4c3d59cc8c1',
 'sources':[{'path':f.relative_to(p).as_posix(),'sha256':hashfile(f),'bytes':f.stat().st_size} for f in sources],
 'renderCondition':'同场景三灯、同相机和静止姿势、Cycles CPU 24 samples、无景深；对比图仅拼接标注，无修图',
 'images':[{'path':f.relative_to(p).as_posix(),'sha256':hashfile(f)} for f in sorted(out.glob('*comparison.png'))],
 'changes':['CC0动画头经比例与鼻颌适配，替换旧头脸/耳/眼','重做球形眼及曲面虹膜，保留项目虹膜纹理','可编辑曲线发型与双马尾自然下垂，修复头顶根线与后脑覆盖','绿裙在保留刀褶/腰部的可逆形态键中收拢下摆并同步缝线','新头脸A/B可调方向保存在独立源'],
 'scopeAndLimits':['尚未达到AAA或照片级近景，不是最终交付','新脸偏成熟、严肃，原角色身份与年龄感尚未确认保留，需要用户选择或调整','头脸与身体尚未焊接，领口和颈肩仍需雕刻整合','新头脸与曲线发型未绑定，旧表情和旧动作未适配；不适用旧技术通过结论','皮肤仍是大形评审材质，缺少区域纹理、细微不对称、毛孔与绒毛','眼睑、睫毛、眉毛、口腔及表情系统仍需重做与验证','头发尚有规则梳理、头顶亮斑和高光过于整齐的问题；没有物理模拟','裙摆为手工静态形变，不是布料仿真；上衣本轮未修改','不更新VRM/GLB或avatar.blend；它们仍对应迁移主源','无新增异机备份，不宣称备份已冻结'],
 'nextDecision':'选择 A 的较明确下颌或 B 的较柔和短下半脸，或指出希望继续调整的部位',
 'nextProduction':['用户方向选择后深化颅面、颈肩与原服装适配','统一全身拓扑和UV，处理隐藏重叠身体','新眼口眉睫/皮肤与曲线梳理','重做制作绑定、面部表演、动作重定向与布料毛发动态','按实际目标软件另做实时交付适配']}
(work/'review-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
statefile=p/'project-state.json'; state=json.loads(statefile.read_text(encoding='utf-8'))
state['reconstructionR01']={'status':'static-candidates-awaiting-appearance-direction','integratedIntoMaster':False,'report':'work/reconstruction-r01/review-summary.json','research':'work/reconstruction-r01/asset-research.json','sources':summary['sources'][1:],'comparison':'previews/reconstruction-r01/face-comparison.png','oldExportsStillMatchMigrationMaster':True,'newCandidateAnimationValidated':False,'acceptance':'not accepted, not AAA/photoreal final'}
statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=p/'.asset-code-paths.json'; data=json.loads(manifest.read_text(encoding='utf-8'))
data['paths']=sorted(set(data['paths']+[f'work/local-model/{name}' for name in ['reconstruction_finish_r01.py','reconstruction_package_r01.py']]))
manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'complete':True,'mainUnchanged':summary['masterUnchanged'],'images':len(summary['images'])}))

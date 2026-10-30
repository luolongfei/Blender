import fs from 'node:fs';
import crypto from 'node:crypto';
import {workspacePath,projectPath} from '../../../pipeline/context.mjs';

const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const base=workspacePath('Character_Rebuild','interactive');
const file=(...parts)=>workspacePath('Character_Rebuild','interactive',...parts);
const staticBlend=workspacePath('Character_Rebuild','character-rebuilt.blend');
const expectedStatic='b3506fb27cdc04633a5bdaddf5e14166059209bab0f115e790614a3b6bd3d4f5';
if(sha(staticBlend)!==expectedStatic||sha(projectPath()+'/work/character.blend')!==expectedStatic)throw Error('Static source/master identity changed');
const glb=file('animated-preview.glb'),blend=file('rigged-preview.blend');
const hash=sha(glb);
const audit=JSON.parse(fs.readFileSync(file('reports','animation-data-audit.json'),'utf8'));
if(!audit.passed||audit.sha256!==hash)throw Error('Animation audit is missing, failed, or stale');
const labels={idle:'自然站立',wave:'招手',look:'转头',walk:'原地轻踏'};
const manifest={
  schemaVersion:1,createdAt:new Date().toISOString(),purpose:'Local interactive motion preview derived from E8 static rebuild',
  model:{url:'animated-preview.glb',sha256:hash,bytes:fs.statSync(glb).size},
  source:{url:'rigged-preview.blend',sha256:sha(blend),bytes:fs.statSync(blend).size},
  downloads:{blend:'rigged-preview.blend'},
  originalStaticSourceSha256:expectedStatic,
  actions:Object.entries(labels).map(([id,label])=>{const clip=audit.clips.find(c=>c.name.toLowerCase()===id);if(!clip?.valid)throw Error('Missing valid clip '+id);return{id,label,clip:clip.name,duration:clip.duration,loop:id!=='wave'};}),
  viewerNote:'基础骨骼动作预览。尚未制作面部表情、独立手指动作、头发和裙子的物理模拟。',
  appearanceAcceptedByUser:false,
  staticDeliveryPreserved:true
};
fs.writeFileSync(file('preview-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
fs.writeFileSync(file('使用说明.txt'),[
  '角色网页动作预览',
  '',
  '当前地址：http://127.0.0.1:52860/interactive/viewer.html（本机预览服务运行时可用）',
  '鼠标左键拖动旋转，滚轮缩放，右键拖动平移。触屏单指旋转、双指缩放或平移。',
  '选择自然站立、招手、转头或原地轻踏。可暂停、拖时间轴、调整速度、切换观察角度。',
  '查看T形姿势还原模型原始姿势；恢复站立继续待机。',
  '',
  'animated-preview.glb：内嵌贴图、蒙皮和4段实际骨骼动画。',
  'rigged-preview.blend：对应可编辑预览源。',
  'preview-manifest.json：文件身份与实际动作列表。reports：验证记录。',
  '原有character-rebuilt.blend、character-rebuilt-textured.glb与项目主源保持不变。',
  '',
  '范围：这是当前角色的基础动作预览。手指为整手跟随，未新增面部表情；头发随头部、裙子随骨盆，没有头发/布料物理模拟。有限幅度动作检查不等于生产绑定、全局碰撞检查或外观验收。',
  '重启本机服务：从工作区根目录运行 node Character_Rebuild/scripts/serve_delivery.mjs --port 52860；服务仅监听127.0.0.1。',
  '',
  'GLB SHA256：'+hash,
  '预览源 SHA256：'+manifest.source.sha256,
].join('\n')+'\n');
console.log(JSON.stringify({directory:base,manifest:'preview-manifest.json',model:manifest.model,actions:manifest.actions},null,2));

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {workspacePath} from '../../pipeline/context.mjs';
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const read=p=>JSON.parse(fs.readFileSync(workspacePath(p),'utf8'));
const out='Character_Rebuild',project='projects/6c8500bf-1c38-481a-babd-e5478a50dd67';
const reimport=read(`${out}/reports/clean-reimport.json`),visual=read(`${out}/reports/final-visual-review.json`),binary=read(`${out}/reports/final-glb-validation.json`);
const delivered=workspacePath(out,'character-rebuilt.blend'),glb=workspacePath(out,'character-rebuilt-textured.glb');
if(!reimport.structuralReimportPassed||!reimport.glbUnchangedDuringReimport||!visual.reviewCompleted||!binary.passed||hash(delivered)!==reimport.blendSha256||hash(glb)!==reimport.glbSha256||visual.glbSha256!==hash(glb))throw Error('Final verification is incomplete or stale');
const sourceState=read(`${project}/project-state.json`),master=workspacePath(project,'work/character.blend');
let prior=null;
if(fs.existsSync(master)&&hash(master)!==hash(delivered)){
  const old=hash(master),relative=`work/stage-backups/character-before-hunyuan-${old.slice(0,16)}.blend`,backup=workspacePath(project,relative);
  fs.mkdirSync(path.dirname(backup),{recursive:true});if(!fs.existsSync(backup))fs.copyFileSync(master,backup);
  if(hash(backup)!==old)throw Error('Prior master backup mismatch');
  prior={...sourceState.workingMaster,actualSha256:old,backupPath:relative,backupBytes:fs.statSync(backup).size};
  fs.copyFileSync(delivered,master);
  if(hash(master)!==reimport.blendSha256)throw Error('New master copy mismatch');
}
const now=new Date().toISOString();
if(prior){sourceState.workingMasterHistory??=[];sourceState.workingMasterHistory.push(prior);sourceState.preHunyuanExports=sourceState.currentExports;sourceState.preHunyuanExportCorrespondence=sourceState.exportCorrespondence;}
sourceState.recordedAt=now;
sourceState.stage='hunyuan-static-rebuild-delivered; file-validation-passed; artistic-refinement-still-needed';
sourceState.route='local-Blender-rebuild-of-user-hunyuan-input';
sourceState.workingMaster={path:'work/character.blend',bytes:fs.statSync(master).size,sha256:hash(master),integratedFrom:{path:'../../Character_Rebuild/character-rebuilt.blend',sha256:reimport.blendSha256},savedAtSource:now};
sourceState.currentExports=[['character-rebuilt.blend',reimport.blendSha256],['character-rebuilt-textured.glb',reimport.glbSha256]].map(([name,sha256])=>({path:'../../Character_Rebuild/'+name,bytes:fs.statSync(workspacePath(out,name)).size,sha256}));
sourceState.exportCorrespondence={sourceSha256:reimport.blendSha256,glbSha256:reimport.glbSha256,sourceStage:read(`${out}/reports/final-source-geometry.json`).sourceStage,sourceReport:'../../Character_Rebuild/reports/final-source-geometry.json',reimportReport:'../../Character_Rebuild/reports/clean-reimport.json',visualReport:'../../Character_Rebuild/reports/final-visual-review.json',sameMasterBytesVerified:true};
sourceState.hunyuanStaticRebuild={input:'../../character-hq-textured.glb',delivery:'../../Character_Rebuild/',currentInputSha256:'105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2',statistics:binary.summary,originalAndStagesPreserved:true,noRig:true,previousR11Work:'Stopped and superseded; no R11 geometry reused',appearanceAcceptedByUser:false,remainingLimitations:'Face expression, strand roots/tips and fine garment folds remain simplified relative to the original art.',migrationManifestStatus:'Older migration manifest is historical; not a frozen manifest of this new workspace state',offMachineBackup:false};
fs.writeFileSync(workspacePath(project,'project-state.json'),JSON.stringify(sourceState,null,2)+'\n');
const current='当前制作进度：按用户新提供的 `character-hq-textured.glb` 完成静态动漫重建交付，主源已更新为本轮可编辑工程，原主源先复制并校验保存。交付位于 `Character_Rebuild/`；GLB已通过空场景重导入、贴图与逐网格内部接触检查，无骨骼或动画。面部神态、发束和细衣褶仍有简化，与原画精细程度有差距，未获客户外观验收。旧R11制作保持停止，旧VRM和缓存均为历史。详见 [最新交接](docs/HANDOFF.md)。';
const readme=workspacePath('README.md');fs.writeFileSync(readme,fs.readFileSync(readme,'utf8').replace(/^当前制作进度.*$/m,current));
const handoff=workspacePath('docs/HANDOFF.md');let history=fs.readFileSync(handoff,'utf8');
const heading='## 当前：Hunyuan静态动漫重建交付';
if(!history.includes(heading)){
  const first=history.indexOf('\n');
  history=history.slice(0,first+1)+`\n${heading}\n\n用户随后明确要求改用根目录新GLB重建，不继续旧R11模型。本轮交付目录为 \`Character_Rebuild/\`，其 \`character-rebuilt.blend\` 与 \`projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/character.blend\` 字节一致。原主源已备份于 \`${prior?.backupPath??'work/stage-backups/（见project-state记录）'}\` 并校验。\n\nGLB：\`${reimport.glbSha256}\`；可编辑源：\`${reimport.blendSha256}\`。已实际在空Blender场景重导入，全部${binary.summary.meshes}个网格逐个内部接触检查为0；文件包含${binary.summary.trianglesAcrossMeshNodes}三角、${binary.summary.materials}材质、${binary.summary.images}张内嵌贴图，无骨骼/动画。具体范围与限制见 \`Character_Rebuild/完成说明.txt\` 和对应最终报告。外观仍需艺术精修，不能视为原画完全还原或客户认可；未验证跨部件全局碰撞及动画。\n\n旧R11、旧VRM、旧.charakit应用状态、旧缓存和迁移清单保留作历史，不得据其覆盖这份新工作主源。以下停止记录仍适用于旧R11路线；本轮新GLB是用户另行授权的新任务。没有异机备份声明。\n\n`+history.slice(first+1);
  fs.writeFileSync(handoff,history);
}
const record={recordedAt:now,masterPath:`${project}/work/character.blend`,masterSha256:hash(master),deliverySha256:reimport.blendSha256,byteCopyVerified:true,priorMasterBackup:prior,appearanceAcceptedByUser:false};
fs.writeFileSync(workspacePath(out,'reports/master-adoption.json'),JSON.stringify(record,null,2)+'\n');
console.log(JSON.stringify({master:record.masterPath,sha256:record.masterSha256,priorBackup:prior?.backupPath,byteCopyVerified:true},null,2));

import fs from 'node:fs';
import crypto from 'node:crypto';
import {workspacePath} from '../../pipeline/context.mjs';
const p=x=>workspacePath('Character_Rebuild',x);
const read=x=>JSON.parse(fs.readFileSync(p(x),'utf8'));
const digest=x=>crypto.createHash('sha256').update(fs.readFileSync(p(x))).digest('hex');
const glb=read('reports/final-glb-validation.json'),imp=read('reports/clean-reimport.json'),vis=read('reports/final-visual-review.json'),cam=read('reports/matched-review-settings.json');
if(!glb.passed||!imp.structuralReimportPassed||!vis.reviewCompleted||!cam.passed)throw Error('Final evidence is incomplete');
if(glb.sha256!==digest('character-rebuilt-textured.glb')||glb.sha256!==imp.glbSha256||glb.sha256!==vis.glbSha256||imp.blendSha256!==digest('character-rebuilt.blend'))throw Error('Evidence does not match delivery');
const s=glb.summary;
const contactSummary=imp.objects.map(o=>({name:o.object,...o.selfContacts}));
const contactTotal=contactSummary.reduce((n,o)=>n+(o.contacts??0),0);
const text=`角色重建交付说明

交付状态
已实际导入根目录 character-hq-textured.glb，重建、贴图、导出，并在全新的空Blender场景重新导入交付GLB完成文件与图像检查。此文件是可编辑的静态动漫重建版本。面部神态、发束层次和局部衣褶仍与原画精细程度有差距，不代表原画完全还原或客户外观验收通过。

文件
character-rebuilt.blend：可编辑分件模型、必要修改器、隐藏原始四视图及检查相机；贴图已打包。
character-rebuilt-textured.glb：实际使用贴图内嵌，排除参考面、辅助相机和灯光。
review.html：浏览检查图、同条件对照及重新导入结果。
checks/final：正、侧、背、斜侧全身，脸部多角度，双手掌背与斜侧近景。
checks/before-matched：与修复后使用相同相机、灯光、距离和色彩设置的原模型对照。
checks/reimport：交付GLB在空场景中重新导入的实际检查图。

实际修改范围
1. 重建头部、颈部、眼眶、眼白、虹膜、眼睑边缘、眉与闭口线；重新安排眼口局部布线，修正侧面眼部朝向与鼻部体积。耳内凹腔为新建的简化结构。眼白/虹膜采用有曲率的表面网格，尚非完整可转动眼球；闭口没有牙齿、舌头及完整口腔。没有加入毛孔、写实皮肤噪点或表情系统。
2. 重建发冠、六组刘海、鬓发及双低马尾主要发束，全部为可导出的网格。发冠细流包含新烘焙的2K颜色和切线法线；法线来自真实Blender NORMAL烘焙，不是颜色图改作法线。
3. 替换原粘连双手，建立每手五根独立可辨的手指、基础关节环、掌体和袖内手腕。五指轻微分开，保持原T-pose。掌纹、指甲和肌腱细节仍作简化处理。
4. 身体、主要衬衫体积、裙褶、腿、袜和鞋保留输入模型。后肩/领口、前圆领、袖口内部和纽扣做了局部结构修复；不是全身四边面重拓扑。原三角衣物仍有少量粗糙褶痕。
5. 身体重新排布有效UV并实际烘焙4K色图；去除领口头发串色与错误背门襟，衬衫改为米白固有色。新脸部2K肤色、1K虹膜及发冠贴图独立使用。眼口厚度面、领片和颈部UV已重新展开，修正颈部朝内法线。

最终统计（直接从交付GLB读取）
三角面：${s.trianglesAcrossMeshNodes.toLocaleString('en-US')}；网格对象：${s.meshes}；材质：${s.materials}。
贴图：${glb.images.map(i=>`${i.name} ${i.width}×${i.height}`).join('；')}。
骨骼/蒙皮：${s.skins}；动画：${s.animations}。本次没有制作绑定和表情，不能作为已验证的动画成品。

验证结果
Blender ${imp.blenderVersion} 空场景实际重新导入成功。对象清单、三角面数与源一致，世界包围盒差值为${imp.maxWorldBoundsDifference}；五张图片均能从GLB内嵌数据读取，材料、法线和头发网格存在。实际检查图中双手各五指可辨。
已完成${contactSummary.filter(o=>o.status==='completed').length}/${contactSummary.length}个网格的内部三角接触检查，记录接触对数${contactTotal}（排除相邻面/共享位置，容差1e-8源单位）。此检查不等于跨部件的全局零交叉证明；隐藏的头发发束、领片与躯体装配按分件设计处理。
身体边界为颈领口与双袖口三个既定闭合环；开放衣物和分件接口没有强制焊成一个封闭体。实际使用的贴图UV无退化；身体保留的旧UV仅供来源追溯，其中119个旧退化三角不用于贴图。
对照图统一将原模型缺省金属参数改为非金属，原始GLB文件本身完全未动。检查使用中性面积灯、真实阴影采样、无景深或后期美颜；设置对应报告见 reports/matched-review-settings.json。

参考与推断
指定的IMG_7225–7228未找到。实际使用既有.charakit中恢复的IMG_7223原立绘和IMG_7224原四视图，按鼻尖/胸廓/鞋尖确认正面、面朝画面左侧、背面、面朝画面右侧。参考板为垂臂姿势，只用于身份/体积，模型保留新输入的T-pose。
耳内结构、手掌内侧、拇指根部、头发遮住的头皮、后领与袖口内部属于保守推断补全，不是扫描还原。保留输入数值尺度、glTF常规轴向转换，没有重新标定真人身高。未复用此前被拒绝的R11角色几何。

文件对应与已知限制
GLB SHA-256：${glb.sha256}
BLEND SHA-256：${imp.blendSha256}
原输入SHA-256：105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2
仍需艺术层面的进一步精雕才能更贴近原画，尤其是面部神态、刘海根部、发尾层次及细小衣褶；技术检查通过不等于这一外观目标已完全达到。无动画形变或真实布料/发束物理验证。所有原始文件和中间副本保留在工作区；本地交付包不代表已完成异机备份。
`;
fs.writeFileSync(p('完成说明.txt'),text,'utf8');
console.log('Wrote Character_Rebuild/完成说明.txt against matching final evidence.');

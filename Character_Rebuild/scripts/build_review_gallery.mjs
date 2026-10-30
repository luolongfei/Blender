/**
 * Build a local, dependency-free gallery of actual Blender check renders.
 * Run from the workspace root only after the final delivery files exist:
 *   node Character_Rebuild/scripts/build_review_gallery.mjs
 * Writes only Character_Rebuild/review.html. It never renders or edits a model.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {workspacePath} from '../../pipeline/context.mjs';

const root = workspacePath('Character_Rebuild');
const file = relative => path.join(root, ...relative.split('/'));
const exists = relative => fs.existsSync(file(relative)) && fs.statSync(file(relative)).isFile();
const read = relative => {
  if (!exists(relative)) return null;
  try { return JSON.parse(fs.readFileSync(file(relative), 'utf8')); }
  catch { return null; }
};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const href = relative => relative.split('/').map(encodeURIComponent).join('/');
const hash = relative => crypto.createHash('sha256').update(fs.readFileSync(file(relative))).digest('hex');
const formatNumber = value => Number.isFinite(value) ? value.toLocaleString('zh-CN') : '待记录';

for (const required of ['character-rebuilt.blend', 'character-rebuilt-textured.glb']) {
  if (!exists(required)) {
    console.error(`尚无交付文件 ${required}；未生成或覆盖 review.html。`);
    process.exit(2);
  }
}

const validation = read('reports/final-glb-validation.json');
const reimport = read('reports/clean-reimport.json');
const glbHash = hash('character-rebuilt-textured.glb');
const blendHash = hash('character-rebuilt.blend');
const validationCurrent = validation?.sha256 === glbHash;
const reimportCurrent = reimport?.glbSha256 === glbHash && reimport?.blendSha256 === blendHash;
const stats = validationCurrent ? validation.summary : null;
const status = !validation ? '导出检查待记录' : !validationCurrent ? '检查记录与当前 GLB 不一致' : validation.passed ? 'GLB 结构检查通过' : 'GLB 结构检查存在问题';
const importStatus = !reimport ? '干净场景重新导入：待记录' : !reimportCurrent ? '干净场景重新导入：记录与当前文件不一致' : reimport.structuralReimportPassed ? '干净场景重新导入：结构核对通过' : '干净场景重新导入：核对尚未通过';
const visualStatus = reimportCurrent && reimport.visualReviewCompleted === true ? '重导入画面已记录人工查看；不等于客户外观验收。' : '图片供直接审阅；自动结构检查不代表外观验收。';

const views = [
  ['front','全身正面'], ['right','全身侧面'], ['back','全身背面'], ['quarter','全身斜侧面'],
  ['face-front','脸部正面'], ['face-side','脸部侧面'], ['face-quarter','脸部斜侧面'], ['face-other-side','脸部另一侧'],
  ['face-back','头部背面'], ['face-back-quarter','头部后侧'],
  ['hand-top','左手 · 手背'], ['hand-palm','左手 · 掌侧'], ['hand-oblique','左手 · 斜侧'],
  ['hand-other-top','右手 · 手背'], ['hand-other-palm','右手 · 掌侧'],
];
const labels = new Map(views);
const photo = (relative, caption, eager = false) => `<figure>${exists(relative)
  ? `<a class="photo" href="${href(relative)}" target="_blank" rel="noopener"><img src="${href(relative)}" alt="${esc(caption)}" loading="${eager ? 'eager' : 'lazy'}" decoding="async"><span class="open-hint">打开原图 ↗</span></a>`
  : `<div class="pending-image" role="status">此视角图片待生成</div>`}<figcaption>${esc(caption)}</figcaption></figure>`;
const gallery = directory => views.filter(([name]) => exists(`${directory}/${name}.png`)).map(([name,label], i) => photo(`${directory}/${name}.png`,label,i<2)).join('') || '<p class="notice">这组检查图片尚未生成。</p>';

// Verify the claimed matching conditions from the settings emitted by render_views.
const beforeSettings = read('checks/before-matched/review-settings.json');
const afterSettings = read('checks/final/review-settings.json');
function equivalent(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a-b) <= 1e-6;
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v,i) => equivalent(v,b[i]));
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const keys = [...new Set([...Object.keys(a),...Object.keys(b)])];
    return keys.every(k => equivalent(a[k],b[k]));
  }
  return a === b;
}
const conditions = ['engine','viewTransform','look','exposure','gamma','resolution','depthOfField','areaShadowJitter','samples'];
const sortLights = s => (s?.lights || []).slice().sort((a,b) => a.name.localeCompare(b.name));
const lightingMatches = Boolean(beforeSettings && afterSettings && conditions.every(k => beforeSettings[k] !== undefined && equivalent(beforeSettings[k],afterSettings[k])) && equivalent(sortLights(beforeSettings),sortLights(afterSettings)));
const comparisons = ['front','face-front','face-side','hand-top'].map(name => {
  const beforeView = beforeSettings?.views?.find(v => v.name === name);
  const afterView = afterSettings?.views?.find(v => v.name === name);
  const matches = lightingMatches && beforeView && afterView && equivalent(beforeView,afterView);
  return `<article class="comparison"><div class="comparison-title"><h3>${esc(labels.get(name))}</h3><span class="condition ${matches ? 'matched' : ''}">${matches ? '同机位 · 同光照 · 同距离已核对' : '机位与光照条件待核对'}</span></div><div class="pair">${photo(`checks/before-matched/${name}.png`,'修复前 · 原模型，材质统一为非金属',name === 'front')}${photo(`checks/final/${name}.png`,'本次重建 · 可编辑工程渲染',name === 'front')}</div></article>`;
}).join('');

const imageStats = validationCurrent && Array.isArray(validation?.images) ? validation.images.map(im => `<li>${esc(im.name || `贴图 ${im.index + 1}`)} <strong>${formatNumber(im.width)} × ${formatNumber(im.height)}</strong></li>`).join('') : '<li>贴图尺寸待检查记录确认</li>';
const links = [['character-rebuilt.blend','可编辑 Blender 工程'],['character-rebuilt-textured.glb','内嵌贴图 GLB'],['完成说明.txt','完成说明']].map(([relative,label]) => exists(relative) ? `<a class="download" href="${href(relative)}" download>${esc(label)} <span>↓</span></a>` : `<span class="download disabled">${esc(label)} · 待生成</span>`).join('');
const reference = 'reports/reference-audit-images/IMG_7224-original.png';
const referenceSection = exists(reference) ? `<section id="references" data-panel><div class="section-title"><h2>原始四视图参考</h2><p>此参考图人物手臂下垂；本次模型保留输入 GLB 的 T-pose。</p></div><div class="reference">${photo(reference,'原始参考：正面、面朝画面左、背面、面朝画面右')}</div></section>` : '';
const notes = exists('完成说明.txt') ? `<details class="completion"><summary>展开完成说明</summary><pre>${esc(fs.readFileSync(file('完成说明.txt'),'utf8'))}</pre></details>` : '<p class="notice">完成说明尚未生成，当前页面不表示整项工作已完成。</p>';
const warnings = validationCurrent && Array.isArray(validation.errors) && validation.errors.length ? `<div class="notice error">检查问题：${validation.errors.map(esc).join('；')}</div>` : '';

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>角色重建 · 检查图与交付文件</title>
<style>
:root{font-family:Inter,"Microsoft YaHei","PingFang SC",system-ui,sans-serif;color:#24332b;background:#f5f5ef;line-height:1.6;font-synthesis:none}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:38px 28px 60px}header{display:flex;justify-content:space-between;align-items:start;gap:24px;border-bottom:1px solid #d8ded6;padding-bottom:25px}.eyebrow{font-size:12px;letter-spacing:.14em;color:#68796e;margin:0 0 6px}h1{font-size:clamp(25px,3vw,36px);line-height:1.25;margin:0 0 12px;font-weight:650}h2{font-size:23px;margin:0}h3{font-size:18px;margin:0}p{margin:8px 0;color:#627267}.status{white-space:nowrap;padding:7px 13px;border:1px solid #b9c8bc;border-radius:22px;background:#edf2eb;font-size:13px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.stat{padding:15px 18px;background:#fff;border:1px solid #e0e4da;border-radius:12px}.stat dt{font-size:12px;color:#68796e}.stat dd{margin:2px 0 0;font-size:23px;font-weight:600}.downloads{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.download{display:inline-flex;gap:20px;align-items:center;background:#344e3f;color:white;text-decoration:none;border-radius:8px;padding:10px 16px;font-size:14px}.download span{margin-left:auto}.download:hover{background:#213d2c}.disabled{background:#e3e7df;color:#748074}.metadata{font-size:13px}.metadata code{overflow-wrap:anywhere;color:#425449}.tabs{position:sticky;top:0;z-index:3;background:#f5f5eff5;border-block:1px solid #d8ded6;display:flex;gap:6px;padding:12px 0;margin:27px 0 24px;overflow-x:auto}.tabs button{border:0;background:transparent;font:inherit;white-space:nowrap;padding:8px 16px;border-radius:7px;color:#5e7063;cursor:pointer}.tabs button[aria-selected=true]{background:#344e3f;color:#fff}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #b2884b;outline-offset:3px}[hidden]{display:none!important}.section-title{margin-bottom:20px}.comparison{margin-bottom:24px;background:#fff;border:1px solid #dbe1d6;border-radius:12px;overflow:hidden}.comparison-title{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 20px}.condition{font-size:12px;color:#8c6b36}.matched{color:#5d7564}.pair,.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.pair{gap:1px;background:#dfe4da}.pair figure{background:white}figure{margin:0;min-width:0}.grid figure{background:white;border:1px solid #dbe1d6;border-radius:10px;overflow:hidden}.photo{position:relative;display:block;background:#efeee7;line-height:0;overflow:hidden}.photo img{display:block;width:100%;aspect-ratio:1;object-fit:contain}.open-hint{position:absolute;right:12px;bottom:12px;line-height:1.5;padding:5px 9px;border-radius:5px;background:#ffffffe0;color:#405946;font-size:11px;opacity:0;transition:opacity .15s}.photo:hover .open-hint,.photo:focus-visible .open-hint{opacity:1}figcaption{padding:11px 16px;color:#66756b;font-size:13px}.pending-image{display:grid;place-items:center;aspect-ratio:1;background:repeating-linear-gradient(135deg,#f2f3ee,#f2f3ee 12px,#ebeee7 12px,#ebeee7 24px);color:#839080;font-size:14px}.notice{padding:12px 15px;border-left:3px solid #b1bcaa;background:#ecefe6;color:#5c6e60;font-size:13px;margin:16px 0}.error{border-color:#b88361;background:#fbf0e6}.reference{max-width:1000px}.reference img{aspect-ratio:auto}.completion{border-top:1px solid #d8ded6;margin-top:28px;padding-top:16px}.completion summary{cursor:pointer;font-weight:600}.completion pre{font-family:inherit;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;color:#5c6e60;max-width:1000px}.texture-details{font-size:13px;margin-top:15px}.texture-details summary{cursor:pointer}.texture-details ul{padding-left:20px}.texture-details li{margin:5px 0}.texture-details strong{margin-left:12px;font-weight:500}.footer{border-top:1px solid #d8ded6;margin-top:35px;padding-top:15px;color:#748074;font-size:12px}section{scroll-margin-top:90px;margin-bottom:30px}@media(min-width:1100px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){main{padding:24px 14px 40px}header{display:block}.status{display:inline-block;margin-top:12px}.stats{grid-template-columns:repeat(2,1fr)}.stat dd{font-size:21px}.comparison-title{display:block;padding:12px}.condition{display:block;margin-top:4px}.pair,.grid{gap:1px}.grid{gap:10px;grid-template-columns:1fr}.pair figcaption{font-size:11px;padding:8px}.tabs button{padding:7px 12px}.downloads{gap:7px}.download{font-size:12px;padding:9px 11px;gap:8px}.photo .open-hint{display:none}}
</style>
</head>
<body><main>
<header><div><p class="eyebrow">CHARACTER REBUILD / LOCAL REVIEW</p><h1>角色重建 · 检查图与交付文件</h1><p>基于 character-hq-textured.glb 的静态动漫角色；保留双低马尾、服装设计与 T-pose。</p></div><span class="status">${esc(status)}</span></header>
<dl class="stats"><div class="stat"><dt>导出三角面</dt><dd>${formatNumber(stats?.trianglesAcrossMeshNodes)}</dd></div><div class="stat"><dt>材质</dt><dd>${formatNumber(stats?.materials)}</dd></div><div class="stat"><dt>内嵌贴图</dt><dd>${formatNumber(stats?.images)}</dd></div><div class="stat"><dt>骨骼</dt><dd>${stats ? stats.containsSkeleton ? '包含' : '无' : '待检查'}</dd></div></dl>
<div class="downloads">${links}</div>
<p class="metadata">${esc(importStatus)}。${esc(visualStatus)}</p>
<p class="metadata">GLB 文件标识：<code>${esc(glbHash)}</code></p>
<details class="texture-details"><summary>查看实际贴图尺寸</summary><ul>${imageStats}</ul></details>
${warnings}
<nav class="tabs" aria-label="检查图分类"><button type="button" data-show="comparison" aria-selected="true">前后对比</button><button type="button" data-show="final" aria-selected="false">工程检查图</button><button type="button" data-show="reimport" aria-selected="false">GLB 重导入</button>${referenceSection ? '<button type="button" data-show="references" aria-selected="false">原始参考</button>' : ''}</nav>
<section id="comparison" data-panel><div class="section-title"><h2>同条件前后对比</h2><p>左侧保留输入模型和原贴图，仅将原材质统一为非金属；右侧是本次重建。每组按保存的相机、光照、曝光与距离设置核对。</p><p>所有图片均为实际 Blender 渲染，点击可以查看原图。</p></div>${comparisons}</section>
<section id="final" data-panel><div class="section-title"><h2>可编辑工程检查图</h2><p>全身、脸部与双手细节。使用中性检查光照，无景深遮挡。</p></div><div class="grid">${gallery('checks/final')}</div></section>
<section id="reimport" data-panel><div class="section-title"><h2>GLB 干净场景重新导入</h2><p>导出模型在空场景中重新导入后拍摄，检查贴图、头发和双手是否保留。</p></div><div class="grid">${gallery('checks/reimport')}</div></section>
${referenceSection}
${notes}
<p class="footer">此页为本地检查图册，不是实时三维查看器。交付文件与图片均使用相对路径；请保留 Character_Rebuild 文件夹结构。制作范围、推断补全与已知问题以完成说明为准。</p>
</main><script>
const buttons = [...document.querySelectorAll('[data-show]')];
const panels = [...document.querySelectorAll('[data-panel]')];
function showPanel(id) {
  if (!panels.some(p => p.id === id)) id = 'comparison';
  for (const panel of panels) panel.hidden = panel.id !== id;
  for (const button of buttons) button.setAttribute('aria-selected', String(button.dataset.show === id));
}
for (const button of buttons) button.addEventListener('click', () => { showPanel(button.dataset.show); history.replaceState(null,'','#'+button.dataset.show); });
showPanel(location.hash.slice(1));
</script></body></html>`;

fs.writeFileSync(file('review.html'), html, 'utf8');
console.log(JSON.stringify({output:'Character_Rebuild/review.html',glbSha256:glbHash,validationCurrent,reimportCurrent,status,lightingMatches,finalImageCount:views.filter(([name])=>exists(`checks/final/${name}.png`)).length,reimportImageCount:views.filter(([name])=>exists(`checks/reimport/${name}.png`)).length},null,2));

import * as THREE from 'three';
import {OrbitControls} from '/public/vendor/OrbitControls.js';
import {GLTFLoader} from '/public/vendor/GLTFLoader.js';

const $=id=>document.getElementById(id);
const projectURL=new URL('../',import.meta.url);
const delivered=new URL(location.href).searchParams.get('version')==='delivery';
const deliveredSource='8fd87255aa7576eeaa486cad396773350b608b27b5e57914332165afa67eb2d9';
const manifestURL=new URL(delivered?'../exports/web-preview-stopped-8fd87255/preview-manifest.json':'../exports/web-preview-r11/preview-manifest.json',import.meta.url);
if(delivered){
  document.title='R11 · 历史停止交接预览';
  $('title').textContent='R11 · 历史停止交接预览';
  $('subtitle').textContent='2026-10-29 历史停止交接 · 不对应当前重建';
  $('badge').textContent='历史停止交接版本';
  $('loading-title').textContent='正在读取交付版本';
  $('refresh-note').textContent='此页面固定对应交付包中的可编辑模型。';
}
const viewport=$('viewport');
let renderer,camera,orbit,model,frameRequest,resizeObserver;
let bounds,ready=false,crop='head',view='oblique';
const expressionControls=[];
const morphTargets=new Map();
const scene=new THREE.Scene();
scene.background=new THREE.Color('#f3f3ef');

function projectAsset(path){
  const url=new URL(path,projectURL);
  if(url.origin!==projectURL.origin||!url.pathname.startsWith(projectURL.pathname))throw Error('预览文件必须位于当前本地角色目录内。');
  return url;
}
function requestFrame(){
  if(!frameRequest)frameRequest=requestAnimationFrame(draw);
}
function draw(){
  frameRequest=undefined;
  const moving=orbit?.update();
  renderer.render(scene,camera);
  if(moving)requestFrame();
}
function resize(){
  const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);
  renderer.setSize(width,height,false);
  camera.aspect=width/height;
  camera.updateProjectionMatrix();
  requestFrame();
}
function frameCamera(){
  if(!bounds)return;
  const center=bounds.getCenter(new THREE.Vector3());
  const size=bounds.getSize(new THREE.Vector3());
  const isHead=crop==='head';
  const aim=new THREE.Vector3(center.x,isHead?bounds.min.y+size.y*.855:center.y,center.z);
  const subjectHeight=size.y*(isHead?.34:1.08);
  const subjectWidth=isHead?Math.min(size.x,size.y*.40):size.x*1.1;
  const halfFov=Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  const distance=Math.max(subjectHeight/(2*halfFov),subjectWidth/(2*halfFov*camera.aspect));
  const angle=({front:0,oblique:.33,side:Math.PI/2,back:Math.PI,'other-side':-Math.PI/2})[view]??0;
  orbit.target.copy(aim);
  camera.position.set(aim.x+Math.sin(angle)*distance,aim.y+distance*.025,aim.z+Math.cos(angle)*distance);
  camera.near=Math.max(.001,size.y*.0005);camera.far=Math.max(50,size.y*30);camera.updateProjectionMatrix();
  orbit.minDistance=Math.max(.03,size.y*.07);orbit.maxDistance=Math.max(5,size.y*5);
  orbit.update();requestFrame();
  document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===view)));
  document.querySelectorAll('[data-crop]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.crop===crop)));
}
function setExpressionValues(){
  for(const entries of morphTargets.values())for(const target of entries)target.object.morphTargetInfluences[target.index]=0;
  function apply(bindings,weight){
    for(const binding of bindings)for(const target of morphTargets.get(binding.name)??[])target.object.morphTargetInfluences[target.index]+=weight*(binding.weight??1);
  }
  for(const control of expressionControls){
    const value=Number(control.input.value);
    if(control.samples){
      const samples=control.samples;
      if(value<=samples[0].value)apply(samples[0].targets,1);
      else if(value>=samples.at(-1).value)apply(samples.at(-1).targets,1);
      else{
        const upperIndex=samples.findIndex(sample=>sample.value>=value);
        const lower=samples[upperIndex-1],upper=samples[upperIndex];
        const mix=(value-lower.value)/(upper.value-lower.value);
        apply(lower.targets,1-mix);apply(upper.targets,mix);
      }
    }else{
      apply(control.targets,value);
    }
  }
  const active=expressionControls.find(control=>Number(control.input.value)>0);
  if(ready)$('status').textContent=active?`${active.label} · ${Math.round(Number(active.input.value)*100)}%`:'预览已就绪 · 中性';
  requestFrame();
}
function addExpressionControls(manifest){
  // Only expose intentional public controls, never proportion keys or internal blink correctives.
  const publicNames={'R11 gentle smile':'小笑','R11 natural soft smile':'自然微笑','R11 warm closed smile':'温柔闭口笑','R11 blink L':'左眼闭合','R11 blink R':'右眼闭合'};
  const entries=Array.isArray(manifest.expressions)?manifest.expressions:Object.entries(publicNames).filter(([name])=>morphTargets.has(name)).map(([name,label])=>({id:name,label,targets:[{name}],exclusiveGroup:name.includes('smile')?'smile':undefined}));
  for(const [index,item] of entries.entries()){
    const samples=Array.isArray(item.samples)?item.samples.map(sample=>({value:Number(sample.value),targets:sample.targets??[]})).sort((a,b)=>a.value-b.value):null;
    if(samples&&(!samples.length||samples.some((sample,i)=>!Number.isFinite(sample.value)||sample.value<0||sample.value>1||(i&&sample.value===samples[i-1].value))))throw Error('表情采样记录无效，需重新导出。');
    const targets=samples?samples.flatMap(sample=>sample.targets):(item.targets??[{name:item.name??item.id}]);
    if(!targets.length)continue;
    if(targets.some(target=>!morphTargets.has(target.name)))throw Error('模型缺少版本记录中的表情：'+(item.label??item.id));
    const wrapper=document.createElement('div');wrapper.className='expression';
    const label=document.createElement('label'),name=document.createElement('span'),output=document.createElement('output'),input=document.createElement('input');
    name.textContent=item.label??item.name??item.id;output.value='0%';
    input.type='range';input.min='0';input.max='1';input.step='.01';input.value='0';input.id='expression-'+index;
    label.htmlFor=input.id;output.htmlFor=input.id;label.append(name,output);wrapper.append(label,input);$('expressions').append(wrapper);
    const control={input,output,targets,samples,label:name.textContent,group:item.exclusiveGroup};expressionControls.push(control);
    input.addEventListener('input',()=>{
      if(control.group&&Number(input.value)>0)for(const other of expressionControls)if(other!==control&&other.group===control.group){other.input.value='0';other.output.value='0%';}
      output.value=Math.round(Number(input.value)*100)+'%';setExpressionValues();
    });
  }
  $('expression-note').textContent=expressionControls.length?(expressionControls.some(control=>control.samples)?'表情取自 Blender 的实际采样，采样点之间插值。每次查看一种表情。':'拖动滑块查看表情变化。组合效果仍需评估。'):'这份快照没有导出可调表情，可先检查静态外观。';
  $('neutral').disabled=!expressionControls.length;
  $('neutral').addEventListener('click',()=>{for(const control of expressionControls){control.input.value='0';control.output.value='0%';}setExpressionValues();});
  setExpressionValues();
}
async function loadBytes(url){
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw Error('模型读取失败（'+response.status+'）。');
  const total=Number(response.headers.get('content-length'))||0;
  if(!response.body)return response.arrayBuffer();
  const reader=response.body.getReader(),chunks=[];let received=0;
  for(;;){
    const {done,value}=await reader.read();if(done)break;
    chunks.push(value);received+=value.byteLength;
    if(total)$('progress').value=received/total;
    $('loading-detail').textContent=total?`已读取 ${(received/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`:`已读取 ${(received/1048576).toFixed(1)} MB`;
  }
  const data=new Uint8Array(received);let offset=0;
  for(const chunk of chunks){data.set(chunk,offset);offset+=chunk.byteLength;}
  return data.buffer;
}

try{
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.5));
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
  renderer.domElement.setAttribute('aria-label','历史 R11 模型，可拖动旋转和滚轮缩放');
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();$('status').textContent='显示已中断';$('loading-title').textContent='显卡暂时无法继续显示';$('loading-detail').textContent='请关闭其他占用较高的视图后重新读取预览。';$('progress').hidden=true;$('retry').hidden=false;$('overlay').hidden=false;});
  viewport.prepend(renderer.domElement);
  camera=new THREE.PerspectiveCamera(32,1,.001,50);camera.position.set(0,1,3);
  orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.dampingFactor=.12;
  orbit.addEventListener('change',requestFrame);
  scene.add(new THREE.HemisphereLight(0xffffff,0xb8beb7,1.05));
  const key=new THREE.DirectionalLight(0xfffcf8,1.30);key.position.set(-3,4,5);scene.add(key);
  const fill=new THREE.DirectionalLight(0xf4f7ff,.55);fill.position.set(3,2,2);scene.add(fill);
  const rim=new THREE.DirectionalLight(0xffffff,.7);rim.position.set(1,3,-3);scene.add(rim);
  resizeObserver=new ResizeObserver(resize);resizeObserver.observe(viewport);resize();
  const response=await fetch(manifestURL,{cache:'no-store'});
  if(!response.ok)throw Error('版本记录尚未就绪（'+response.status+'）。');
  const manifest=await response.json();
  if(delivered&&manifest.source?.sha256!==deliveredSource)throw Error('网页模型与交付版本不一致，已停止载入。');
  const modelURL=projectAsset(manifest.export?.path??'exports/web-preview-r11/avatar-preview.glb');
  if(!modelURL.pathname.toLowerCase().endsWith('.glb'))throw Error('这份预览需要 GLB 模型。');
  $('title').textContent=delivered?'R11 · 历史停止交接预览':'R11 · 历史制作预览';
  $('status').textContent='正在读取模型';
  const bytes=await loadBytes(modelURL);
  $('loading-title').textContent='正在准备模型画面';$('loading-detail').textContent='解析几何、贴图与已导出的表情，首次载入可能需要一些时间。';$('progress').removeAttribute('value');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(value=>value.toString(16).padStart(2,'0')).join('');
  if(manifest.export?.sha256&&hash!==manifest.export.sha256.toLowerCase())throw Error('模型文件与版本记录不一致，请重新读取或等待导出完成。');
  const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(!url.startsWith('blob:')&&!url.startsWith('data:'))throw Error('预览包含未内嵌的资源，需重新完整导出。');return url;});
  const gltf=await new GLTFLoader(manager).parseAsync(bytes,'');model=gltf.scene;scene.add(model);
  let meshes=0,triangles=0;
  model.traverse(object=>{
    if(object.isMesh){meshes++;triangles+=(object.geometry.index?.count??object.geometry.attributes.position?.count??0)/3;}
    if(object.morphTargetDictionary&&object.morphTargetInfluences)for(const [name,index] of Object.entries(object.morphTargetDictionary)){
      if(!morphTargets.has(name))morphTargets.set(name,[]);morphTargets.get(name).push({object,index});
    }
  });
  model.updateMatrixWorld(true);bounds=new THREE.Box3().setFromObject(model);
  if(bounds.isEmpty()||!Number.isFinite(bounds.max.y))throw Error('模型没有可显示的有效几何。');
  const size=bounds.getSize(new THREE.Vector3());
  const ground=new THREE.Mesh(new THREE.CircleGeometry(Math.max(3,size.y*3),64),new THREE.MeshBasicMaterial({color:0xe9ece5}));
  ground.rotation.x=-Math.PI/2;ground.position.set(bounds.getCenter(new THREE.Vector3()).x,bounds.min.y-.003,bounds.getCenter(new THREE.Vector3()).z);scene.add(ground);
  addExpressionControls(manifest);
  $('limits').replaceChildren();
  const limits=Array.isArray(manifest.limitations)?manifest.limitations:['制作中的阶段快照，尚未作为正式交付。','网页灯光与 Blender 渲染不同，适合交互检查。'];
  for(const text of limits){const li=document.createElement('li');li.textContent=String(text);$('limits').append(li);}
  const attributions=Array.isArray(manifest.attribution)?manifest.attribution:[];
  for(const item of attributions){
    const li=document.createElement('li'),title=document.createElement('strong'),description=document.createElement('div');
    title.textContent=String(item.asset??'素材');
    description.textContent=[item.author?`作者：${item.author}`:null,item.license?`许可：${item.license}`:null,item.note??null].filter(Boolean).join('；');
    li.append(title,description);
    const source=typeof item.source==='object'?item.source?.url:item.source;
    if(source){
      try{
        const url=new URL(source);
        if(['http:','https:'].includes(url.protocol)){
          const link=document.createElement('a');link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';link.textContent=item.source?.title??'查看原始来源';li.append(link);
        }
      }catch{/* Keep the supplied attribution text even if its optional link is invalid. */}
    }
    $('attributions').append(li);
  }
  $('attribution-section').hidden=!attributions.length;
  const sourceName=manifest.source?.path?.split(/[\\/]/).at(-1)??'未记录';
  $('version-info').textContent=`来源：${sourceName}。源版本：${manifest.source?.sha256?.slice(0,12)??'未记录'}。预览版本：${hash.slice(0,12)}。`;
  $('version-info').dataset.sourceSha256=manifest.source?.sha256??'';$('version-info').dataset.sha256=hash;
  $('badge').textContent=delivered?'历史停止交接版本':`R11 历史预览 · ${manifest.source?.sha256?.slice(0,8)??hash.slice(0,8)}`;
  $('download').href=modelURL.href;$('download').hidden=false;
  document.querySelectorAll('[data-view],[data-crop],#reset-view').forEach(button=>button.disabled=false);
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.view;frameCamera();}));
  document.querySelectorAll('[data-crop]').forEach(button=>button.addEventListener('click',()=>{crop=button.dataset.crop;frameCamera();}));
  $('reset-view').addEventListener('click',()=>{crop='head';view='oblique';frameCamera();});
  ready=true;frameCamera();renderer.render(scene,camera);
  $('overlay').hidden=true;$('status').textContent='预览已就绪 · 中性';
  viewport.dataset.ready='true';viewport.dataset.meshes=String(meshes);viewport.dataset.triangles=String(Math.round(triangles));viewport.dataset.expressions=String(expressionControls.length);
}catch(error){
  console.error(error);
  $('status').textContent='暂时无法显示';$('loading-title').textContent='预览未能载入';
  $('loading-detail').textContent=error.message?.includes('WebGL')?'当前浏览器无法启用 WebGL，请尝试支持硬件加速的浏览器。':error.message;
  $('progress').hidden=true;$('retry').hidden=false;$('overlay').hidden=false;
}
$('reload').addEventListener('click',()=>location.reload());$('retry').addEventListener('click',()=>location.reload());
window.addEventListener('pagehide',()=>{if(frameRequest)cancelAnimationFrame(frameRequest);resizeObserver?.disconnect();orbit?.dispose();renderer?.dispose();});

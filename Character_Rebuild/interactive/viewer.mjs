import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {GLTFLoader} from '/vendor/GLTFLoader.js';

const $=id=>document.getElementById(id);
const viewport=$('viewport');
const directory=new URL('./',import.meta.url);
const scene=new THREE.Scene();scene.background=new THREE.Color('#f1f4ee');
let renderer,camera,orbit,model,mixer,observer,frameRequest,lastTime=0,lastStatus=0;
let ready=false,playing=false,seeking=false,resumeAfterSeek=false,ended=false;
let current=null,currentEntry=null,bindPoseAction=null,crop='body',view='front',bounds,manifest;
let meshCount=0,skinCount=0,boneCount=0,frameCount=0,modelHash='';
const entries=[],actions=new Map(),skeletons=new Set(),initialTransforms=new Map();
const alias={idle:'自然站立',wave:'招手',look:'转头',walk:'原地轻踏',tpose:'T 形姿势'};
const iconPaths={
  idle:'M12 3.8a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM8.5 11l-1 5M15.5 11l1 5M9 10h6l-1 5 1 6M10 15l-1 6',
  wave:'M11 3.8a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM8 11l-1 5M9 10h5l3-3 1-4M10 15l-1 6M13 15l1 6M20 3l1 2M20 7l1-1',
  look:'M9 4a6 6 0 1 0 7 10M13 4c4 0 6 3 6 6M16 8l3 2 2-3M8 15v5M8 20h8M12 10h.01',
  walk:'M14 3a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM12 9l-2 5 4 3 1 4M10 14l-3 6M11 11l-4 1M13 9l4 3 3-1',
  other:'M8 5l11 7-11 7Z'
};

function requestFrame(){if(!frameRequest)frameRequest=requestAnimationFrame(draw);}
function isFiniteBounds(box){return !box.isEmpty()&&[...box.min,...box.max].every(Number.isFinite);}
function signalState(){
  if(!ready)return;
  const duration=currentEntry?.duration??0,time=current?.time??0;
  const json={ready:true,playing,currentAction:currentEntry?.id??'tpose',clip:currentEntry?.clip??null,time,duration,speed:Number($('speed').value),meshes:meshCount,skinnedMeshes:skinCount,bones:boneCount,modelSha256:modelHash,frameCount,camera:camera.position.toArray(),target:orbit.target.toArray(),modelRotation:model.rotation.toArray().slice(0,3),modelTranslation:model.position.toArray()};
  const hand=model.getObjectByName('Rebuilt_Hand_R')??model.getObjectByName('Rebuilt_Hand_L');
  if(hand?.isSkinnedMesh&&hand.geometry.attributes.position?.count){const point=new THREE.Vector3();hand.getVertexPosition(0,point);point.applyMatrix4(hand.matrixWorld);json.skinnedHandVertex=point.toArray();}
  $('runtime-state').textContent=JSON.stringify(json);
  viewport.dataset.playing=String(playing);viewport.dataset.action=json.currentAction;viewport.dataset.time=time.toFixed(4);viewport.dataset.frame=String(frameCount);
  if(!seeking)$('timeline').value=duration?String(Math.round(Math.min(1,time/duration)*1000)):'0';
  const clock=t=>`${Math.floor(Math.max(0,t)/60)}:${String(Math.floor(Math.max(0,t)%60)).padStart(2,'0')}`;
  $('time-label').value=`${clock(time)} / ${clock(duration)}`;
}
function draw(now){
  frameRequest=undefined;const delta=lastTime?Math.min((now-lastTime)/1000,.05):0;lastTime=now;
  if(playing&&!seeking&&mixer)mixer.update(delta);
  const moving=orbit?.update();
  renderer.render(scene,camera);frameCount++;
  if(now-lastStatus>100){lastStatus=now;signalState();}
  if(playing||moving)requestFrame();else lastTime=0;
}
function resize(){
  const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);
  renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();requestFrame();
}
function frameCamera(reset=false){
  if(!bounds)return;
  if(reset){crop='body';view='front';}
  const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());
  const head=crop==='head';
  const aim=new THREE.Vector3(center.x,head?bounds.min.y+size.y*.875:center.y,center.z);
  const height=size.y*(head?.34:1.10);
  const width=head?size.y*.36:(currentEntry?Math.min(size.x,size.y*.70):size.x)*1.08;
  const fov=Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  const distance=Math.max(height/(2*fov),width/(2*fov*camera.aspect));
  const angle=({front:0,oblique:.40,side:Math.PI/2,back:Math.PI})[view]??0;
  orbit.target.copy(aim);camera.position.set(aim.x+Math.sin(angle)*distance,aim.y+distance*.015,aim.z+Math.cos(angle)*distance);
  camera.near=Math.max(.001,size.y*.0005);camera.far=Math.max(30,size.y*30);camera.updateProjectionMatrix();
  orbit.minDistance=size.y*.11;orbit.maxDistance=size.y*6;orbit.update();
  document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===view)));
  document.querySelectorAll('[data-crop]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.crop===crop)));
  requestFrame();signalState();
}
function setPlaying(value){
  if(!current)return;
  if(value&&ended){current.reset().play();ended=false;}
  playing=value;current.paused=!value;
  $('pause-symbol').toggleAttribute('hidden',!value);$('play-symbol').toggleAttribute('hidden',value);
  $('play-pause').setAttribute('aria-label',value?'暂停动作':ended?'重新播放动作':'继续播放动作');
  $('status').textContent=`${currentEntry.label} · ${value?'播放中':ended?'已结束':'已暂停'}`;
  lastTime=0;requestFrame();signalState();
}
function selectAction(entry,{restart=false}={}){
  if(!ready)return;
  if(entry===currentEntry&&!restart){setPlaying(!playing);return;}
  const next=actions.get(entry.id);if(!next)return;
  const previous=current;
  if(!previous)mixer.stopAllAction();
  for(const action of actions.values())if(action!==previous&&action!==next)action.stop();
  next.reset();next.enabled=true;next.clampWhenFinished=true;
  next.setLoop(entry.loop?THREE.LoopRepeat:THREE.LoopOnce,entry.loop?Infinity:1);
  next.setEffectiveTimeScale(1).setEffectiveWeight(1).play();next.paused=false;
  if(previous&&previous!==next){previous.paused=false;previous.crossFadeTo(next,.30,false);}
  else if(!previous)next.fadeIn(.20);
  current=next;currentEntry=entry;ended=false;playing=true;
  $('active-motion').textContent=entry.label;
  $('timeline').disabled=false;$('play-pause').disabled=false;$('speed').disabled=false;
  document.querySelectorAll('[data-action]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.action===entry.id)));
  $('tpose').setAttribute('aria-pressed','false');
  mixer.update(0);model.updateMatrixWorld(true);setPlaying(true);
}
function restorePose(){
  if(!ready)return;
  mixer?.stopAllAction();playing=false;current=null;currentEntry=null;ended=false;
  // Stopping time alone would preserve the last pose. Restore the actual bind
  // transforms, including unkeyed bones, rather than inventing a T pose.
  for(const [object,state] of initialTransforms){object.position.copy(state.position);object.quaternion.copy(state.quaternion);object.scale.copy(state.scale);}
  model.updateMatrixWorld(true);
  for(const skeleton of skeletons)skeleton.pose();
  if(bindPoseAction){bindPoseAction.reset().setLoop(THREE.LoopOnce,1).setEffectiveWeight(1).play();bindPoseAction.clampWhenFinished=true;mixer.update(0);bindPoseAction.paused=true;}
  model.updateMatrixWorld(true);
  $('active-motion').textContent='T 形姿势';$('status').textContent='初始姿势 · 可自由观察';
  $('play-pause').disabled=true;$('timeline').disabled=true;$('speed').disabled=true;
  $('pause-symbol').toggleAttribute('hidden',true);$('play-symbol').toggleAttribute('hidden',false);$('timeline').value='0';
  document.querySelectorAll('[data-action]').forEach(button=>button.setAttribute('aria-pressed','false'));
  $('tpose').setAttribute('aria-pressed','true');frameCamera();signalState();requestFrame();
}
function localURL(relative){
  const url=new URL(relative,directory);
  if(url.origin!==directory.origin||!url.pathname.startsWith(directory.pathname))throw Error('动作文件不在本次预览目录中。');
  return url;
}
async function loadBytes(url){
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw Error(response.status===404?'动作模型还在准备中，请稍后重新载入。':`动作模型读取失败（${response.status}）。`);
  const total=Number(response.headers.get('content-length'))||0;
  if(!response.body)return response.arrayBuffer();
  const reader=response.body.getReader(),chunks=[];let received=0;
  for(;;){const {value,done}=await reader.read();if(done)break;chunks.push(value);received+=value.byteLength;
    if(total)$('progress').value=received/total;
    $('loading-detail').textContent=total?`已读取 ${(received/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`:`已读取 ${(received/1048576).toFixed(1)} MB`;
  }
  const bytes=new Uint8Array(received);let offset=0;for(const part of chunks){bytes.set(part,offset);offset+=part.byteLength;}return bytes.buffer;
}
function makeActionEntries(clips){
  const declared=Array.isArray(manifest.actions)?manifest.actions:null;
  const desired=declared??clips.map(clip=>{const id=['idle','wave','look','walk','tpose'].find(id=>new RegExp(id,'i').test(clip.name))??clip.name;return{id,clip:clip.name,label:alias[id]??clip.name,loop:true};});
  for(const item of desired){
    const clip=clips.find(clip=>clip.name===(item.clip??item.name??item.id));
    if(!clip||!Number.isFinite(clip.duration)||clip.duration<=0||!clip.tracks.length)continue;
    const id=String(item.id??clip.name);if(id.toLowerCase()==='tpose'||clip.name.toLowerCase()==='tpose'||entries.some(entry=>entry.id===id))continue;
    const entry={id,clip:clip.name,label:String(item.label??alias[id]??clip.name),duration:clip.duration,loop:item.loop!==false&&item.loop!=='once'};
    entries.push(entry);actions.set(id,mixer.clipAction(clip));
    const button=document.createElement('button');button.className='action-button';button.dataset.action=id;button.setAttribute('aria-pressed','false');
    const icon=document.createElement('span');icon.className='action-icon';icon.setAttribute('aria-hidden','true');
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');const line=document.createElementNS('http://www.w3.org/2000/svg','path');line.setAttribute('d',iconPaths[id]??iconPaths.other);svg.append(line);icon.append(svg);
    const label=document.createElement('span');label.textContent=entry.label;button.append(icon,label);button.addEventListener('click',()=>selectAction(entry));$('actions').append(button);
  }
}
function fail(error){
  console.error(error);playing=false;ready=false;viewport.dataset.ready='false';viewport.dataset.error=error.message??String(error);
  $('status').textContent='暂时无法显示';$('loading-title').textContent='预览尚未就绪';
  $('loading-detail').textContent=/WebGL|context/i.test(error.message??'')?'浏览器暂时无法显示三维画面，请启用硬件加速后重新载入。':error.message??'请重新载入预览。';
  $('loading-symbol').style.animation='none';$('progress').hidden=true;$('retry').hidden=false;$('overlay').hidden=false;
}

$('retry').addEventListener('click',()=>location.reload());
$('play-pause').addEventListener('click',()=>setPlaying(!playing));
$('reset-idle').addEventListener('click',()=>{const idle=entries.find(entry=>entry.id==='idle')??entries[0];if(idle)selectAction(idle,{restart:true});else restorePose();});
$('tpose').addEventListener('click',restorePose);
$('speed').addEventListener('input',()=>{const speed=Number($('speed').value);$('speed-value').value=`${Number.isInteger(speed)?speed.toFixed(1):String(Number(speed.toFixed(2)))}×`;if(mixer)mixer.timeScale=speed;signalState();});
$('reset-camera').addEventListener('click',()=>frameCamera(true));
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.view;frameCamera();}));
document.querySelectorAll('[data-crop]').forEach(button=>button.addEventListener('click',()=>{crop=button.dataset.crop;frameCamera();}));
$('timeline').addEventListener('pointerdown',()=>{if(!current)return;seeking=true;resumeAfterSeek=playing;setPlaying(false);});
$('timeline').addEventListener('input',()=>{if(!current)return;for(const action of actions.values())if(action!==current)action.stop();current.stopFading();current.setEffectiveWeight(1);current.time=(Number($('timeline').value)/1000)*currentEntry.duration;ended=false;mixer.update(0);model.updateMatrixWorld(true);signalState();requestFrame();});
window.addEventListener('pointerup',()=>{if(!seeking)return;seeking=false;if(resumeAfterSeek)setPlaying(true);else signalState();});
window.addEventListener('keydown',event=>{if(event.code==='Space'&&event.target===renderer?.domElement&&current){event.preventDefault();setPlaying(!playing);}});
document.addEventListener('visibilitychange',()=>{lastTime=0;if(!document.hidden)requestFrame();});

try{
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NeutralToneMapping;renderer.toneMappingExposure=1.12;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  const canvas=renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('aria-label','角色三维视图。拖动旋转，右键平移，滚轮缩放。');
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();fail(Error('WebGL context lost'));});viewport.prepend(canvas);
  camera=new THREE.PerspectiveCamera(32,1,.001,60);camera.position.set(0,0,4);
  orbit=new OrbitControls(camera,canvas);orbit.enableDamping=true;orbit.dampingFactor=.12;orbit.screenSpacePanning=true;orbit.maxPolarAngle=Math.PI*.97;orbit.minPolarAngle=Math.PI*.03;orbit.listenToKeyEvents(canvas);
  orbit.addEventListener('change',requestFrame);
  scene.add(new THREE.HemisphereLight(0xffffff,0xc0c8bb,1.12));
  const key=new THREE.DirectionalLight(0xfffcf7,2.0);key.position.set(-3,4,5);key.castShadow=true;key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-2;key.shadow.camera.right=2;key.shadow.camera.top=2;key.shadow.camera.bottom=-2;key.shadow.camera.near=.1;key.shadow.camera.far=12;key.shadow.normalBias=.015;key.shadow.bias=-.00015;key.shadow.radius=3;scene.add(key);
  const fill=new THREE.DirectionalLight(0xf4f7ff,.60);fill.position.set(3,1,3);scene.add(fill);
  const back=new THREE.DirectionalLight(0xffffff,.65);back.position.set(1,3,-4);scene.add(back);
  observer=new ResizeObserver(resize);observer.observe(viewport);resize();
  const response=await fetch(localURL('preview-manifest.json'),{cache:'no-store'});
  if(!response.ok)throw Error(response.status===404?'动作预览正在制作中，请稍后点击“重新载入”。':`预览记录读取失败（${response.status}）。`);
  manifest=await response.json();
  const modelURL=localURL(manifest.model?.url??manifest.model?.path??manifest.export?.url??manifest.export?.path??'animated-preview.glb');
  const bytes=await loadBytes(modelURL);$('loading-title').textContent='正在准备角色与动作';$('loading-detail').textContent='整理贴图和骨骼，首次载入可能需要片刻。';$('progress').removeAttribute('value');
  modelHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(byte=>byte.toString(16).padStart(2,'0')).join('');
  const expected=manifest.model?.sha256??manifest.export?.sha256??manifest.glbSha256;
  if(expected&&expected.toLowerCase()!==modelHash)throw Error('动作模型与预览记录不一致，请重新载入。');
  const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(!url.startsWith('blob:')&&!url.startsWith('data:'))throw Error('动作模型引用了外部资源，需要重新完整导出。');return url;});
  const gltf=await new GLTFLoader(manager).parseAsync(bytes,'');model=gltf.scene;scene.add(model);
  const bones=new Set();
  model.traverse(object=>{
    initialTransforms.set(object,{position:object.position.clone(),quaternion:object.quaternion.clone(),scale:object.scale.clone()});
    if(object.isMesh){meshCount++;object.castShadow=true;object.receiveShadow=false;}
    if(object.isSkinnedMesh){skinCount++;object.frustumCulled=false;skeletons.add(object.skeleton);object.skeleton.bones.forEach(bone=>bones.add(bone));}
  });boneCount=bones.size;
  model.updateMatrixWorld(true);bounds=new THREE.Box3().setFromObject(model,true);if(!isFiniteBounds(bounds))throw Error('模型没有可显示的有效形状。');
  const size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
  const ground=new THREE.Mesh(new THREE.CircleGeometry(Math.max(3,size.y*2),80),new THREE.MeshStandardMaterial({color:0xe9eee5,roughness:1,metalness:0}));ground.rotation.x=-Math.PI/2;ground.position.set(center.x,bounds.min.y-.003,center.z);ground.receiveShadow=true;scene.add(ground);
  mixer=new THREE.AnimationMixer(model);mixer.timeScale=Number($('speed').value);
  const bindClip=gltf.animations.find(clip=>clip.name.toLowerCase()==='tpose');if(bindClip&&skinCount>0)bindPoseAction=mixer.clipAction(bindClip);
  if(skinCount>0&&boneCount>0)makeActionEntries(gltf.animations);
  mixer.addEventListener('finished',event=>{if(event.action===current){ended=true;setPlaying(false);}});
  ready=true;viewport.dataset.ready='true';viewport.dataset.skinnedMeshes=String(skinCount);viewport.dataset.bones=String(boneCount);viewport.dataset.clips=String(entries.length);viewport.dataset.sha256=modelHash;
  document.querySelectorAll('[data-view],[data-crop],#reset-camera,#tpose,#reset-idle').forEach(button=>button.disabled=false);
  $('download').href=modelURL.href;$('download').download=modelURL.pathname.split('/').at(-1);$('download').textContent=entries.length?'下载动作模型':'下载预览模型';$('download').hidden=false;
  if(manifest.downloads?.blend){$('download-blend').href=localURL(manifest.downloads.blend).href;$('download-blend').hidden=false;}
  $('action-note').textContent=entries.length?'可暂停、拖动时间轴，逐帧查看动作。':'这个文件尚未包含可播放的骨骼动作，可先旋转查看角色。';
  $('footer-status').textContent=entries.length?'可旋转观察 · 可播放动作':'可旋转观察';
  if(manifest.viewerNote)$('scope-note').textContent=String(manifest.viewerNote);
  const idle=entries.find(entry=>entry.id==='idle')??entries[0];if(idle)selectAction(idle);else restorePose();
  frameCamera();renderer.render(scene,camera);$('overlay').hidden=true;signalState();requestFrame();
}catch(error){fail(error);}
window.addEventListener('pagehide',()=>{if(frameRequest)cancelAnimationFrame(frameRequest);observer?.disconnect();orbit?.dispose();renderer?.dispose();});

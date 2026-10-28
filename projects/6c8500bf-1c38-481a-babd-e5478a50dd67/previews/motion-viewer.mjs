import * as THREE from 'three';
import {OrbitControls} from '/public/vendor/OrbitControls.js';
import {PreviewAvatar} from '/public/preview-avatar.mjs';
import {NaturalMotion} from './natural-motion.mjs';

const $=id=>document.getElementById(id);
const viewport=$('viewport'),controls=[...document.querySelectorAll('button,input')];
const modeLabels={idle:['自然站姿','轻微呼吸与重心变化'],walk:['柔和步行','重心转移、落脚与肩肘配合'],
  greet:['挥手招呼','手指、视线与微笑'],standard:['原标准步行','原动作对照']};
let renderer,avatar,natural,orbit,camera,observer,keyLight;
let ready=false,time=0,remainder=0,mode='idle',playing=true,speed=1,blink=true,expressive=true;
let nextBlink=2.6,blinkStart=null,filmStart=null,filmPhase='',recorder=null,recordChunks=[];
let recordingCanvas=null,recordingContext=null;
let view='oblique',angle=.42,portrait=0,followZ=0,frameDistance=3,recordFrames=0,recordStarted=0;
const scene=new THREE.Scene();scene.background=new THREE.Color('#f5f4ef');
const toggle=(id,value)=>{$(id).setAttribute('aria-pressed',String(value));$(id).textContent=value?'开启':'关闭';};
const smooth=(t,a,b)=>THREE.MathUtils.smoothstep(t,a,b);

function setMode(next,{transition=true}={}){
  const previousMode=mode;mode=next;
  if(next==='standard'){
    avatar.stop();natural.originZ=0;natural.currentTravel=0;avatar.play();
    avatar.mixer.update(0);avatar.vrm.update(0);avatar.springManager?.reset();
  }else{
    if(previousMode==='standard'){avatar.stop();natural.originZ=0;transition=false;}
    avatar.playing=false;
    natural.setMode(next,time,{transition});
  }
  document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===next)));
  $('motion-name').textContent=modeLabels[next][0];$('motion-detail').textContent=modeLabels[next][1];
}
function stopFilm(){
  filmStart=null;filmPhase='';$('film').textContent='播放 18 秒演示';
  if(recorder?.state==='recording')recorder.stop();
  if(ready){controls.forEach(c=>c.disabled=false);$('status').textContent='动作已就绪';}
}
function startFilm(){
  playing=true;toggle('playing',true);speed=1;$('speed').value=100;$('speed-value').value='1.0×';
  blink=true;expressive=true;toggle('blink',true);toggle('expression',true);
  avatar.setPhysics(true);toggle('physics',true);
  avatar.stop();avatar.playing=false;
  natural.originZ=0;natural.currentTravel=0;natural.mode='idle';natural.modeStart=time;
  natural.apply(time);avatar.vrm.update(0);avatar.springManager?.reset();
  mode='idle';setMode('idle',{transition:false});filmStart=time;filmPhase='idle';
  $('film').textContent='结束演示';$('status').textContent='演示播放中';
}
async function recordFilm(){
  if(!renderer.domElement.captureStream||!globalThis.MediaRecorder){
    $('record-note').textContent='当前浏览器不支持视频录制，可使用系统录屏。';return;
  }
  stopFilm();
  const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
  if(!mime){$('record-note').textContent='当前浏览器没有可用的 WebM 编码器。';return;}
  startFilm();recordFrames=0;recordStarted=performance.now();recordChunks=[];
  renderer.setPixelRatio(1);renderer.setSize(1280,720,false);camera.aspect=1280/720;camera.updateProjectionMatrix();
  recordingCanvas=document.createElement('canvas');
  recordingCanvas.width=1280;recordingCanvas.height=720;
  recordingContext=recordingCanvas.getContext('2d',{alpha:false});
  const stream=recordingCanvas.captureStream(30);
  recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:6500000});
  recorder.addEventListener('dataavailable',event=>{if(event.data.size)recordChunks.push(event.data);});
  recorder.addEventListener('error',event=>{$('record-note').textContent='录制失败：'+event.error.message;});
  recorder.addEventListener('stop',()=>{
    stream.getTracks().forEach(track=>track.stop());
    const blob=new Blob(recordChunks,{type:mime});
    const link=document.createElement('a'),url=URL.createObjectURL(blob);
    link.href=url;link.download='character-motion-review.webm';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    const seconds=(performance.now()-recordStarted)/1000;
    $('record-note').textContent=`已录制 ${seconds.toFixed(1)} 秒，${recordFrames} 次画面更新。视频已保存。`;
    recorder=null;recordingCanvas=null;recordingContext=null;resize();controls.forEach(c=>c.disabled=false);
  },{once:true});
  recorder.start(1000);controls.forEach(c=>c.disabled=true);
  $('status').textContent='正在录制 · 18 秒';
}
function frameCamera(delta=1){
  const z=mode==='standard'?0:natural.currentTravel;
  followZ=THREE.MathUtils.lerp(followZ,z,1-Math.exp(-delta*10));
  const targetY=THREE.MathUtils.lerp(.83,1.445,portrait);
  const tangent=Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  const fullDistance=Math.max(1.82/(2*tangent),1.0/(2*tangent*camera.aspect));
  frameDistance=THREE.MathUtils.lerp(fullDistance,.94,portrait);
  orbit.target.set(0,targetY,followZ);
  camera.position.set(Math.sin(angle)*frameDistance,targetY+.055,
    followZ+Math.cos(angle)*frameDistance);
  orbit.update();
}
function resize(){
  if(recorder?.state==='recording')return;
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.7));
  renderer.setSize(viewport.clientWidth,Math.max(1,viewport.clientHeight),false);
  camera.aspect=viewport.clientWidth/Math.max(1,viewport.clientHeight);camera.updateProjectionMatrix();
  if(ready)frameCamera();
}
function expressions(){
  const manager=avatar.vrm.expressionManager;manager.resetValues();
  const local=filmStart===null?null:time-filmStart;
  if(expressive){
    if(mode==='greet')manager.setValue('happy',.20+.035*Math.sin(time*1.2));
    else if(mode==='idle')manager.setValue('relaxed',.045+.015*Math.sin(time*.5));
    if(local!==null&&local>13.2){
      const syllable=(local-13.2)*2.8,index=Math.floor(syllable)%5,
        value=Math.sin(Math.PI*(syllable%1))**1.5*.52;
      manager.setValue(['aa','ih','ou','ee','oh'][index],value);
    }
  }
  if(blink){
    if(blinkStart===null&&time>=nextBlink)blinkStart=time;
    if(blinkStart!==null){
      const elapsed=time-blinkStart;
      const amount=elapsed<.085?elapsed/.085:elapsed<.12?1:Math.max(0,1-(elapsed-.12)/.13);
      manager.setValue('blink',Math.min(1,amount));
      if(elapsed>=.25){blinkStart=null;nextBlink=time+3.0+.7*Math.sin(time*1.7);}
    }
  }
  if(manager.getExpression('blush'))manager.setValue('blush',0);
}
function simulate(dt){
  time+=dt;
  if(filmStart!==null){
    const t=time-filmStart;
    if(t>=18){stopFilm();setMode('idle');return;}
    const next=t<2?'idle':t<9?'walk':t<13?'greet':'idle';
    if(next!==filmPhase){filmPhase=next;setMode(next);}
    portrait=smooth(t,12.6,14.1);
    angle=t<9?.35+.38*Math.sin(t*.45):THREE.MathUtils.lerp(-.25,.26,smooth(t,10,16));
  }
  expressions();
  if(mode==='standard'){
    if(avatar.playing)avatar.mixer?.update(dt);
  }else natural.apply(time);
  avatar.vrm.update(dt);
}
function seek(seconds){
  stopFilm();playing=false;toggle('playing',false);
  avatar.stop();natural.originZ=0;natural.currentTravel=0;
  natural.mode=mode==='standard'?'idle':mode;natural.modeStart=0;natural.transition=null;
  if(mode==='standard')avatar.play();
  time=0;nextBlink=2.6;blinkStart=null;remainder=0;
  expressions();
  if(mode==='standard')avatar.mixer?.setTime(0);else natural.apply(0);
  avatar.vrm.update(0);
  avatar.springManager?.reset();
  const frames=Math.floor(seconds*60);
  for(let i=1;i<=frames;i++){
    time=i/60;expressions();
    if(mode==='standard')avatar.mixer?.setTime(time);else natural.apply(time);
    avatar.vrm.update(1/60);
  }
  followZ=mode==='standard'?0:natural.currentTravel;frameCamera();
  $('status').textContent=`已暂停在 ${time.toFixed(2)} 秒`;
}

try{
  renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:false});
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute('aria-label','可旋转的角色动作');viewport.prepend(renderer.domElement);
  camera=new THREE.PerspectiveCamera(32,1,.01,80);
  orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.dampingFactor=.10;
  orbit.minDistance=.55;orbit.maxDistance=5;orbit.maxPolarAngle=Math.PI*.54;
  scene.add(new THREE.HemisphereLight(0xfffcf6,0xc8ccbe,1.25));
  keyLight=new THREE.DirectionalLight(0xfff8ee,1.55);keyLight.position.set(-2,4,4);
  keyLight.castShadow=true;keyLight.shadow.mapSize.set(2048,2048);
  Object.assign(keyLight.shadow.camera,{left:-1.4,right:1.4,top:1.6,bottom:-1.3,near:.1,far:10});
  keyLight.shadow.normalBias=.002;keyLight.shadow.bias=-.0001;
  keyLight.shadow.radius=3;scene.add(keyLight,keyLight.target);
  const fill=new THREE.DirectionalLight(0xf1f5ff,.85);fill.position.set(3,2,1);scene.add(fill);
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(200,200),
    new THREE.MeshBasicMaterial({color:0xf0f0e9}));
  ground.rotation.x=-Math.PI/2;ground.position.y=-.0015;scene.add(ground);
  const contactShadow=new THREE.Mesh(ground.geometry,new THREE.ShadowMaterial({opacity:.15}));
  contactShadow.rotation.x=-Math.PI/2;contactShadow.position.y=-.0013;
  contactShadow.receiveShadow=true;scene.add(contactShadow);
  const grid=new THREE.GridHelper(200,400,0xe3e6da,0xe3e6da);
  grid.position.y=-.001;grid.material.transparent=true;grid.material.opacity=.55;scene.add(grid);
  scene.fog=new THREE.Fog(0xf5f4ef,8,16);
  avatar=new PreviewAvatar({scene});avatar.setActive(true);
  const projectURL=new URL('../',import.meta.url);
  const candidate=new URLSearchParams(location.search).get('model');
  const modelURL=candidate?new URL(candidate,projectURL):
    new URL('../exports/expression-review/avatar-preview.vrm',import.meta.url);
  if(modelURL.origin!==projectURL.origin||!modelURL.pathname.startsWith(projectURL.pathname)){
    throw Error('动作预览只能载入当前本地项目中的模型。');
  }
  const response=await fetch(modelURL,{cache:'no-store'});
  if(!response.ok)throw Error('角色读取失败（'+response.status+'）');
  const bytes=await response.arrayBuffer();
  await avatar.loadVRM(new File([bytes],'avatar-preview.vrm'));
  // The normalized hips travel while exported mesh nodes stay at the origin.
  // A rest-pose bounding sphere would otherwise cull the walking character.
  avatar.model.traverse(object=>{
    if(object.isSkinnedMesh)object.frustumCulled=false;
    if(object.isMesh){object.castShadow=true;object.receiveShadow=false;}
  });
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)))
    .map(b=>b.toString(16).padStart(2,'0')).join('');
  $('model-info').textContent='本地角色版本 '+hash.slice(0,12);
  $('model-info').dataset.sha256=hash;
  const motionResponse=await fetch('/public/elyth_base_walk_v1.vrma');
  if(!motionResponse.ok)throw Error('原标准步行读取失败');
  await avatar.loadVRMA(new File([await motionResponse.arrayBuffer()],'standard-walk.vrma'));
  natural=new NaturalMotion(avatar.vrm);natural.apply(0);avatar.vrm.update(0);
  avatar.springManager?.reset();
  observer=new ResizeObserver(resize);observer.observe(viewport);resize();
  ready=true;controls.forEach(c=>c.disabled=false);$('overlay').hidden=true;
  $('status').textContent='动作已就绪';frameCamera();
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{
    stopFilm();setMode(button.dataset.mode);playing=true;toggle('playing',true);
  }));
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
    view=button.dataset.view;portrait=view==='face'?1:0;
    angle=({front:0,oblique:.42,profile:Math.PI/2,face:0})[view];frameCamera();
    document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  }));
  $('playing').addEventListener('click',()=>{playing=!playing;toggle('playing',playing);});
  $('physics').addEventListener('click',()=>{avatar.setPhysics(!avatar.physics);toggle('physics',avatar.physics);});
  $('blink').addEventListener('click',()=>{blink=!blink;toggle('blink',blink);});
  $('expression').addEventListener('click',()=>{expressive=!expressive;toggle('expression',expressive);});
  $('speed').addEventListener('input',()=>{speed=Number($('speed').value)/100;$('speed-value').value=speed.toFixed(2)+'×';});
  $('film').addEventListener('click',()=>filmStart===null?startFilm():stopFilm());
  $('record').addEventListener('click',recordFilm);
  $('seek').addEventListener('click',()=>{
    const value=Number($('seek-time').value);
    if(Number.isFinite(value))seek(THREE.MathUtils.clamp(value,0,30));
  });
  let manual=false;orbit.addEventListener('start',()=>{manual=true;stopFilm();});
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{manual=false;}));
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{manual=false;}));
  $('film').addEventListener('click',()=>{manual=false;});$('record').addEventListener('click',()=>{manual=false;});
  let previous=performance.now();
  renderer.setAnimationLoop(now=>{
    const delta=Math.min(Math.max((now-previous)/1000,0),.1);previous=now;
    if(playing){remainder+=delta*speed;
      while(remainder>=1/60){simulate(1/60);remainder-=1/60;}
    }
    if(!manual)frameCamera(delta);else orbit.update();
    const travel=mode==='standard'?0:natural.currentTravel;
    keyLight.position.set(-2,4,travel+4);keyLight.target.position.set(0,.8,travel);
    renderer.render(scene,camera);
    if(recorder?.state==='recording'){
      const c=recordingContext;
      c.drawImage(renderer.domElement,0,0,1280,720);
      const elapsed=time-filmStart,face=elapsed>=13;
      c.fillStyle='rgba(245,244,239,.9)';c.fillRect(26,24,268,76);
      c.fillStyle='#29352c';c.font='500 20px "PingFang SC", sans-serif';
      c.fillText(face?'面部与连续口型':modeLabels[mode][0],44,54);
      c.fillStyle='#788075';c.font='13px "PingFang SC", sans-serif';
      c.fillText(face?'自然眨眼 · 轻微表情':modeLabels[mode][1],44,79);
      recordFrames++;
    }
  });
  document.addEventListener('visibilitychange',()=>{previous=performance.now();});
  window.addEventListener('pagehide',()=>{
    if(recorder?.state==='recording')recorder.stop();
    renderer.setAnimationLoop(null);observer.disconnect();orbit.dispose();avatar.dispose();renderer.dispose();
  },{once:true});
}catch(error){
  controls.forEach(c=>c.disabled=true);$('status').textContent='预览待就绪';
  $('loading-title').textContent='暂时无法显示动作';
  $('loading-detail').textContent=error.message;console.error(error);
}

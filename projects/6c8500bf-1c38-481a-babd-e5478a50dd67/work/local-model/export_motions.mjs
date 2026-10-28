import {projectPath} from '../../../../pipeline/context.mjs';
// Export the authored normalized pose animation as self-contained VRMA.
// The model and its source Blend are never modified.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),option=n=>args.includes(n)?args[args.indexOf(n)+1]:null;
const input=path.resolve(option('--input')||path.join(P,'exports/expression-review/avatar-preview.vrm'));
const output=path.resolve(option('--output-dir')||path.join(P,'exports/motions'));
const {avatar,inputSha}=await loadRuntime(input);avatar.setPhysics(false);
const motion=new NaturalMotion(avatar.vrm),names=Object.keys(motion.nodes),fps=60;
const presets=['happy','relaxed','blink'];

function encode(config){
  const nodes=names.map(name=>({name,translation:motion.rest[name].position.toArray(),
    rotation:motion.rest[name].quaternion.toArray()}));
  const index=new Map(names.map((name,i)=>[motion.nodes[name],i]));
  const roots=[];
  names.forEach((name,i)=>{
    const parent=index.get(motion.nodes[name].parent);
    if(parent===undefined)roots.push(i);else(nodes[parent].children??=[]).push(i);
  });
  const expressionNodes=Object.fromEntries(presets.map(name=>{
    const node=nodes.length;nodes.push({name:'Expression_'+name,translation:[0,0,0]});roots.push(node);
    return[name,{node}];
  }));
  const chunks=[],bufferViews=[],accessors=[];
  let bytes=0;
  function accessor(values,size,minmax=false){
    const array=new Float32Array(values),chunk=Buffer.from(array.buffer);
    const view=bufferViews.length;
    bufferViews.push({buffer:0,byteOffset:bytes,byteLength:chunk.length});
    chunks.push(chunk);bytes+=chunk.length;
    const row={bufferView:view,componentType:5126,count:values.length/size,
      type:({1:'SCALAR',3:'VEC3',4:'VEC4'})[size]};
    if(minmax){
      row.min=Array.from({length:size},(_,k)=>Math.min(...values.filter((_,i)=>i%size===k)));
      row.max=Array.from({length:size},(_,k)=>Math.max(...values.filter((_,i)=>i%size===k)));
    }
    accessors.push(row);return accessors.length-1;
  }
  const count=Math.round(config.duration*fps),times=Array.from({length:count+1},(_,i)=>i/count*config.duration);
  const rotations=Object.fromEntries(names.map(name=>[name,[]])),translations=[],expressions=Object.fromEntries(presets.map(n=>[n,[]]));
  motion.originZ=0;motion.mode=config.mode;motion.modeStart=0;motion.transition=null;
  for(let i=0;i<=count;i++){
    const t=times[i];motion.apply(t);
    for(const name of names)rotations[name].push(...motion.nodes[name].quaternion.toArray());
    const hip=motion.nodes.hips.position.clone();
    if(config.inPlace)hip.z-=motion.currentTravel;
    translations.push(...hip.toArray());
    const b=t%3.7;
    const blink=b>2.4&&b<2.66?Math.sin((b-2.4)/.26*Math.PI)**1.2:0;
    expressions.blink.push(blink,0,0);
    expressions.happy.push(config.mode==='greet'?.20+.035*Math.sin(t*1.2):0,0,0);
    expressions.relaxed.push(config.mode==='idle'?.045+.015*Math.sin(t*.5):0,0,0);
  }
  // Idle / greeting have multiple independent slow rhythms. A soft ending
  // brings their complete pose and expression back to the first sample.
  if(config.loop&&config.mode!=='walk'){
    const tail=Math.round(.7*fps);
    for(let i=count-tail;i<=count;i++){
      const x=(i-(count-tail))/tail,w=x*x*x*(x*(x*6-15)+10);
      for(const name of names){
        const values=rotations[name],q=new THREE.Quaternion().fromArray(values,i*4);
        q.slerp(new THREE.Quaternion().fromArray(values,0),w).toArray(values,i*4);
      }
      for(let k=0;k<3;k++)translations[i*3+k]=THREE.MathUtils.lerp(translations[i*3+k],translations[k],w);
      for(const values of Object.values(expressions))values[i*3]=THREE.MathUtils.lerp(values[i*3],values[0],w);
    }
  }
  const timeAccessor=accessor(times,1,true),samplers=[],channels=[];
  const add=(values,size,node,channel)=>{
    const output=accessor(values,size);
    channels.push({sampler:samplers.length,target:{node,path:channel}});
    samplers.push({input:timeAccessor,output,interpolation:'LINEAR'});
  };
  names.forEach((name,i)=>add(rotations[name],4,i,'rotation'));
  add(translations,3,names.indexOf('hips'),'translation');
  presets.forEach(name=>add(expressions[name],3,expressionNodes[name].node,'translation'));
  const json={asset:{version:'2.0',generator:'Charakuru local character motion'},
    extensionsUsed:['VRMC_vrm_animation'],extensions:{VRMC_vrm_animation:{
      specVersion:'1.0',humanoid:{humanBones:Object.fromEntries(names.map((n,i)=>[n,{node:i}]))},
      expressions:{preset:expressionNodes,custom:{}}}},
    scene:0,scenes:[{nodes:roots}],nodes,
    animations:[{name:config.name,samplers,channels}],
    bufferViews,accessors,buffers:[{byteLength:bytes}],
    extras:{sourceVrmSha256:inputSha,fps,loop:config.loop,inPlace:config.inPlace,
      note:config.mode==='walk'?'Heel-to-forefoot support with skeleton-sized two-bone IK.':'Authored relaxed motion with face expressions.'}};
  const binary=Buffer.concat(chunks),encoded=Buffer.from(JSON.stringify(json)),padded=Math.ceil(encoded.length/4)*4;
  const file=Buffer.alloc(28+padded+binary.length,32);
  file.writeUInt32LE(0x46546c67,0);file.writeUInt32LE(2,4);file.writeUInt32LE(file.length,8);
  file.writeUInt32LE(padded,12);file.writeUInt32LE(0x4e4f534a,16);encoded.copy(file,20);
  file.writeUInt32LE(binary.length,20+padded);file.writeUInt32LE(0x004e4942,24+padded);binary.copy(file,28+padded);
  return{file,reference:{times,rotations,translations,expressions},frames:count+1,channels:channels.length};
}

await fs.mkdir(output,{recursive:true});
const files=[];
try{
  for(const config of [
    {name:'natural-walk-in-place',mode:'walk',duration:motion.period,inPlace:true,loop:true},
    {name:'natural-walk-forward',mode:'walk',duration:motion.period*4,inPlace:false,loop:false},
    {name:'natural-idle',mode:'idle',duration:16,inPlace:false,loop:true},
    {name:'friendly-greeting',mode:'greet',duration:6.1,inPlace:false,loop:true},
  ]){
    const {file,reference,frames,channels}=encode(config);
    const filename=config.name+'.vrma',destination=path.join(output,filename);
    await fs.writeFile(destination,file);
    // Re-load through the actual app's VRMA loader and compare bone pose.
    await avatar.loadVRMA(new File([file],filename));avatar.play();
    let rotationError=0,translationError=0,expressionError=0;
    const sampledIndices=new Set([0,Math.round((frames-1)*.23),
      Math.round((frames-1)*.57),frames-2]);
    // Include each expression's real peak, so a missing blink track cannot pass
    // merely because the generic pose samples happen between blinks.
    for(const name of presets){
      let peak=0;
      for(let i=1;i<frames;i++)
        if(reference.expressions[name][i*3]>reference.expressions[name][peak*3])peak=i;
      sampledIndices.add(peak);
    }
    for(const i of sampledIndices){
      avatar.mixer.setTime(reference.times[i]);avatar.vrm.update(0);
      for(const name of names){
        const expected=new THREE.Quaternion().fromArray(reference.rotations[name],i*4).normalize();
        rotationError=Math.max(rotationError,avatar.vrm.humanoid.getNormalizedBoneNode(name).quaternion.angleTo(expected));
      }
      translationError=Math.max(translationError,avatar.vrm.humanoid.getNormalizedBoneNode('hips').position
        .distanceTo(new THREE.Vector3().fromArray(reference.translations,i*3)));
      for(const name of presets)
        expressionError=Math.max(expressionError,Math.abs(
          avatar.vrm.expressionManager.getValue(name)-reference.expressions[name][i*3]));
    }
    if(rotationError>2e-3||translationError>2e-5)throw Error('VRMA roundtrip mismatch: '+filename);
    if(expressionError>2e-5)throw Error('VRMA expression roundtrip mismatch: '+filename);
    files.push({...config,file:destination,sha256:sha(file),bytes:file.length,frames,channels,
      roundTripSampledTimes:[...sampledIndices].sort((a,b)=>a-b).map(i=>reference.times[i]),
      roundTripMaxExpressionWeightError:expressionError,
      roundTripMaxRotationErrorRadians:rotationError,roundTripMaxHipsTranslationErrorMeters:translationError});
    avatar.stop();
  }
  const unchanged=sha(await fs.readFile(input))===inputSha;
  if(!unchanged)throw Error('VRM changed during animation export');
  const motionScript=path.join(P,'previews/natural-motion.mjs');
  const report={vrmPath:input,vrmSha256:inputSha,sourceUnchanged:true,fps,
    motionScript,motionScriptSha256:sha(await fs.readFile(motionScript)),files,
    limits:'These motions fit this character proportions. In-place is a treadmill loop; forward includes root displacement and should play once or accumulate root motion. Independent hosts may vary in spring simulation and retargeting.'};
  await fs.writeFile(path.join(output,'motions-report.json'),JSON.stringify(report,null,2)+'\n');
  await fs.writeFile(path.join(output,'使用说明.txt'),[
    '本地角色动作（VRM Animation 1.0）',
    '',
    'natural-idle.vrma：16 秒自然站姿，包含呼吸、偏侧承重、小幅注视与眨眼，可循环。',
    'natural-walk-in-place.vrma：1.52 秒原地步行循环。用于跟随角色的游戏/应用；场景位移由宿主控制。',
    'natural-walk-forward.vrma：6.08 秒向前步行，包含实际髋部位移。单次播放；循环时需由宿主累加根位移。',
    'friendly-greeting.vrma：6.1 秒招手，含轻微笑意与眨眼，可循环；开始/结束建议与站姿混合。',
    '',
    '在支持 VRMA 的查看器中先载入本角色 VRM，再载入上述动作。',
    '动作按本角色骨架尺寸制作。头发与裙摆仍由 VRM 的 SpringBone 在宿主中实时模拟。',
    '不同骨架比例、物理实现或高速动作需要重新检查脚接地和裙腿遮挡。',
    '',
    '实际输入版本和回读验证记录见 motions-report.json。',
  ].join('\n')+'\n');
  console.log(JSON.stringify(report,null,2));
}finally{avatar.dispose();}

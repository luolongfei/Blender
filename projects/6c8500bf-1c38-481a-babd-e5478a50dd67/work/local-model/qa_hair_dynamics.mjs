import {projectPath} from '../../../../pipeline/context.mjs';
// Compare actual exported ponytail skin with physics ON/OFF, then hold a pose
// to measure settling. No model/source/physics parameters are modified.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input',path.join(P,'exports/expression-review/avatar-preview.vrm')));
const output=path.resolve(arg('--output',path.join(P,'work/qa-hair-physics/hair-dynamics.json')));
const data=await fs.readFile(input),json=JSON.parse(data.subarray(20,20+data.readUInt32LE(12)));
const motionPath=path.join(P,'previews/natural-motion.mjs'),motionSHA=sha(await fs.readFile(motionPath));
const hairIndices=json.nodes.map((n,i)=>[n,i]).filter(([n])=>
  n.mesh!==undefined&&n.extras?.part==='hair'&&n.extras?.hair_region==='ponytail').map(([,i])=>i);
const on=await loadRuntime(input),off=await loadRuntime(input);
try{
  on.avatar.setPhysics(true);off.avatar.setPhysics(false);
  const cases=await Promise.all([on,off].map(async({avatar})=>{
    const meshes=[];
    for(const i of hairIndices){
      const node=await avatar.resources.parser.getDependency('node',i);
      meshes.push(...node.getObjectsByProperty('isMesh',true));
    }
    const natural=new NaturalMotion(avatar.vrm);
    avatar.model.updateMatrixWorld(true);
    const restSamples=[],restPoint=new THREE.Vector3();
    for(const mesh of meshes){
      mesh.skeleton?.update();
      for(let i=0;i<mesh.geometry.attributes.position.count;i+=8){
        mesh.getVertexPosition(i,restPoint).applyMatrix4(mesh.matrixWorld);
        restSamples.push(restPoint.y);
      }
    }
    const tipThreshold=Math.min(...restSamples)+.025;
    const tipIndices=restSamples.map((y,i)=>y<=tipThreshold?i:-1).filter(i=>i>=0);
    natural.mode='walk';natural.modeStart=0;natural.originZ=0;natural.transition=null;
    natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
    return{avatar,natural,meshes,tipIndices,tipThreshold};
  }));
  function capture(item){
    item.avatar.model.updateMatrixWorld(true);
    const inverse=item.avatar.vrm.humanoid.getRawBoneNode('head').matrixWorld.clone().invert();
    const result=[],point=new THREE.Vector3();
    for(const mesh of item.meshes){
      mesh.skeleton?.update();
      for(let i=0;i<mesh.geometry.attributes.position.count;i+=8){
        mesh.getVertexPosition(i,point).applyMatrix4(mesh.matrixWorld).applyMatrix4(inverse);
        if(![point.x,point.y,point.z].every(Number.isFinite))throw Error('Nonfinite ponytail vertex');
        result.push(point.x,point.y,point.z);
      }
    }
    return Float64Array.from(result);
  }
  function compare(a,b){
    let max=0,squared=0;
    for(let i=0;i<a.length;i+=3){
      const d=(a[i]-b[i])**2+(a[i+1]-b[i+1])**2+(a[i+2]-b[i+2])**2;
      max=Math.max(max,Math.sqrt(d));squared+=d;
    }
    const tipY=cases[0].tipIndices.map(i=>a[i*3+1]-b[i*3+1]);
    return{maximumMeters:max,rmsMeters:Math.sqrt(squared/(a.length/3)),
      tipMeanHeadFrameRiseMeters:tipY.reduce((a,b)=>a+b,0)/tipY.length,
      tipMaximumHeadFrameRiseMeters:Math.max(...tipY)};
  }
  const samples=[],settle=[],dt=1/60;
  let previous=null,sampledVertices=0;
  for(let i=1;i<=16*60;i++){
    const time=i/60;
    for(const item of cases){
      if(i===6*60)item.natural.setMode('idle',time,{transition:true});
      // After 8 seconds the complete humanoid pose is fixed; only spring
      // dynamics advance. This separates settling from continued breathing.
      item.natural.apply(Math.min(time,8));
      item.avatar.vrm.update(dt);
    }
    if(i%6!==0&&time<8)continue;
    const enabled=capture(cases[0]),disabled=capture(cases[1]);
    sampledVertices=enabled.length/3;
    if(i%6===0)samples.push({time,...compare(enabled,disabled)});
    if(time>=8){
      if(previous)settle.push({time,...compare(enabled,previous)});
      previous=enabled;
    }
  }
  const last=settle.filter(s=>s.time>=15),first=settle.filter(s=>s.time<=9);
  const report={vrmPath:input,vrmSha256:on.inputSha,motionPath,motionScriptSha256:motionSHA,
    physicsModes:[true,false],timestep:dt,sampledVerticesPerState:sampledVertices,
    tipSampledVertices:cases[0].tipIndices.length,
    tipSelection:'Rest-space bottom 25mm of the sampled ponytail vertices; rise measured along head-frame +Y.',
    sequence:[{start:0,end:6,mode:'walk'},{start:6,end:8,mode:'idle transition'},
      {start:8,end:16,mode:'humanoid pose held; spring simulation advances'}],
    maxPhysicsOnOffDisplacementMeters:Math.max(...samples.map(s=>s.maximumMeters)),
    finalPhysicsOnOffDifference:samples.at(-1),
    firstHeldSecondMaxFrameDisplacementMeters:Math.max(...first.map(s=>s.maximumMeters)),
    lastHeldSecondMaxFrameDisplacementMeters:Math.max(...last.map(s=>s.maximumMeters)),
    lastHeldSecondRmsFrameDisplacementMeters:Math.max(...last.map(s=>s.rmsMeters)),
    scope:'Actual exported skinned ponytail vertices sampled every 8th index, compared in the raw head bone frame. Final 8s holds humanoid pose; springs continue updating at 60Hz.',
    limitations:['ON/OFF difference includes collision deflection and gravity, not only oscillation.',
      'Does not prove garment clearance, valid aesthetics, or stability under every motion.',
      'Sampling every 8th vertex can miss unsampled local anomalies.'],
    sourceUnchanged:sha(await fs.readFile(input))===on.inputSha,
    motionUnchanged:sha(await fs.readFile(motionPath))===motionSHA,
    onOffPerTenthSecond:samples,settlePerFrame:settle};
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({...report,onOffPerTenthSecond:undefined,settlePerFrame:undefined},null,2));
}finally{on.avatar.dispose();off.avatar.dispose();}

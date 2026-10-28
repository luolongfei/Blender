import {projectPath} from '../../../../pipeline/context.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');

export async function testTransitions(avatar){
  avatar.stop();avatar.clearMotion();
  avatar.vrm.humanoid.resetNormalizedPose();avatar.vrm.update(0);
  avatar.model.updateMatrixWorld(true);
  const footwear=[],point=new THREE.Vector3();
  avatar.model.traverse(mesh=>{
    if(!mesh.isMesh)return;
    mesh.skeleton?.update();
    const indices=[];
    for(let i=0;i<mesh.geometry.attributes.position.count;i++){
      mesh.getVertexPosition(i,point).applyMatrix4(mesh.matrixWorld);
      if(point.y<.115)indices.push(i);
    }
    if(indices.length)footwear.push({mesh,indices});
  });
  const natural=new NaturalMotion(avatar.vrm);
  natural.mode='idle';natural.modeStart=0;natural.originZ=0;natural.transition=null;
  natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
  const events=new Map([[120,'walk'],[540,'greet'],[780,'idle']]);
  const report={frames:1080,timestep:1/60,footwearVertices:
    footwear.reduce((n,e)=>n+e.indices.length,0),events:[
    {time:2,mode:'walk'},{time:9,mode:'greet'},{time:13,mode:'idle'}],
    minShoeY:Infinity,maxShoeMinimumY:-Infinity,belowGroundFrames:0,
    maxIKReachError:0,belowGroundSamples:[],bothFeetElevatedFrames:0,worstElevation:null,
    scope:'Full 18s film pose timeline with 0.8s mode transitions; every skinned footwear vertex whose rest world height is below 0.115m. Does not test fabric or hair intersections.'};
  for(let i=0;i<1080;i++){
    const time=i/60;
    if(events.has(i))natural.setMode(events.get(i),time,{transition:true});
    natural.apply(time);avatar.vrm.update(1/60);avatar.model.updateMatrixWorld(true);
    let floor=Infinity;
    for(const {mesh,indices}of footwear){
      mesh.skeleton?.update();
      for(const index of indices){
        mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld);
        if(![point.x,point.y,point.z].every(Number.isFinite))throw Error('Nonfinite shoe');
        floor=Math.min(floor,point.y);
      }
    }
    report.minShoeY=Math.min(report.minShoeY,floor);
    if(floor>report.maxShoeMinimumY){
      report.maxShoeMinimumY=floor;
      report.worstElevation={time,mode:natural.mode,minimumY:floor,
        feet:Object.fromEntries(['left','right'].map(side=>[side,
          natural.nodes[side+'Foot'].getWorldPosition(new THREE.Vector3()).toArray()]))};
    }
    if(floor>.008)report.bothFeetElevatedFrames++;
    report.maxIKReachError=Math.max(report.maxIKReachError,natural.lastReachError);
    if(floor<-.001){
      report.belowGroundFrames++;
      if(report.belowGroundSamples.length<20)report.belowGroundSamples.push({time,mode:natural.mode,minimumY:floor});
    }
  }
  return report;
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
  const args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
  const input=path.resolve(arg('--input',path.join(P,'exports/expression-review/avatar-preview.vrm')));
  const output=path.resolve(arg('--output',path.join(P,'work/qa-motion-vrma/transitions-report.json')));
  const motionPath=path.join(P,'previews/natural-motion.mjs');
  const motionSha=sha(await fs.readFile(motionPath));
  const {avatar,inputSha}=await loadRuntime(input);
  try{
    const report=await testTransitions(avatar);
    Object.assign(report,{input,vrmSha256:inputSha,motionScriptSha256:motionSha,
      sourceUnchanged:sha(await fs.readFile(input))===inputSha,
      motionUnchanged:sha(await fs.readFile(motionPath))===motionSha});
    await fs.mkdir(path.dirname(output),{recursive:true});
    await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
  }finally{avatar.dispose();}
}

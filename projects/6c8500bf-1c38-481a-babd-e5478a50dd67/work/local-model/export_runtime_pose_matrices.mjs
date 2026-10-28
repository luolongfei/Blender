// Exact raw skin-bone rest->posed world deformation for independent Blender QA.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input','')),output=path.resolve(arg('--output',''));
const mode=arg('--mode','greet'),time=Number(arg('--time','4.2'));
const runtime=await loadRuntime(input);
try{
  const {avatar,json}=runtime;
  avatar.model.updateMatrixWorld(true);
  const inverseBinds=new Map(),samples=[];
  avatar.model.traverse(mesh=>{
    if(!mesh.isSkinnedMesh)return;
    mesh.skeleton.update();
    // This export has identity skin mesh transforms. Assert the convention
    // instead of silently applying world-space bind matrices to local points.
    const factor=mesh.bindMatrix.clone().multiply(mesh.matrixWorld.clone().invert());
    if(factor.elements.some((v,i)=>Math.abs(v-(i%5===0?1:0))>1e-6))
      throw Error('Nonidentity mesh bind factor needs explicit treatment');
    mesh.skeleton.bones.forEach((bone,i)=>{
      const matrix=mesh.skeleton.boneInverses[i];
      const previous=inverseBinds.get(bone.name);
      if(previous&&matrix.elements.some((v,k)=>Math.abs(v-previous.elements[k])>1e-6))
        throw Error('Different inverse bind for same bone '+bone.name);
      inverseBinds.set(bone.name,matrix.clone());
    });
    if(mesh.geometry.morphAttributes.position?.length)return;
    for(let index=0;index<mesh.geometry.attributes.position.count;index+=257){
      const raw=new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position,index)
        .applyMatrix4(mesh.matrixWorld);
      const si=mesh.geometry.attributes.skinIndex,sw=mesh.geometry.attributes.skinWeight;
      samples.push({mesh,index,raw,weights:Array.from({length:4},(_,c)=>({
        bone:mesh.skeleton.bones[si.getComponent(index,c)].name,weight:sw.getComponent(index,c),
      })).filter(w=>w.weight>1e-8)});
    }
  });
  const joints=[...new Set(json.skins.flatMap(s=>s.joints))],bones=[];
  for(const index of joints){
    const node=await avatar.resources.parser.getDependency('node',index);
    bones.push({name:json.nodes[index].name,index,node,rest:node.matrixWorld.clone()});
  }
  const motion=new NaturalMotion(avatar.vrm);
  motion.mode=mode;motion.modeStart=0;motion.originZ=0;motion.transition=null;
  avatar.setPhysics(true);motion.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
  const frames=Math.round(time*60);
  for(let i=1;i<=frames;i++){motion.apply(i/60);avatar.vrm.update(1/60);}
  avatar.model.updateMatrixWorld(true);
  const runtimeToBlender=new THREE.Matrix4().set(
    1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1);
  const toRuntime=runtimeToBlender.clone().invert();
  const rows=bones.map(({name,index,node,rest})=>{
    const deform=node.matrixWorld.clone().multiply(inverseBinds.get(name));
    return{name,nodeIndex:index,restWorldRuntime:rest.toArray(),
      posedWorldRuntime:node.matrixWorld.toArray(),restToPosedRuntime:deform.toArray(),
      inverseBindRuntime:inverseBinds.get(name).toArray(),
      loadedRestToPosedRuntime:node.matrixWorld.clone().multiply(rest.clone().invert()).toArray(),
      restToPosedBlender:runtimeToBlender.clone().multiply(deform).multiply(toRuntime).toArray()};
  });
  const byName=new Map(rows.map(row=>[row.name,new THREE.Matrix4().fromArray(row.restToPosedRuntime)]));
  let replayError=0;
  for(const sample of samples){
    const replay=new THREE.Vector3();
    for(const w of sample.weights)
      replay.addScaledVector(sample.raw.clone().applyMatrix4(byName.get(w.bone)),w.weight);
    sample.mesh.skeleton.update();
    const actual=sample.mesh.getVertexPosition(sample.index,new THREE.Vector3())
      .applyMatrix4(sample.mesh.matrixWorld);
    replayError=Math.max(replayError,replay.distanceTo(actual));
  }
  if(replayError>1e-5)throw Error('Raw bind LBS replay disagrees with runtime '+replayError);
  const report={version:2,input,vrmSha256:runtime.inputSha,mode,time:frames/60,
    timestep:1/60,physics:true,matrixStorage:'column-major 16 numeric values, column vectors',
    runtimeToBlender:runtimeToBlender.toArray(),bones:rows,
    sampledLbsReplayVertices:samples.length,maxRawBindLbsReplayErrorMeters:replayError,
    scope:'Version2: restToPosedRuntime maps RAW POSITION source bind vertices, using posed bone world × inverseBindMatrices. loadedRestToPosedRuntime separately describes loaded VRM rest. Blender conjugation C×D×inverse(C); sum(weight×D_bone×raw source vertex), not dual-quaternion skinning.',
    sourceUnchanged:sha(await fs.readFile(input))===runtime.inputSha};
  await fs.writeFile(output,JSON.stringify(report,null,2));
  console.log(JSON.stringify({output,vrmSha256:runtime.inputSha,mode,time,bones:rows.length}));
}finally{runtime.avatar.dispose();}

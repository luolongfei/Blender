// Add a measured local back coverage candidate to a disposable VRM only.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const args=process.argv.slice(2);
const arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input','')),output=path.resolve(arg('--output',''));
if(!args.includes('--input')||!args.includes('--output')||input===output)
  throw Error('Use explicit distinct input and QA output.');
const original=await fs.readFile(input);
const jsonLength=original.readUInt32LE(12);
const document=JSON.parse(original.subarray(20,20+jsonLength));
const bin=original.subarray(28+jsonLength);
const runtime=await loadRuntime(input);
try{
  runtime.avatar.model.updateMatrixWorld(true);
  const boneIndex=document.extensions.VRMC_vrm.humanoid.humanBones.chest.node;
  const bone=await runtime.avatar.resources.parser.getDependency('node',boneIndex);
  const inverse=bone.matrixWorld.clone().invert();
  const local=p=>new THREE.Vector3(p[0],p[2],-p[1]).applyMatrix4(inverse).toArray();
  const sb=document.extensions.VRMC_springBone;
  const specs=[1,-1].map(sign=>({
    name:`Back ${sign===1?'L':'R'}`,bone:'chest',
    center:[sign*.090,.025,1.090],tail:[sign*.090,.025,1.230],radius:.060,
  }));
  const indices=specs.map(spec=>{
    const index=sb.colliders.length;
    sb.colliders.push({node:boneIndex,shape:{capsule:{
      offset:local(spec.center),tail:local(spec.tail),radius:spec.radius}}});
    return index;
  });
  const group=sb.colliderGroups.length;
  sb.colliderGroups.push({name:'Back garment · ponytails',colliders:indices});
  const hitChanges=[];
  for(const spring of sb.springs.filter(s=>s.name.startsWith('ponytail_'))){
    spring.colliderGroups.push(group);
    if(args.includes('--measured-hit-radius')){
      for(const joint of spring.joints){
        const name=document.nodes[joint.node].name;
        if(/^hair_[LR]_[23](?:_|$)/.test(name)){
          hitChanges.push({bone:name,before:joint.hitRadius,after:.040});
          joint.hitRadius=.040;
        }
      }
    }
  }
  const encoded=Buffer.from(JSON.stringify(document)),padded=Math.ceil(encoded.length/4)*4;
  const bytes=Buffer.alloc(28+padded+bin.length,32);
  bytes.writeUInt32LE(0x46546c67,0);bytes.writeUInt32LE(2,4);bytes.writeUInt32LE(bytes.length,8);
  bytes.writeUInt32LE(padded,12);bytes.writeUInt32LE(0x4e4f534a,16);encoded.copy(bytes,20);
  bytes.writeUInt32LE(bin.length,20+padded);bytes.writeUInt32LE(0x004e4942,24+padded);
  bin.copy(bytes,28+padded);
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,bytes);
  const report={input,inputSha256:sha(original),output,outputSha256:sha(bytes),specs,hitChanges,
    sourceUnchanged:sha(await fs.readFile(input))===sha(original),
    reason:'The measured t=.4 standard-walk contact maps to chest-rest (x=.098..100,y=.068..071,z=1.194..1.200). Existing shoulder/chest spheres leave this local rear garment surface uncovered.'};
  await fs.writeFile(output+'.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
}finally{runtime.avatar.dispose();}

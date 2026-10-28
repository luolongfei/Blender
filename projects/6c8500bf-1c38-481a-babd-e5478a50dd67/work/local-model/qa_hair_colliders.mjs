import {projectPath} from '../../../../pipeline/context.mjs';
// Material-preserving QA copies only. The source and exports are untouched.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const source=path.join(P,'exports/expression-review/avatar-preview.vrm');
const output=path.join(P,'work/qa-hair-physics');await fs.mkdir(output,{recursive:true});
const bytes=await fs.readFile(source),jsonLength=bytes.readUInt32LE(12);
const original=JSON.parse(bytes.subarray(20,20+jsonLength)),binary=bytes.subarray(28+jsonLength);
const {avatar,inputSha}=await loadRuntime(source);
const node=avatar.vrm.humanoid.getRawBoneNode('upperChest');
avatar.model.updateMatrixWorld(true);
const positions=[-1,1].map(sign=>node.worldToLocal(new THREE.Vector3(sign*.12,1.23,-.004)).toArray());
avatar.dispose();
const reports=[];
for(const [radius,hit]of [[.065,.025],[.070,.025],[.070,.030],[.075,.025]]){
  const json=structuredClone(original),spring=json.extensions.VRMC_springBone;
  const colliderIndices=[];
  const boneIndex=json.extensions.VRMC_vrm.humanoid.humanBones.upperChest.node;
  for(const offset of positions){
    colliderIndices.push(spring.colliders.length);
    spring.colliders.push({node:boneIndex,shape:{sphere:{offset,radius}}});
  }
  const group=spring.colliderGroups.length;
  spring.colliderGroups.push({name:'QA shoulders for ponytails',colliders:colliderIndices});
  for(const item of spring.springs.filter(s=>s.name.startsWith('ponytail_'))){
    item.colliderGroups.push(group);
    for(const joint of item.joints)joint.hitRadius=hit;
  }
  const text=Buffer.from(JSON.stringify(json)),padded=Math.ceil(text.length/4)*4;
  const file=Buffer.alloc(28+padded+binary.length,32);
  file.writeUInt32LE(0x46546c67,0);file.writeUInt32LE(2,4);file.writeUInt32LE(file.length,8);
  file.writeUInt32LE(padded,12);file.writeUInt32LE(0x4e4f534a,16);text.copy(file,20);
  file.writeUInt32LE(binary.length,20+padded);file.writeUInt32LE(0x004e4942,24+padded);binary.copy(file,28+padded);
  const filename=`shoulder-r${Math.round(radius*1000).toString().padStart(3,'0')}-h${Math.round(hit*1000).toString().padStart(3,'0')}.vrm`;
  await fs.writeFile(path.join(output,filename),file);
  reports.push({file:path.join(output,filename),sha256:sha(file),radius,hitRadius:hit,
    attachBone:'upperChest',vrmLocalOffsets:positions,
    worldCentersBlender:[[-.12,.004,1.23],[.12,.004,1.23]]});
}
if(sha(await fs.readFile(source))!==inputSha)throw Error('Source changed during QA');
await fs.writeFile(path.join(output,'candidate-provenance.json'),
  JSON.stringify({source,sourceSha256:inputSha,sourceUnchanged:true,candidates:reports},null,2));
console.log(JSON.stringify(reports,null,2));

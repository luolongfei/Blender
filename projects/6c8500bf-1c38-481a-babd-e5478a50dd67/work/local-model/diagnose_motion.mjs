import {projectPath, toolPath} from '../../../../pipeline/context.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime} from './runtime_cpu.mjs';
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const {avatar,inputSha}=await loadRuntime(path.join(P,'exports/expression-review/avatar-preview.vrm'));
const h=avatar.vrm.humanoid;
avatar.setPhysics(false);
const names=['hips','spine','chest','upperChest','neck','head',
  'leftShoulder','leftUpperArm','leftLowerArm','leftHand',
  'leftUpperLeg','leftLowerLeg','leftFoot','leftToes',
  'rightUpperLeg','rightLowerLeg','rightFoot','rightToes'];
const rest={};
avatar.model.updateMatrixWorld(true);
for(const name of names){
  const n=h.getNormalizedBoneNode(name),r=h.getRawBoneNode(name);
  rest[name]={normalizedWorld:n.getWorldPosition(new THREE.Vector3()).toArray(),
    normalizedQuaternion:n.quaternion.toArray(),rawWorld:r.getWorldPosition(new THREE.Vector3()).toArray(),
    localPosition:n.position.toArray()};
}
const motion=await fs.readFile(toolPath('public/elyth_base_walk_v1.vrma'));
await avatar.loadVRMA(new File([motion],'walk.vrma'));
avatar.play();
const tracks={};for(const name of names)tracks[name]=[];
const duration=avatar.action.getClip().duration;
for(let i=0;i<=120;i++){
  avatar.mixer.setTime(i/120*duration);avatar.vrm.update(0);
  avatar.model.updateMatrixWorld(true);
  for(const name of names){
    const n=h.getNormalizedBoneNode(name),r=h.getRawBoneNode(name);
    tracks[name].push({p:r.getWorldPosition(new THREE.Vector3()).toArray(),
      e:new THREE.Euler().setFromQuaternion(n.quaternion,'XYZ').toArray().slice(0,3)});
  }
}
const ranges={};
for(const [name,frames]of Object.entries(tracks)){
  ranges[name]={worldMin:[0,1,2].map(k=>Math.min(...frames.map(f=>f.p[k]))),
    worldMax:[0,1,2].map(k=>Math.max(...frames.map(f=>f.p[k]))),
    normalizedDegreesMin:[0,1,2].map(k=>THREE.MathUtils.radToDeg(Math.min(...frames.map(f=>f.e[k])))),
    normalizedDegreesMax:[0,1,2].map(k=>THREE.MathUtils.radToDeg(Math.max(...frames.map(f=>f.e[k]))))};
}
const report={vrmSha256:inputSha,duration,rest,ranges,tracks};
await fs.writeFile(path.join(P,'work/local-model/motion-diagnosis.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({rest,ranges},null,2));
avatar.dispose();

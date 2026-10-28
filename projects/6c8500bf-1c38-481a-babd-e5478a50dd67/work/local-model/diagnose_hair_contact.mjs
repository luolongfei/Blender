import {projectPath, toolPath} from '../../../../pipeline/context.mjs';
// Read-only mapping from actual runtime collision samples to source-space skin.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input',path.join(P,'exports/expression-review/avatar-preview.vrm')));
const collisions=JSON.parse(await fs.readFile(arg('--collisions',
  path.join(P,'exports/expression-review/avatar-hair-walk-report.json')),'utf8'));
const time=Number(arg('--time','5.9'));
const mode=arg('--mode',collisions.mode||'walk');
// Historical reports may record the previous computer's absolute path.
const motionPath=mode==='standard'?toolPath('public/elyth_base_walk_v1.vrma'):
  path.join(P,'previews/natural-motion.mjs');
const motionBytes=await fs.readFile(motionPath);
const motionSha=sha(motionBytes);
const recordedMotionSha=collisions.motionSha256||collisions.motionScriptSha256;
if(!recordedMotionSha||motionSha!==recordedMotionSha)
  throw Error('Collision report motion differs from the migrated motion source.');
const output=arg('--output',path.join(P,'work/qa-hair-physics/contact-binding-5-90.json'));
const source=await fs.readFile(input),json=JSON.parse(source.subarray(20,20+source.readUInt32LE(12)));
if(sha(source)!==collisions.vrmSha256)throw Error('Collision report has different VRM');
const {avatar,inputSha}=await loadRuntime(input);
try{
  const frame=collisions.perFrame.find(row=>Math.abs(row.time-time)<1e-8);
  if(!frame)throw Error('No sampled frame at time');
  const natural=new NaturalMotion(avatar.vrm);
  avatar.vrm.humanoid.resetNormalizedPose();avatar.vrm.update(0);avatar.model.updateMatrixWorld(true);
  const restBodyMatrices = Object.fromEntries(['chest','upperChest'].map(name=>[
    name,avatar.vrm.humanoid.getRawBoneNode(name).matrixWorld.clone()]));
  const records=[];
  for(const hit of frame.examples){
    const ni=json.nodes.findIndex(n=>n.name===hit.hair);
    const node=await avatar.resources.parser.getDependency('node',ni);
    const mesh=node.getObjectsByProperty('isMesh',true)[0];
    mesh.skeleton.update();
    const vertices=(hit.hairEdge||hit.hairTriangle).map(index=>{
      const rest=mesh.getVertexPosition(index,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
      const skinIndex=mesh.geometry.attributes.skinIndex,skinWeight=mesh.geometry.attributes.skinWeight;
      const weights=Array.from({length:4},(_,channel)=>{
        const indexOfBone=skinIndex.getComponent(index,channel);
        return{bone:mesh.skeleton.bones[indexOfBone].name,weight:skinWeight.getComponent(index,channel)};
      }).filter(w=>w.weight>1e-8);
      return{index,restWorldRuntime:rest.toArray(),restBlender:[rest.x,-rest.z,rest.y],weights};
    });
    records.push({...hit,meshObject:mesh,vertices});
  }
  const colliderNodes=[];
  for(const collider of json.extensions.VRMC_springBone.colliders){
    if(!collider.shape.sphere)continue;
    const node=await avatar.resources.parser.getDependency('node',collider.node);
    colliderNodes.push({node,nodeIndex:collider.node,shape:collider.shape.sphere});
  }
  natural.mode=mode;natural.modeStart=0;natural.originZ=0;natural.transition=null;
  natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
  if(mode==='standard'){
    await avatar.loadVRMA(new File([motionBytes],'standard-walk.vrma'));
    avatar.setPhysics(true);avatar.play();avatar.mixer.update(0);
    avatar.vrm.update(0);avatar.springManager?.reset();
  }
  for(let i=1;i<=Math.round(time*60);i++){
    if(mode==='standard')avatar.tick(1/60);
    else{natural.apply(i/60);avatar.vrm.update(1/60);}
  }
  avatar.model.updateMatrixWorld(true);
  const colliderPose=colliderNodes.map(({node,nodeIndex,shape})=>({
    bone:json.nodes[nodeIndex].name,radius:shape.radius,
    center:new THREE.Vector3(...shape.offset).applyMatrix4(node.matrixWorld).toArray()}));
  for(const record of records){
    record.meshObject.skeleton.update();
    for(const vertex of record.vertices){
      const moved=record.meshObject.getVertexPosition(vertex.index,new THREE.Vector3())
        .applyMatrix4(record.meshObject.matrixWorld);
      vertex.posedWorldRuntime=moved.toArray();
      vertex.relativeBodyRestBlender=Object.fromEntries(Object.entries(restBodyMatrices).map(([name,matrix])=>{
        const body=avatar.vrm.humanoid.getRawBoneNode(name);
        const point=moved.clone().applyMatrix4(body.matrixWorld.clone().invert()).applyMatrix4(matrix);
        return[name,[point.x,-point.z,point.y]];
      }));
      vertex.colliderDistances=colliderPose.map(c=>({bone:c.bone,
        center:c.center,distance:moved.distanceTo(new THREE.Vector3(...c.center)),
        signedSurfaceDistance:moved.distanceTo(new THREE.Vector3(...c.center))-c.radius}));
    }
    delete record.meshObject;
  }
  const jointPose=[];
  for(const spring of json.extensions.VRMC_springBone.springs.filter(s=>s.name.startsWith('ponytail_'))){
    jointPose.push({name:spring.name,joints:await Promise.all(spring.joints.map(async joint=>{
      const node=await avatar.resources.parser.getDependency('node',joint.node);
      return{bone:json.nodes[joint.node].name,hitRadius:joint.hitRadius,
        position:node.getWorldPosition(new THREE.Vector3()).toArray()};
    }))});
  }
  const report={vrmPath:input,vrmSha256:inputSha,time,mode,physics:true,
    motionScriptSha256:motionSha,
    records,colliders:colliderPose,jointPose,
    sourceUnchanged:sha(await fs.readFile(input))===inputSha,
    note:'Rest Blender coordinates convert runtime world [x,y,z] to Blender [x,-z,y] for this identity root. Joint/vertex distances are actual runtime frames; colliders constrain chain tail particles, not every skinned surface point.'};
  await fs.writeFile(output,JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
}finally{avatar.dispose();}

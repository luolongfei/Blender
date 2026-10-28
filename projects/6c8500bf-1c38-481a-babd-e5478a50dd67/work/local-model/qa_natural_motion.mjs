import {projectPath, toolPath} from '../../../../pipeline/context.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
import {testTransitions} from './qa_motion_transitions.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),option=n=>args.includes(n)?args[args.indexOf(n)+1]:null;
const input=path.resolve(option('--input')||path.join(P,'exports/expression-review/avatar-preview.vrm'));
const output=path.resolve(option('--output')||path.join(P,'work/local-model/natural-motion-report.json'));
const {avatar,inputSha}=await loadRuntime(input);
const motionScript=path.join(P,'previews/natural-motion.mjs');
const report={input,vrmPath:input,vrmSha256:inputSha,
  motionScript,motionScriptSha256:sha(await fs.readFile(motionScript)),stepSeconds:1/60,scope:
  'Real three-vrm bones, spring simulation and selected footwear skinned vertices. '
  +'Ground contact tests do not test all mesh intersections, garment clearance, visual quality or universal animation compatibility.'};
try{
  avatar.model.updateMatrixWorld(true);
  const shoeSamples=[],soleTracking=[],v=new THREE.Vector3();
  avatar.model.traverse(mesh=>{
    if(!mesh.isMesh)return;mesh.skeleton?.update();
    const indices=[];
    for(let i=0;i<mesh.geometry.attributes.position.count;i++){
      mesh.getVertexPosition(i,v).applyMatrix4(mesh.matrixWorld);
      if(v.y<.115)indices.push(i);
      if(v.y<.006&&i%8===0)soleTracking.push({mesh,index:i,side:v.x>0?'left':'right'});
    }
    if(indices.length)shoeSamples.push({mesh,indices});
  });
  report.footwearVertices=shoeSamples.reduce((n,s)=>n+s.indices.length,0);
  const restBounds=new THREE.Box3().setFromObject(avatar.model);
  report.restMinimumY=restBounds.min.y;
  function sample(){
    avatar.model.updateMatrixWorld(true);
    let minimumY=Infinity;
    for(const {mesh,indices}of shoeSamples){
      mesh.skeleton?.update();
      for(const i of indices){
        mesh.getVertexPosition(i,v).applyMatrix4(mesh.matrixWorld);
        if(![v.x,v.y,v.z].every(Number.isFinite))throw Error('Nonfinite footwear vertex');
        minimumY=Math.min(minimumY,v.y);
      }
    }
    return minimumY;
  }
  const motion=await fs.readFile(toolPath('public/elyth_base_walk_v1.vrma'));
  await avatar.loadVRMA(new File([motion],'standard-walk.vrma'));avatar.play();
  avatar.mixer.update(0);avatar.vrm.update(0);avatar.springManager?.reset();
  report.animationFirstPosePrimedBeforePhysics=true;
  const baseline={frames:180,minShoeY:Infinity,maxShoeMinimumY:-Infinity,belowGroundFrames:0};
  for(let i=0;i<180;i++){
    avatar.tick(1/60);
    const floor=sample();baseline.minShoeY=Math.min(baseline.minShoeY,floor);
    baseline.maxShoeMinimumY=Math.max(baseline.maxShoeMinimumY,floor);
    if(floor<-.001)baseline.belowGroundFrames++;
  }
  report.standard=baseline;avatar.stop();avatar.clearMotion();
  const natural=new NaturalMotion(avatar.vrm);
  report.natural={};
  for(const mode of ['idle','walk','greet']){
    avatar.vrm.humanoid.resetNormalizedPose();avatar.vrm.update(0);
    natural.originZ=0;natural.transition=null;natural.mode=mode;natural.modeStart=0;
    const result={frames:240,minShoeY:Infinity,maxShoeMinimumY:-Infinity,
      belowGroundFrames:0,maxIKReachError:0,maxStanceAnchorError:0,stanceSamples:0,
      flatSupportSoleHorizontalDriftMeters:0,flatSupportVertexComparisons:0,
      maxKneeFlexDegrees:0,flatSupportScope:'Actual sole vertices with rest y<6mm, every 8th index; contiguous phase .15-.42 within each support cycle.'};
    const supportHistory=new Map();
    natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
    for(let i=1;i<=240;i++){
      natural.apply(i/60);avatar.vrm.update(1/60);
      const floor=sample();result.minShoeY=Math.min(result.minShoeY,floor);
      result.maxShoeMinimumY=Math.max(result.maxShoeMinimumY,floor);
      if(floor<-.001)result.belowGroundFrames++;
      result.maxIKReachError=Math.max(result.maxIKReachError,natural.lastReachError);
      for(const side of ['left','right']){
        const knee=avatar.vrm.humanoid.getNormalizedBoneNode(side+'LowerLeg');
        result.maxKneeFlexDegrees=Math.max(result.maxKneeFlexDegrees,
          Math.abs(THREE.MathUtils.radToDeg(new THREE.Euler().setFromQuaternion(knee.quaternion).x)));
      }
      for(const [side,contact]of Object.entries(natural.lastContacts)){
        if(!contact)continue;
        const foot=avatar.vrm.humanoid.getNormalizedBoneNode(side+'Foot');
        const point=new THREE.Vector3(...contact.localPivot).applyMatrix4(foot.matrixWorld);
        result.maxStanceAnchorError=Math.max(result.maxStanceAnchorError,
          point.distanceTo(new THREE.Vector3(...contact.point)));
        result.stanceSamples++;
      }
      for(const entry of soleTracking){
        const contact=natural.lastContacts[entry.side];
        if(!contact||contact.phase<.15||contact.phase>.42)continue;
        const key=entry.mesh.uuid+':'+entry.index+':'+contact.cycle;
        entry.mesh.getVertexPosition(entry.index,v).applyMatrix4(entry.mesh.matrixWorld);
        const previous=supportHistory.get(key);
        if(previous){
          result.flatSupportSoleHorizontalDriftMeters=Math.max(
            result.flatSupportSoleHorizontalDriftMeters,
            Math.hypot(v.x-previous.x,v.z-previous.z));
          result.flatSupportVertexComparisons++;
        }else supportHistory.set(key,v.clone());
      }
    }
    report.natural[mode]=result;
  }
  report.transitions=await testTransitions(avatar);
  report.sourceUnchanged=sha(await fs.readFile(input))===inputSha;
  report.motionUnchanged=sha(await fs.readFile(motionScript))===report.motionScriptSha256;
  if(!report.sourceUnchanged)throw Error('Source changed during test');
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
}finally{avatar.dispose();}

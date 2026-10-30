// Actual locked Three/three-vrm CPU simulation. No Blender/render/model writes.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {projectPath,toolPath} from '../../../../pipeline/context.mjs';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const P=projectPath(),args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input',path.join(P,'work/runtime-r06/preview-tail-colliders-r09/review-preview.vrm')));
const output=path.resolve(arg('--output',path.join(P,'work/runtime-r06/r09-tail-motion-baseline.json')));
const duration=Number(arg('--duration','12')),dt=Number(arg('--dt',String(1/60)));
const overrides=JSON.parse(args.includes('--settings-file')?await fs.readFile(arg('--settings-file'),'utf8'):arg('--settings','{}'));
const surface=args.includes('--surface'),surfaceEvery=args.includes('--surface-all');
const loaded=await loadRuntime(input),{avatar,inputSha,json}=loaded,vrm=avatar.vrm;
const manager=vrm.springBoneManager,parser=avatar.resources.parser;
const v=()=>new THREE.Vector3(),point=v(),direction=v();
const records=[],sourceBytes=await fs.readFile(input),original=JSON.parse(sourceBytes.subarray(20,20+sourceBytes.readUInt32LE(12)));
const nameByObject=new Map();
for(let i=0;i<json.nodes.length;i++)nameByObject.set(await parser.getDependency('node',i),json.nodes[i].name);
const jointByBone=new Map([...manager.joints].map(j=>[j.bone,j]));
const tailJoints=[...manager.joints].filter(j=>nameByObject.get(j.bone)?.startsWith('RT06 tail_'));
for(const j of tailJoints){
  for(const [key,value] of Object.entries(overrides)){
    if(key==='sleeveCapsules')continue;
    if(key==='hitRadiiBySide')j.settings.hitRadius=value[nameByObject.get(j.bone).match(/tail_([LR])_/)[1]][Number(nameByObject.get(j.bone).match(/_(\d+)$/)[1])];
    else if(key==='hitRadii')j.settings.hitRadius=value[Number(nameByObject.get(j.bone).match(/_(\d+)$/)[1])];
    else if(key==='stiffnesses')j.settings.stiffness=value[Number(nameByObject.get(j.bone).match(/_(\d+)$/)[1])];
    else if(key==='gravityDir')j.settings.gravityDir.fromArray(value);
    else if(key==='center')j.center=vrm.humanoid.getRawBoneNode(value);
    else if(key in j.settings)j.settings[key]=value;
    else throw Error('Unsupported explicit diagnostic setting '+key);
  }
}
if(overrides.sleeveCapsules){
  const {VRMSpringBoneCollider,VRMSpringBoneColliderShapeCapsule}=await import(pathToFileURL(toolPath('public/vendor/three-vrm.module.js')));
  const group={name:'R09 measured puff sleeves',colliders:[]};
  for(const row of overrides.sleeveCapsules){
    const parent=vrm.humanoid.getRawBoneNode(row.bone),shape=new VRMSpringBoneColliderShapeCapsule({offset:new THREE.Vector3(...row.offset),tail:new THREE.Vector3(...row.tail),radius:row.radius});
    const collider=new VRMSpringBoneCollider(shape);parent.add(collider);group.colliders.push(collider);
  }
  for(const joint of tailJoints)joint.colliderGroups=[...joint.colliderGroups,group];
  manager._isSortedJointsDirty=true;
}
function pose(time,motion){
  vrm.humanoid.resetNormalizedPose();
  for(const [name,z] of [['leftUpperArm',-1.15],['rightUpperArm',1.15]])vrm.humanoid.getNormalizedBoneNode(name).rotation.z=z;
  if(motion){const n=vrm.humanoid.getNormalizedBoneNode('head');n.rotation.y=.35*Math.sin(time*1.2);n.rotation.x=.10*Math.sin(time*.9);vrm.humanoid.getNormalizedBoneNode('spine').rotation.z=.06*Math.sin(time*1.7);}
  vrm.expressionManager.resetValues();vrm.humanoid.update();vrm.expressionManager.update();vrm.nodeConstraintManager?.update();vrm.scene.updateMatrixWorld(true);
}
function update(time,motion){pose(time,motion);manager.update(dt);vrm.scene.updateMatrixWorld(true);}
function reset(){pose(0,false);manager.reset();vrm.scene.updateMatrixWorld(true);}
for(const side of ['L','R']){
  const name='RT06 Cards / R02 Strands / R02 Twin tail '+side,ni=json.nodes.findIndex(n=>n.name===name);
  const node=await parser.getDependency('node',ni),meshes=node.getObjectsByProperty('isMesh',true);
  if(meshes.length!==1)throw Error('Expected one tail primitive');
  const mesh=meshes[0],joints=tailJoints.filter(j=>nameByObject.get(j.bone).startsWith('RT06 tail_'+side+'_')).sort((a,b)=>nameByObject.get(a.bone).localeCompare(nameByObject.get(b.bone)));
  if(joints.length!==6)throw Error('Expected six actual simulated tail joints');
  const groupInfo=mesh.geometry.attributes.skinIndex,weights=mesh.geometry.attributes.skinWeight,used=new Set();
  for(let i=0;i<weights.count;i++)for(let c=0;c<4;c++)if(weights.getComponent(i,c)>1e-7)used.add(nameByObject.get(mesh.skeleton.bones[groupInfo.getComponent(i,c)]));
  records.push({side,name,mesh,joints,positions:Array.from({length:weights.count},v),weightedBones:[...used].sort()});
}
const body=[],alphaImages=new Map();let alphaManifest;
if(surface){
  const folder=path.join(P,'work/runtime-r06/r09-tail-alpha-audit');
  alphaManifest=JSON.parse(await fs.readFile(path.join(folder,'manifest.json'),'utf8'));
  if(alphaManifest.sourceSha256!==inputSha)throw Error('Alpha manifest refers to another VRM');
  for(const [id,desc] of Object.entries(alphaManifest.images)){
    const data=await fs.readFile(path.join(folder,desc.file));if(sha(data)!==desc.sha256)throw Error('Alpha bytes changed');
    alphaImages.set(Number(id),{...desc,data});
  }
  for(let i=0;i<json.nodes.length;i++){
    const name=json.nodes[i].name;
    if(!['Blouse_Tucked_Bodice','Blouse_Puff_Sleeve_L','Blouse_Puff_Sleeve_R','Blouse_PeterPan_Collar_L','Blouse_PeterPan_Collar_R'].includes(name))continue;
    const node=await parser.getDependency('node',i);
    for(const mesh of node.getObjectsByProperty('isMesh',true))body.push({name,mesh,positions:Array.from({length:mesh.geometry.attributes.position.count},v)});
  }
}
function contactSample(){
  const cell=.025,grid=new Map(),faces=[];
  function cells(ps,cb){
    const ranges=['x','y','z'].map(axis=>[Math.floor(Math.min(...ps.map(p=>p[axis]))/cell),Math.floor(Math.max(...ps.map(p=>p[axis]))/cell)]);
    for(let x=ranges[0][0];x<=ranges[0][1];x++)for(let y=ranges[1][0];y<=ranges[1][1];y++)for(let z=ranges[2][0];z<=ranges[2][1];z++)cb(`${x},${y},${z}`);
  }
  for(const item of body){
    item.mesh.skeleton?.update();for(let i=0;i<item.positions.length;i++)item.mesh.getVertexPosition(i,item.positions[i]).applyMatrix4(item.mesh.matrixWorld);
    const ids=item.mesh.geometry.index.array;
    for(let i=0;i<ids.length;i+=3){
      const ps=[item.positions[ids[i]],item.positions[ids[i+1]],item.positions[ids[i+2]]];
      if(Math.max(...ps.map(p=>p.y))<1.10)continue;
      const index=faces.length;faces.push({body:item.name,ps,indices:[ids[i],ids[i+1],ids[i+2]]});
      cells(ps,key=>{if(!grid.has(key))grid.set(key,[]);grid.get(key).push(index);});
    }
  }
  const dir=v(),e1=v(),e2=v(),pvec=v(),tvec=v(),qvec=v();
  function intersect(a,b,c,d,e){
    dir.subVectors(b,a);e1.subVectors(d,c);e2.subVectors(e,c);pvec.crossVectors(dir,e2);const det=e1.dot(pvec);
    if(Math.abs(det)<1e-13)return null;tvec.subVectors(a,c);const u=tvec.dot(pvec)/det;if(u<0||u>1)return null;
    qvec.crossVectors(tvec,e1);const w=dir.dot(qvec)/det;if(w<0||u+w>1)return null;const t=e2.dot(qvec)/det;
    return t>=0&&t<=1?t:null;
  }
  const rows=[];
  for(const item of records){
    const ids=item.mesh.geometry.index.array,uv=item.mesh.geometry.attributes.uv,seen=new Set(),examples=[],byBody={};let raw=0,visible=0;
    const desc=alphaManifest.meshes[item.name],alpha=alphaImages.get(desc.image);
    for(let i=0;i<ids.length;i+=3)for(let corner=0;corner<3;corner++){
      const a=ids[i+corner],b=ids[i+(corner+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;if(seen.has(key))continue;seen.add(key);
      const ps=[item.positions[a],item.positions[b]],candidates=new Set();cells(ps,key=>{for(const j of grid.get(key)||[])candidates.add(j);});
      for(const j of candidates){
        const face=faces[j],t=intersect(...ps,...face.ps);if(t===null)continue;raw++;
        const u=uv.getX(a)*(1-t)+uv.getX(b)*t,w=uv.getY(a)*(1-t)+uv.getY(b)*t;
        const x=Math.max(0,Math.min(alpha.width-1,Math.floor(u*alpha.width))),y=Math.max(0,Math.min(alpha.height-1,Math.floor(w*alpha.height)));
        const opacity=alpha.data[y*alpha.width+x]/255*desc.alphaFactor;if(opacity<.15)continue;
        visible++;byBody[face.body]=(byBody[face.body]||0)+1;
        if(examples.length<8)examples.push({body:face.body,edge:[a,b],bodyFace:face.indices,point:ps[0].clone().lerp(ps[1],t).toArray(),alpha:opacity});
      }
    }
    rows.push({side:item.side,geometricIntersectionPairs:raw,alphaFilteredIntersectionPairs:visible,byBody,examples});
  }
  return {bodyTriangles:faces.length,alphaThreshold:.15,sides:rows,scope:'Actual skinned tail edges versus bodice, two sleeves and two collar halves; alpha sampled at each crossing. One direction only; no coplanar test.'};
}
function colliderDistance(c,p,r=0){return c.shape.calculateCollision(c.colliderMatrix,p,r,direction);}
function restWorld(bone,cache){
  if(cache.has(bone))return cache.get(bone);
  const own=jointByBone.get(bone),parent=bone.parent;
  let result;
  if(own)result=restWorld(parent,cache).clone().multiply(own._initialLocalMatrix);
  else if(parent&&jointByBone.has(parent))result=restWorld(parent,cache).clone().multiply(bone.matrix);
  else result=bone.matrixWorld;
  cache.set(bone,result);return result;
}
function sample(time,motion,label){
  vrm.scene.updateMatrixWorld(true);const result={time,motion,label,headYawRadians:motion?.35*Math.sin(time*1.2):0,headPitchRadians:motion?.1*Math.sin(time*.9):0,spineRollRadians:motion?.06*Math.sin(time*1.7):0,tails:[]};
  for(const item of records){
    const {mesh,joints}=item;mesh.skeleton.update();
    const positions=item.positions;for(let i=0;i<positions.length;i++)mesh.getVertexPosition(i,positions[i]).applyMatrix4(mesh.matrixWorld);
    const heads=joints.map(j=>j.bone.getWorldPosition(v()));heads.push(joints.at(-1).child.getWorldPosition(v()));
    const dirs=heads.slice(1).map((p,i)=>p.clone().sub(heads[i]).normalize());
    const angles=dirs.slice(1).map((d,i)=>THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(d.dot(dirs[i]),-1,1))));
    const groups=joints[0].colliderGroups,clearance={};
    for(const group of groups){
      let min=Infinity,index=-1,inside=0;
      for(let i=0;i<positions.length;i++){
        const distance=Math.min(...group.colliders.map(c=>colliderDistance(c,positions[i])));
        if(distance<min){min=distance;index=i;}if(distance<0)inside++;
      }
      clearance[group.name]={minimumSurfaceSignedDistanceMeters:min,verticesInside:inside,totalVertices:positions.length,worstVertex:index,worstWorld:positions[index].toArray()};
    }
    const particles=joints.map((joint,i)=>({bone:nameByObject.get(joint.bone),position:heads[i+1].toArray(),
      hitRadius:joint.settings.hitRadius,minimumColliderMarginMeters:Math.min(...groups.flatMap(g=>g.colliders.map(c=>colliderDistance(c,heads[i+1],joint.settings.hitRadius)))),
      angleFromRestLocalDegrees:THREE.MathUtils.radToDeg(joint.bone.quaternion.angleTo(joint._initialLocalRotation))}));
    const cache=new Map(),matrix=mesh.matrixWorld.clone().multiply(mesh.bindMatrixInverse),boneMatrices=mesh.skeleton.bones.map((b,i)=>restWorld(b,cache).clone().multiply(mesh.skeleton.boneInverses[i]));
    const attr=mesh.geometry.attributes,si=attr.skinIndex,sw=attr.skinWeight,off=v(),temp=v();let maxMove=0,tipSum=v(),tipOff=v(),tipCount=0;
    const threshold=Math.min(...positions.map(p=>p.y))+.025;
    for(let i=0;i<positions.length;i++){
      point.fromBufferAttribute(attr.position,i).applyMatrix4(mesh.bindMatrix);off.set(0,0,0);
      for(let c=0;c<4;c++){const w=sw.getComponent(i,c);if(w)off.addScaledVector(temp.copy(point).applyMatrix4(boneMatrices[si.getComponent(i,c)]),w);}
      off.applyMatrix4(matrix);maxMove=Math.max(maxMove,off.distanceTo(positions[i]));
      if(positions[i].y<=threshold){tipSum.add(positions[i]);tipOff.add(off);tipCount++;}
    }
    result.tails.push({side:item.side,jointWorldHeads:heads.map(p=>p.toArray()),interSegmentAnglesDegrees:angles,maxInterSegmentAngleDegrees:Math.max(...angles),
      particles,clearance,maximumOnOffDisplacementMeters:maxMove,lowest25mmMeanWorld:tipSum.divideScalar(tipCount).toArray(),sameVerticesPhysicsOffMeanWorld:tipOff.divideScalar(tipCount).toArray()});
  }
  if(surface&&(surfaceEvery||!motion||Math.abs(time-1.2)<1e-6||Math.abs(time-4.4)<1e-6))result.garmentContact=contactSample();
  return result;
}
try{
  reset();
  if(args.includes('--fit-only')){
    const fits=[];
    for(const side of ['L','R']){
      const part=side==='L'?'left':'right',upper=vrm.humanoid.getRawBoneNode(part+'UpperArm'),lower=vrm.humanoid.getRawBoneNode(part+'LowerArm');
      const inverse=upper.matrixWorld.clone().invert(),axis=lower.getWorldPosition(v()).applyMatrix4(inverse).normalize();
      const row=body.find(b=>b.name==='Blouse_Puff_Sleeve_'+side),mesh=row.mesh;mesh.skeleton.update();
      const points=Array.from({length:mesh.geometry.attributes.position.count},(_,i)=>mesh.getVertexPosition(i,v()).applyMatrix4(mesh.matrixWorld).applyMatrix4(inverse));
      const ts=points.map(p=>p.dot(axis)),min=Math.min(...ts),max=Math.max(...ts);
      const center=points.reduce((sum,p,i)=>sum.add(p.clone().addScaledVector(axis,-ts[i])),v()).divideScalar(points.length);
      const radials=points.map((p,i)=>p.clone().addScaledVector(axis,-ts[i]).distanceTo(center)).sort((a,b)=>a-b);
      fits.push({side,bone:part+'UpperArm',axis:axis.toArray(),radialCenter:center.toArray(),minimumAlong:min,maximumAlong:max,
        radius95:radials[Math.floor(radials.length*.95)],radiusMax:radials.at(-1),vertices:points.length});
    }
    const tailWidths=records.map(item=>{
      const mesh=item.mesh,attr=mesh.geometry.attributes,buckets=new Map(),coordinates={head:2};
      [7,13,19,25,30,34].forEach((row,i)=>coordinates[`RT06 tail_${item.side}_${String(i).padStart(2,'0')}`]=row);
      mesh.skeleton.update();
      for(let i=0;i<attr.position.count;i++){
        let parameter=0;for(let c=0;c<4;c++){const w=attr.skinWeight.getComponent(i,c);if(w)parameter+=w*coordinates[nameByObject.get(mesh.skeleton.bones[attr.skinIndex.getComponent(i,c)])];}
        const row=Math.round(parameter);if(!buckets.has(row))buckets.set(row,[]);
        buckets.get(row).push(mesh.getVertexPosition(i,v()).applyMatrix4(mesh.matrixWorld));
      }
      return {side:item.side,rows:[...buckets].sort((a,b)=>a[0]-b[0]).map(([row,points])=>{
        const center=points.reduce((sum,p)=>sum.add(p),v()).divideScalar(points.length),radii=points.map(p=>p.distanceTo(center)).sort((a,b)=>a-b);
        return {row,parameter:row/34,center:center.toArray(),cardEnvelopeRadiusMax:radii.at(-1),cardEnvelopeRadius95:radii[Math.floor(radii.length*.95)],vertices:points.length};
      })};
    });
    await fs.writeFile(output,JSON.stringify({inputSha,fits,tailWidths,scope:'Actual arms-down sleeve skin in each raw upper-arm local frame; exact bounds. Card envelope at rest uses inverse authored skin interpolation parameter; first three anchored rows share parameter2.'},null,2),{flag:'wx'});
    console.log(JSON.stringify({output,fits,tailWidths},null,2));
  }else{
  const samples=[sample(0,false,'physics reset')];
  for(let f=1;f<=Math.round(2/dt);f++)update(0,false);
  samples.push(sample(0,false,'stationary after 2s settling'));
  for(let f=1;f<=Math.round(duration/dt);f++){
    const t=f*dt;update(t,true);
    if(f%Math.round(.2/dt)===0)samples.push(sample(t,true,'viewer motion'));
  }
  const summary=records.map(({side,name,mesh,weightedBones,joints})=>{
    const frames=samples.map(frame=>({time:frame.time,label:frame.label,...frame.tails.find(t=>t.side===side)}));
    const worst=frames.toSorted((a,b)=>b.maxInterSegmentAngleDegrees-a.maxInterSegmentAngleDegrees)[0];
    return {side,name,vertices:mesh.geometry.attributes.position.count,weightedBones,settings:joints.map(j=>({bone:nameByObject.get(j.bone),...j.settings,gravityDir:j.settings.gravityDir.toArray(),center:j.center?nameByObject.get(j.center):null})),
      maxInterSegmentAngle:worst.maxInterSegmentAngleDegrees,worstAngleTime:worst.time,
      minimumSurfaceDistanceByGroup:Object.fromEntries(joints[0].colliderGroups.map(g=>[g.name,Math.min(...frames.map(f=>f.clearance[g.name].minimumSurfaceSignedDistanceMeters))])),
      maxParticlePenetrationMeters:Math.min(...frames.flatMap(f=>f.particles.map(p=>p.minimumColliderMarginMeters)))};
  });
  const report={input,inputSha,dt,duration,overrides,viewerSha:sha(await fs.readFile(path.join(P,'work/local-model/runtime_viewer_r07.html'))),
    threeSha:sha(await fs.readFile(toolPath('public/vendor/three.module.js'))),threeVrmSha:sha(await fs.readFile(toolPath('public/vendor/three-vrm.module.js'))),
    exactViewerMotion:'head.y=.35*sin(t*1.2), head.x=.10*sin(t*.9), spine.z=.06*sin(t*1.7); upperarms.z= -/+1.15; same humanoid/constraint/matrix/spring update order',
    sourceUnchanged:sha(await fs.readFile(input))===inputSha,sourceSprings:original.extensions.VRMC_springBone,summary,samples,
    limits:['CPU loader strips only materials/textures, retains actual geometry, hierarchy, skin and springs.',
      'Surface clearance uses every card vertex without alpha masking; negative clearance is a conservative geometric indication, not visible opacity intersection proof.',
      surface?'Selected frames include actual skinned garment intersection tests filtered by embedded alpha.':'No actual garment-surface test in this first probe. Existing collision volumes may not contain the garment.',
      'Deterministic 60Hz and motion starting at t=0; browser clock phase/timestep may differ.']};
  await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2),{flag:'wx'});
  console.log(JSON.stringify({output,inputSha,summary},null,2));
  }
}finally{avatar.dispose();}

// Actual upper-sleeve plane intersections and perimeter PCA; read-only CPU.
import fs from 'node:fs/promises';import path from 'node:path';
import {projectPath} from '../../../../pipeline/context.mjs';
import {THREE,loadRuntime} from './runtime_cpu.mjs';
const P=projectPath(),input=path.join(P,'work/runtime-r06/preview-tail-colliders-r09/review-preview.vrm');
const {avatar,json,inputSha}=await loadRuntime(input),vrm=avatar.vrm;
const vec=()=>new THREE.Vector3();
try{
  vrm.humanoid.resetNormalizedPose();
  for(const [name,z] of [['leftUpperArm',-1.15],['rightUpperArm',1.15]])vrm.humanoid.getNormalizedBoneNode(name).rotation.z=z;
  vrm.humanoid.update();vrm.nodeConstraintManager?.update();vrm.scene.updateMatrixWorld(true);vrm.springBoneManager.reset();
  const rows=[];
  for(const side of ['L','R']){
    const boneName=side==='L'?'leftUpperArm':'rightUpperArm',bone=vrm.humanoid.getRawBoneNode(boneName),inverse=bone.matrixWorld.clone().invert();
    const ni=json.nodes.findIndex(n=>n.name==='Blouse_Puff_Sleeve_'+side),node=await avatar.resources.parser.getDependency('node',ni),mesh=node.getObjectsByProperty('isMesh',true)[0];mesh.skeleton.update();
    const co=Array.from({length:mesh.geometry.attributes.position.count},(_,i)=>mesh.getVertexPosition(i,vec()).applyMatrix4(mesh.matrixWorld).applyMatrix4(inverse));
    const ids=mesh.geometry.index.array,sections=[];
    for(const y of [-.02,0,.02,.04,.06,.08,.10,.12,.14,.16,.18,.20,.22]){
      const segments=[];
      for(let i=0;i<ids.length;i+=3){
        const points=[];for(let k=0;k<3;k++){
          const a=co[ids[i+k]],b=co[ids[i+(k+1)%3]];
          if((a.y<=y&&b.y>y)||(b.y<=y&&a.y>y)){const t=(y-a.y)/(b.y-a.y);points.push([a.x+t*(b.x-a.x),a.z+t*(b.z-a.z)]);}
        }
        if(points.length===2)segments.push(points);
      }
      if(!segments.length){sections.push({y,segments:0});continue;}
      const samples=[];for(const [a,b] of segments){const length=Math.hypot(b[0]-a[0],b[1]-a[1]),n=Math.max(1,Math.ceil(length/.001));for(let k=0;k<n;k++){const t=(k+.5)/n;samples.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),length/n]);}}
      const weight=samples.reduce((s,p)=>s+p[2],0),center=[0,0];for(const p of samples){center[0]+=p[0]*p[2]/weight;center[1]+=p[1]*p[2]/weight;}
      let xx=0,xz=0,zz=0;for(const p of samples){const x=p[0]-center[0],z=p[1]-center[1];xx+=x*x*p[2]/weight;xz+=x*z*p[2]/weight;zz+=z*z*p[2]/weight;}
      const angle=.5*Math.atan2(2*xz,xx-zz),major=[Math.cos(angle),Math.sin(angle)],minor=[-major[1],major[0]];
      const projections=samples.map(p=>[(p[0]-center[0])*major[0]+(p[1]-center[1])*major[1],(p[0]-center[0])*minor[0]+(p[1]-center[1])*minor[1]]);
      const majorRange=[Math.min(...projections.map(p=>p[0])),Math.max(...projections.map(p=>p[0]))],minorRange=[Math.min(...projections.map(p=>p[1])),Math.max(...projections.map(p=>p[1]))];
      sections.push({y,segments:segments.length,centerXZ:center,covarianceXZ:[[xx,xz],[xz,zz]],majorAxisXZ:major,minorAxisXZ:minor,majorRange,minorRange,
        majorExtent:majorRange[1]-majorRange[0],minorExtent:minorRange[1]-minorRange[0],
        rawXRange:[Math.min(...samples.map(p=>p[0])),Math.max(...samples.map(p=>p[0]))],rawZRange:[Math.min(...samples.map(p=>p[1])),Math.max(...samples.map(p=>p[1]))],
        planeSegmentsXZ:segments});
    }
    rows.push({side,bone:boneName,boneWorldMatrix:bone.matrixWorld.toArray(),sections});
  }
  const output=path.join(P,'work/runtime-r06/r09-sleeve-cross-sections.json');
  await fs.writeFile(output,JSON.stringify({input,inputSha,rows,scope:'Actual posed/arm-skinned triangles intersecting bone-local Y planes. PCA weighted by segment length; includes both Solidify layers. No model changes.'},null,2),{flag:'wx'});
  console.log(JSON.stringify({output,rows:rows.map(r=>({...r,sections:r.sections.map(({planeSegmentsXZ,...s})=>s)}))},null,2));
}finally{avatar.dispose();}

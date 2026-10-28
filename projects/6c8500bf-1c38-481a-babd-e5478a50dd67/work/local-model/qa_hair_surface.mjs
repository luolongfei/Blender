import {projectPath, toolPath} from '../../../../pipeline/context.mjs';
// Exact skinned ponytail-edge / upper-garment-triangle diagnostic.
// All candidate parameters are already in the supplied QA VRM; no mutation.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),arg=(key,def)=>args.includes(key)?args[args.indexOf(key)+1]:def;
const input=path.resolve(arg('--input',path.join(P,'exports/expression-review/avatar-preview.vrm')));
const output=path.resolve(arg('--output',path.join(P,'work/qa-hair-physics/hair-surface.json')));
const mode=arg('--mode','walk'),end=Number(arg('--end','8')),every=Number(arg('--every','6'));
const staticOnly=args.includes('--static-only');
const bidirectional=args.includes('--bidirectional');
const primeStart=args.includes('--prime-start');
const motionPath=mode==='standard'?toolPath('public/elyth_base_walk_v1.vrma'):
  path.join(P,'previews/natural-motion.mjs');
const motionSHA=sha(await fs.readFile(motionPath));
const {avatar,inputSha}=await loadRuntime(input);
try{
  const source=await fs.readFile(input),json=JSON.parse(source.subarray(20,20+source.readUInt32LE(12)));
  const groups={hair:[],body:[]},point=new THREE.Vector3();
  for(let i=0;i<json.nodes.length;i++){
    const node=json.nodes[i];
    if(node.mesh===undefined)continue;
    const type=node.extras?.part==='hair'&&node.extras?.hair_region==='ponytail'?'hair':
      node.extras?.part==='upper_body'||node.extras?.part==='neck'||
      node.extras?.deform_role==='completed_anatomy'?'body':null;
    if(!type)continue;
    const object=await avatar.resources.parser.getDependency('node',i);
    for(const mesh of object.getObjectsByProperty('isMesh',true)){
      const vertices=Array.from({length:mesh.geometry.attributes.position.count},()=>new THREE.Vector3());
      groups[type].push({mesh,name:node.name,vertices,indices:mesh.geometry.index.array});
    }
  }
  const update=()=>{
    avatar.model.updateMatrixWorld(true);
    for(const group of [...groups.hair,...groups.body]){
      group.mesh.skeleton?.update();
      for(let i=0;i<group.vertices.length;i++)
        group.mesh.getVertexPosition(i,group.vertices[i]).applyMatrix4(group.mesh.matrixWorld);
    }
  };
  update();
  const faces=[],edges=[],reverseFaces=[],reverseEdges=[];
  for(const body of groups.body){
    for(let i=0;i<body.indices.length;i+=3){
      const ids=[body.indices[i],body.indices[i+1],body.indices[i+2]];
      if(Math.max(...ids.map(k=>body.vertices[k].y))<1.11)continue;
      faces.push({group:body,ids});
    }
  }
  for(const hair of groups.hair){
    const seen=new Set();
    for(let i=0;i<hair.indices.length;i+=3){
      const ids=[hair.indices[i],hair.indices[i+1],hair.indices[i+2]];
      reverseFaces.push({group:hair,ids});
      for(let k=0;k<3;k++){
        const a=ids[k],b=ids[(k+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
        if(!seen.has(key)){seen.add(key);edges.push({group:hair,ids:[a,b]});}
      }
    }
  }
  if(bidirectional){
    const seenByGroup=new Map();
    for(const face of faces){
      let seen=seenByGroup.get(face.group);
      if(!seen){seen=new Set();seenByGroup.set(face.group,seen);}
      for(let k=0;k<3;k++){
        const a=face.ids[k],b=face.ids[(k+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
        if(!seen.has(key)){seen.add(key);reverseEdges.push({group:face.group,ids:[a,b]});}
      }
    }
  }
  const cell=.025;
  function cells(ps,cb){
    const ranges=['x','y','z'].map(axis=>[
      Math.floor(Math.min(...ps.map(p=>p[axis]))/cell),
      Math.floor(Math.max(...ps.map(p=>p[axis]))/cell)]);
    for(let x=ranges[0][0];x<=ranges[0][1];x++)
      for(let y=ranges[1][0];y<=ranges[1][1];y++)
        for(let z=ranges[2][0];z<=ranges[2][1];z++)cb(`${x},${y},${z}`);
  }
  const dir=new THREE.Vector3(),e1=new THREE.Vector3(),e2=new THREE.Vector3(),
    pvec=new THREE.Vector3(),tvec=new THREE.Vector3(),qvec=new THREE.Vector3();
  function intersect(start,end,p0,p1,p2){
    dir.subVectors(end,start);e1.subVectors(p1,p0);e2.subVectors(p2,p0);
    pvec.crossVectors(dir,e2);const det=e1.dot(pvec);
    if(Math.abs(det)<1e-13)return null;
    tvec.subVectors(start,p0);const u=tvec.dot(pvec)/det;
    if(u<0||u>1)return null;
    qvec.crossVectors(tvec,e1);const v=dir.dot(qvec)/det;
    if(v<0||u+v>1)return null;
    const t=e2.dot(qvec)/det;
    return t>=0&&t<=1?start.clone().addScaledVector(dir,t):null;
  }
  function measure(time){
    update();const grid=new Map();
    faces.forEach((face,index)=>cells(face.ids.map(i=>face.group.vertices[i]),key=>{
      if(!grid.has(key))grid.set(key,[]);grid.get(key).push(index);
    }));
    let count=0;const examples=[],objects={};
    for(const edge of edges){
      const ps=edge.ids.map(i=>edge.group.vertices[i]),candidates=new Set();
      cells(ps,key=>{for(const index of grid.get(key)||[])candidates.add(index);});
      for(const index of candidates){
        const face=faces[index],hit=intersect(...ps,...face.ids.map(i=>face.group.vertices[i]));
        if(!hit)continue;
        count++;objects[face.group.name]=(objects[face.group.name]||0)+1;
        if(examples.length<6)examples.push({hair:edge.group.name,body:face.group.name,
          point:hit.toArray(),hairEdge:edge.ids,bodyTriangle:face.ids});
      }
    }
    const forwardCount=count;
    if(bidirectional){
      grid.clear();
      reverseFaces.forEach((face,index)=>cells(face.ids.map(i=>face.group.vertices[i]),key=>{
        if(!grid.has(key))grid.set(key,[]);grid.get(key).push(index);
      }));
      for(const edge of reverseEdges){
        const ps=edge.ids.map(i=>edge.group.vertices[i]),candidates=new Set();
        cells(ps,key=>{for(const index of grid.get(key)||[])candidates.add(index);});
        for(const index of candidates){
          const face=reverseFaces[index],hit=intersect(...ps,...face.ids.map(i=>face.group.vertices[i]));
          if(!hit)continue;
          count++;objects[edge.group.name]=(objects[edge.group.name]||0)+1;
          if(examples.length<6)examples.push({hair:face.group.name,body:edge.group.name,
            point:hit.toArray(),bodyEdge:edge.ids,hairTriangle:face.ids});
        }
      }
    }
    return{time,intersectionPairs:count,forwardIntersectionPairs:forwardCount,
      reverseIntersectionPairs:count-forwardCount,bodyObjects:objects,examples};
  }
  const natural=new NaturalMotion(avatar.vrm);
  const staticFrames=[];
  if(staticOnly)staticFrames.push({...measure(0),pose:'exported rest, physics not advanced'});
  natural.mode=staticOnly?'idle':mode;natural.modeStart=0;natural.originZ=0;natural.transition=null;
  natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
  if(mode==='standard'&&!staticOnly){
    await avatar.loadVRMA(new File([await fs.readFile(motionPath)],'standard-walk.vrma'));
    avatar.setPhysics(true);avatar.play();avatar.springManager?.reset();
    if(primeStart){
      avatar.mixer.update(0);avatar.vrm.update(0);avatar.springManager?.reset();
    }
  }
  if(staticOnly){
    staticFrames.push({...measure(0),pose:'relaxed idle at t=0, physics not advanced'});
    for(let i=1;i<=240;i++){natural.apply(0);avatar.vrm.update(1/60);}
    staticFrames.push({...measure(4),pose:'same relaxed pose held, physics advanced 4s'});
  }
  const frames=[];
  for(let i=1;!staticOnly&&i<=Math.round(end*60);i++){
    if(mode==='standard')avatar.tick(1/60);
    else{natural.apply(i/60);avatar.vrm.update(1/60);}
    if(i%every===0){const row=measure(i/60);frames.push(row);
      console.log(JSON.stringify({time:row.time,intersectionPairs:row.intersectionPairs}));}
  }
  if(staticOnly)frames.push(...staticFrames);
  const report={input,vrmSha256:inputSha,motionPath,motionSha256:motionSHA,
    mode:staticOnly?'static':mode,end:staticOnly?4:end,timestep:1/60,sampleEveryFrames:staticOnly?null:every,frames:frames.length,
    hairMeshes:groups.hair.map(g=>g.name),bodyMeshes:groups.body.map(g=>g.name),
    testedHairEdges:edges.length,testedUpperGarmentTriangles:faces.length,
    bidirectional,testedReverseBodyEdges:bidirectional?reverseEdges.length:0,
    primedAnimationStart:primeStart,
    framesWithIntersection:frames.filter(f=>f.intersectionPairs>0).length,
    maximumIntersectionPairs:Math.max(0,...frames.map(f=>f.intersectionPairs)),
    scope:'Actual skinned ponytail edges intersecting upper garments, neck and any completed anatomy triangles with rest maxY >=1.11m, 25mm broadphase grid.',
    limitations:[bidirectional?'Bidirectional triangle-edge tests do not detect coplanar overlap.':
      'One-direction edge tests may miss reverse-contained triangle crossings.',
      'Does not test coplanar overlap or measure penetration depth.',
      'Samples specified frames only; does not prove aesthetic or universal collision success.'],
    sourceUnchanged:sha(await fs.readFile(input))===inputSha,
    motionUnchanged:sha(await fs.readFile(motionPath))===motionSHA,perFrame:frames};
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({...report,perFrame:undefined},null,2));
}finally{avatar.dispose();}

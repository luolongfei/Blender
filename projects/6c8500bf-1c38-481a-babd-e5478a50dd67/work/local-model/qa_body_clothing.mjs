import {projectPath, toolPath} from '../../../../pipeline/context.mjs';
// Read-only, bidirectional surface intersections for the completed anatomy.
// Only the actual supplied VRM is loaded; no collider/skin overrides or masks.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const {NaturalMotion}=await import('../../previews/natural-motion.mjs');
const P=projectPath('6c8500bf-1c38-481a-babd-e5478a50dd67');
const args=process.argv.slice(2),arg=(key,def)=>args.includes(key)?args[args.indexOf(key)+1]:def;
const input=path.resolve(arg('--input',path.join(P,'exports/expression-review/avatar-preview.vrm')));
const output=path.resolve(arg('--output',path.join(P,'work/qa-body-motion/body-clothing.json')));
const mode=arg('--mode','walk'),end=Number(arg('--end','8')),every=Number(arg('--every','6'));
const staticOnly=args.includes('--static-only');
const motionPath=mode==='standard'?toolPath('public/elyth_base_walk_v1.vrma'):
  path.join(P,'previews/natural-motion.mjs');
const motionSHA=sha(await fs.readFile(motionPath));
const {avatar,inputSha,json}=await loadRuntime(input);

try{
  const groups={skin:[],clothing:[]};
  let completeAnatomyNodes=0;
  for(let i=0;i<json.nodes.length;i++){
    const node=json.nodes[i],extra=node.extras||{};
    if(node.mesh===undefined)continue;
    const complete=extra.deform_role==='completed_anatomy';
    const type=complete||['leg','hand','neck'].includes(extra.part)?'skin':
      ['upper_body','lower_body'].includes(extra.part)?'clothing':null;
    if(!type)continue;
    if(complete)completeAnatomyNodes++;
    const object=await avatar.resources.parser.getDependency('node',i);
    for(const mesh of object.getObjectsByProperty('isMesh',true)){
      const vertices=Array.from({length:mesh.geometry.attributes.position.count},
        ()=>new THREE.Vector3());
      const indices=mesh.geometry.index?.array;
      if(!indices)throw Error('Expected indexed exported mesh: '+node.name);
      const faces=[],edges=[],seen=new Set();
      for(let k=0;k<indices.length;k+=3){
        const ids=[indices[k],indices[k+1],indices[k+2]];
        faces.push(ids);
        for(let j=0;j<3;j++){
          const a=ids[j],b=ids[(j+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
          if(!seen.has(key)){seen.add(key);edges.push([a,b]);}
        }
      }
      groups[type].push({mesh,name:node.name,role:extra.deform_role||extra.part,
        complete,vertices,faces,edges});
    }
  }
  if(!completeAnatomyNodes&&!args.includes('--allow-incomplete')){
    throw Error('No completed_anatomy nodes: refusing to label a partial body complete.');
  }
  if(!groups.skin.length||!groups.clothing.length)throw Error('Missing inspected skin/clothing groups.');
  const update=()=>{
    avatar.model.updateMatrixWorld(true);
    for(const group of [...groups.skin,...groups.clothing]){
      group.mesh.skeleton?.update();
      for(let i=0;i<group.vertices.length;i++)
        group.mesh.getVertexPosition(i,group.vertices[i]).applyMatrix4(group.mesh.matrixWorld);
    }
  };
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
  function intersection(start,end,p0,p1,p2){
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
  function direction(edgeGroups,faceGroups,reverse){
    const grid=new Map(),faces=[];
    for(const group of faceGroups)for(const ids of group.faces){
      const index=faces.length;faces.push({group,ids});
      cells(ids.map(i=>group.vertices[i]),key=>{
        if(!grid.has(key))grid.set(key,[]);grid.get(key).push(index);
      });
    }
    let count=0,completeBodyCount=0;
    const pairs={},examples=[],examplesByPair={},pairBounds={};
    for(const group of edgeGroups)for(const edge of group.edges){
      const ps=edge.map(i=>group.vertices[i]),candidates=new Set();
      cells(ps,key=>{for(const index of grid.get(key)||[])candidates.add(index);});
      for(const index of candidates){
        const face=faces[index],hit=intersection(...ps,...face.ids.map(i=>face.group.vertices[i]));
        if(!hit)continue;
        const skin=reverse?face.group:group,clothing=reverse?group:face.group;
        count++;if(skin.complete)completeBodyCount++;
        const key=skin.name+' / '+clothing.name;
        pairs[key]=(pairs[key]||0)+1;
        pairBounds[key]??={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
        hit.toArray().forEach((value,k)=>{
          pairBounds[key].min[k]=Math.min(pairBounds[key].min[k],value);
          pairBounds[key].max[k]=Math.max(pairBounds[key].max[k],value);
        });
        const example={skin:skin.name,clothing:clothing.name,
          completedAnatomy:skin.complete,position:hit.toArray(),
          direction:reverse?'clothing-edge/skin-face':'skin-edge/clothing-face',
          edge,triangle:face.ids};
        if(examples.length<12)examples.push(example);
        examplesByPair[key]??=[];
        if(examplesByPair[key].length<6)examplesByPair[key].push(example);
      }
    }
    return{count,completeBodyCount,pairs,examples,examplesByPair,pairBounds};
  }
  function measure(time,pose){
    update();
    const a=direction(groups.skin,groups.clothing,false),
      b=direction(groups.clothing,groups.skin,true),pairs={...a.pairs};
    for(const [key,count]of Object.entries(b.pairs))pairs[key]=(pairs[key]||0)+count;
    const examplesByPair={...a.examplesByPair},pairBounds=structuredClone(a.pairBounds);
    for(const [key,rows]of Object.entries(b.examplesByPair))
      examplesByPair[key]=[...(examplesByPair[key]||[]),...rows].slice(0,10);
    for(const [key,bounds]of Object.entries(b.pairBounds)){
      if(!pairBounds[key])pairBounds[key]=bounds;
      else for(let k=0;k<3;k++){
        pairBounds[key].min[k]=Math.min(pairBounds[key].min[k],bounds.min[k]);
        pairBounds[key].max[k]=Math.max(pairBounds[key].max[k],bounds.max[k]);
      }
    }
    return{time,pose,intersectionPairs:a.count+b.count,
      completedAnatomyIntersectionPairs:a.completeBodyCount+b.completeBodyCount,
      skinEdgeIntersections:a.count,clothingEdgeIntersections:b.count,pairs,
      examplesByPair,pairBounds,examples:[...a.examples,...b.examples].slice(0,18)};
  }
  const natural=new NaturalMotion(avatar.vrm),frames=[];
  if(staticOnly)frames.push(measure(0,'exported rest, physics not advanced'));
  natural.mode=staticOnly?'idle':mode;
  natural.modeStart=0;natural.originZ=0;natural.transition=null;
  natural.apply(0);avatar.vrm.update(0);avatar.springManager?.reset();
  if(mode==='standard'&&!staticOnly){
    await avatar.loadVRMA(new File([await fs.readFile(motionPath)],'standard-walk.vrma'));
    avatar.setPhysics(true);avatar.play();
    avatar.mixer.update(0);avatar.vrm.update(0);avatar.springManager?.reset();
  }
  if(staticOnly){
    frames.push(measure(0,'relaxed t0, physics not advanced'));
    for(let i=1;i<=240;i++){natural.apply(0);avatar.vrm.update(1/60);}
    frames.push(measure(4,'same relaxed pose held with physics for 4s'));
  }else{
    for(let i=1;i<=Math.round(end*60);i++){
      if(mode==='standard')avatar.tick(1/60);
      else{natural.apply(i/60);avatar.vrm.update(1/60);}
      if(i%every===0){
        const row=measure(i/60,mode);frames.push(row);
        console.log(JSON.stringify({time:row.time,intersectionPairs:row.intersectionPairs,
          completedAnatomyIntersectionPairs:row.completedAnatomyIntersectionPairs}));
      }
    }
  }
  const report={input,vrmSha256:inputSha,motionPath,motionSha256:motionSHA,
    mode:staticOnly?'static':mode,completeAnatomyNodes,
    groups:Object.fromEntries(Object.entries(groups).map(([type,values])=>[type,
      values.map(g=>({name:g.name,role:g.role,complete:g.complete,vertices:g.vertices.length,
        triangles:g.faces.length,edges:g.edges.length}))])),
    physics:true,primedAnimationStart:mode==='standard',timestep:1/60,
    sampleEveryFrames:staticOnly?null:every,frames:frames.length,
    framesWithIntersection:frames.filter(f=>f.intersectionPairs>0).length,
    framesWithCompletedAnatomyIntersection:frames.filter(f=>f.completedAnatomyIntersectionPairs>0).length,
    maximumIntersectionPairs:Math.max(0,...frames.map(f=>f.intersectionPairs)),
    scope:'Actual skinned full completed anatomy, existing legs/hands/neck against every exported upper/lower clothing mesh, including skirt, socks and shoes. Bidirectional triangle-edge tests, 25mm broadphase grid, no spatial exclusions.',
    limitations:['Coplanar surface overlap and tangential contacts are not robustly detected.',
      'Counts are edge/triangle pairs, not distinct penetrations or penetration depth.',
      'A fully enclosed skin volume is normal under clothing and is not an intersection.',
      'Reported collar/cuff/shoe seams need geometric review; no automatic exceptions were masked.',
      'The sampled motions and frames do not prove collision freedom for all animation.'],
    sourceUnchanged:sha(await fs.readFile(input))===inputSha,
    motionUnchanged:sha(await fs.readFile(motionPath))===motionSHA,perFrame:frames};
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({...report,groups:undefined,perFrame:undefined}));
}finally{avatar.dispose();}

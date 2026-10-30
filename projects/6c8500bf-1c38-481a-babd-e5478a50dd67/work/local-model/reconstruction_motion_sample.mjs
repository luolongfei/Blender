// Read-only sampling of this project's existing authored NaturalMotion.
// It is procedural choreography with two-bone IK, NOT motion capture.
import fs from 'node:fs/promises';
import path from 'node:path';
import {projectPath} from '../../../../pipeline/context.mjs';
import {THREE, loadRuntime, sha} from './runtime_cpu.mjs';
import {NaturalMotion} from '../../previews/natural-motion.mjs';

const args=process.argv.slice(2);
const option=(name,fallback)=>args.includes(name)?args[args.indexOf(name)+1]:fallback;
const P=projectPath();
const mode=option('--mode','walk');
const fps=Number(option('--fps','30'));
const duration=Number(option('--duration',mode==='walk'?'6.08':mode==='greet'?'6.1':'8'));
const inPlace=args.includes('--in-place');
if(!['idle','walk','greet'].includes(mode))throw Error('Use idle, walk or greet.');
if(![24,30,60].includes(fps)||!(duration>0&&duration<=20))throw Error('Invalid bounded sampling range.');
const relOutput=option('--output',`work/reconstruction-r02/motion-${mode}${inPlace?'-in-place':''}.json`);
if(path.isAbsolute(relOutput)||path.win32.isAbsolute(relOutput)||relOutput.split(/[\\/]/).includes('..'))
  throw Error('Output must be a project-relative path without traversal.');
const input=path.join(P,'exports/expression-review/avatar-preview.vrm');
const output=path.join(P,relOutput);
const runtime=await loadRuntime(input);
try{
  const {avatar,json}=runtime;
  const inverseBinds=new Map();
  avatar.model.updateMatrixWorld(true);
  avatar.model.traverse(mesh=>{
    if(!mesh.isSkinnedMesh)return;
    mesh.skeleton.update();
    const factor=mesh.bindMatrix.clone().multiply(mesh.matrixWorld.clone().invert());
    if(factor.elements.some((v,i)=>Math.abs(v-(i%5===0?1:0))>1e-6))
      throw Error('Nonidentity skin bind factor is not supported.');
    mesh.skeleton.bones.forEach((bone,i)=>{
      const matrix=mesh.skeleton.boneInverses[i];
      const old=inverseBinds.get(bone.name);
      if(old&&matrix.elements.some((v,k)=>Math.abs(v-old.elements[k])>1e-6))
        throw Error('Inconsistent bind matrix for '+bone.name);
      inverseBinds.set(bone.name,matrix.clone());
    });
  });
  const bones=[];
  for(const index of [...new Set(json.skins.flatMap(s=>s.joints))]){
    const name=json.nodes[index].name;
    // Hair and skirt simulation belongs to the newly fitted production model.
    if(/^(hair_|skirt_)/.test(name))continue;
    const node=await avatar.resources.parser.getDependency('node',index);
    bones.push({name,node,index});
  }
  const C=new THREE.Matrix4().set(1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1);
  const Ci=C.clone().invert();
  const motion=new NaturalMotion(avatar.vrm);
  motion.mode=mode;motion.modeStart=0;motion.originZ=0;motion.transition=null;
  avatar.setPhysics(false);
  const poses=[];
  const count=Math.round(duration*fps);
  for(let i=0;i<=count;i++){
    const t=i/fps;
    motion.apply(t);avatar.vrm.update(0);avatar.model.updateMatrixWorld(true);
    const offset=new THREE.Matrix4().makeTranslation(0,0,inPlace?-motion.currentTravel:0);
    const rows=bones.map(({name,node,index})=>{
      const D=offset.clone().multiply(node.matrixWorld).multiply(inverseBinds.get(name));
      return {name,nodeIndex:index,restToPosedBlender:C.clone().multiply(D).multiply(Ci).toArray()};
    });
    poses.push({time:t,frame:i,bones:rows});
  }
  if(sha(await fs.readFile(input))!==runtime.inputSha)throw Error('Source changed during sampling.');
  const script=path.join(P,'previews/natural-motion.mjs');
  const report={version:3,provenance:'Existing project-authored NaturalMotion, two-bone IK; not mocap',
    input:path.relative(P,input).replaceAll('\\','/'),vrmSha256:runtime.inputSha,
    motionScript:'previews/natural-motion.mjs',motionScriptSha256:sha(await fs.readFile(script)),
    mode,fps,duration:count/fps,inPlace,physics:false,boneCount:bones.length,
    matrixStorage:'column-major 16 numeric values, column vectors',runtimeToBlender:C.toArray(),
    poses,sourceUnchanged:true,
    limitations:['Body motion sampled from unchanged migration VRM proportions.',
      'No new face expression transfer, no new hair/cloth simulation.',
      'Blender import must compare evaluated pose matrices and inspect contact/volume.',
      'No claim of final quality or motion-capture provenance.']};
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report)+'\n');
  console.log(JSON.stringify({output,poses:poses.length,bones:bones.length,mode,fps,duration:count/fps}));
}finally{runtime.avatar.dispose();}

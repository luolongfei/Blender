// Read-only actual VRM skinned geometry capture for independent shape renders.
// Rendering these snapshots in another engine is not a browser material check.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE, loadRuntime, sha} from './runtime_cpu.mjs';
const {NaturalMotion} = await import('../../previews/natural-motion.mjs');
const args = process.argv.slice(2);
const arg = (n,d) => args.includes(n) ? args[args.indexOf(n)+1] : d;
const input = path.resolve(arg('--input',''));
const output = path.resolve(arg('--output',''));
const mode = arg('--mode','walk'), time = Number(arg('--time','5.9'));
const physics = !args.includes('--physics-off');
if (!args.includes('--input') || !args.includes('--output'))
  throw Error('Pass exact --input VRM and --output snapshot JSON.');
if (!['walk','idle','greet','settled'].includes(mode)) throw Error('Unknown mode');
const data = await fs.readFile(input);
const document = JSON.parse(data.subarray(20,20+data.readUInt32LE(12)));
const runtime = await loadRuntime(input);
try {
  const {avatar} = runtime;
  const motion = new NaturalMotion(avatar.vrm);
  motion.mode = mode === 'settled' ? 'walk' : mode;
  motion.modeStart = 0;
  motion.originZ = 0;
  motion.transition = null;
  avatar.setPhysics(physics);
  motion.apply(0);
  avatar.vrm.update(0);
  avatar.springManager?.reset();
  const frames = Math.round(time*60);
  for (let i=1;i<=frames;i++) {
    const seconds=i/60;
    if (mode==='settled' && i===360) motion.setMode('idle',seconds,{transition:true});
    motion.apply(mode==='settled'?Math.min(seconds,8):seconds);
    avatar.vrm.update(1/60);
  }
  avatar.model.updateMatrixWorld(true);
  const hips = avatar.vrm.humanoid.getRawBoneNode('hips').getWorldPosition(new THREE.Vector3());
  const origin = new THREE.Vector3(hips.x,0,hips.z);
  const meshes=[];
  for (let nodeIndex=0;nodeIndex<document.nodes.length;nodeIndex++) {
    const node=document.nodes[nodeIndex];
    if (node.mesh===undefined || !['head','hair','neck','upper_body'].includes(node.extras?.part))
      continue;
    const loaded=await avatar.resources.parser.getDependency('node',nodeIndex);
    for (const mesh of loaded.getObjectsByProperty('isMesh',true)) {
      mesh.skeleton?.update();
      const positions=[],point=new THREE.Vector3();
      for(let i=0;i<mesh.geometry.attributes.position.count;i++) {
        mesh.getVertexPosition(i,point).applyMatrix4(mesh.matrixWorld).sub(origin);
        if(![point.x,point.y,point.z].every(Number.isFinite)) throw Error('Nonfinite vertex');
        positions.push(...point.toArray());
      }
      const index=mesh.geometry.index;
      const indices=index?Array.from(index.array):
        Array.from({length:mesh.geometry.attributes.position.count},(_,i)=>i);
      const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];
      const groups=mesh.geometry.groups.length?mesh.geometry.groups:
        [{start:0,count:indices.length,materialIndex:0}];
      for(let groupIndex=0;groupIndex<groups.length;groupIndex++){
        const group=groups[groupIndex];
        meshes.push({
          name:mesh.name+(groups.length>1?`_material_${groupIndex}`:''),
          part:node.extras?.part,facial_role:node.extras?.facial_role,
          hair_region:node.extras?.hair_region,side:node.extras?.side,
          positions,triangles:indices.slice(group.start,group.start+group.count),
          materialName:materials[group.materialIndex||0]?.name||'',
        });
      }
    }
  }
  const report={
    input,vrmSha256:runtime.inputSha,mode,time:frames/60,physics,timestep:1/60,
    coordinates:'Actual morphed/skinned Three.js world vertices, Y up. Only global hips X/Z translation removed.',
    originShift:origin.toArray(),meshes,
    scope:'Actual exported geometry and spring simulation. Material names only; any independent render uses approximate materials.',
    sourceUnchanged:sha(await fs.readFile(input))===runtime.inputSha,
  };
  await fs.mkdir(path.dirname(output),{recursive:true});
  await fs.writeFile(output,JSON.stringify(report));
  console.log(JSON.stringify({output,vrmSha256:report.vrmSha256,mode,time,physics,
    meshes:meshes.length,vertices:meshes.reduce((n,m)=>n+m.positions.length/3,0),
    sourceUnchanged:report.sourceUnchanged}));
}finally{runtime.avatar.dispose();}

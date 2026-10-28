// Map reported runtime contacts back to exact exported rest coordinates/weights.
// No pose/model/source changes are saved.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE,loadRuntime,sha} from './runtime_cpu.mjs';
const args=process.argv.slice(2),arg=(n,d)=>args.includes(n)?args[args.indexOf(n)+1]:d;
const input=path.resolve(arg('--input','')),reportPath=path.resolve(arg('--report',''));
const output=path.resolve(arg('--output',''));
const report=JSON.parse(await fs.readFile(reportPath,'utf8'));
const runtime=await loadRuntime(input);
if(runtime.inputSha!==report.vrmSha256)throw Error('Exact input/report hash mismatch');
try{
  runtime.avatar.model.updateMatrixWorld(true);
  const frame=args.includes('--frame-index')?report.perFrame[Number(arg('--frame-index','0'))]:
    report.perFrame.reduce((a,b)=>a.intersectionPairs>=b.intersectionPairs?a:b);
  const records=[];
  async function vertices(name,indices){
    const nodeIndex=runtime.json.nodes.findIndex(n=>n.name===name&&n.mesh!==undefined);
    if(nodeIndex<0)throw Error('Unknown reported mesh '+name);
    const node=await runtime.avatar.resources.parser.getDependency('node',nodeIndex);
    const meshes=node.getObjectsByProperty('isMesh',true);
    if(meshes.length!==1)throw Error('Ambiguous primitive mesh: '+name);
    const mesh=meshes[0];mesh.skeleton?.update();
    return indices.map(index=>{
      const p=new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position,index)
        .applyMatrix4(mesh.matrixWorld);
      const loaded=mesh.getVertexPosition(index,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
      const si=mesh.geometry.attributes.skinIndex,sw=mesh.geometry.attributes.skinWeight;
      const weights=Array.from({length:4},(_,c)=>({
        bone:mesh.skeleton.bones[si.getComponent(index,c)].name,weight:sw.getComponent(index,c),
      })).filter(w=>w.weight>1e-8);
      return {index,restRuntime:p.toArray(),restBlender:[p.x,-p.z,p.y],
        loadedRestBlender:[loaded.x,-loaded.z,loaded.y],weights};
    });
  }
  for(const [pair,examples]of Object.entries(frame.examplesByPair)){
    for(const hit of examples.slice(0,3)){
      const reverse=hit.direction==='clothing-edge/skin-face';
      records.push({pair,hit,
        skin:await vertices(hit.skin,reverse?hit.triangle:hit.edge),
        clothing:await vertices(hit.clothing,reverse?hit.edge:hit.triangle)});
    }
  }
  const data={version:2,input,vrmSha256:runtime.inputSha,reportPath,
    reportSha256:sha(await fs.readFile(reportPath)),mode:report.mode,time:frame.time,
    pose:frame.pose,intersectionPairs:frame.intersectionPairs,records,
    scope:'Version2: raw POSITION accessor source-bind rest coordinates and skin weights of exact reported vertices. Loaded runtime rest coordinates are separate, since exporter canonical pose can differ from bind.',
    sourceUnchanged:sha(await fs.readFile(input))===runtime.inputSha};
  await fs.writeFile(output,JSON.stringify(data,null,2));
  console.log(JSON.stringify({output,time:data.time,intersectionPairs:data.intersectionPairs,
    records:records.length,vrmSha256:data.vrmSha256}));
}finally{runtime.avatar.dispose();}

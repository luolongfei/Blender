import {toolPath} from '../../../../pipeline/context.mjs';
import {register} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const threeURL=pathToFileURL(toolPath('public/vendor/three.module.js')).href;
register('data:text/javascript,'+encodeURIComponent(`export async function resolve(s,c,next){return s==='three'?{url:${JSON.stringify(threeURL)},shortCircuit:true}:next(s,c);}`),import.meta.url);
const THREE=await import('three');
const {PreviewAvatar}=await import(pathToFileURL(toolPath('public/preview-avatar.mjs')));
const here=path.dirname(fileURLToPath(import.meta.url));
const args=process.argv.slice(2);
const option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const inputPath=path.resolve(option('--input',path.join(here,'qa-runtime-optimized.vrm')));
const outputPath=path.resolve(option('--output',path.join(here,'qa-runtime-walk.json')));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const input=await fs.readFile(inputPath),inputSha256=sha(input);
const jl=input.readUInt32LE(12),json=JSON.parse(input.subarray(20,20+jl));
const bin=input.subarray(28+jl);
json.materials=json.materials.map(m=>({name:m.name,pbrMetallicRoughness:{baseColorFactor:[1,1,1,1]}}));delete json.images;delete json.textures;
for(const k of ['extensionsUsed','extensionsRequired'])json[k]=(json[k]||[]).filter(x=>!['VRMC_materials_mtoon','KHR_texture_transform','KHR_materials_unlit'].includes(x));
const text=Buffer.from(JSON.stringify(json));const len=Math.ceil(text.length/4)*4;const bytes=Buffer.alloc(28+len+bin.length,32);
bytes.writeUInt32LE(0x46546c67,0);bytes.writeUInt32LE(2,4);bytes.writeUInt32LE(bytes.length,8);bytes.writeUInt32LE(len,12);bytes.writeUInt32LE(0x4e4f534a,16);text.copy(bytes,20);bytes.writeUInt32LE(bin.length,20+len);bytes.writeUInt32LE(0x004e4942,24+len);bin.copy(bytes,28+len);
const avatar=new PreviewAvatar({scene:new THREE.Scene()});avatar.setActive(true);
try {
 await avatar.loadVRM(new File([bytes],'qa.vrm'));
 const motionPath=toolPath('public/elyth_base_walk_v1.vrma');
 const motion=await fs.readFile(motionPath);
 await avatar.loadVRMA(new File([motion],'walk.vrma'));
 const bone=avatar.vrm.humanoid.getRawBoneNode('leftFoot');
 avatar.model.updateMatrixWorld(true);const before=bone.getWorldPosition(new THREE.Vector3());let max=0;
 avatar.play();for(let i=0;i<120;i++){avatar.tick(1/60);avatar.model.updateMatrixWorld(true);const p=bone.getWorldPosition(new THREE.Vector3());if(!p.toArray().every(Number.isFinite))throw Error('Nonfinite walking pose');max=Math.max(max,p.distanceTo(before));}
 if(max<.01)throw Error('Standard walk did not move the foot');
 avatar.setPhysics(false);if(avatar.vrm.springBoneManager!==undefined)throw Error('Physics OFF not detached');
 avatar.tick(1/60);avatar.setPhysics(true);if(avatar.vrm.springBoneManager!==avatar.springManager)throw Error('Physics ON not restored');
 if(sha(await fs.readFile(inputPath))!==inputSha256)throw Error('Source VRM changed during the motion check');
 const report={version:1,vrmPath:inputPath,vrmSha256:inputSha256,sourceUnchanged:true,motionPath,motionSha256:sha(motion),frames:120,framesPerSecond:60,standardWalkLoaded:true,clipDuration:avatar.action.getClip().duration,leftFootMaxDisplacementMeters:max,physicsToggleVerified:true,finiteMotionVerified:true,scope:'Exact runtime skin and animation with materials omitted for CPU test; browser appearance, clothing collision freedom, expressions, other motions and longer sessions are not tested.'};
 await fs.writeFile(outputPath,JSON.stringify(report,null,2));console.log(JSON.stringify(report));
} finally {avatar.dispose();}

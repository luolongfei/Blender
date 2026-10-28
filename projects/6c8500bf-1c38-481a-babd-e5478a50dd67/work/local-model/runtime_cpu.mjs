import {toolPath} from '../../../../pipeline/context.mjs';
// Load the actual VRM geometry, hierarchy, skins and springs without browser images.
// This is a numerical-audit helper; it never rewrites the model on disk.
import {register} from 'node:module';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

register('data:text/javascript,' + encodeURIComponent(
  `export async function resolve(s,c,next){return s==='three'?{url:${
    JSON.stringify(pathToFileURL(toolPath('public/vendor/three.module.js')).href)
  },shortCircuit:true}:next(s,c);}`
), import.meta.url);
export const THREE = await import('three');
const {PreviewAvatar} = await import(pathToFileURL(toolPath('public/preview-avatar.mjs')));
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');

export async function loadRuntime(inputPath) {
  const input = await fs.readFile(inputPath), inputSha = sha(input);
  const jsonLength = input.readUInt32LE(12);
  const json = JSON.parse(input.subarray(20,20+jsonLength));
  const bin = input.subarray(28+jsonLength);
  json.materials = (json.materials||[]).map(m=>({
    name:m.name,pbrMetallicRoughness:{baseColorFactor:[1,1,1,1]}
  }));
  delete json.images;delete json.textures;
  for(const key of ['extensionsUsed','extensionsRequired']){
    json[key]=(json[key]||[]).filter(n=>![
      'VRMC_materials_mtoon','KHR_texture_transform','KHR_materials_unlit'
    ].includes(n));
  }
  const encoded=Buffer.from(JSON.stringify(json)),padded=Math.ceil(encoded.length/4)*4;
  const bytes=Buffer.alloc(28+padded+bin.length,32);
  bytes.writeUInt32LE(0x46546c67,0);bytes.writeUInt32LE(2,4);bytes.writeUInt32LE(bytes.length,8);
  bytes.writeUInt32LE(padded,12);bytes.writeUInt32LE(0x4e4f534a,16);encoded.copy(bytes,20);
  bytes.writeUInt32LE(bin.length,20+padded);bytes.writeUInt32LE(0x004e4942,24+padded);
  bin.copy(bytes,28+padded);
  const avatar=new PreviewAvatar({scene:new THREE.Scene()});
  avatar.setActive(true);
  await avatar.loadVRM(new File([bytes],'runtime-audit.vrm'));
  return {avatar,inputSha,json};
}

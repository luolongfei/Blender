// Read-only UV statistics for the actual new GLB. No image or model edits.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {workspacePath} from '../../pipeline/context.mjs';
const file=workspacePath('character-hq-textured.glb'), bytes=fs.readFileSync(file);
const jsonBytes=bytes.readUInt32LE(12), doc=JSON.parse(bytes.subarray(20,20+jsonBytes));
const bin=bytes.subarray(28+jsonBytes);
function acc(n){const a=doc.accessors[n],v=doc.bufferViews[a.bufferView],dims={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[a.type],size={5123:2,5125:4,5126:4}[a.componentType];if(!dims||!size)throw Error('Unsupported accessor');const out=[];for(let i=0;i<a.count;i++)for(let k=0;k<dims;k++){const off=(v.byteOffset||0)+(a.byteOffset||0)+i*(v.byteStride||dims*size)+k*size;out.push(a.componentType===5126?bin.readFloatLE(off):a.componentType===5125?bin.readUInt32LE(off):bin.readUInt16LE(off));}return out;}
const primitive=doc.meshes[0].primitives[0],p=acc(primitive.attributes.POSITION),uv=acc(primitive.attributes.TEXCOORD_0),idx=acc(primitive.indices);
const total={triangles:0,surfaceArea:0,uvArea:0,zeroUvArea:0,zeroSurfaceArea:0,uvOutsideUnit:0};
const regions={head_neck:{},hands:{},garments_legs_shoes:{}};
for(const r of Object.values(regions))Object.assign(r,total);
let uvMin=[Infinity,Infinity],uvMax=[-Infinity,-Infinity];
for(let i=0;i<uv.length;i++) {let k=i%2;uvMin[k]=Math.min(uvMin[k],uv[i]);uvMax[k]=Math.max(uvMax[k],uv[i]);}
for(let i=0;i<idx.length;i+=3){const ids=idx.slice(i,i+3),a=ids.map(k=>p.slice(3*k,3*k+3)),t=ids.map(k=>uv.slice(2*k,2*k+2));let ab=a[1].map((x,k)=>x-a[0][k]),ac=a[2].map((x,k)=>x-a[0][k]);let cross=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];let area=Math.hypot(...cross)/2,ta=Math.abs((t[1][0]-t[0][0])*(t[2][1]-t[0][1])-(t[1][1]-t[0][1])*(t[2][0]-t[0][0]))/2;let cent=[0,1,2].map(k=>(a[0][k]+a[1][k]+a[2][k])/3);let region=cent[1]>.63?'head_neck':Math.abs(cent[0])>.78?'hands':'garments_legs_shoes';for(const r of [total,regions[region]]){r.triangles++;r.surfaceArea+=area;r.uvArea+=ta;r.zeroUvArea+=ta<1e-12?1:0;r.zeroSurfaceArea+=area<1e-12?1:0;r.uvOutsideUnit+=t.some(a=>a.some(x=>x<0||x>1))?1:0;}}
for(const r of Object.values(regions)){r.uvAreaFraction=r.uvArea/total.uvArea;r.surfaceAreaFraction=r.surfaceArea/total.surfaceArea;r.relativeLinearTexelDensity=Math.sqrt(r.uvAreaFraction/r.surfaceAreaFraction);}
const report={source:'character-hq-textured.glb',sourceSha256:crypto.createHash('sha256').update(bytes).digest('hex'),materials:doc.materials,images:doc.images.map(im=>({mimeType:im.mimeType,bufferView:im.bufferView})),uvMin,uvMax,total,regions,interpretation:['Region bounds are only approximate original glTF coordinate groups, not segmentation masks.','uvArea is the sum of triangle UV areas, not unique occupied texels; overlapping islands are not excluded.','A 4K input texture does not prove local feature sharpness. Valid garment details should be preserved, while replacement face/hand regions need dedicated UVs.','No texture pixels were changed and no Blender material validation was performed.']};
fs.mkdirSync(workspacePath('Character_Rebuild','reports'),{recursive:true});
fs.writeFileSync(workspacePath('Character_Rebuild','reports','texture-uv-audit.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));

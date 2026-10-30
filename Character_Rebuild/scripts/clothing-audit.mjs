// Read-only, dependency-free input inspection. Produces unlit orthographic diagnostics;
// this is not a Blender render, a texture edit, or a repaired model.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import {workspacePath} from '../../pipeline/context.mjs';

const source=workspacePath('character-hq-textured.glb');
const b=fs.readFileSync(source), n=b.readUInt32LE(12);
const j=JSON.parse(b.subarray(20,20+n)), bin=b.subarray(28+n);
const out=workspacePath('Character_Rebuild','diagnostics','clothing');
fs.mkdirSync(out,{recursive:true});
function accessor(id){
  const a=j.accessors[id],v=j.bufferViews[a.bufferView],k={SCALAR:1,VEC2:2,VEC3:3}[a.type];
  if(v.byteStride)throw Error('Interleaved accessor not implemented');
  const bytes=bin.subarray((v.byteOffset||0)+(a.byteOffset||0),(v.byteOffset||0)+(a.byteOffset||0)+a.count*k*4);
  return a.componentType===5126?new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)):new Uint32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
}
const p=j.meshes[0].primitives[0], pos=accessor(p.attributes.POSITION), uv=accessor(p.attributes.TEXCOORD_0), ids=accessor(p.indices);
const iv=j.bufferViews[j.images[0].bufferView],png=bin.subarray(iv.byteOffset,iv.byteOffset+iv.byteLength);
const w=png.readUInt32BE(16),h=png.readUInt32BE(20),ch=png[25]===2?3:4;
if(png[24]!==8||png[28]!==0||![2,6].includes(png[25]))throw Error('Unsupported PNG');
const chunks=[];for(let q=8;q+12<=png.length;){const l=png.readUInt32BE(q),t=png.subarray(q+4,q+8).toString();if(t==='IDAT')chunks.push(png.subarray(q+8,q+8+l));q+=12+l;if(t==='IEND')break;}
const raw=zlib.inflateSync(Buffer.concat(chunks)),tex=Buffer.alloc(w*h*ch),stride=w*ch;
function paeth(a,b,c){const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);return pa<=pb&&pa<=pc?a:pb<=pc?b:c;}
for(let y=0;y<h;y++){const f=raw[y*(stride+1)];for(let x=0;x<stride;x++){const a=x>=ch?tex[y*stride+x-ch]:0,up=y?tex[(y-1)*stride+x]:0,c=y&&x>=ch?tex[(y-1)*stride+x-ch]:0;tex[y*stride+x]=(raw[y*(stride+1)+1+x]+[0,a,up,Math.floor((a+up)/2),paeth(a,up,c)][f])&255;}}
function sample(u,v){const x=Math.max(0,Math.min(w-1,Math.round(u*(w-1)))),y=Math.max(0,Math.min(h-1,Math.round(v*(h-1))));return [tex[(y*w+x)*ch],tex[(y*w+x)*ch+1],tex[(y*w+x)*ch+2]];}
const colors=Array.from({length:pos.length/3},(_,i)=>sample(uv[i*2],uv[i*2+1]));
const crcTable=Uint32Array.from({length:256},(_,n)=>{for(let i=0;i<8;i++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
function chunk(t,d){const b=Buffer.alloc(d.length+12);b.writeUInt32BE(d.length);b.write(t,4);d.copy(b,8);let c=0xffffffff;for(const x of b.subarray(4,8+d.length))c=crcTable[(c^x)&255]^(c>>>8);b.writeUInt32BE((c^0xffffffff)>>>0,d.length+8);return b;}
function savePNG(file,rgb,W,H){const ih=Buffer.alloc(13);ih.writeUInt32BE(W);ih.writeUInt32BE(H,4);ih[8]=8;ih[9]=2;const scan=Buffer.alloc((W*3+1)*H);for(let y=0;y<H;y++)rgb.copy(scan,y*(W*3+1)+1,y*W*3,(y+1)*W*3);fs.writeFileSync(file,Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',ih),chunk('IDAT',zlib.deflateSync(scan)),chunk('IEND',Buffer.alloc(0))]));}
function render(name,axis,sign){const W=1000,H=1000,scale=460,pixels=Buffer.alloc(W*H*3,238),depth=new Float32Array(W*H).fill(-Infinity),projected=[];
 for(let i=0;i<pos.length/3;i++){const x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];projected.push(axis==='z'?[W/2+x*scale*sign,H/2-y*scale,z*sign]:[W/2+z*scale*sign,H/2-y*scale,-x*sign]);}
 for(let f=0;f<ids.length;f+=3){const ia=ids[f],ib=ids[f+1],ic=ids[f+2],a=projected[ia],b=projected[ib],c=projected[ic],d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(d)<.00001)continue;
 const x0=Math.max(0,Math.floor(Math.min(a[0],b[0],c[0]))),x1=Math.min(W-1,Math.ceil(Math.max(a[0],b[0],c[0]))),y0=Math.max(0,Math.floor(Math.min(a[1],b[1],c[1]))),y1=Math.min(H-1,Math.ceil(Math.max(a[1],b[1],c[1])));
 for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/d,v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/d,s=1-u-v;if(u<0||v<0||s<0)continue;const z=u*a[2]+v*b[2]+s*c[2],k=y*W+x;if(z<depth[k])continue;depth[k]=z;const col=sample(u*uv[ia*2]+v*uv[ib*2]+s*uv[ic*2],u*uv[ia*2+1]+v*uv[ib*2+1]+s*uv[ic*2+1]);for(let q=0;q<3;q++)pixels[k*3+q]=col[q];}}
 savePNG(path.join(out,name+'.png'),pixels,W,H);
}
render('unlit-z-positive','z',1);render('unlit-z-negative','z',-1);render('unlit-x-positive','x',-1);render('unlit-x-negative','x',1);
const bands=[];for(let lo=-1;lo<1;lo+=.1){const ar=[];for(let i=0;i<pos.length/3;i++)if(pos[i*3+1]>=lo&&pos[i*3+1]<lo+.1)ar.push(i);const mean=[0,0,0];for(const i of ar)colors[i].forEach((v,k)=>mean[k]+=v/ar.length);bands.push({yMin:+lo.toFixed(2),count:ar.length,meanRGB:mean.map(v=>+v.toFixed(1)),minX:Math.min(...ar.map(i=>pos[i*3])),maxX:Math.max(...ar.map(i=>pos[i*3]))});}
fs.writeFileSync(path.join(out,'height-bands.json'),JSON.stringify({coordinates:'original glTF, Y up; no rescale',textureSampling:'UV v=0 at top, original embedded PNG',bands},null,2));
console.log(JSON.stringify({out,images:4,bands}));
export {pos,uv,ids,colors};

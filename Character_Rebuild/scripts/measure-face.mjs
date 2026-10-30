import fs from 'node:fs';
import zlib from 'node:zlib';
import {workspacePath} from '../../pipeline/context.mjs';
const raw=fs.readFileSync(workspacePath('character-hq-textured.glb'));
const len=raw.readUInt32LE(12),j=JSON.parse(raw.subarray(20,20+len)),bin=raw.subarray(28+len);
function acc(id){const a=j.accessors[id],v=j.bufferViews[a.bufferView],n={VEC2:2,VEC3:3,SCALAR:1}[a.type],o=(v.byteOffset||0)+(a.byteOffset||0),bytes=bin.subarray(o,o+a.count*n*4);return a.componentType===5126?new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)):new Uint32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));}
const pr=j.meshes[0].primitives[0],p=acc(pr.attributes.POSITION),uv=acc(pr.attributes.TEXCOORD_0),indices=acc(pr.indices);
const iv=j.bufferViews[j.images[0].bufferView],png=bin.subarray(iv.byteOffset,iv.byteOffset+iv.byteLength),w=png.readUInt32BE(16),h=png.readUInt32BE(20),ch=png[25]===2?3:4;
const chunks=[];for(let o=8;o+12<=png.length;){const n=png.readUInt32BE(o),t=png.subarray(o+4,o+8).toString();if(t==='IDAT')chunks.push(png.subarray(o+8,o+8+n));o+=n+12;if(t==='IEND')break;}
const scan=zlib.inflateSync(Buffer.concat(chunks)),tex=Buffer.alloc(w*h*ch),stride=w*ch;
function paeth(a,b,c){const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);return pa<=pb&&pa<=pc?a:pb<=pc?b:c;}
for(let y=0;y<h;y++){const f=scan[y*(stride+1)];for(let x=0;x<stride;x++){const a=x>=ch?tex[y*stride+x-ch]:0,b=y?tex[(y-1)*stride+x]:0,c=y&&x>=ch?tex[(y-1)*stride+x-ch]:0;tex[y*stride+x]=(scan[y*(stride+1)+1+x]+[0,a,b,Math.floor((a+b)/2),paeth(a,b,c)][f])&255;}}
function color(i){const x=Math.max(0,Math.min(w-1,Math.round(uv[2*i]*(w-1)))),y=Math.max(0,Math.min(h-1,Math.round(uv[2*i+1]*(h-1))));return [...tex.subarray((y*w+x)*ch,(y*w+x)*ch+3)];}
const verts=[];for(let i=0;i<p.length/3;i++){if(p[3*i+1]<.59)continue;const c=color(i);verts.push({x:p[3*i],y:-p[3*i+2],z:p[3*i+1],c,skin:c[0]>130&&c[1]>95&&c[2]>65&&c[0]>c[1]+5&&c[1]>c[2]+6});}
const q=(ar,f)=>{if(!ar.length)return null;ar.sort((a,b)=>a-b);return +ar[Math.floor((ar.length-1)*f)].toFixed(6);};
const bounds=ar=>({n:ar.length,x:[q(ar.map(v=>v.x),0),q(ar.map(v=>v.x),1)],y:[q(ar.map(v=>v.y),0),q(ar.map(v=>v.y),1)],z:[q(ar.map(v=>v.z),0),q(ar.map(v=>v.z),1)]});
const rows=[];for(let z=.60;z<=1;z+=.01){const ar=verts.filter(v=>Math.abs(v.z-z)<.004),skin=ar.filter(v=>v.skin),front=skin.filter(v=>Math.abs(v.x)<.008&&v.y<0),side=skin.filter(v=>Math.abs(v.x)>.07);rows.push({z:+z.toFixed(3),all:bounds(ar),skin:bounds(skin),centralFront:q(front.map(v=>v.y),.12),centralFrontCount:front.length,skinWidthQ95:q(skin.map(v=>Math.abs(v.x)),.95),side:bounds(side)});}
const areas=[];for(const [name,x0,x1,z0,z1] of [['eyeL',-.080,-.022,.775,.808],['eyeR',.022,.080,.775,.808],['mouth',-.032,.032,.717,.741],['nose',-.015,.015,.743,.767]]){const ar=verts.filter(v=>v.x>=x0&&v.x<=x1&&v.z>=z0&&v.z<=z1&&v.y<-.04);areas.push({name,bounds:bounds(ar),samples:ar.filter(v=>name.startsWith('eye')?v.c[0]<130:name==='mouth'?v.c[0]>v.c[1]*1.2:true).map(v=>[+v.x.toFixed(6),+v.y.toFixed(6),+v.z.toFixed(6),...v.c])});}
function sampleUV(u,v){const x=Math.max(0,Math.min(w-1,Math.round(u*(w-1)))),y=Math.max(0,Math.min(h-1,Math.round(v*(h-1))));return [...tex.subarray((y*w+x)*ch,(y*w+x)*ch+3)];}
const envelope=[];
for(let z=.67;z<=.825;z+=.005){
 const faces=[];for(let n=0;n<indices.length;n+=3){const a=indices[n],b=indices[n+1],c=indices[n+2];if(z<Math.min(p[3*a+1],p[3*b+1],p[3*c+1])||z>Math.max(p[3*a+1],p[3*b+1],p[3*c+1]))continue;faces.push([a,b,c]);}
 const samples=[];for(let x=-.125;x<=.1251;x+=.005){let nearest=Infinity,hit=null;for(const [ia,ib,ic]of faces){const ax=p[ia*3],az=p[ia*3+1],bx=p[ib*3],bz=p[ib*3+1],cx=p[ic*3],cz=p[ic*3+1],det=(bz-cz)*(ax-cx)+(cx-bx)*(az-cz);if(Math.abs(det)<1e-12)continue;const u=((bz-cz)*(x-cx)+(cx-bx)*(z-cz))/det,v=((cz-az)*(x-cx)+(ax-cx)*(z-cz))/det,s=1-u-v;if(u<0||v<0||s<0)continue;const y=-(u*p[3*ia+2]+v*p[3*ib+2]+s*p[3*ic+2]);if(y<nearest){nearest=y;hit=sampleUV(u*uv[2*ia]+v*uv[2*ib]+s*uv[2*ic],u*uv[2*ia+1]+v*uv[2*ib+1]+s*uv[2*ic+1]);}}samples.push([+x.toFixed(4),Number.isFinite(nearest)?+nearest.toFixed(6):null,hit]);}
 envelope.push({z:+z.toFixed(3),samples});
}
const report={coordinates:'Blender imported: X=glTF X,Y=-glTF Z,Z=glTF Y; original normalized scale unchanged',rows,areas,frontSurfaceEnvelope:envelope};
fs.mkdirSync(workspacePath('Character_Rebuild/reports'),{recursive:true});fs.writeFileSync(workspacePath('Character_Rebuild/reports/face-measurements.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(rows.map(r=>({z:r.z,width:r.skinWidthQ95,front:r.centralFront,skin:r.skin.n,allY:r.all.y,skinY:r.skin.y}))));

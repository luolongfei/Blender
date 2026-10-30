import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {workspacePath} from '../../pipeline/context.mjs';

const input=workspacePath('character-hq-textured.glb');
const bytes=fs.readFileSync(input);
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
if(bytes.readUInt32LE(0)!==0x46546c67||bytes.readUInt32LE(4)!==2||bytes.readUInt32LE(8)!==bytes.length) throw Error('Invalid GLB header');
let json,bin; const chunks=[];
for(let o=12;o<bytes.length;){const n=bytes.readUInt32LE(o),t=bytes.readUInt32LE(o+4);chunks.push({offset:o,length:n,type:t}); const b=bytes.subarray(o+8,o+8+n); if(t===0x4e4f534a)json=JSON.parse(b.toString()); else if(t===0x004e4942)bin=b; o+=n+8;}
const sizes={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4};
const widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
const reads={5120:'readInt8',5121:'readUInt8',5122:'readInt16LE',5123:'readUInt16LE',5125:'readUInt32LE',5126:'readFloatLE'};
function accessor(id){const a=json.accessors[id],v=json.bufferViews[a.bufferView];if(a.sparse)throw Error('Unexpected sparse accessor'); const w=widths[a.type],s=sizes[a.componentType],step=v.byteStride||w*s,start=(v.byteOffset||0)+(a.byteOffset||0);return Array.from({length:a.count},(_,i)=>Array.from({length:w},(_,k)=>bin[reads[a.componentType]](start+i*step+k*s)));}
const imageDir=workspacePath('Character_Rebuild/diagnostics/original-textures');fs.mkdirSync(imageDir,{recursive:true});
const images=(json.images||[]).map((im,i)=>{const bv=json.bufferViews[im.bufferView],data=bin.subarray(bv.byteOffset||0,(bv.byteOffset||0)+bv.byteLength);const ext=im.mimeType==='image/png'?'png':'bin';const name=`base-color-${i}.${ext}`;const out=path.join(imageDir,name);if(fs.existsSync(out)&&sha(fs.readFileSync(out))!==sha(data))throw Error('Existing extraction differs');fs.writeFileSync(out,data);return {index:i,mimeType:im.mimeType,embedded:true,bytes:data.length,sha256:sha(data),width:ext==='png'?data.readUInt32BE(16):null,height:ext==='png'?data.readUInt32BE(20):null,path:`Character_Rebuild/diagnostics/original-textures/${name}`};});
const audit=[];
for(const [mi,m] of json.meshes.entries())for(const [pi,p] of m.primitives.entries()){
 const pos=accessor(p.attributes.POSITION),uv=accessor(p.attributes.TEXCOORD_0),norm=accessor(p.attributes.NORMAL),idx=accessor(p.indices).map(x=>x[0]);
 const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};let nonFinitePositions=0,nonFiniteUVs=0,nonFiniteNormals=0;
 pos.forEach(v=>v.forEach((x,k)=>{if(!Number.isFinite(x))nonFinitePositions++;bounds.min[k]=Math.min(bounds.min[k],x);bounds.max[k]=Math.max(bounds.max[k],x);}));
 uv.forEach(v=>v.forEach(x=>{if(!Number.isFinite(x))nonFiniteUVs++;}));norm.forEach(v=>v.forEach(x=>{if(!Number.isFinite(x))nonFiniteNormals++;}));
 bounds.extents=bounds.max.map((v,k)=>v-bounds.min[k]);
 const spatial=new Map(),weld=[];let unique=0;for(const v of pos){const key=v.map(x=>Math.round(x/1e-7)).join(',');if(!spatial.has(key))spatial.set(key,unique++);weld.push(spatial.get(key));}
 const parent=Array.from({length:unique},(_,i)=>i),weight=new Uint32Array(unique).fill(1);const find=x=>{while(parent[x]!==x){parent[x]=parent[parent[x]];x=parent[x];}return x;};function join(a,b){a=find(a);b=find(b);if(a===b)return;if(weight[a]<weight[b])[a,b]=[b,a];parent[b]=a;weight[a]+=weight[b];}
 const edgesRaw=new Map(),edgesWeld=new Map();let zeroArea=0,invalidIndex=0,orientationAgainstNormals=0,volume=0;
 function edge(map,a,b){const key=a<b?`${a},${b}`:`${b},${a}`;const old=map.get(key)||{count:0,flow:0};old.count++;old.flow+=a<b?1:-1;map.set(key,old);}
 for(let i=0;i<idx.length;i+=3){const ids=idx.slice(i,i+3);if(ids.some(j=>j<0||j>=pos.length)){invalidIndex++;continue;}const [ia,ib,ic]=ids,[a,b,c]=ids.map(j=>pos[j]),[wa,wb,wc]=ids.map(j=>weld[j]);join(wa,wb);join(wb,wc);edge(edgesRaw,ia,ib);edge(edgesRaw,ib,ic);edge(edgesRaw,ic,ia);edge(edgesWeld,wa,wb);edge(edgesWeld,wb,wc);edge(edgesWeld,wc,wa);
 const u=b.map((v,k)=>v-a[k]),v=c.map((v,k)=>v-a[k]),cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],area2=Math.hypot(...cross);if(area2<1e-12)zeroArea++;const n=[0,1,2].map(k=>norm[ia][k]+norm[ib][k]+norm[ic][k]);if(cross.reduce((s,x,k)=>s+x*n[k],0)<0)orientationAgainstNormals++;
 volume+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;
 }
 const components=new Map();for(let i=0;i<pos.length;i++){const r=find(weld[i]);if(!components.has(r))components.set(r,{triangles:0,rawVertices:0,min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]});const c=components.get(r);c.rawVertices++;pos[i].forEach((x,k)=>{c.min[k]=Math.min(c.min[k],x);c.max[k]=Math.max(c.max[k],x);});}for(let i=0;i<idx.length;i+=3)components.get(find(weld[idx[i]])).triangles++;
 const edgeStats=map=>({edges:map.size,boundaryEdges:[...map.values()].filter(v=>v.count===1).length,nonManifoldOverTwoFaces:[...map.values()].filter(v=>v.count>2).length,inconsistentWindingTwoFaces:[...map.values()].filter(v=>v.count===2&&v.flow!==0).length});
 audit.push({mesh:mi,primitive:pi,name:m.name,vertices:pos.length,triangles:idx.length/3,material:p.material,attributes:Object.keys(p.attributes),bounds,nonFinitePositions,nonFiniteUVs,nonFiniteNormals,invalidIndex,zeroAreaTriangles:zeroArea,trianglesOppositeAveragedVertexNormals:orientationAgainstNormals,signedVolume:volume,indexedTopology:edgeStats(edgesRaw),positionWeldedTopology:{tolerance:1e-7,uniquePositions:unique,duplicatePositions:pos.length-unique,...edgeStats(edgesWeld)},spatialComponents:[...components.values()].sort((a,b)=>b.triangles-a.triangles),uvBounds:{min:[uv.reduce((a,v)=>Math.min(a,v[0]),Infinity),uv.reduce((a,v)=>Math.min(a,v[1]),Infinity)],max:[uv.reduce((a,v)=>Math.max(a,v[0]),-Infinity),uv.reduce((a,v)=>Math.max(a,v[1]),-Infinity)]}});
}
const report={schemaVersion:1,input:'character-hq-textured.glb',inputBytes:bytes.length,inputSha256:sha(bytes),readOnlyInput:true,generatedAt:new Date().toISOString(),asset:json.asset,chunks,counts:{scenes:json.scenes?.length||0,nodes:json.nodes?.length||0,meshes:json.meshes?.length||0,materials:json.materials?.length||0,textures:json.textures?.length||0,images:json.images?.length||0,skins:json.skins?.length||0,animations:json.animations?.length||0,accessors:json.accessors?.length||0},nodes:json.nodes,materials:json.materials,materialNotes:['metallicFactor absent: glTF 2.0 default is 1.0; importer may therefore render the character metallic unless explicitly corrected.','No normal/roughness image texture; one base-color texture only.'],images,meshes:audit,limitations:['Structural parser diagnostic only, not a visual or self-intersection acceptance.','Position welding used only for diagnostics; UV seam splits are not necessarily defects.','No geometric edit, remesh, smoothing, export or Blender source write was performed.','Units follow glTF metre convention, but source was normalized to approximately two units tall; real intended height cannot be established from binary alone.']};
const out=workspacePath('Character_Rebuild/reports/glb-input-audit.json');fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');
if(sha(fs.readFileSync(input))!==report.inputSha256)throw Error('Input changed during read-only audit');
console.log(JSON.stringify({report:out,sha256:report.inputSha256,counts:report.counts,images,mesh:audit.map(({spatialComponents,...r})=>({...r,componentCount:spatialComponents.length,largestComponents:spatialComponents.slice(0,8)}))},null,2));

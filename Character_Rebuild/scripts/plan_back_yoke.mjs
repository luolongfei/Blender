// Build an exact patch plan from the actual D-stage local mesh capture.
import fs from 'node:fs';
import {workspacePath} from '../../pipeline/context.mjs';
const data=JSON.parse(fs.readFileSync(workspacePath('Character_Rebuild/reports/d-yoke-local.json'),'utf8'));
const vs=new Map(data.vertices.map(p=>[p[0],p.slice(1)]));
const deleteFaces=[],keep=[];
for(const p of data.polygons){const c=[0,0,0];for(const i of p.slice(1))vs.get(i).forEach((x,k)=>c[k]+=x/(p.length-1));if(Math.abs(c[0])<.207&&c[1]>-.035&&c[2]>.550)deleteFaces.push(p[0]);else keep.push(p);}
const edges=new Map();for(const f of keep){const p=f.slice(1);for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length],key=a<b?`${a},${b}`:`${b},${a}`;edges.set(key,(edges.get(key)||0)+1);}}
const adj=new Map();for(const [key,n]of edges){if(n!==1)continue;const[a,b]=key.split(',').map(Number);if(!adj.has(a))adj.set(a,[]);if(!adj.has(b))adj.set(b,[]);adj.get(a).push(b);adj.get(b).push(a);}
const pending=new Set(adj.keys()),components=[];while(pending.size){const q=[pending.values().next().value],r=[];pending.delete(q[0]);while(q.length){const i=q.pop();r.push(i);for(const n of adj.get(i))if(pending.has(n)){pending.delete(n);q.push(n);}}components.push(r);}
const candidates=components.filter(p=>Math.min(...p.map(i=>vs.get(i)[2]))>.53);
if(candidates.length!==1)throw Error('Expected one exact D-stage yoke boundary');
const component=candidates[0];if(component.some(i=>adj.get(i).length!==2))throw Error('Yoke boundary branches');
const ordered=[component[0]];let prev=null,cur=ordered[0];do{const next=adj.get(cur).find(i=>i!==prev);prev=cur;cur=next;if(cur===ordered[0])break;ordered.push(cur);}while(ordered.length<=component.length);
if(ordered.length!==component.length)throw Error('Boundary traversal failed');
const original=ordered.map(i=>vs.get(i));let outer=original.map(p=>p.slice());
for(let it=0;it<5;it++)outer=outer.map((p,i)=>p.map((x,k)=>.6*x+.2*outer[(i+outer.length-1)%outer.length][k]+.2*outer[(i+1)%outer.length][k]));
const N=outer.length,rx=.055,ry=.058,cy=.030;
const theta=outer.map(p=>Math.atan2((p[1]-cy)/ry,p[0]/rx));
const delta=theta.map((t,i)=>{let d=theta[(i+1)%N]-t;while(d>Math.PI)d-=Math.PI*2;while(d< -Math.PI)d+=Math.PI*2;return d;});
const direction=Math.sign(delta.reduce((a,b)=>a+b,0));
const weights=delta.map(d=>Math.max(1e-4,d*direction)),total=weights.reduce((a,b)=>a+b,0);
const angles=[theta[0]];for(let i=1;i<N;i++)angles.push(angles[i-1]+direction*2*Math.PI*weights[i-1]/total);
// Remove the one measured tiny polar reversal left by the irregular imported
// boundary. Radial distance and height stay fixed; patch rows cannot fold in XY.
outer=outer.map((p,i)=>{const r=Math.hypot(p[0]/rx,(p[1]-cy)/ry);return [rx*r*Math.cos(angles[i]),cy+ry*r*Math.sin(angles[i]),p[2]];});
const inner=angles.map(a=>[rx*Math.cos(a),cy+ry*Math.sin(a),.618+.003*Math.sin(a)]);
const verts=[],faces=[],uv=[];const levels=[0,.20,.42,.65,.84,1];
function point(i,t){const p=outer[i].map((x,k)=>x*(1-t)+inner[i][k]*t);p[2]+=.004*Math.sin(Math.PI*t);return p;}
for(const t of levels)for(let i=0;i<N;i++){const p=point(i,t);verts.push(p);uv.push([(p[0]+.23)/.46,(p[1]+.05)/.19]);}
for(let k=0;k<levels.length-1;k++)for(let i=0;i<N;i++)faces.push([k*N+i,(k+1)*N+i,(k+1)*N+(i+1)%N,k*N+(i+1)%N]);
// Back collar band follows the same reconstructed yoke surface. Original front
// Peter Pan collar remains on the retained mesh. The band has actual thickness.
const selected=angles.map((a,i)=>({a,i})).filter(q=>Math.sin(q.a)>-.06).sort((a,b)=>Math.atan2(Math.sin(a.a),Math.cos(a.a))-Math.atan2(Math.sin(b.a),Math.cos(b.a)));
// atan sorting puts a tiny negative-angle segment first, then 0..pi; discard any
// discontinuous wrap fragment on the far side of pi for a single solid back band.
const seq=selected.filter(q=>{let a=Math.atan2(Math.sin(q.a),Math.cos(q.a));return a>-.06&&a<Math.PI-.001});
const cv=[],cf=[],cu=[],widthLevels=[0,.33,.66,1];
for(let underside=0;underside<2;underside++)for(const w of widthLevels)for(const {i}of seq){const d=Math.hypot(outer[i][0]-inner[i][0],outer[i][1]-inner[i][1]);const t=1-w*Math.min(.85,.021/Math.max(.006,d));const p=point(i,t);p[1]+=.0012;p[2]+=.0020-underside*.0016;cv.push(p);cu.push([seq.findIndex(q=>q.i===i)/Math.max(1,seq.length-1),w]);}
const M=seq.length,R=widthLevels.length,S=M*R;
for(let s=0;s<2;s++)for(let k=0;k<R-1;k++)for(let i=0;i<M-1;i++){const a=s*S+k*M+i,b=a+1,c=a+M+1,d=a+M;cf.push(s?[d,c,b,a]:[a,b,c,d]);}
for(let i=0;i<M-1;i++){cf.push([i,S+i,S+i+1,i+1]);const a=(R-1)*M+i;cf.push([a,a+1,S+a+1,S+a]);}
for(let k=0;k<R-1;k++){const a=k*M,b=(k+1)*M;cf.push([a,b,S+b,S+a]);const c=k*M+M-1,d=(k+1)*M+M-1;cf.push([c,S+c,S+d,d]);}
let signedVolume=0;for(const f of cf)for(let i=1;i<f.length-1;i++){const[a,b,c]=[f[0],f[i],f[i+1]].map(j=>cv[j]);signedVolume+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;}
if(signedVolume<0)for(const f of cf)f.reverse();
const plan={source:'D-integrated-geometry.blend',sourceObject:data.object,cut:{absX:.207,minY:-.035,minZ:.55},deleteFaces,boundaryVertexIndices:ordered,boundaryOriginal:original,boundarySmoothed:outer,maxBoundaryDisplacement:Math.max(...outer.map((p,i)=>Math.hypot(...p.map((x,k)=>x-original[i][k])))),patch:{vertices:verts,faces,uv},collar:{vertices:cv,faces:cf,uv:cu},note:'Actual D-stage boundary fitted. New rear cloth is inferred reconstruction; front collar faces retained. No Blender execution or appearance acceptance claimed.'};
const file=workspacePath('Character_Rebuild/reports/back-yoke-plan.json');fs.writeFileSync(file,JSON.stringify(plan));console.log(JSON.stringify({file,deleteFaces:deleteFaces.length,boundary:N,maxBoundaryDisplacement:plan.maxBoundaryDisplacement,patchQuads:faces.length,collarQuads:cf.length}));

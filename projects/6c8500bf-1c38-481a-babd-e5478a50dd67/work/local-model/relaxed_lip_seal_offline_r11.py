from pathlib import Path
import runpy,numpy as np,json,sys
p=next(p for p in Path(__file__).resolve().parents if (p/'project-state.json').exists());out=p/'work/reconstruction-r02'
api=lambda n:runpy.run_path(str(p/'work/local-model'/(n+'.py')))
d=np.load(out/'lower-face-current-surface-r11.npz');v=np.load(out/'reference-adult-lower-face-continuity-v2-r11.lower-face-continuity-v2.npz')
ids=d['state_0_originalVertexIds'];w=v['afterWarm0.0World'][ids];f=d['state_0_faces'];g=d['state_0_groups'];c=d['state_0_currentMixedCageWorld'];_,land,_=api('natural_skin_adult_r11')['_donor']()
surf=api('relaxed_lip_seal_surface_r11');a=surf['plan'](w,f,g,c,land);states=[v['afterWarm'+str(t)+'World'][ids] for t in [0.,.5,1.]];angles,bounds=surf['bounded_angles'](states,f,g,a);de=a['worldOffsets'].copy();de[a['lowerMembership']]=0;de+=surf['rotate_lower'](w,a,np.interp(w[:,0],a['profileXs'],angles));a['worldOffsets']=de;a['profileAngles']=angles;a['contactBounds']=bounds;print('angles',angles.min(),angles.max(),flush=True);active=np.any(de,1);q=api('lower_face_continuity_surface_r11')['quality'];oral=api('adult_oral_observe_r11');vis=api('relaxed_lip_seal_visibility_r11')
report={'active':int(active.sum()),'maximumMeters':float(np.linalg.norm(de,axis=1).max()),'states':[]}
for warm in [0.,.5,1.]:
 before=v['afterWarm'+str(warm)+'World'][ids];after=before+de;row={'warm':warm,'quality':q(before,after,f,active),'sections':[]}
 for x in np.linspace(-.018,.018,19):
  ss=oral['section_segments'](after,f,g,c,float(x));r=oral['section_lenses'](ss,1.);row['sections'].append({'x':float(x),'crossings':len(r['crossings']),'lenses':r['localLenses']})
 print('row',warm,row['quality'],'crossings',sum(r['crossings'] for r in row['sections']),flush=True)
 grid=vis['grid_for'](before,f,g);row['visibilityBefore']=vis['measure'](before,f,g,grid)[0];row['visibilityAfter']=vis['measure'](after,f,g,grid)[0];report['states'].append(row)
np.savez_compressed(out/'relaxed-lip-seal-prototype-r11.npz',originalVertexIds=ids,world=w,faces=f,groups=g,**a)
(out/'relaxed-lip-seal-prototype-r11.json').write_text(json.dumps(report,indent=2))
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(16,7))
for ax,x in zip(axs,[-.012,0,.012]):
 for world,al in [(w,.3),(w+de,1.)]:
  for group,color in [(7,'#7040b0'),(8,'#2070c0'),(17,'#bb3020'),(22,'#13852b')]:
   ss=surf['cut_triangles'](world,f[g==group],x)
   for s in ss:ax.plot(s[:,0]*1000,s[:,1]*1000,color=color,lw=1.5,alpha=al)
 ax.set_xlim(-117,-102);ax.set_ylim(1361,1378);ax.set_aspect('equal');ax.grid();ax.set_title(f'x={x*1000:.0f}mm, light original / strong candidate')
fig.tight_layout();fig.savefig(out/'relaxed-lip-seal-prototype-r11.png',dpi=140)
print(json.dumps(report)[:1000])

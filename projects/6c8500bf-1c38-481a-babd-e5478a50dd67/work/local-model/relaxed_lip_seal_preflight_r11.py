from pathlib import Path
import runpy,numpy as np,json,hashlib
p=next(q for q in Path(__file__).resolve().parents if (q/'project-state.json').exists());out=p/'work/reconstruction-r02';api=lambda n:runpy.run_path(str(p/'work/local-model'/(n+'.py')))
a=np.load(out/'relaxed-lip-seal-plan-r11.npz');v=np.load(out/'reference-adult-lower-face-continuity-v2-r11.lower-face-continuity-v2.npz');ids=a['originalVertexIds'];f=a['faces'];g=a['groups'];off=a['worldOffsets'];upper=off.copy();upper[~a['upperMembership']]=0.;active=np.any(off,1);tri=np.concatenate([f[:,[0,1,2]],f[:,[0,2,3]]]);contact=api('relaxed_lip_seal_contact_r11')['contact'];quality=api('lower_face_continuity_surface_r11')['quality'];vis=api('relaxed_lip_seal_visibility_r11');fullf=a['fullFaces'];fg=np.zeros(len(fullf),np.int32);fg[a['roiOriginalFaceIds']]=g
states={t:v['afterWarm'+str(t)+'World'] for t in [0.,.5,1.]};grid=vis['union_grid']([vis['grid_for'](w,fullf,fg) for w in states.values()]);rows=[]
for warm,c in [(0.,.25),(0.,.5),(0.,.75),(0.,1.),(.5,1.),(1.,.25),(1.,.5),(1.,.75),(1.,1.)]:
 before=states[warm][ids];delta=off*c+upper*c*(1-c);after=before+delta;row={'warm':warm,'seal':c,'quality':quality(before,after,f,active)}
 row['contact']=contact(before,after,f,g,tri,active,a['neutralCageWorld']);print('contact',warm,c,'new',row['contact']['newPairCount'],'outside',len(row['contact']['newPairsOutsideBaselineContactTwoRing']),'worse',len(row['contact']['worsenedSections']),flush=True)
 if c==1.:
  fullafter=states[warm].copy();fullafter[ids]+=delta;row['visibilityBefore']=vis['measure'](states[warm],fullf,fg,grid)[0];row['visibilityAfter']=vis['measure'](fullafter,fullf,fg,grid)[0]
 rows.append(row);(out/'relaxed-lip-seal-preflight-r11.json').write_text(json.dumps(rows,indent=2))
print('ALL',all(r['contact']['guardPassed'] for r in rows),flush=True)

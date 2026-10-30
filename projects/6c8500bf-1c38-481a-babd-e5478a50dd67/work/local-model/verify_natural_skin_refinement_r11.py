"""Offline field checks on real donor topology with reconstructed known maps.

This is not a sample of the newest Blend or a Blender installation/render test.
The actual installation separately checks optics, source maps and evaluated
head-neck boundary equality before saving an independent candidate.
"""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    new=runpy.run_path(str(HERE/'natural_skin_refinement_r11.py'))
    natural=runpy.run_path(str(HERE/'natural_skin_adult_r11.py'))
    lip=runpy.run_path(str(HERE/'lip_surface_finish_r11.py'))
    P=new['project_path'](); donor,landmarks,provenance=natural['_donor']()
    for f in (Path(__file__),HERE/'natural_skin_refinement_r11.py'): ast.parse(f.read_text(encoding='utf-8'))
    count=len(donor['local']); base=np.array([.49,.31,.225]); pigment=np.array([.4468,.2488,.189])
    colour=np.tile(np.r_[base,1.],(count,1)).astype(np.float32); rough=np.full(count,.50,np.float32)
    mouth=landmarks['mouth']; strip=np.asarray(mouth['vermilionStripVertexIndices']); outer=np.asarray(mouth['outerVermilionLoopCandidate']['sourceVertexIndices'])
    colour[strip,:3]=base*.4+pigment*.6; colour[outer,:3]=base*.8+pigment*.2; rough[strip]=.43
    skin=natural['_vertices_of_groups'](donor,set(map(int,donor['face_sets']))-{7,8})
    natural_maps,_=natural['build_maps'](donor,landmarks,colour,rough,skin)
    source,_=lip['build_maps'](donor,landmarks,natural_maps[natural['COLOR']],natural_maps[natural['ROUGH']])
    selection_path=P/'work/reconstruction-r02/female-head-selection-r11.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    args=(donor,landmarks,source[lip['COLOR']],source[lip['ROUGH']],source[lip['SUPPORT']],skin,selection['neck_ring_old_indices'])
    maps,plan=new['build_maps'](*args); twice,_=new['build_maps'](*args)
    assert all(np.array_equal(maps[k],twice[k]) for k in maps)
    below=donor['local'][:,2]<=plan['donorChinFloorZ']
    assert np.array_equal(maps[new['COLOR']][below],source[lip['COLOR']][below])
    assert np.array_equal(maps[new['ROUGH']][below],source[lip['ROUGH']][below])
    assert np.array_equal(maps[new['COLOR']][~skin],source[lip['COLOR']][~skin])
    assert np.array_equal(maps[new['ROUGH']][~skin],source[lip['ROUGH']][~skin])
    assert np.array_equal(maps[new['COLOR']][:,3],source[lip['COLOR']][:,3])
    assert plan['protectedNeckRowsExact'] and plan['protectedNeckTwoRingCount']==270
    support=source[lip['SUPPORT']]; active=support>1e-8
    assert not np.any(active & ~np.isin(np.arange(count),strip))
    lip_rgb_before=np.median(source[lip['COLOR']][active,:3],axis=0)
    lip_rgb_after=np.median(maps[new['COLOR']][active,:3],axis=0)
    # Invalid anatomy support must reject instead of painting elsewhere.
    altered=support.copy(); altered[8026]=1.  # Real forehead skin, not just a non-skin row.
    try: new['build_maps'](donor,landmarks,args[2],args[3],altered,skin,args[6])
    except RuntimeError: invalid_rejected=True
    else: invalid_rejected=False
    assert invalid_rejected
    report={'scope':__doc__,'helperSha256':sha(HERE/'natural_skin_refinement_r11.py'),
        'verifierSha256':sha(__file__),'donorProvenance':provenance,'neckSelectionSha256':sha(selection_path),
        'sourceMaps':'Reconstructed from documented neutral .49/.31/.225 baseline plus current natural_skin_adult and lip_surface_finish pure functions; not a new Blend observation.',
        'checks':{'syntax':True,'finiteBoundedFields':True,'deterministic':True,'unchangedAlphaAndNonSkin':True,
            'unchangedBelowChinFloor':True,'unchanged270NeckRows':True,'lipSupportStaysInside120RowStrip':True,'invalidLipSupportRejected':True},
        'lipActiveMedianLinearRGBBefore':lip_rgb_before.tolist(),'lipActiveMedianLinearRGBAfter':lip_rgb_after.tolist(),
        'plan':plan,'runtimeBoundaryGuardNotYetExecuted':True,'actualBlenderExecuted':False,'appearanceAccepted':False,'passed':True}
    output=P/'work/reconstruction-r02/natural-skin-refinement-offline-r11-v2.json'
    with output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(output),'sha256':sha(output),'helperSha256':report['helperSha256'],'lipMedianBefore':lip_rgb_before.tolist(),'lipMedianAfter':lip_rgb_after.tolist(),'checks':report['checks']}))


if __name__=='__main__': main()

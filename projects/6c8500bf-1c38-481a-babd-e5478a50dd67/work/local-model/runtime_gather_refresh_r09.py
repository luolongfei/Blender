"""Refresh both measured root-gather meshes into one shared coverage atlas.

Explicit source payload and runtime apply stages only; no launch/save/render.
The source must retain each pre-PN Curves data under the documented key.
"""
import importlib.util,json,re
from pathlib import Path


def _side(side):
    spec=importlib.util.spec_from_file_location('gather09_'+side,Path(__file__).with_name('runtime_scalp_refresh_r09.py'))
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    api.SOURCE='R06 full gathered roots '+side;api.TARGET='RT06 Cards / '+api.SOURCE
    api.REGION='gather';api.POLICY='replace_same_cohorts';api.PRIOR_DATA_KEY='R09_root_projection_original_data'
    api.MARKER='RT09_gather_'+side+'_refresh_manifest';api.BACKUP='RT09_gather_'+side+'_mesh_backup'
    api.PROPERTY_PREFIX='RT09_gather_'+side
    return api


def prepare_payload(source_relative,stage='r09-gather'):
    if not re.fullmatch(r'[a-z0-9-]+',stage):raise ValueError('Simple explicit stage name required')
    first=_side('L');path=first._path('work/runtime-r06/'+stage+'-manifest.json','work/runtime-r06')
    if path.exists():raise FileExistsError(path)
    rows={}
    for side in ('L','R'):
        api=_side(side);report=api.prepare_payload(source_relative,stage+'-'+side.lower())
        rows[side]={'manifest':'work/runtime-r06/'+stage+'-'+side.lower()+'/manifest.json',
            'manifestSha256':api._sha(api._path('work/runtime-r06/'+stage+'-'+side.lower()+'/manifest.json','work/runtime-r06')),
            'sourceSha256':report['sourceSha256'],'sourceRigSha256':report['sourceRig']['sha256']}
    if len({r['sourceSha256'] for r in rows.values()})!=1 or len({r['sourceRigSha256'] for r in rows.values()})!=1:
        raise RuntimeError('Gather sides do not share one source and rest rig')
    result={'schema':1,'source':source_relative,'sourceSha256':rows['L']['sourceSha256'],'sides':rows,
        'scope':'Both fixed gathered roots only; one shared newly projected gather atlas; tail meshes/bones untouched.'}
    first._json(path,result);return result


def apply_payload(manifest_relative):
    apis={s:_side(s) for s in ('L','R')};first=apis['L'];first._guard('work/runtime-r06')
    path=first._path(manifest_relative,'work/runtime-r06');spec=json.loads(path.read_text(encoding='utf-8'))
    if spec.get('schema')!=1 or set(spec['sides'])!={'L','R'}:raise RuntimeError('Expected two-side gather manifest')
    cards=first._api('runtime_haircards_r06');packets={};constants=[]
    for side,api in apis.items():
        row=spec['sides'][side];manifest=api._path(row['manifest'],'work/runtime-r06')
        if api._sha(manifest)!=row['manifestSha256']:raise RuntimeError('Gather manifest changed')
        _,data,paths,radii,_=api._read_payload(row['manifest'])
        if data['sourceSha256']!=spec['sourceSha256']:raise RuntimeError('Gather source mismatch')
        packets[api.SOURCE]=cards.card_packet(paths,radii,data['spec'])
        obj=api.bpy.data.objects[api.TARGET]
        if obj.get(api.BACKUP):raise RuntimeError('Gather already refreshed')
        if len(obj.data.materials)!=1:raise RuntimeError('Expected one existing gather material')
        bs=next(n for n in obj.data.materials[0].node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        names=('Roughness','Metallic','Specular IOR Level','Coat Weight','Coat Roughness')
        if any(bs.inputs[n].is_linked for n in names):raise RuntimeError('Unexpected linked gather scalar')
        constants.append({n:float(bs.inputs[n].default_value) for n in names})
    if constants[0]!=constants[1]:raise RuntimeError('Two gathered roots require different material constants; cannot silently merge')
    folder=path.with_suffix('');index=1
    while folder.exists():folder=path.with_suffix('').with_name(path.stem+f'-apply-{index:03d}');index+=1
    folder.mkdir();materials,atlas=cards._projected_materials(folder,packets)
    shared={'materials':materials,'atlas':atlas,'packets':packets};applied=[];reports={}
    try:
        for side,api in apis.items():
            reports[side]=api.apply_payload(spec['sides'][side]['manifest'],_shared=shared);applied.append(side)
        targets=[api.bpy.data.objects[api.TARGET] for api in apis.values()]
        if targets[0].data.materials[0]!=targets[1].data.materials[0]:raise RuntimeError('Gather atlas was not shared')
        result={'source':spec['source'],'sourceSha256':spec['sourceSha256'],'sides':reports,
            'sharedAtlas':atlas,'tailBonesAndMeshUnchanged':True,'appearanceAccepted':False,'savedBlend':False}
        first._json(folder/'apply-report.json',result);return result
    except Exception:
        for side in reversed(applied):apis[side].restore()
        raise


def restore():return {side:_side(side).restore() for side in ('R','L')}


if __name__=='__main__':print('Loaded only. prepare_payload/apply_payload/restore; never saves or starts Blender.')

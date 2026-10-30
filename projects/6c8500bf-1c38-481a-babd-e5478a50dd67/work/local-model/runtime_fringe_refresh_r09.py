"""Independent fringe-only adapter using retained pre-edit native fingerprints.

Uses the same transfer implementation as scalp but a separate explicit policy,
target object, atlas region, rollback marker and source-data backup. Never
launches Blender or saves/exports. Does not apply the scalp replacement.
"""
import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('fringe09_shared_transfer',Path(__file__).with_name('runtime_scalp_refresh_r09.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.SOURCE='R03 Forelocks from crown';base.TARGET='RT06 Cards / '+base.SOURCE
base.REGION='fringe';base.POLICY='replace_same_cohorts';base.PRIOR_DATA_KEY='R09_forelock_original_data'
base.MARKER='RT09_fringe_refresh_manifest';base.BACKUP='RT09_fringe_mesh_backup';base.PROPERTY_PREFIX='RT09_fringe'
_path=base._path;_sha=base._sha;_json=base._json

def _source_layers():
    """Trace R03 -> flow -> optional volume OR ellipse, without rebasing.

    The current runtime still owns the initial R03 points. Either final layer's
    immediate backup is the intermediate flow candidate, not that runtime
    baseline. Keep the first original-data key for the actual card comparison.
    """
    base._guard('work/reconstruction-r02');obj=base.bpy.data.objects[base.SOURCE]
    api=base._api('posterior_groom_r09');data=base.bpy.data.hair_curves
    original=data.get(obj.get('R09_forelock_original_data',''))
    if original is None:raise RuntimeError('Initial pre-flow fringe data is required')
    record=json.loads(obj.get('R09_forelock_restore_record','null'))
    flow=json.loads(obj.get('R09_forelock_flow','null'))
    initial_hash=api._data_hash(original)
    if not record or not flow or initial_hash!=record['originalHash'] or initial_hash!=flow['sourceDataSha256']:
        raise RuntimeError('Initial fringe restore hash no longer matches the flow source')
    rows=[{'layer':'runtime R03 baseline','data':original.name,'attributeTopologySha256':initial_hash,
           'backupKey':'R09_forelock_original_data'}]
    variants={}
    for variant in ('volume','ellipse'):
        prefix='R09_forelock_'+variant
        keys=(prefix+'_candidate',prefix+'_original_data',prefix+'_restore_record')
        present=[key in obj for key in keys]
        if any(present) and not all(present):raise RuntimeError('Incomplete '+variant+' layer metadata')
        if all(present):
            if any(not obj[key] for key in keys):raise RuntimeError('Empty '+variant+' layer metadata')
            variants[variant]={'report':json.loads(obj[keys[0]]),'backupKey':keys[1],
                'previous':json.loads(obj[keys[2]])}
    if len(variants)>1:raise RuntimeError('Fringe volume and ellipse layers are mutually exclusive; refuse a mixed or stale source')
    optional=next(iter(variants),None)
    if optional:
        entry=variants[optional];candidate=entry['report'];prior=data.get(obj[entry['backupKey']])
        previous=entry['previous']
        if prior is None or not previous:raise RuntimeError(optional+' layer has no retained immediate source')
        prior_hash=api._data_hash(prior)
        if prior_hash!=previous['originalHash'] or prior_hash!=flow['candidateDataSha256'] or prior_hash!=candidate['sourceDataSha256']:
            raise RuntimeError('Fringe '+optional+' layer does not descend from the recorded flow candidate')
        rows.append({'layer':'R09 flow intermediate','data':prior.name,'attributeTopologySha256':prior_hash,
                     'backupKey':entry['backupKey']})
        expected=candidate['candidateDataSha256']
    else:
        expected=flow['candidateDataSha256']
    current_hash=api._data_hash(obj.data)
    if current_hash!=expected:raise RuntimeError('Final fringe differs from its documented current layer')
    rows.append({'layer':'final source paths','data':obj.data.name,'attributeTopologySha256':current_hash})
    return {'layers':rows,'runtimeBaselineKey':'R09_forelock_original_data',
        'volumeLayerPresent':'volume' in variants,'ellipseLayerPresent':'ellipse' in variants,'optionalFinalLayer':optional,
        'mutuallyExclusiveFinalLayersValidated':True,
        'comparison':'Initial R03 runtime cards directly to final source paths; no intermediate rebase or second application.',
        'hashNote':'These are attribute+topology hashes. The numerical payload separately records rig-local point SHA-256.'}


def prepare_payload(source_relative,stage='r09-fringe'):
    return base.prepare_payload(source_relative,stage,_source_lineage=_source_layers())

def apply_payload(manifest_relative):return base.apply_payload(manifest_relative)

def restore():return base.restore()

if __name__=='__main__':print('Loaded only. Explicit fringe prepare_payload/apply_payload/restore, independent of scalp.')

"""Explicit, reversible INTERNAL-delivery metadata; no work on import.

Prepared for VRM Add-on 4.10.2 / VRM 1.0. This is not appearance acceptance.
Only the integrator may call apply() in a saved disposable runtime copy after
the R09 appearance/integration review. Never saves, exports or updates state.

    proposal = plan()                           # ordinary Python, read only
    record = apply(rig, runtime_path, runtime_sha256,
                   production_path, production_sha256)  # Blender, in memory
    # Integrator saves a NEW runtime checkpoint, then uses a deliberate final
    # export route. runtime_export_r06.export() currently forces preview naming;
    # it must not be used unchanged to produce a formal internal-delivery file.
    inspect_document(actual_gltf_json, record)  # after actual fresh export
    restore(rig)                               # exact guarded rollback

The companion internal-delivery-terms-r09.txt accompanies the whole package.
onlySeparatelyLicensedPerson requires that separate scope document; the enum
itself is not a recipient identity/access control system. Do not invent a URL.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

MARKER = 'RT09_internal_delivery_metadata'
TERMS = 'work/runtime-r06/internal-delivery-terms-r09.txt'
AUTHOR = 'Original character supplied by project owner (credit name not provided)'
NAME = '双马尾角色 · 内部项目交付'
LICENSE_URL = 'https://vrm.dev/licenses/1.0/'
SCHEMA_URL = 'https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/schema/VRMC_vrm.meta.schema.json'
COPYRIGHT = (
    'Original character artwork/design supplied by the project owner; credit name not provided. '
    'Prepared as an internal editable deliverable for this project owner and recipients '
    'separately designated by that owner for this project. Internal project use and editing '
    'only; no public authorization to third parties and no redistribution of this character '
    'or modified character. See the accompanying internal-delivery-terms-r09.txt. '
    'Original third-party component terms remain applicable. This statement records source '
    'and delivery scope, not copyright registration, ownership adjudication, or appearance acceptance.'
)
THIRD_PARTY = (
    'Human Base Meshes 1.4.1, Blender Studio/community: CC0; adapted geometry. '
    'Source: https://www.blender.org/download/demo-files/ . Original per-asset '
    'attribution and local source hashes are retained in the project asset records. '
    'The internal character-delivery restrictions do not restrict independent rights '
    'in that underlying CC0 component.\n'
    'Production-source motion dependency where included: CMU Graphics Lab Motion Capture '
    'Database, subject 07 / trial 07_01, https://mocap.cs.cmu.edu/ and '
    'https://mocap.cs.cmu.edu/faqs.php . Its original terms allow copying/modification '
    'and inclusion in products, but prohibit direct resale of the data, including '
    'converted data. The runtime VRM export does not contain the production action/cache. '
    'See assets/mocap-cmu/07/source-record.json for exact provenance and scope.\n'
    'The adopted R07 iris image is generated within this workspace, with no external '
    'image pixels and no separate public license granted. Historical assets are not '
    'automatically included or cleared by this notice. No claim of new third-party authorization.'
)
FIELDS = {
    'vrm_name': 'name', 'version': 'version',
    'copyright_information': 'copyrightInformation',
    'third_party_licenses': 'thirdPartyLicenses',
    'avatar_permission': 'avatarPermission',
    'allow_excessively_violent_usage': 'allowExcessivelyViolentUsage',
    'allow_excessively_sexual_usage': 'allowExcessivelySexualUsage',
    'commercial_usage': 'commercialUsage',
    'allow_political_or_religious_usage': 'allowPoliticalOrReligiousUsage',
    'allow_antisocial_or_hate_usage': 'allowAntisocialOrHateUsage',
    'credit_notation': 'creditNotation',
    'allow_redistribution': 'allowRedistribution',
    'modification': 'modification', 'other_license_url': 'otherLicenseUrl',
}
ENUMS = {
    'avatar_permission': {'onlyAuthor', 'onlySeparatelyLicensedPerson', 'everyone'},
    'commercial_usage': {'personalNonProfit', 'personalProfit', 'corporation'},
    'credit_notation': {'required', 'unnecessary'},
    'modification': {'prohibited', 'allowModification', 'allowModificationRedistribution'},
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def plan():
    """Pure proposal. Values were expressly adopted by the project integrator."""
    return {
        'schema': 1, 'status': 'prepared-not-applied',
        'authors': [AUTHOR],
        'fields': {
            'vrm_name': NAME, 'version': 'R09',
            'copyright_information': COPYRIGHT, 'third_party_licenses': THIRD_PARTY,
            'avatar_permission': 'onlySeparatelyLicensedPerson',
            'commercial_usage': 'corporation', 'credit_notation': 'required',
            'allow_redistribution': False, 'modification': 'allowModification',
            'allow_excessively_violent_usage': False,
            'allow_excessively_sexual_usage': False,
            'allow_political_or_religious_usage': False,
            'allow_antisocial_or_hate_usage': False,
            'other_license_url': '',
        },
        'licenseUrl': LICENSE_URL, 'schemaReference': SCHEMA_URL,
        'termsFile': TERMS,
        'policyReasons': {
            'authors': 'Truthful source description expressly selected for this project; no personal name invented.',
            'avatarPermission': 'Only the owner and separately designated project recipients; not everyone.',
            'commercialUsage': 'VRM1 corporation permits organizational recipients; the accompanying scope confines their use to this project.',
            'modification': 'Editable delivery permits continued project edits, not redistribution of modifications.',
            'copyrightInformation': 'Source and scope only; not a registration or adjudication claim.',
            'otherLicenseUrl': 'Empty because no independently supplied, published license URL exists.',
        },
        'preserved': ['references', 'contact_information', 'thumbnail_image', 'existing third-party notices (appended verbatim)'],
        'appearanceAccepted': False, 'thirdPartyAuthorizationNewlyObtained': False,
        'requiresFreshExportAndActualJsonInspection': True,
    }


def _snapshot(meta):
    return {'fields': {key: getattr(meta, key) for key in FIELDS},
            'authors': [entry.value for entry in meta.authors]}


def _write(meta, snapshot):
    for key, value in snapshot['fields'].items():
        setattr(meta, key, value)
    meta.authors.clear()
    for author in snapshot['authors']:
        meta.authors.add().value = author


def _validate_rna(meta, fields):
    for key, value in fields.items():
        prop = meta.bl_rna.properties.get(key)
        if prop is None:
            raise RuntimeError('VRM metadata property missing: ' + key)
        if key in ENUMS:
            available = {item.identifier for item in prop.enum_items}
            if not ENUMS[key].issubset(available) or value not in available:
                raise RuntimeError('Installed metadata enum contract differs: ' + key)
        elif isinstance(value, bool):
            if prop.type != 'BOOLEAN':
                raise RuntimeError('Expected Boolean metadata: ' + key)
        elif prop.type != 'STRING':
            raise RuntimeError('Expected String metadata: ' + key)


def _checked_file(relative, expected_sha, subfolder):
    project = project_path().resolve()
    path = (project / relative).resolve()
    path.relative_to((project / subfolder).resolve())
    if not path.is_file() or len(expected_sha) != 64 or _sha(path) != expected_sha:
        raise RuntimeError('Explicit source file/hash differs: ' + str(relative))
    return path


def apply(rig, runtime_path, runtime_sha256, production_path, production_sha256):
    """Mutate metadata only, in memory. Caller owns saving a NEW checkpoint."""
    import bpy
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Use a saved disposable background runtime copy in Object Mode')
    runtime = _checked_file(runtime_path, runtime_sha256, 'work/runtime-r06')
    production = _checked_file(production_path, production_sha256, 'work')
    if Path(bpy.data.filepath).resolve() != runtime:
        raise RuntimeError('Open the explicitly hashed runtime source')
    if rig.type != 'ARMATURE' or rig.library or rig.data.library or bpy.context.scene.objects.get(rig.name) != rig:
        raise RuntimeError('An explicit local scene armature is required')
    if not rig.data.vrm_addon_extension.is_vrm1():
        raise RuntimeError('VRM 1.0 armature required; no automatic conversion')
    if MARKER in rig.data:
        raise RuntimeError('Metadata already applied; inspect or restore first')
    terms = project_path() / TERMS
    if not terms.is_file() or AUTHOR not in terms.read_text(encoding='utf-8'):
        raise RuntimeError('Companion internal-delivery terms missing or incompatible')
    proposal = plan()
    meta = rig.data.vrm_addon_extension.vrm1.meta
    before = _snapshot(meta)
    after = {'fields': dict(proposal['fields']), 'authors': list(proposal['authors'])}
    existing = before['fields']['third_party_licenses'].strip()
    if existing and existing != THIRD_PARTY:
        after['fields']['third_party_licenses'] = existing + '\n\nR09 additional provenance/scope notice:\n' + THIRD_PARTY
    _validate_rna(meta, after['fields'])
    record = {
        'schema': 1, 'status': 'applied-in-memory-not-saved-or-exported',
        'runtimeBeforeMetadata': str(runtime.relative_to(project_path())).replace('\\', '/'),
        'runtimeBeforeMetadataSha256': runtime_sha256,
        'productionSource': str(production.relative_to(project_path())).replace('\\', '/'),
        'productionSourceSha256': production_sha256,
        'termsFile': TERMS, 'termsSha256': _sha(terms),
        'before': before, 'after': after, 'policyReasons': proposal['policyReasons'],
        'appearanceAccepted': False, 'thirdPartyAuthorizationNewlyObtained': False,
        'requiresFreshExportAndActualJsonInspection': True,
    }
    try:
        _write(meta, after)
        if _snapshot(meta) != after:
            raise RuntimeError('Metadata did not round-trip exactly')
        rig.data[MARKER] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    except Exception:
        _write(meta, before)
        if MARKER in rig.data:
            del rig.data[MARKER]
        raise
    return record


def restore(rig):
    """No saving; refuse to erase unrelated metadata edits made after apply()."""
    record = json.loads(rig.data[MARKER])
    meta = rig.data.vrm_addon_extension.vrm1.meta
    if _snapshot(meta) != record['after']:
        raise RuntimeError('Metadata changed after application; do not discard it')
    _write(meta, record['before'])
    if _snapshot(meta) != record['before']:
        raise RuntimeError('Original metadata did not restore exactly')
    del rig.data[MARKER]
    return {'restored': True, 'saved': False}


def inspect_document(document, record=None):
    """Read-only exact contract for JSON decoded from a freshly exported VRM."""
    expected = record['after'] if record else plan()
    meta = document.get('extensions', {}).get('VRMC_vrm', {}).get('meta')
    if not isinstance(meta, dict):
        raise ValueError('Actual VRM1 metadata missing')
    if meta.get('licenseUrl') != LICENSE_URL or meta.get('authors') != expected['authors']:
        raise ValueError('License URL or truthful credit differs')
    for python_name, json_name in FIELDS.items():
        value = expected['fields'][python_name]
        actual = meta.get(json_name, '') if python_name == 'other_license_url' else meta.get(json_name)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('Actual exported metadata differs: ' + json_name)
    if '__CHARAKIT_REVIEW__' in json.dumps(meta, ensure_ascii=False):
        raise ValueError('Preview placeholder remains; cannot promote by filename')
    return {'metadataExact': True, 'name': meta['name'], 'version': meta['version'],
            'appearanceAccepted': False, 'filePublicationPerformed': False}

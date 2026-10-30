"""Fresh INTERNAL-delivery staging, separate from the unchanged preview exporter.

No work on import, no process launch, no writes to exports/avatar.*. The
integrator runs these explicit stages in separate background Blender processes:

1. With final PRODUCTION open, before runtime payload preparation:
   -- source-audit --production work/reconstruction-r02/FINAL.blend
      --production-sha SHA --report work/runtime-r06/FINAL-source-layers.json
   R10_crown_volume_candidate is recorded and BLOCKED until a dedicated adapter
   transports final root/scalp/fringe/gather positions and rebuilds root maps.
   This helper never assumes that R09 payloads represent a new R10 cage layer.

2. After all integration/material stages, with that saved RUNTIME open:
   -- prepare --bundle work/runtime-r06/FINAL-integration.json
      --runtime work/runtime-r06/IN.blend --runtime-sha SHA
      --production work/reconstruction-r02/FINAL.blend --production-sha SHA
      --source-audit work/runtime-r06/FINAL-source-layers.json
      --output work/runtime-r06/INTERNAL-ready.blend
      --report work/runtime-r06/INTERNAL-preparation.json
   Applies the expressly selected internal metadata, configures expressions,
   saves a NEW runtime, and records its exact hash. Does not export.

3. CLOSE that process; reopen INTERNAL-ready.blend in a fresh background process:
   -- export --preparation work/runtime-r06/INTERNAL-preparation.json
      --output-folder work/runtime-r06/internal-delivery-r09
   Reads the saved scene without reconfiguring expressions. Fresh exports are
   avatar.vrm + avatar.glb inside a NEW STAGING folder. Exact VRM1 metadata,
   explicit meshes, PBR/alpha/pixels, expressions and springs are checked.
   A complete manifest is written last and the whole staging directory is
   renamed into the requested fresh folder. Failed staging remains for diagnosis.

The production/runtime/VRM SHA link is explicit. No filenames are promoted from
old previews. Final fixed-name copies, reopened dependencies, actual target-
viewer evidence and appearance status remain separate integrator-owned steps.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

try:
    import bpy
except ImportError:
    bpy = None
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CROWN_OBJECTS = ('R06 attached scalp root substrate', 'R06 continuous scalp flow',
                 'R03 Forelocks from crown', 'R06 full gathered roots L', 'R06 full gathered roots R')
REQUIRED_STAGES = {'face', 'scalp', 'fringe', 'gather', 'root', 'tail', 'skin-bake', 'root-bake', 'iris'}
LAYER_GAP = ('R10 crown cage is not adapted by this R09 entry. Preserve final native point '
             'projection/lineage for root, scalp, fringe and both gathers, keep ties/tails '
             'protected, refresh their regional cards/atlases, and rebake changed root maps '
             'before implementing and validating a new explicit export contract.')


def _api(name):
    spec = importlib.util.spec_from_file_location('internal_delivery_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(relative, area='work/runtime-r06'):
    path = (project_path() / relative).resolve()
    path.relative_to((project_path() / area).resolve())
    return path


def _relative(path):
    return str(Path(path).resolve().relative_to(project_path().resolve())).replace('\\', '/')


def _identity(path):
    return {'file': _relative(path), 'sha256': _sha(path), 'bytes': Path(path).stat().st_size}


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _checked(relative, sha, area='work/runtime-r06'):
    path = _path(relative, area)
    if not path.is_file() or len(sha) != 64 or _sha(path) != sha:
        raise RuntimeError('Source file/hash differs: ' + relative)
    return path


def _open(path, fresh=False):
    if bpy is None or not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Explicit saved background Blender file in Object Mode required')
    if Path(bpy.data.filepath).resolve() != Path(path).resolve():
        raise RuntimeError('Open the exact explicitly identified source file')
    if fresh and bpy.data.is_dirty:
        raise RuntimeError('Freshly reopen the saved internal-ready runtime before export; unsaved scene changes are not accepted')


def audit_source_layers(production_path, production_sha256, report_relative):
    """Read CURRENT native objects in the exact production file, without editing."""
    source = _checked(production_path, production_sha256, 'work')
    _open(source, fresh=True)
    output = _path(report_relative)
    if output.exists():
        raise FileExistsError(output)
    rows = []; markers = []
    for name in CROWN_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError('Production groom object missing: ' + name)
        rows.append({'object': name, 'data': obj.data.name, 'type': obj.type, 'hideRender': bool(obj.hide_render)})
        for owner, block in (('object', obj), ('data', obj.data)):
            for key in block.keys():
                if str(key).startswith('R10_'):
                    value = str(block[key])
                    markers.append({'object': name, 'owner': owner, 'key': key,
                                    'value': value, 'valueSha256': hashlib.sha256(value.encode()).hexdigest()})
    for key in bpy.context.scene.keys():
        if str(key).startswith('R10_crown'):
            value = str(bpy.context.scene[key])
            markers.append({'object': '<scene>', 'owner': 'scene', 'key': key,
                            'value': value, 'valueSha256': hashlib.sha256(value.encode()).hexdigest()})
    if _sha(source) != production_sha256:
        raise RuntimeError('Production file changed during the read-only layer audit')
    report = {'schema': 1, 'source': _identity(source), 'currentNativeObjects': rows,
              'observedR10Markers': markers, 'requiresNewCageAdapter': bool(markers),
              'cageAdaptationPolicy': LAYER_GAP, 'openedFresh': True, 'sceneChanged': False, 'blenderSaved': False,
              'auditHelperSha256': _sha(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    return report


def _validate_layers(audit, production):
    if audit.get('schema') != 1 or audit.get('source') != _identity(production):
        raise RuntimeError('Source-layer audit belongs to another exact production file')
    if audit.get('openedFresh') is not True or audit.get('auditHelperSha256') != _sha(__file__):
        raise RuntimeError('Repeat the small source-layer audit with this helper on a freshly opened production file')
    if {row['object'] for row in audit.get('currentNativeObjects', [])} != set(CROWN_OBJECTS):
        raise RuntimeError('Source-layer audit did not examine all five active crown regions')
    if audit.get('requiresNewCageAdapter') or audit.get('observedR10Markers'):
        raise RuntimeError(LAYER_GAP)
    if audit.get('requiresNewCageAdapter') is not False:
        raise RuntimeError('Missing explicit source-layer audit outcome')


def _bundle_contract(bundle_path, production, rig):
    bundle = _read(bundle_path)
    if bundle.get('schema') != 1 or set(bundle.get('members', {})) != {'face', 'scalp', 'fringe', 'gather', 'root'}:
        raise RuntimeError('Explicit completed integration bundle required')
    if _path(bundle['source'], 'work') != production or bundle['sourceSha256'] != _sha(production):
        raise RuntimeError('Production identity is not the exact integrated payload source')
    initial = _checked(bundle['initialRuntime'], bundle['initialRuntimeSha256'])
    for region, row in bundle['members'].items():
        member = _checked(row['file'], row['sha256'])
        data = _read(member)
        source_hash = data.get('sourceSha256') or (data.get('source') or {}).get('sha256')
        if source_hash != bundle['sourceSha256']:
            raise RuntimeError('Region payload has another production source: ' + region)
    marker = json.loads(rig.get('RT09_integration_bundle', '{}'))
    if not marker.get('file') or _path(marker['file']) != bundle_path or marker.get('sha256') != _sha(bundle_path):
        raise RuntimeError('Runtime integration marker differs from bundle')
    done = json.loads(rig.get('RT09_integration_completed', '[]'))
    if not isinstance(done, list) or len(done) != len(set(done)) or REQUIRED_STAGES - set(done):
        raise RuntimeError('Runtime region/material integration is incomplete')
    return bundle, initial, done


def _supplemental_meta(rig):
    meta = rig.data.vrm_addon_extension.vrm1.meta
    image = meta.thumbnail_image
    return {'references': [r.value for r in meta.references if r.value],
            'contactInformationInSource': meta.contact_information,
            'contactExportLimit': 'Installed VRM Add-on 4.10.2 omits contactInformation; retained here as source metadata only.',
            'thumbnailImage': image.name if image else None,
            'thumbnailSource': _api('runtime_export_r06')._image_source(SimpleNamespace(image=image)) if image else None}


def prepare_delivery(bundle_relative, runtime_path, runtime_sha256, production_path,
                     production_sha256, source_audit_relative, output_relative, report_relative):
    """Configure, apply deliberate metadata and save one NEW editable runtime."""
    runtime = _checked(runtime_path, runtime_sha256)
    production = _checked(production_path, production_sha256, 'work')
    _open(runtime, fresh=True)
    output = _path(output_relative); report_path = _path(report_relative)
    if output.suffix.lower() != '.blend' or output.exists() or report_path.exists() or output == report_path:
        raise FileExistsError('New independent runtime Blend and preparation report required')
    bundle_path = _path(bundle_relative); audit_path = _path(source_audit_relative)
    _validate_layers(_read(audit_path), production)
    tech = _api('runtime_export_r06'); metadata = _api('runtime_internal_metadata_r09')
    rig, meshes = _api('runtime_assembly_r06').current_meshes()
    bundle, initial, completed = _bundle_contract(bundle_path, production, rig)
    lineage = [_relative(production), _relative(initial)]
    # Explicit preparation may configure; only the saved result is deliverable.
    contract = tech.prepare(lineage, require_baked=True, write=False, configure_expressions=True)
    semantics = tech.read_vrm_semantics(rig)
    supplementary = _supplemental_meta(rig)
    record = metadata.apply(rig, runtime_path, runtime_sha256, production_path, production_sha256)
    if _supplemental_meta(rig) != supplementary:
        raise RuntimeError('Metadata preparation changed preserved fields')
    if _sha(runtime) != runtime_sha256 or _sha(production) != production_sha256:
        raise RuntimeError('An input changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if 'FINISHED' not in result or not output.is_file():
        raise RuntimeError('New metadata-bearing runtime save failed')
    saved = tech.prepare(lineage, require_baked=True, write=False, configure_expressions=False)
    if saved['sceneContractSha256'] != contract['sceneContractSha256'] or tech.read_vrm_semantics(rig) != semantics:
        raise RuntimeError('Read-only saved runtime contract differs after preparation')
    report = {'schema': 1, 'stage': 'internal-delivery-prepared-not-exported',
              'production': _identity(production), 'runtime': _identity(output),
              'inputRuntime': _identity(runtime), 'integrationBundle': _identity(bundle_path),
              'sourceLayerAudit': _identity(audit_path), 'completedIntegrationStages': completed,
              'technicalContract': saved, 'semanticContract': semantics,
              'metadataRecord': record, 'supplementalMetadata': supplementary,
              'terms': _identity(project_path() / metadata.TERMS),
              'code': [_identity(Path(__file__).with_name(name + '.py')) for name in
                       ('runtime_internal_export_r09', 'runtime_export_r06', 'runtime_internal_metadata_r09',
                        'runtime_assembly_r06', 'runtime_face_snapshot_r06', 'vrm_helpers')],
              'requireFreshReopen': True, 'appearanceAccepted': False,
              'publicRedistributionAuthorized': False, 'fixedOutputFilesChanged': False}
    _json(report_path, report)
    return report


def _checked_coat_fix(tech, path, contract):
    """The one documented exporter correction, verified as JSON-only and bounded."""
    before, binary = tech.read_container(path)
    corrections = tech._restore_default_coat_roughness(path, contract)
    after, after_binary = tech.read_container(path)
    if hashlib.sha256(binary).digest() != hashlib.sha256(after_binary).digest():
        raise RuntimeError('Coat-default correction unexpectedly changed BIN data')
    expected = json.loads(json.dumps(before))
    for row in corrections:
        material = next(m for m in expected['materials'] if m.get('name') == row['material'])
        coat = material['extensions']['KHR_materials_clearcoat']
        if 'clearcoatRoughnessFactor' in coat or abs(row['sourceValue'] - .03) > 1e-5:
            raise RuntimeError('Unapproved coat correction')
        coat['clearcoatRoughnessFactor'] = row['sourceValue']
    if after != expected:
        raise RuntimeError('Coat correction changed unrelated exported JSON')
    return {'corrections': corrections, 'binarySha256': hashlib.sha256(binary).hexdigest(), 'binaryUnchanged': True}


def _inspect_supplemental(document, expected):
    meta = document['extensions']['VRMC_vrm']['meta']
    if meta.get('references', []) != expected['references']:
        raise RuntimeError('Saved reference metadata changed in export')
    if bool('thumbnailImage' in meta) != bool(expected['thumbnailImage']):
        raise RuntimeError('Saved thumbnail presence changed in export')
    return {'referencesExact': True, 'thumbnailPresenceExact': True,
            'contactInformation': expected['contactExportLimit']}


def export_delivery(preparation_relative, output_relative):
    """Fresh export only. Never calls the preview export() or applies metadata."""
    preparation_path = _path(preparation_relative); prep = _read(preparation_path)
    if prep.get('schema') != 1 or prep.get('stage') != 'internal-delivery-prepared-not-exported':
        raise RuntimeError('Explicit internal-delivery preparation report required')
    runtime = _checked(prep['runtime']['file'], prep['runtime']['sha256'])
    production = _checked(prep['production']['file'], prep['production']['sha256'], 'work')
    _open(runtime, fresh=True)
    for row in prep['code']:
        _checked(row['file'], row['sha256'], 'work')
    if (_sha(ROOT / 'toolchain.lock.json') != prep['technicalContract']['toolchainLockSha256'] or
            bpy.app.version_string != prep['technicalContract']['blender']):
        raise RuntimeError('Prepared toolchain lock or Blender version changed')
    bundle_path = _checked(prep['integrationBundle']['file'], prep['integrationBundle']['sha256'])
    audit_path = _checked(prep['sourceLayerAudit']['file'], prep['sourceLayerAudit']['sha256'])
    terms = _checked(prep['terms']['file'], prep['terms']['sha256'])
    _validate_layers(_read(audit_path), production)
    tech = _api('runtime_export_r06'); metadata = _api('runtime_internal_metadata_r09')
    rig, meshes = _api('runtime_assembly_r06').current_meshes()
    _bundle_contract(bundle_path, production, rig)
    record = json.loads(rig.data.get(metadata.MARKER, '{}'))
    if record != prep['metadataRecord'] or metadata._snapshot(rig.data.vrm_addon_extension.vrm1.meta) != record['after']:
        raise RuntimeError('Saved internal-delivery metadata differs from prepared values')
    if tech.read_vrm_semantics(rig) != prep['semanticContract'] or _supplemental_meta(rig) != prep['supplementalMetadata']:
        raise RuntimeError('Saved semantic or supplementary metadata contract changed')
    current = tech.prepare([r['file'] for r in prep['technicalContract']['sourceLineage']],
                           require_baked=True, write=False, configure_expressions=False)
    if (current['savedRuntimeCopy'].replace('\\', '/') != prep['runtime']['file'] or
            current['savedRuntimeSha256'] != prep['runtime']['sha256'] or
            current['sceneContractSha256'] != prep['technicalContract']['sceneContractSha256'] or
            current['sourceLineage'] != prep['technicalContract']['sourceLineage']):
        raise RuntimeError('Exact reopened runtime identity or technical contract changed')
    destination = _path(output_relative)
    if destination.exists():
        raise FileExistsError('Use a fresh internal-delivery staging destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + destination.name + '.incomplete-' + uuid.uuid4().hex[:10])
    staging.mkdir()
    selected = list(bpy.context.selected_objects); active = bpy.context.view_layer.objects.active
    meshes = [bpy.data.objects[row['name']] for row in current['meshes']]
    hidden = {obj: obj.hide_get() for obj in [rig, *meshes]}
    try:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in [rig, *meshes]:
            obj.hide_set(False); obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        glb = staging / 'avatar.glb'; vrm = staging / 'avatar.vrm'
        result = bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', use_selection=True,
            export_materials='EXPORT', export_image_format='AUTO', export_animations=False, export_current_frame=True,
            export_rest_position_armature=True, export_apply=False, export_morph=True, export_morph_normal=True,
            export_skins=True, export_all_influences=False, export_influence_nb=4, export_yup=True,
            export_cameras=False, export_lights=False, export_extras=True, export_try_sparse_sk=False)
        if 'FINISHED' not in result:
            raise RuntimeError('Fresh internal GLB export failed')
        result = bpy.ops.export_scene.vrm(filepath=str(vrm), armature_object_name=rig.name,
            export_only_selections=True, export_invisibles=False, use_addon_preferences=False,
            enable_advanced_preferences=True, export_all_influences=False, export_lights=False,
            export_gltf_animations=False, export_try_sparse_sk=False)
        if 'FINISHED' not in result:
            raise RuntimeError('Fresh internal VRM export failed')
        checks = {}; scratch = staging / 'pixel-audit'; scratch.mkdir()
        for path, is_vrm in ((glb, False), (vrm, True)):
            corrections = _checked_coat_fix(tech, path, current)
            checked, document = tech.inspect(path, current, vrm=is_vrm)
            checked['constantCorrectionAudit'] = corrections
            checked['pixelAudit'] = tech._pixel_audit(path, current, checked, scratch)
            checked['file'] = _relative(destination / path.name)
            if is_vrm:
                checked['internalMetadata'] = metadata.inspect_document(document, record)
                checked['supplementalMetadata'] = _inspect_supplemental(document, prep['supplementalMetadata'])
            _json(staging / (path.name + '.json'), document)
            checks[path.suffix[1:]] = checked
        # No scene repairs or new metadata configuration occur in this phase.
        after = tech.prepare([r['file'] for r in current['sourceLineage']],
                             require_baked=True, write=False, configure_expressions=False)
        if (after['sceneContractSha256'] != current['sceneContractSha256'] or
                tech.read_vrm_semantics(rig) != prep['semanticContract'] or
                metadata._snapshot(rig.data.vrm_addon_extension.vrm1.meta) != record['after'] or
                _supplemental_meta(rig) != prep['supplementalMetadata'] or
                _sha(runtime) != prep['runtime']['sha256'] or _sha(production) != prep['production']['sha256']):
            raise RuntimeError('Export changed a source identity or recorded scene contract')
        # Keep source correspondence inside the same staging package. Actual
        # editable source files are NOT copied here by this export-only entry.
        shutil.copyfile(terms, staging / terms.name)
        if _sha(staging / terms.name) != prep['terms']['sha256']:
            raise RuntimeError('Companion terms copy verification failed')
        # Pixel samples are diagnostics, not release assets. Leave their path
        # explicitly identified rather than deleting any staging files.
        manifest = {'schema': 1, 'status': 'fresh-internal-export-technical-checks-complete',
            'production': prep['production'], 'runtime': prep['runtime'], 'outputs': checks,
            'preparation': _identity(preparation_path), 'integrationBundle': prep['integrationBundle'],
            'sourceLayerAudit': prep['sourceLayerAudit'], 'metadataRecord': record,
            'supplementalMetadata': prep['supplementalMetadata'],
            'terms': {'file': terms.name, 'sha256': prep['terms']['sha256']},
            'toolchainLockSha256': _sha(ROOT / 'toolchain.lock.json'), 'blender': bpy.app.version_string,
            'materialPolicy': 'Actual PBR atlases and constants preserved; no MToon conversion.',
            'losses': tech.LOSSES, 'technicalScope': 'Exact saved Blend hashes plus explicit mesh/material/morph/spring and PNG grid checks; not a complete semantic proof of every UV/weight/modifier.',
            'viewerValidation': 'Pending actual review of this exact new VRM SHA.',
            'appearanceAccepted': False, 'clientAcceptanceClaimed': False,
            'publicRedistributionAuthorized': False, 'fixedOutputFilesChanged': False,
            'productionAndRuntimeFilesCopiedIntoPackage': False,
            'diagnosticPixelPngFolder': 'pixel-audit',
            'completionPolicy': 'Only the requested final staging directory with no FAILED.json is complete. An .incomplete-* directory is never a completed output, even if its technical-check manifest exists.'}
        _json(staging / 'avatar-release-manifest.json', manifest)
        # Same parent filesystem: complete directory appears only after all
        # checks and the manifest succeed. A failed folder stays .incomplete-*.
        staging.rename(destination)
        return manifest
    except Exception as exc:
        _json(staging / 'FAILED.json', {'status': 'failed-not-a-release', 'reason': str(exc),
                                      'production': prep['production'], 'runtime': prep['runtime']})
        raise
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects:
                obj.select_set(True)
        if active and active.name in bpy.context.view_layer.objects:
            bpy.context.view_layer.objects.active = active
        for obj, value in hidden.items():
            obj.hide_set(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('source-audit', 'prepare', 'export'))
    for name in ('production', 'production-sha', 'runtime', 'runtime-sha', 'bundle', 'source-audit',
                 'output', 'report', 'preparation', 'output-folder'):
        parser.add_argument('--' + name)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    required = {'source-audit': ('production', 'production_sha', 'report'),
                'prepare': ('bundle', 'runtime', 'runtime_sha', 'production', 'production_sha', 'source_audit', 'output', 'report'),
                'export': ('preparation', 'output_folder')}[args.phase]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error('Explicit arguments required: ' + ', '.join('--' + name.replace('_', '-') for name in missing))
    if args.phase == 'source-audit':
        result = audit_source_layers(args.production, args.production_sha, args.report)
    elif args.phase == 'prepare':
        result = prepare_delivery(args.bundle, args.runtime, args.runtime_sha, args.production,
                                  args.production_sha, args.source_audit, args.output, args.report)
    else:
        result = export_delivery(args.preparation, args.output_folder)
    print('INTERNAL_DELIVERY_STAGE', json.dumps({'phase': args.phase,
          'status': result.get('status', result.get('stage', 'source-audited')), 'appearanceAccepted': False}), flush=True)


if __name__ == '__main__':
    main()

"""Offline evidence dossier; never changes assets, Blender files or shared state."""
import argparse
import datetime
import hashlib
import html
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

P = project_path()
DEMO = 'https://www.blender.org/download/demo-files/'


def within(relative):
    path = (P / relative).resolve()
    path.relative_to(P.resolve())
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def item(relative, expected=None):
    path = within(relative)
    result = {'path': path.relative_to(P).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    if expected and result['sha256'] != expected:
        raise RuntimeError('Evidence hash differs: ' + relative)
    return result


def read(relative):
    return json.loads(within(relative).read_text(encoding='utf-8-sig'))


def png_info(relative):
    with within(relative).open('rb') as stream:
        header = stream.read(33)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise RuntimeError('Not PNG IHDR: ' + relative)
    return {'width': struct.unpack('>I', header[16:20])[0],
            'height': struct.unpack('>I', header[20:24])[0],
            'bitDepth': header[24], 'pngColorType': header[25], 'method': 'Read-only PNG IHDR; pixels not edited'}


def website_evidence():
    request = urllib.request.Request(DEMO, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(3000000)
        if len(raw) >= 3000000:
            raise RuntimeError('Unexpected demo page size')
    source = raw.decode('utf-8')
    anchor = source.index('id="hair"')
    region = source[anchor:anchor + 9000]
    marker = region.index('CC-BY-SA')
    # Only the first hair card; avoid attributing a neighbouring asset's license.
    card = region[:marker + region[marker:].index('</span>') + 7]
    notice = re.search(r'CC-BY-SA.*?</p>', card, re.S).group(0)
    links = sorted(set(html.unescape(s) for s in re.findall(r'href="([^"]+)"', card)))
    return {'url': DEMO + '#hair', 'retrievedAtUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'responseSha256': hashlib.sha256(raw).hexdigest(),
            'publishedNotice': html.unescape(re.sub('<[^>]+>', '', notice)).strip(),
            'cardLinks': links,
            'versionedCreativeCommonsLinkInHairCard': any('creativecommons.org/licenses/' in s for s in links),
            'exactVersion': None,
            'scope': 'Parsed the Hair Styles card only; no version is inferred from other demos or the author name.'}


def build(output):
    out = within(output)
    if out.exists():
        raise FileExistsError('Preserve earlier dossier; choose a new output')
    guides_path = 'work/reconstruction-r02/hair-donor-guides-r10.json'
    guides = read(guides_path)
    text = next(t for t in guides['texts'] if t['name'] == 'Hair demo file info')
    credit = text['content'].split('Credits:', 1)[1].split('------------------------------', 1)[0].strip()
    if 'Daniel Bystedt. CC BY-SA license' not in credit:
        raise RuntimeError('Original credit was not found')
    donor_report_path = 'work/reconstruction-r02/reference-adult-donor-groom-r11.donor-fit.json'
    donor_report = read(donor_report_path)
    donor = item('assets/blender-hair-styles-bystedt-2023/hair_nodes-female_hair_styles.blend',
                 '1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48')
    fitted = item(donor_report['output'].replace('\\', '/'), donor_report['outputSha256'])
    generated = []
    for stem, expected in [
        ('references/generated/realistic-character-three-views-r11', '330c2c5ab2033217e36a54fe3b367367cbe29a0d68d04a28293f8a3d269176c3'),
        ('textures/generated-r11/iris-dark-brown-r11', '8aa48663c77fec6c668c4bc67c0673776fde6d3ae9c4a3fed81a64b776ae3e0e')]:
        metadata = read(stem + '.json')
        if 'not exposed' not in metadata['actualModel']:
            raise RuntimeError('Generated model provenance requires review')
        generated.append({'image': item(stem + '.png', expected), 'metadata': item(stem + '.json'),
                          'actualSize': png_info(stem + '.png'), 'requestedModel': metadata['requestedModel'],
                          'actualModel': metadata['actualModel'], 'purpose': metadata['purpose'],
                          'hasFullPrompt': bool(metadata.get('prompt')), 'sourceMetadataUnchanged': True})
    iris_path = 'work/reconstruction-r02/reference-adult-dark-iris-r11.iris.json'
    iris = read(iris_path)
    iris_source = item(iris['output'].replace('\\', '/'), iris['outputSha256'])
    if not iris['candidateSaved'] or not iris['imagePacked'] or iris['textureSha256'] != generated[1]['image']['sha256']:
        raise RuntimeError('Iris application evidence mismatch')
    head_path = 'work/reconstruction-r02/female-basemesh-actual-r11.json'
    head = read(head_path)
    head_donor = item(head['source'].replace('\\', '/'), head['source_sha256'])
    report = {
        'schemaVersion': 1, 'status': 'verified-source-evidence; not final delivery clearance',
        'recordedAtUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'helper': item(Path(__file__).relative_to(P).as_posix()),
        'officialHairCard': website_evidence(),
        'hair': {'originalAsset': donor, 'embeddedTextEvidence': item(guides_path), 'embeddedCredit': credit,
                 'publishedLicense': 'CC BY-SA', 'specificVersion': None,
                 'versionFinding': 'Official card and all exported donor Text blocks do not establish a specific CC BY-SA version.',
                 'retainedNamedNodeGroup': [g['name'] for g in guides['nodeGroups'] if 'Bystedt' in g['name']],
                 'nodeMetadataLimit': 'The old node export captures name/type/interface/nodes/links, not asset_data author/license/copyright; current saved-file audit must read those fields.',
                 'actuallySavedFit': fitted, 'fitReport': item(donor_report_path),
                 'actualGeometryAndNodesUsed': donor_report['donor']['actualGeometryAndNodesUsed'],
                 'modificationsInFit': ['Mapped native author growth mesh and guide points to the R11 adult head.',
                                        'Preserved author interpolation/clump/noise/profile node workflow.',
                                        'Adjusted dark hair material; retained original author data.',
                                        'Trimmed six source-buried secondary arcs; applied bounded skin clearance correction.'],
                 'creditDraft': 'Hair Styles by Daniel Bystedt — CC BY-SA (specific version not stated in the supplied notice). Source: https://www.blender.org/download/demo-files/#hair . Adapted to this character’s scalp, guide fit and hair colour. Original author notices and the applicable ShareAlike terms are retained.',
                 'distributionScope': 'Do not label this donor or its derivative hair CC0. Do not assign a version without evidence. The exact-version gap remains recorded for final distribution; this dossier is not a grant or legal clearance.'},
        'humanBaseMeshes': {'asset': head_donor, 'version': '1.4.1 downloaded bundle; embedded README says Version 1.4',
                            'officialSource': DEMO, 'publisher': 'Blender Studio and community contributions',
                            'license': 'CC0', 'embeddedEvidence': item(head_path),
                            'readmeNotice': 'All provided assets are public domain under the CC0 license.',
                            'rainTextScope': 'The separate Rain Rig CC BY 4.0 Text is preserved historical data; it does not override the Human Base Meshes README or imply a Rain rig was adopted.',
                            'notClaimed': ['Scan acquisition', 'Individual female-base author inferred from unrelated object metadata', 'Whole mixed character licensed CC0']},
        'generatedImages': generated,
        'irisApplicationSupplement': {'originalMetadataStatusIsHistorical': True,
                                      'actualSavedSource': iris_source, 'applicationReport': item(iris_path),
                                      'packed': True, 'actualSize': iris['actualSize'], 'colorSpace': iris['actualColorspace'],
                                      'note': 'Original generation metadata said not yet applied. This later saved-file evidence supersedes that status without rewriting the hash-locked generation record.'},
        'inheritedSourcesRecord': item('work/reconstruction-r02/delivery-sources-r07.json'),
        'scope': ['No Blender process, source save, texture edit, packaging, export or shared project-state change.',
                  'Existing source reports were linked and their referenced files hashed; no assertion that all these resources are in the current final visible object set.',
                  'Original project reference, CMU motion and legacy texture unknowns remain in the inherited source record.',
                  'Generated artwork is design/texture material, never actual Blender render, calibrated photo or visual acceptance evidence.',
                  'No final delivery, appearance, motion, platform, backup or rights clearance claim.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': output, 'sha256': sha(out), 'hairLicenseVersion': None,
                      'generatedModelVerified': False, 'irisActuallySaved': iris['candidateSaved']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)

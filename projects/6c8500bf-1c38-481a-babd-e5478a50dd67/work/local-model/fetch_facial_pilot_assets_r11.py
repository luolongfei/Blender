"""Fetch only pinned CC0 facial pilot data into a new isolated directory."""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

EXTRA = '7eaba3453134385bb5ea9811ef0b33b85b4b556d'
MPFB = 'd0a32e57a7f915cb2f2b95410e2117648c7bbb7e'
UNITS = ('browInnerUp', 'browDownLeft', 'eyeBlinkLeft', 'eyeSquintLeft', 'mouthSmileLeft', 'mouthFunnel')


def get(url, limit=2500000):
    req = urllib.request.Request(url, headers={'User-Agent': 'Blender-workspace-CC0-provenance-check'})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Expected only a small asset or metadata file')
    return data


def main():
    output = project_path() / 'assets/mpfb-facial-pilot-r11'
    if output.exists():
        raise FileExistsError('Preserve existing source files: ' + str(output))
    repos = {'makehumancommunity/extra-targets': EXTRA, 'makehumancommunity/mpfb2': MPFB}
    trees = {}
    for repo, commit in repos.items():
        raw = get('https://api.github.com/repos/' + repo + '/git/trees/' + commit + '?recursive=1', 5000000)
        tree = json.loads(raw)
        if tree.get('truncated'):
            raise ValueError('Incomplete repository tree')
        trees[repo] = {item['path']: item for item in tree['tree']}
    entries = [
        ('makehumancommunity/mpfb2', 'src/mpfb/data/3dobjs/base.obj', 'base-hm08.obj', 'neutral-basemesh'),
        ('makehumancommunity/mpfb2', 'LICENSE.md', 'licenses/mpfb-LICENSE.md', 'license-scope'),
        ('makehumancommunity/mpfb2', 'LICENSE.ASSETS.md', 'licenses/mpfb-LICENSE.ASSETS.md', 'CC0-license'),
        ('makehumancommunity/extra-targets', 'LICENSE', 'licenses/extra-targets-LICENSE', 'CC0-license'),
        ('makehumancommunity/extra-targets', 'README.md', 'licenses/extra-targets-README.md', 'source-quality-note')
    ]
    entries += [('makehumancommunity/extra-targets', 'assets/faceunits01/targets/faceunits/' + u + '.target',
                 'targets/' + u + '.target', 'facial-expression-target') for u in UNITS]
    records = []
    downloaded = []
    # Verify every byte before creating the directory; a failure leaves no partial asset set.
    for repo, original, local, kind in entries:
        commit = repos[repo]
        url = 'https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + original
        item = trees[repo][original]
        if item['size'] > 2500000:
            raise ValueError('Unexpected large file')
        data = get(url)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != item['size'] or blob != item['sha']:
            raise ValueError('Source byte count/blob checksum mismatch: ' + original)
        if kind == 'neutral-basemesh' and (b'explicitly released as CC0' not in data[:1000] or b'basemesh hm08' not in data[:1000]):
            raise ValueError('Expected explicit CC0 hm08 header')
        downloaded.append((local, data))
        records.append({'path': local, 'role': kind, 'sourceUrl': url,
                        'sourcePage': 'https://github.com/' + repo + '/blob/' + commit + '/' + original,
                        'repo': repo, 'commit': commit, 'upstreamPath': original,
                        'bytes': len(data), 'gitBlobSha1': blob, 'sha256': hashlib.sha256(data).hexdigest(),
                        'license': 'CC0-1.0' if kind in ('neutral-basemesh', 'facial-expression-target') else 'upstream legal/provenance text'})
    output.mkdir()
    for local, data in downloaded:
        target = output / local
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(data)
    manifest = {
        'schemaVersion': 1, 'retrievedAt': datetime.now(timezone.utc).isoformat(),
        'scope': 'Only six pinned CC0 targets, the exact hm08 base asset, licenses and README; no addon, install, new head candidate or Blender process.',
        'mpfbTag': 'v2.0.18', 'mpfbCommit': MPFB, 'extraTargetsCommit': EXTRA,
        'licenseEvidence': {'neutral': 'base-hm08.obj own header explicitly releases this asset as CC0 in September 2020; MPFB LICENSE.md includes the base mesh in CC0 assets.',
            'targets': 'extra-targets repository LICENSE is CC0-1.0; official functional-packs page also specifically labels Faceunits01 CC0.',
            'officialFunctionalPackUrl': 'https://static.makehumancommunity.org/assets/assetpacks/index.html#functional-asset-packs'},
        'basisCompatibility': {'type': 'canonical MakeHuman hm08 basis',
            'verifiedBlob': 'd26635e9326e3cca30778fd7b9c00062b03cce09',
            'identicalOfficialMakehumanCommit': 'a8bc2d54ff0ac92e78ff71431b1023eda42bf482',
            'lastMpfbBaseChange': 'f60b86a059d057022db20cfa13559417ead46de4 (2021-03-26)',
            'evidence': 'Pinned MPFB v2.0.18 and official MakeHuman base.obj have identical Git blob identities; MPFB documents loading these target packs into its basemesh. Targets must preserve hm08 vertex order.',
            'limit': 'The extra-targets author does not pin a separate personalized donor-neutral shape. This is the canonical base/topology, not evidence of a character-specific expression sculpt or current HMB correspondence.'},
        'coordinateEvidence': {'targetParserUrl': 'https://github.com/makehumancommunity/mpfb2/blob/' + MPFB + '/src/mpfb/services/targetservice.py',
            'conversion': 'MPFB target parser converts original (x,y,z) to Blender (x,-z,y), then multiplies by configured scale_factor. Donor/target must use identical transform; translation never applies to deltas.',
            'unitStatus': 'OBJ is original MakeHuman dm-scale coordinates; exact meter factor is checked separately against pinned human-service code before deriving metric plots.'},
        'files': records, 'totalBytes': sum(x['bytes'] for x in records),
        'allDownloadsVerifiedAgainstPinnedGitBlob': True, 'installed': False,
        'blenderExecuted': False, 'characterModified': False, 'deliveryReady': False
    }
    with (output / 'source-manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({'directory': str(output.relative_to(project_path())), 'files': len(records), 'bytes': manifest['totalBytes'],
                      'baseSha256': records[0]['sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

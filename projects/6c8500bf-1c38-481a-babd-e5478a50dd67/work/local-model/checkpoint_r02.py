"""Record an explicit saved reconstruction source, without promoting it to master."""
import json,hashlib,sys,datetime
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run(source):
    p=project_path();source=p/source
    if not source.is_file():raise FileNotFoundError(source)
    identify=lambda f:{'path':f.relative_to(p).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    candidate=identify(source)
    combined_path=p/'previews/reconstruction-r02/combined-r04/render-report.json'
    combined=json.loads(combined_path.read_text(encoding='utf-8')) if combined_path.is_file() else {}
    state_path=p/'project-state.json';state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    if state.get('masterIntegrationR09'):
        raise RuntimeError('R09 master integration exists; this historical checkpoint must not reset its provenance. Use master_integrate_r09.')
    state['reconstructionR01']['status']='superseded-by-reference-led-r02; preserved historical candidates'
    state['stage']='reference-led-reconstruction-in-progress'
    state['recordedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state['reconstructionR02']={
      'status':'active; user delegated appearance and routine production decisions',
      'integratedIntoMaster':False,'currentSavedCandidate':candidate,
      'masterVerifiedUnchanged':identify(p/'work/character.blend')['sha256']==state['workingMaster']['sha256'],
      'reference':identify(p/'references/user-original-portrait-2026-10-28.png'),
      'visibleReviews':'previews/reconstruction-r02/',
      'implemented':[
        'Reference-led head, dark eyes, cranial groom and native hair strands',
        'Independent blink, jaw/oral cavity and expanded facial controls',
        'R04 neck interface stitching and erroneous neck/arm skinning repairs',
        'R04 eyeballs retracted 3 mm with matching orbital and eyelash adjustments; eye pivots recentered',
        'IK/FK controls and authored diagnostic actions; authored clips remain explicitly distinct from motion capture',
        '301-frame authored full-body cloth simulation baked and converted to an external Blender cache',
        'Matching 301-frame twin-tail offsets embedded as native shape keys and Geometry Nodes, without playback handlers or external hair-cache dependency',
        'CMU 07_01 human motion-capture retarget v4; actual 631-sample foot audit passed its finite probe thresholds'],
      'checks':{
        'blink':'Historical closed-eye and expression reviews exist; new R04 combinations still need final appearance review',
        'ikFk':'Four limb probe reports saved; not all poses certified',
        'cloth':'301-frame authored body cache externalized; frame 229 internal/external vertex comparison was exactly equal. External path present after combined-source reopen; final delivery relocation still unverified',
        'nativeHair':'301-frame authored-body cache evaluated in Blender at frames 1, 61, 101.375, 151, 229 and 301 with zero point error. Saved-source reopen comparison at 1, 101.375, 229 and 301 also had zero error. Supports only its recorded action; selecting another action does not automatically select matching dynamics',
        'combinedReopen':{
          'report':'previews/reconstruction-r02/combined-r04/render-report.json',
          'reportedSourceSha256':combined.get('sha256'),
          'matchesCurrentSavedCandidate':bool(combined.get('sha256')) and combined['sha256']==candidate['sha256'],
          'renderedFrames':[1,61,151,229],
          'scope':'Four sparse rendered frames and sampled coordinate parity; not a continuous-motion, collision, visual-likeness or client-acceptance certificate'},
        'cmuMocap':'v1/v2 had contact failures; v4 actual 0.125-frame audit covered 631 samples with no penetration or sliding flags (2 mm / 0.05 m/s thresholds). Numerical contact does not establish naturalness; head gaze and matching cloth remain in progress',
        'continuousVideo':'R03 facial video and 10.067-second R04 combined body video exist. Both describe historical sources; R06 CMU body-only video has mismatched physics disabled and is diagnostic',
        'source':'R04 work checkpoints are separate from the formal master; migration master and old exports remain preserved'},
      'evidence':{
        'nativeHair':'work/reconstruction-r02/native-hair-body-r04.json',
        'clothBake':'work/reconstruction-r02/cloth-dynamic-r04-report.json',
        'clothExternalComparison':'work/reconstruction-r02/cloth-cache-portability-r04.json',
        'combinedReopen':'previews/reconstruction-r02/combined-r04/render-report.json',
        'cmuFootAudit':'work/reconstruction-r02/cmu-v4-foot-audit-r06-eighth.json',
        'r04BodyVideo':'previews/reconstruction-r02/continuous-body-r04/performance-preview.mp4'},
      'remaining':[
        'Final reference likeness, face/eye/neck appearance and full deformation review on one integrated source',
        'CMU foot-contact correction and actual motion-capture deformation review',
        'Current-source combined expressions and continuous video',
        'Hair/cloth collision and motion review for every delivered action; existing native tail cache covers the authored body diagnostic only',
        'Final source integration, export correspondence, relocated cache validation and delivery package'],
      'acceptance':{'userAppearanceApproval':False,'photorealFinalReady':False},
      'oldExports':'Still correspond to migration master; not proof of reconstructed character validation',
      'backup':'Working checkpoints saved locally; no new off-machine backup asserted'}
    if any(stage in source.name for stage in ('r05','r06')):
        state['reconstructionR02']['latestGeometryPass']={
          'stage':'R05 reference proportion and aperture refinement; visual checks in progress',
          'changes':['Head and all facial/eye/oral/hair attachments scaled coherently by 1.10 and lowered 15 mm',
                     'Neck transition restricted to central axis; arm vertices excluded',
                     'Rounder neutral eye aperture, cancelling at closed blink; combined blink/squint clamped for lids and lashes'],
          'hairCache':'The previous R04 native tail playback was restored to authored raw curves before geometry editing. It is NOT active on R05; changed groom and attachment require a new bake.',
          'previousReports':'R04 dynamic and reopen reports describe their exact historical sources only; they do not certify this R05 geometry.',
          'pending':['Face/front/side/full-body/closed-eye renders','New tail dynamics after groom changes','Combined deformation and cache relocation review']}
    if 'r06' in source.name:
        state['reconstructionR02']['latestGeometryPass'].update({
          'stage':'R06 scalp/root integration and motion/physics validation in progress',
          'newGeometry':['R05 tail volume/length and fringe coverage refined with original tail pins preserved',
                         'R06 attached scalp root substrate, 6720 scalp fibres and 2480 gathered root fibres per side',
                         'Narrow transparent root edge; original scalp layers preserved hidden'],
          'visualEvidence':['previews/reconstruction-r02/scalp-r06/','previews/reconstruction-r02/root-feather-r06/',
                            'previews/reconstruction-r02/expressions-r05/'],
          'expressionReview':'Closed, half-blink, wink/smile, open mouth and gaze up/down rendered; no obvious globe breakthrough or corner tearing in these six frames. Not exhaustive combination certification.',
          'pending':['New tail dynamics after groom/attachment changes','Walking cloth and upright head-gaze adjustment',
                     'Independent realtime adaptation, actual viewer review and final source/cache integration']})
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    registered=p/'.asset-code-paths.json';paths=json.loads(registered.read_text(encoding='utf-8-sig'))
    paths['paths']=sorted(set(paths['paths'])|{f.relative_to(p).as_posix() for f in (p/'work/local-model').glob('*') if f.suffix in {'.py','.js','.mjs'}})
    registered.write_text(json.dumps(paths,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(state['reconstructionR02']['currentSavedCandidate']))

if __name__=='__main__':run(sys.argv[1])

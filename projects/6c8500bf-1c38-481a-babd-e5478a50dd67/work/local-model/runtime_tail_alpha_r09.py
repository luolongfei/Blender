"""Read actual embedded tail alpha for CPU intersection visibility filtering."""
import hashlib,io,json,runpy,sys
from pathlib import Path
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run():
    p=project_path();source=p/'work/runtime-r06/preview-tail-colliders-r09/review-preview.vrm'
    api=runpy.run_path(str(Path(__file__).with_name('runtime_binary_audit_r09.py')));d,b=api['read'](source)
    folder=p/'work/runtime-r06/r09-tail-alpha-audit';folder.mkdir(exist_ok=False)
    images={};meshes={}
    for node in d['nodes']:
        if not node.get('name','').startswith('RT06 Cards / R02 Strands / R02 Twin tail '):continue
        primitive=d['meshes'][node['mesh']]['primitives'][0];mat=d['materials'][primitive['material']]
        tex=mat['pbrMetallicRoughness']['baseColorTexture'];image_id=d['textures'][tex['index']]['source']
        if tex.get('texCoord',0)!=0 or tex.get('extensions'):raise RuntimeError('Explicit UV adaptation required')
        if image_id not in images:
            desc=d['images'][image_id];view=d['bufferViews'][desc['bufferView']];start=view.get('byteOffset',0)
            alpha=Image.open(io.BytesIO(b[start:start+view['byteLength']])).convert('RGBA').getchannel('A')
            raw=alpha.tobytes();name=f'alpha-{image_id}.bin';(folder/name).write_bytes(raw)
            images[image_id]={'file':name,'width':alpha.width,'height':alpha.height,'sha256':hashlib.sha256(raw).hexdigest()}
        meshes[node['name']]={'image':image_id,'alphaFactor':mat['pbrMetallicRoughness'].get('baseColorFactor',[1,1,1,1])[3]}
    report={'source':str(source),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'images':images,'meshes':meshes,
        'scope':'Exact decoded RGBA alpha bytes; glTF texture UV origin is top-left, no image modification.'}
    (folder/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(str(folder/'manifest.json'))
if __name__=='__main__':run()

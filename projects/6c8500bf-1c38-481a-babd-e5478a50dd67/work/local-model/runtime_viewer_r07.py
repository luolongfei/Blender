"""Loopback-only actual VRM/PBR review using the locked Three/VRM libraries."""
import argparse, hashlib, json, mimetypes, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--port',type=int,default=8768)
    args=parser.parse_args();project=project_path();model=(project/args.model).resolve()
    model.relative_to(project.resolve())
    if model.suffix.lower()!='.vrm' or not model.is_file():raise ValueError('Existing project VRM required')
    lock=json.loads((ROOT/'toolchain.lock.json').read_text(encoding='utf-8-sig'))
    vendor=(ROOT/lock['charakuru']['path']/'public/vendor').resolve()
    metadata=json.dumps({'file':model.relative_to(project.resolve()).as_posix(),'sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'bytes':model.stat().st_size}).encode()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request=unquote(urlsplit(self.path).path)
            if request=='/metadata.json':data=metadata;kind='application/json'
            else:
                if request=='/':file=Path(__file__).with_name('runtime_viewer_r07.html')
                elif request=='/model.vrm':file=model
                elif request.startswith('/vendor/'):
                    file=(vendor/request[len('/vendor/'):]).resolve()
                    if not file.is_relative_to(vendor) or file.suffix not in {'.js','.mjs'}:self.send_error(404);return
                else:self.send_error(404);return
                if not file.is_file():self.send_error(404);return
                data=file.read_bytes();kind='text/javascript' if file.suffix in {'.js','.mjs'} else mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    print('Actual VRM review:',model,flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__=='__main__':main()

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {pipeline} from 'node:stream/promises';
import {workspacePath,toolPath} from '../../pipeline/context.mjs';
const base=fs.realpathSync(workspacePath('Character_Rebuild'));
const vendor=fs.realpathSync(toolPath('public','vendor'));
const portArgument=process.argv.indexOf('--port');
const requestedPort=portArgument<0?0:Number(process.argv[portArgument+1]);
if(!Number.isInteger(requestedPort)||requestedPort<0||requestedPort>65535)throw Error('Invalid port');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.png':'image/png','.json':'application/json; charset=utf-8','.txt':'text/plain; charset=utf-8','.glb':'model/gltf-binary','.blend':'application/octet-stream','.zip':'application/zip'};
const server=http.createServer(async(req,res)=>{
  try {
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405).end();return;}
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const root=pathname.startsWith('/vendor/')?vendor:base;
    const relativeURL=root===vendor?pathname.slice('/vendor'.length):(pathname==='/'?'/interactive/viewer.html':pathname);
    const requested=path.resolve(root,'.'+relativeURL);
    const actual=await fs.promises.realpath(requested);
    const relative=path.relative(root,actual);
    if(relative.startsWith('..')||path.isAbsolute(relative))throw Error('Outside delivery');
    const mime=types[path.extname(actual).toLowerCase()];
    if(!mime)throw Error('Unsupported file');
    const stat=await fs.promises.stat(actual);if(!stat.isFile())throw Error('Not a file');
    res.writeHead(200,{'Content-Type':mime,'Content-Length':stat.size,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    if(req.method==='HEAD'){res.end();return;}
    await pipeline(fs.createReadStream(actual),res);
  }catch(error){if(!res.headersSent)res.writeHead(error.code==='ENOENT'?404:403);res.end();}
});
server.listen(requestedPort,'127.0.0.1',()=>{
  const state={url:`http://127.0.0.1:${server.address().port}/interactive/viewer.html`,galleryUrl:`http://127.0.0.1:${server.address().port}/review.html`,pid:process.pid,localOnly:true,startedAt:new Date().toISOString()};
  fs.writeFileSync(path.join(base,'reports/preview-server.json'),JSON.stringify(state,null,2));
  console.log(JSON.stringify(state));
});

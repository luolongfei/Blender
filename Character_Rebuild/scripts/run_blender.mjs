import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {workspaceRoot,workspacePath} from '../../pipeline/context.mjs';
const local={...JSON.parse(fs.readFileSync(workspacePath('config/local.json'),'utf8')),...JSON.parse(fs.readFileSync(workspacePath(`config/local.${process.platform}.json`),'utf8'))};
const script=workspacePath('Character_Rebuild/scripts',process.argv[2]);
const child=spawn(local.blenderExecutable,['--background','--threads','8','--python-exit-code','1','--python',script,...(process.argv.length>3?['--',...process.argv.slice(3)]:[])],{cwd:workspaceRoot,stdio:'inherit',windowsHide:true});
child.on('exit',code=>process.exitCode=code??1);

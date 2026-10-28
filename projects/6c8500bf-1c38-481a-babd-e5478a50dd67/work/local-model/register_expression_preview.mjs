import {toolPath} from '../../../../pipeline/context.mjs';
// Register the independently authored model as an expression preview using the
// app's own import function. This does not advance the Tripo/production workflow.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const projectDir=path.resolve(here,'../..');
const load=rel=>import(pathToFileURL(toolPath(rel)).href);
const {unpackProject,packProject}=await load('public/project.mjs');
const {importExpressionModel}=await load('public/expression-editor-data.mjs');
const {writeProject,digest}=await load('scripts/project-write.mjs');
const canonical=path.join(projectDir,'project.charakit');
const source=path.join(projectDir,'exports/expression-review/avatar-preview.vrm');
const before=await fs.readFile(canonical);
const model=await fs.readFile(source);
const checkpoint=path.join(projectDir,'work/checkpoints/expression-redesign-start/project.charakit');
await fs.copyFile(canonical,checkpoint,fs.constants.COPYFILE_EXCL).catch(error=>{
  if(error.code!=='EEXIST')throw error;
});
const {project,files}=await unpackProject(before);
const originalWorkflow=JSON.stringify({workflow:project.workflow,production:project.production,springSetup:project.springSetup});
const preview=await importExpressionModel(project,files,model,'avatar-expressions-review.vrm');
if(originalWorkflow!==JSON.stringify({workflow:project.workflow,production:project.production,springSetup:project.springSetup}))
  throw Error('Expression preview must not advance production stages.');
const packed=await packProject(project,files);
await writeProject(canonical,new Uint8Array(await packed.blob.arrayBuffer()),{input:canonical,expected:digest(before)});
const report={project:canonical,preview,modelSha256:digest(model),projectBefore:digest(before),
  projectAfter:digest(await fs.readFile(canonical)),productionStagesUnchanged:true,userReview:'pending'};
await fs.writeFile(path.join(projectDir,'work/expression-authoring/project-preview-registration.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));

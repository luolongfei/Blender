"""Strict CMU ASF/AMC source parsing and forward kinematics, using NumPy only.

No bpy import, Blender process, retarget, scene mutation, or automatic file write.
Column-vector convention: C for XYZ is Rz @ Ry @ Rx; each local motion is
C @ Mdof @ C.T, and global motion is parent_global @ local_motion.
The format reference uses row vectors, hence its apparently reversed notation:
https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
CMU units: https://mocap.cs.cmu.edu/info.php
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SOURCE_TO_BLENDER=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
FORMAT_REFERENCE='https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html'
UNITS_REFERENCE='https://mocap.cs.cmu.edu/info.php'


def _lines(path):
    for index,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        text=line.split('#',1)[0].strip()
        if text: yield index,text


def _numbers(tokens,count,context):
    if len(tokens)!=count: raise ValueError(f'{context}: expected {count} numbers, got {len(tokens)}')
    try: values=np.array([float(v) for v in tokens],dtype=np.float64)
    except ValueError as exc: raise ValueError(context+': invalid numeric value') from exc
    if not np.all(np.isfinite(values)): raise ValueError(context+': non-finite numeric value')
    return values


def _axis_matrix(axis,angle):
    c,s=math.cos(angle),math.sin(angle)
    if axis=='x': return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=np.float64)
    if axis=='y': return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float64)
    if axis=='z': return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)
    raise ValueError('Unknown rotation axis: '+axis)


def _axis_rotation(angles_xyz,order):
    """Apply listed axis rotations in sequence to column vectors."""
    if sorted(order.lower())!=['x','y','z']: raise ValueError('Axis order must be a permutation of XYZ')
    result=np.eye(3)
    for axis in order.lower(): result=_axis_matrix(axis,float(angles_xyz['xyz'.index(axis)]))@result
    return result


def _dof_rotation(channels,values):
    result=np.eye(3)
    for channel,value in zip(channels,values):
        if channel not in {'rx','ry','rz'}: raise ValueError('Unsupported animated non-rotation bone DOF: '+channel)
        result=_axis_matrix(channel[1],float(value))@result
    return result


def parse_asf(path):
    """Parse all motion-bearing ASF fields and validate a single rooted tree."""
    sections={};section=None
    for lineno,text in _lines(path):
        if text.startswith(':'):
            token,*tail=text[1:].split(maxsplit=1);section=token.lower()
            if section in sections: raise ValueError(f'ASF line {lineno}: repeated section {section}')
            sections[section]=[]
            if tail: sections[section].append((lineno,tail[0]))
        elif section is None: raise ValueError(f'ASF line {lineno}: text outside a section')
        else: sections[section].append((lineno,text))
    required={'version','units','root','bonedata','hierarchy'}
    if not required.issubset(sections): raise ValueError('ASF required sections missing: '+str(sorted(required-set(sections))))
    version=sections['version'][0][1]
    if version!='1.10': raise ValueError('This parser is scoped to Acclaim ASF 1.10')
    units={}
    for lineno,text in sections['units']:
        key,*values=text.split();key=key.lower()
        if key in units or len(values)!=1: raise ValueError(f'ASF line {lineno}: malformed unit')
        units[key]=values[0] if key=='angle' else float(_numbers(values,1,'unit '+key)[0])
    if units.get('angle','').lower() not in {'deg','degrees','rad','radians'} or units.get('length',0)<=0:
        raise ValueError('ASF must declare supported angle units and a positive length scale')
    to_radians=math.pi/180 if units['angle'].lower() in {'deg','degrees'} else 1.0
    root={}
    for lineno,text in sections['root']:
        key,*values=text.split();key=key.lower()
        if key in root: raise ValueError(f'ASF line {lineno}: repeated root field')
        if key=='order': root[key]=[v.lower() for v in values]
        elif key=='axis':
            if len(values)!=1 or sorted(values[0].upper())!=list('XYZ'): raise ValueError('Malformed root axis')
            root[key]=values[0].upper()
        elif key in {'position','orientation'}: root[key]=_numbers(values,3,'root '+key)
        else: raise ValueError('Unsupported root field: '+key)
    if set(root)!={'order','axis','position','orientation'}: raise ValueError('Incomplete root description')
    if len(root['order'])!=6 or set(root['order'])!={'tx','ty','tz','rx','ry','rz'}:
        raise ValueError('CMU root must contain exactly TX TY TZ RX RY RZ channels')
    root['orientation']=root['orientation']*to_radians
    root['axis_matrix']=_axis_rotation(root['orientation'],root['axis'])
    bones={};bone_ids=set();current=None
    for lineno,text in sections['bonedata']:
        if text.lower()=='begin':
            if current is not None: raise ValueError(f'ASF line {lineno}: nested bone')
            current={'dof':[],'limits':[]};continue
        if text.lower()=='end':
            if current is None: raise ValueError(f'ASF line {lineno}: unmatched end')
            mandatory={'id','name','direction','length','axis','axis_order','dof','limits'}
            if not mandatory.issubset(current): raise ValueError('Incomplete bone description: '+str(current.get('name')))
            name=current['name']
            if name=='root' or name in bones or current['id'] in bone_ids: raise ValueError('Duplicate bone name or ID')
            if len(current['limits']) not in (0,len(current['dof'])): raise ValueError('DOF limit count mismatch: '+name)
            if current['dof'] and not current['limits']: raise ValueError('Missing DOF limits: '+name)
            norm=float(np.linalg.norm(current['direction']))
            if abs(norm-1)>0.001: raise ValueError(f'Bone direction is not approximately unit length: {name} ({norm})')
            current['raw_direction']=current['direction'].copy()
            current['direction']=current['direction']/norm
            current['direction_normalization_error']=abs(norm-1)
            current['axis_matrix']=_axis_rotation(current['axis']*to_radians,current['axis_order'])
            current['limits']=np.array(current['limits'],dtype=float).reshape(-1,2)*to_radians
            bones[name]=current;bone_ids.add(current['id']);current=None;continue
        if current is None: raise ValueError(f'ASF line {lineno}: bone data outside begin/end')
        key,*values=text.split();key=key.lower()
        if key=='limits' or text.startswith('('):
            pairs=re.findall(r'\(([^()]*)\)',text[len('limits'):] if key=='limits' else text)
            if not pairs: raise ValueError(f'ASF line {lineno}: malformed limits')
            for pair in pairs:
                bounds=_numbers(pair.split(),2,'DOF limit')
                if bounds[0]>bounds[1]: raise ValueError('Reversed DOF limits')
                current['limits'].append(bounds)
            continue
        if key in current and key not in {'dof'}: raise ValueError('Repeated bone field: '+key)
        if key=='id':
            if len(values)!=1 or not values[0].isdigit(): raise ValueError('Malformed bone ID')
            current[key]=int(values[0])
        elif key=='name':
            if len(values)!=1: raise ValueError('Malformed bone name')
            current[key]=values[0]
        elif key=='direction': current[key]=_numbers(values,3,'bone direction')
        elif key=='length':
            current[key]=float(_numbers(values,1,'bone length')[0])
            if current[key]<=0: raise ValueError('Nonpositive bone length')
        elif key=='axis':
            if len(values)!=4 or sorted(values[3].upper())!=list('XYZ'): raise ValueError('Malformed bone axis')
            current[key]=_numbers(values[:3],3,'bone axis');current['axis_order']=values[3].upper()
        elif key=='dof':
            if current['dof'] or not values or len(set(values))!=len(values) or not set(values)<={'rx','ry','rz'}:
                raise ValueError('Invalid/unsupported bone DOF declaration')
            current[key]=values
        else: raise ValueError('Unsupported ASF bone field: '+key)
    if current is not None: raise ValueError('Unclosed bone block')
    if not bones: raise ValueError('No skeleton bones')
    parents={'root':None};children={n:[] for n in ['root',*bones]};inside=False;closed=False
    for lineno,text in sections['hierarchy']:
        if text.lower()=='begin':
            if inside or closed: raise ValueError('Invalid hierarchy begin')
            inside=True;continue
        if text.lower()=='end':
            if not inside: raise ValueError('Invalid hierarchy end')
            inside=False;closed=True;continue
        if not inside: raise ValueError('Hierarchy data outside begin/end')
        parent,*kids=text.split()
        if parent not in children or not kids: raise ValueError('Invalid hierarchy parent/children')
        for child in kids:
            if child not in bones or child in parents: raise ValueError('Unknown or multiply parented child: '+child)
            parents[child]=parent;children[parent].append(child)
    if inside or not closed or set(parents)!={'root',*bones}: raise ValueError('Incomplete hierarchy')
    names=[];visiting=set()
    def visit(name):
        if name in visiting: raise ValueError('Cyclic ASF hierarchy')
        visiting.add(name);names.append(name)
        for child in children[name]: visit(child)
        visiting.remove(name)
    visit('root')
    if len(names)!=len(bones)+1 or len(set(names))!=len(names): raise ValueError('Disconnected/cyclic hierarchy')
    return {'version':version,'units':units,'root':root,'bones':bones,'parents':parents,'children':children,'names':names}


def parse_amc(path,skeleton):
    frames=[];samples=[];current=None;frame=None;headers=set()
    channels={'root':skeleton['root']['order'],**{n:b['dof'] for n,b in skeleton['bones'].items()}}
    expected={n for n,c in channels.items() if c}
    def finish():
        if current is None:return
        if not expected.issubset(current): raise ValueError(f'AMC frame {frame}: missing channels {sorted(expected-set(current))}')
        frames.append(frame);samples.append(current)
    for lineno,text in _lines(path):
        if text.startswith(':'):
            if frames or current is not None: raise ValueError('AMC header inside motion frames')
            headers.add(text[1:].upper());continue
        if text.isdigit():
            finish();frame=int(text)
            if frame<1 or (frames and frame!=frames[-1]+1): raise ValueError('AMC frame IDs must be positive and contiguous')
            current={};continue
        if current is None: raise ValueError(f'AMC line {lineno}: channels before a frame number')
        name,*tokens=text.split()
        if name not in channels or name in current: raise ValueError(f'AMC line {lineno}: unknown or repeated joint {name}')
        current[name]=_numbers(tokens,len(channels[name]),f'AMC {frame} {name}')
    finish()
    if not {'FULLY-SPECIFIED','DEGREES'}.issubset(headers) or headers-{'FULLY-SPECIFIED','DEGREES'}:
        raise ValueError('This CMU parser requires FULLY-SPECIFIED, DEGREES AMC data')
    if len(frames)<2: raise ValueError('At least two AMC frames required')
    values={n:np.array([sample.get(n,[]) for sample in samples],dtype=np.float64).reshape(len(frames),len(c))
            for n,c in channels.items()}
    return np.array(frames,dtype=np.int32),values


def _source_record(asf,amc):
    record_path=asf.parent/'source-record.json'
    record=json.loads(record_path.read_text(encoding='utf-8')) if record_path.is_file() else None
    files=[]
    for path in (asf,amc):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        match=next((r for r in (record or {}).get('files',[]) if r.get('path')==path.name),None)
        if match and match.get('sha256')!=digest: raise ValueError('Source-record hash mismatch: '+path.name)
        files.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest,
                      'sourceUrl':match.get('sourceUrl') if match else None,'matchesSourceRecord':bool(match)})
    return files,record


def load_clip(asf_path=None,amc_path=None,source_fps=120):
    """Return the agreed endpoint/global-motion arrays in Blender meters.

    positions[:,j] is this named bone's ENDPOINT (root is its position).
    Its start is positions[:,index[parents[name]]]. rotations[:,j] is the
    GLOBAL MOTION rotation, identity with zero channels; it is not a bone frame.
    fixed no-channel bones are retained as real offsets. Source origin and
    ground heights are preserved, without a hidden recentering or retarget.
    """
    base=project_path()/'assets/mocap-cmu/07'
    asf=Path(asf_path).resolve() if asf_path is not None else base/'07.asf'
    amc=Path(amc_path).resolve() if amc_path is not None else base/'07_01.amc'
    if not math.isfinite(source_fps) or source_fps<=0: raise ValueError('Positive source_fps required; AMC itself does not encode fps')
    skeleton=parse_asf(asf);frames,channels=parse_amc(amc,skeleton)
    files,record=_source_record(asf,amc)
    if all(f['matchesSourceRecord'] for f in files) and record:
        recorded_fps=record.get('format',{}).get('frameRate')
        if recorded_fps and abs(source_fps-float(recorded_fps))>1e-9:
            raise ValueError('source_fps conflicts with the verified source record')
    scale=0.0254/skeleton['units']['length']
    names=skeleton['names'];index={n:i for i,n in enumerate(names)};F,J=len(frames),len(names)
    positions=np.zeros((F,J,3),dtype=np.float64);rotations=np.tile(np.eye(3),(F,J,1,1))
    rest=np.zeros((J,3),dtype=np.float64);directions=np.zeros((J,3),dtype=np.float64)
    rest[0]=skeleton['root']['position'];lengths={'root':0.0}
    root=skeleton['root'];root_values={channel:channels['root'][:,i] for i,channel in enumerate(root['order'])}
    positions[:,0]=root['position']+np.column_stack([root_values['t'+a] for a in 'xyz'])
    for f in range(F):
        angles=np.array([root_values['r'+a][f] for a in 'xyz'])*math.pi/180
        rotations[f,0]=root['axis_matrix']@_axis_rotation(angles,root['axis'])@root['axis_matrix'].T
    violations={}
    for name in names[1:]:
        j=index[name];parent=index[skeleton['parents'][name]];bone=skeleton['bones'][name]
        directions[j]=bone['direction'];offset=bone['direction']*bone['length']
        rest[j]=rest[parent]+offset;lengths[name]=bone['length']*scale
        values=channels[name]*math.pi/180;C=bone['axis_matrix']
        for f in range(F):
            local=C@_dof_rotation(bone['dof'],values[f])@C.T
            rotations[f,j]=rotations[f,parent]@local
            positions[f,j]=positions[f,parent]+rotations[f,j]@offset
        if bone['dof']:
            outside=(values<bone['limits'][:,0]-math.radians(.1))|(values>bone['limits'][:,1]+math.radians(.1))
            if np.any(outside): violations[name]={ch:int(outside[:,k].sum()) for k,ch in enumerate(bone['dof']) if outside[:,k].any()}
    B=SOURCE_TO_BLENDER
    positions=np.einsum('ij,fkj->fki',B,positions)*scale
    rotations=np.einsum('ab,fjbc,dc->fjad',B,rotations,B)
    rest=(B@rest.T).T*scale;directions=(B@directions.T).T
    source={'files':files,'sourceFps':float(source_fps),'sourceFrameRange':[int(frames[0]),int(frames[-1])],
            'sourceSampleCount':F,'durationSeconds':float((frames[-1]-frames[0])/source_fps),
            'metersPerUnit':scale,'meterScaleFormula':'0.0254 / ASF units.length',
            'coordinateMap':'(x,y,z) -> (x,-z,y)','formatReference':FORMAT_REFERENCE,'unitsReference':UNITS_REFERENCE,
            'rotationConvention':'column vectors; axis XYZ = Rz @ Ry @ Rx; local = C @ Mdof @ C.T; global = parent @ local',
            'positionsConvention':'named-bone endpoints; root is root translation; origin unmodified',
            'rotationsConvention':'global motion rotation, rest identity; not bone-direction frames',
            'rawChannelUnits':'Root translations are file length units; rotation channels are degrees',
            'fixedBonesRetained':[n for n in names[1:] if not skeleton['bones'][n]['dof']],
            'maxDirectionNormalizationError':max(b['direction_normalization_error'] for b in skeleton['bones'].values()),
            'sourceLimitViolationsOver0_1Degree':violations,'limitsWereClamped':False,
            'retargeted':False,'blenderOpened':False,'animationAccepted':False,
            'sourceLimitations':(record or {}).get('skeleton',{}).get('sourceLimitations',[])}
    clip={'names':names,'parents':skeleton['parents'].copy(),'frames':frames,
          'times':(frames.astype(np.float64)-frames[0])/source_fps,'positions':positions,'rotations':rotations,
          'rest_positions':rest,'rest_directions':directions,'lengths':lengths,'source':source,
          'skeleton':skeleton,'raw_channels':channels}
    clip['checks']=check_clip(clip)
    if (not clip['checks']['finite'] or clip['checks']['maxBoneLengthErrorMeters']>1e-10
            or clip['checks']['maxRotationOrthogonalityError']>1e-10
            or max(abs(v-1) for v in clip['checks']['rotationDeterminantRange'])>1e-10):
        raise RuntimeError('Source forward-kinematics invariants failed')
    return clip


def check_clip(clip):
    names=clip['names'];index={n:i for i,n in enumerate(names)}
    p=clip['positions'];r=clip['rotations'];length_error=0.;endpoint_error=0.
    for name in names[1:]:
        j=index[name];parent=index[clip['parents'][name]];segment=p[:,j]-p[:,parent]
        length_error=max(length_error,float(np.max(np.abs(np.linalg.norm(segment,axis=1)-clip['lengths'][name]))))
        implied=np.einsum('fij,j->fi',r[:,j],clip['rest_directions'][j])*clip['lengths'][name]
        endpoint_error=max(endpoint_error,float(np.max(np.linalg.norm(implied-segment,axis=1))))
    product=np.swapaxes(r,-1,-2)@r
    determinant=np.linalg.det(r);feet={}
    for name in ('ltibia','rtibia','lfoot','rfoot','ltoes','rtoes'):
        if name in index:
            height=p[:,index[name],2]
            feet[name]={'minWorldHeightMeters':float(height.min()),'maxWorldHeightMeters':float(height.max()),
                        'samplesWithin20mmOfSourceZ0':int((np.abs(height)<=.02).sum()),
                        'samplesBelowSourceZ0By20mm':int((height<-.02).sum())}
    root_delta=p[-1,0]-p[0,0];duration=float(clip['times'][-1]-clip['times'][0])
    return {'sampleCount':len(p),'jointCount':len(names),'finite':bool(np.isfinite(p).all() and np.isfinite(r).all()),
            'durationSeconds':duration,'maxBoneLengthErrorMeters':length_error,'maxEndpointRotationConsistencyErrorMeters':endpoint_error,
            'maxRotationOrthogonalityError':float(np.max(np.abs(product-np.eye(3)))),
            'rotationDeterminantRange':[float(determinant.min()),float(determinant.max())],
            'rootDisplacementMeters':root_delta.tolist(),'rootAverageTravelSpeedMetersPerSecond':float(np.linalg.norm(root_delta)/duration),
            'footEndpointHeights':feet,
            'groundInterpretation':'Source Z=0 is a diagnostic reference only; no floor fitting/recentering applied, and ASF endpoints are not shoe contact surfaces.'}


def _matrix_quaternions(matrices):
    shape=matrices.shape[:-2];flat=matrices.reshape(-1,3,3);out=np.empty((len(flat),4))
    for i,m in enumerate(flat):
        trace=np.trace(m)
        if trace>0:
            s=math.sqrt(trace+1)*2;out[i]=[.25*s,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s]
        else:
            axis=int(np.argmax(np.diag(m)))
            if axis==0:
                s=math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2;out[i]=[(m[2,1]-m[1,2])/s,.25*s,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s]
            elif axis==1:
                s=math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2;out[i]=[(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,.25*s,(m[1,2]+m[2,1])/s]
            else:
                s=math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2;out[i]=[(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,.25*s]
    out/=np.linalg.norm(out,axis=1,keepdims=True)
    return out.reshape(*shape,4)


def _quaternion_matrices(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    r=np.empty((*q.shape[:-1],3,3))
    r[...,0,0]=1-2*(y*y+z*z);r[...,0,1]=2*(x*y-z*w);r[...,0,2]=2*(x*z+y*w)
    r[...,1,0]=2*(x*y+z*w);r[...,1,1]=1-2*(x*x+z*z);r[...,1,2]=2*(y*z-x*w)
    r[...,2,0]=2*(x*z-y*w);r[...,2,1]=2*(y*z+x*w);r[...,2,2]=1-2*(x*x+y*y)
    return r


def sample_clip(clip,target_fps=30,include_endpoint=True):
    """Resample positions linearly and global rotations by shortest-arc SLERP.

    Use returned times/target_frames for keyframes. For 2.625 s at 30 Hz the
    exact last sample is target frame 79.75, not an integer frame. frames is
    only a sample counter in this returned object. include_endpoint=False
    returns only the regular grid (last time 2.6 s for this source).
    Linear endpoint interpolation at non-source sample times can slightly change
    segment lengths; check_clip reports that effect rather than hiding it.
    """
    if not math.isfinite(target_fps) or target_fps<=0: raise ValueError('Positive target_fps required')
    source_times=clip['times'];start,end=float(source_times[0]),float(source_times[-1])
    count=math.floor((end-start)*target_fps+1e-10)+1
    times=start+np.arange(count)/target_fps
    if include_endpoint and end-times[-1]>1e-10: times=np.append(times,end)
    elif abs(end-times[-1])<=1e-10: times[-1]=end
    hi=np.clip(np.searchsorted(source_times,times,side='right'),1,len(source_times)-1);lo=hi-1
    weight=(times-source_times[lo])/(source_times[hi]-source_times[lo]);weight=np.clip(weight,0,1)
    p=clip['positions'];positions=p[lo]*(1-weight[:,None,None])+p[hi]*weight[:,None,None]
    q0=_matrix_quaternions(clip['rotations'][lo]);q1=_matrix_quaternions(clip['rotations'][hi])
    dot=np.sum(q0*q1,axis=-1,keepdims=True);q1=np.where(dot<0,-q1,q1);dot=np.clip(np.abs(dot),0,1)
    angle=np.arccos(dot);den=np.maximum(np.sin(angle),1e-12);t=weight[:,None,None]
    spherical=q0*np.sin((1-t)*angle)/den+q1*np.sin(t*angle)/den
    linear=q0*(1-t)+q1*t
    quaternion=np.where(dot>.9995,linear,spherical)
    rotations=_quaternion_matrices(quaternion)
    result=dict(clip)
    result.update(frames=np.arange(1,len(times)+1,dtype=np.int32),times=times,positions=positions,rotations=rotations,
                  target_frames=1+(times-start)*target_fps,
                  source_frame_numbers=np.interp(times,source_times,clip['frames']))
    result['source']=dict(clip['source'],resampling={'targetFps':float(target_fps),'sampleCount':len(times),
        'includeEndpoint':include_endpoint,'finalTargetFrame':float(result['target_frames'][-1]),
        'durationSeconds':float(times[-1]-times[0]),'positionMethod':'linear endpoint interpolation',
        'rotationMethod':'normalized quaternion shortest-arc SLERP','framesMeaning':'sample counter; use times or target_frames for timing'})
    result['checks']=check_clip(result)
    return result


if __name__=='__main__':
    print('Parser loaded only. Call load_clip(), then sample_clip(); no Blender or file mutation occurs.')

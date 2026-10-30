"""Reference-led mass/length correction of the existing native R05 groom.

Run by the sole integrator after restoring old native/PBD playback and applying
the coherent head transform. No saves, renders, background processes or cache
bakes are performed. Existing native curve topology and the first three tail
points are retained exactly. Mesh hair ties are refitted to the measured roots.
Data copies are retained for restore(); dynamics must be regenerated afterwards.
"""
import json
import math
import bpy
import numpy as np

TAILS = ('R02 Strands / R02 Twin tail L', 'R02 Strands / R02 Twin tail R')
BANGS = 'R03 Forelocks from crown'
MARKER = 'R05_reference_groom'
BACKUP = 'R05_reference_groom_backup'


def _smooth(value):
    t = np.clip(value, 0., 1.)
    return t*t*(3.-2.*t)


def _normal(value):
    return value/max(float(np.linalg.norm(value)), 1e-12)


def _to_world(local, matrix):
    return local @ matrix[:3,:3].T+matrix[:3,3]


def _to_local(world, matrix):
    inv = np.linalg.inv(matrix)
    return world @ inv[:3,:3].T+inv[:3,3]


def _read(obj):
    points = obj.data.position_data if obj.type == 'CURVES' else obj.data.vertices
    prop = 'vector' if obj.type == 'CURVES' else 'co'
    local = np.empty(len(points)*3, dtype=np.float64)
    points.foreach_get(prop, local)
    local = local.reshape(-1, 3)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    result = {'object':obj, 'local':local, 'world':_to_world(local,matrix), 'matrix':matrix}
    if obj.type == 'CURVES':
        result['sizes'] = np.array([c.points_length for c in obj.data.curves],dtype=np.int32)
        result['starts'] = np.array([c.first_point_index for c in obj.data.curves],dtype=np.int32)
        radius = np.empty(len(points),dtype=np.float64)
        obj.data.attributes['radius'].data.foreach_get('value',radius)
        result['radius'] = radius
    return result


def _sample(path, t):
    u = np.clip(t,0,1)*(len(path)-1)
    i = np.minimum(u.astype(int),len(path)-2)
    f = u-i
    return path[i]*(1-f[:,None])+path[i+1]*f[:,None]


def _check(record):
    new, old = record['new_world'], record['world']
    if not np.all(np.isfinite(new)):
        raise RuntimeError('Non-finite groom: '+record['object'].name)
    if record['object'].type != 'CURVES':
        return
    maximum = 0.0
    worst = None
    for i, (start,count) in enumerate(zip(record['starts'],record['sizes'])):
        oldstep = np.linalg.norm(np.diff(old[start:start+count],axis=0),axis=1)
        newstep = np.linalg.norm(np.diff(new[start:start+count],axis=0),axis=1)
        segment = int(np.argmax(newstep))
        if float(newstep[segment]) > maximum:
            maximum = float(newstep[segment]); worst = (i,segment)
        if newstep[segment] > .020:
            diagnostic = {'object':record['object'].name,'strand':i,'segment':segment,
                          'oldStepMm':float(oldstep[segment]*1000),'newStepMm':float(newstep[segment]*1000),
                          'oldWorldPoints':old[start+segment:start+segment+2].tolist(),
                          'newWorldPoints':new[start+segment:start+segment+2].tolist(),
                          'writesPerformed':False}
            raise RuntimeError('R05 continuity guard: '+json.dumps(diagnostic))
    record['maximum_step_mm'] = maximum*1000
    record['worst_segment'] = worst


def _tail(record, volume, length):
    if len(record['sizes']) != 2480 or np.any(record['sizes'] != 35):
        raise RuntimeError('Expected the existing 2480 x 35 tail topology')
    old = record['world'].reshape(2480,35,3)
    mean = old.mean(axis=0)
    groups = old.reshape(8,310,35,3).mean(axis=1)
    # Coordinates below are proportions of the actual tail, not absolute head Z.
    side = 1 if mean[0,0] > 0 else -1
    drop = max(.10,float(mean[2,2]-mean[-1,2]))
    t = np.arange(35)/34
    activate = _smooth((t-t[2])/.25)
    mass = np.sin(np.pi*t)**.70*activate
    mean_new = mean.copy()
    # Shorter vertical gesture, with the original shoulderward depth retained
    # enough to avoid moving the hair abruptly back into the neck or garment.
    mean_new[:,2] += (1-length)*(mean[2,2]-mean[:,2])*activate
    mean_new[:,1] = mean[:,1]+(mean[2,1]-mean[:,1])*(1-length)*.65*activate
    mean_new[:,0] += side*drop*(.030*np.sin(np.pi*t)+.014*np.sin(2*np.pi*t))*activate
    new = old.copy()
    for group in range(8):
        begin = group*310
        group_offset = groups[group]-mean
        cross = group_offset.copy()
        # Enlarge transverse volume, preserving the already distinct length ends.
        cross[:,:2] *= (1+(volume-1)*mass)[:,None]
        cluster = group//2
        end_weight = _smooth((t-.66)/.34)*activate
        end_x = [-.028,.020,.040,-.010][cluster]*drop
        end_z = [.023,-.030,.040,-.046][cluster]*drop
        cross[:,0] += side*end_x*end_weight
        cross[:,2] += end_z*end_weight
        core = mean_new+cross
        within = old[begin:begin+310]-groups[group]
        within[:,:,:2] *= (1+(volume-1)*mass)[None,:,None]
        new[begin:begin+310] = core[None,:,:]+within
    # The fixed PBD attachment is an exact copy, not a tolerance-based pin.
    new[:,:3] = old[:,:3]
    record['new_world'] = new.reshape(-1,3)
    record['new_radius'] = record['radius'].reshape(2480,35).copy()
    record['new_radius'] *= (1+.06*mass)[None,:]
    record['new_radius'] = record['new_radius'].ravel()
    record['pin_points'] = 3
    record['tail_rest'] = new
    record['tail_summary'] = {
        'root_world':mean[0].tolist(), 'root_pin_delta_mm':float(np.abs(new[:,:3]-old[:,:3]).max()*1000),
        'old_vertical_span_mm':float((old[:,:,2].max()-old[:,:,2].min())*1000),
        'new_vertical_span_mm':float((new[:,:,2].max()-new[:,:,2].min())*1000),
        'old_mid_x_width_mm':float(np.ptp(old[:,12:23,0])*1000),
        'new_mid_x_width_mm':float(np.ptp(new[:,12:23,0])*1000)}


def _bangs(record, width):
    if len(record['sizes']) != 3360:
        raise RuntimeError('Expected the existing seven 480-strand forelocks')
    new = record['world'].copy()
    radius = record['radius'].copy()
    sample_t = np.linspace(0,1,61)
    for group in range(7):
        ids = range(group*480,(group+1)*480)
        samples = [_sample(record['world'][record['starts'][i]:record['starts'][i]+record['sizes'][i]],sample_t)
                   for i in ids]
        center = np.mean(samples,axis=0)
        tangent = np.gradient(center,axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
        lateral = np.tile([1.,0.,0.],(len(center),1))-tangent*tangent[:,0,None]
        lateral /= np.maximum(np.linalg.norm(lateral,axis=1,keepdims=True),1e-9)
        for i in ids:
            start,count = record['starts'][i],record['sizes'][i]
            t = np.arange(count)/(count-1)
            path = record['world'][start:start+count]
            core = _sample(center,t)
            across = _sample(lateral,t)
            across /= np.maximum(np.linalg.norm(across,axis=1,keepdims=True),1e-9)
            delta = path-core
            transverse = (delta*across).sum(axis=1)
            growth = _smooth((t-.075)/.19)*(1-_smooth((t-.87)/.13))
            broadened = path+across*(transverse*(width-1)*growth)[:,None]
            # Existing longitudinal/depth layering is retained. Width and fibre
            # radius grow together enough to close the broad forehead gaps.
            broadened[:3] = path[:3]
            new[start:start+count] = broadened
            radius[start:start+count] *= 1+.28*growth
    record['new_world'] = new
    record['new_radius'] = radius
    record['pin_points'] = 3


def _fit_tie(record, tail, band):
    root_paths = tail['tail_rest'][:,:3]
    root_centers = root_paths.mean(axis=0)
    axis = _normal(root_centers[2]-root_centers[0])
    root_span = float(np.linalg.norm(root_centers[2]-root_centers[0]))
    distance = root_span*(.12 if band == 0 else .32)
    center = root_centers[0]+axis*distance
    # Sample every fixed strand at the ring's own transverse plane.
    sections = []
    for path in root_paths:
        along = (path-center)@axis
        pair = 0 if along[1] >= 0 else 1
        fraction = float(np.clip(-along[pair]/max(along[pair+1]-along[pair],1e-9),0,1))
        sections.append(path[pair]*(1-fraction)+path[pair+1]*fraction)
    sections = np.array(sections)-center
    radial = sections-np.outer(sections@axis,axis)
    fit_radius = float(np.quantile(np.linalg.norm(radial,axis=1),.995))
    old = record['world']; old_center = old.mean(axis=0); delta = old-old_center
    _, axes = np.linalg.eigh(delta.T@delta/len(delta))
    old_axis = axes[:,0]
    if old_axis@axis < 0:
        old_axis *= -1
    old_axial = delta@old_axis
    old_radial = delta-np.outer(old_axial,old_axis)
    r = np.linalg.norm(old_radial,axis=1)
    old_major = (float(r.max())+float(r.min()))*.5
    old_minor = max(.0005,(float(r.max())-float(r.min()))*.5)
    minor = old_minor*.78
    # The hole fits the actual fixed roots; do not force a tiny ring through them.
    major = fit_radius+minor+.00035
    old_u = _normal(old_radial[0]); old_v = _normal(np.cross(old_axis,old_u))
    # A fitted ellipse removes the empty space of an oversized circular hoop
    # while still containing the unchanged root bundle.
    _, root_axes = np.linalg.eigh(radial.T@radial)
    new_u = _normal(root_axes[:,-1]); new_v = _normal(np.cross(axis,new_u))
    section_uv = np.column_stack((radial@new_u,radial@new_v))
    ellipse = np.maximum(np.quantile(np.abs(section_uv),.98,axis=0),.001)
    ellipse *= float(np.quantile(np.linalg.norm(section_uv/ellipse,axis=1),.995))
    major_uv = ellipse+minor+.00035
    unit = old_radial/np.maximum(r[:,None],1e-9)
    tube = minor/old_minor
    record['new_world'] = (center
                           +np.outer((unit@old_u)*(major_uv[0]+(r-old_major)*tube),new_u)
                           +np.outer((unit@old_v)*(major_uv[1]+(r-old_major)*tube),new_v)
                           +np.outer(old_axial*tube,axis))
    record['tie_summary'] = {'root_fitting_quantile':.995,'center_world':center.tolist(),
                            'axis_world':axis.tolist(),'fitted_root_radius_mm':fit_radius*1000,
                            'old_outer_radius_mm':(old_major+old_minor)*1000,
                            'new_outer_semiaxes_mm':((major_uv+minor)*1000).tolist(),
                            'new_inner_semiaxes_mm':((major_uv-minor)*1000).tolist()}


def run(tail_volume=1.55, tail_length=.83, bang_width=1.38, diagnostics_only=False):
    if not bpy.data.filepath or bpy.data.filepath.replace('\\','/').endswith('/work/character.blend'):
        raise RuntimeError('Use the saved R05 candidate, not the master')
    rig = bpy.data.objects['Character_Rig']
    if rig.get('R04_native_hair_guide'):
        raise RuntimeError('Restore previous native hair playback before changing the groom')
    if not 1.0 <= tail_volume <= 1.8 or not .72 <= tail_length <= 1 or not 1 <= bang_width <= 1.6:
        raise ValueError('Groom controls outside the intended local correction range')
    objects = [bpy.data.objects[name] for name in (*TAILS,BANGS)]
    ties = []
    for side in (1,-1):
        pair = [bpy.data.objects.get(f'R02 Tie {side} {j}') for j in (-1,1)]
        if any(o is None or o.type!='MESH' for o in pair):
            raise RuntimeError('Expected both existing mesh ties for side '+str(side))
        ties.extend(pair)
    for obj in objects+ties:
        if obj.get(MARKER):
            raise RuntimeError('Already applied: '+obj.name)
        if obj.library or obj.data.library or obj.data.users != 1:
            raise RuntimeError('Local single-user candidate data required: '+obj.name)
        if getattr(obj.data,'shape_keys',None):
            raise RuntimeError('Unexpected groom shape keys: '+obj.name)
        if any(m.show_render or m.show_viewport for m in obj.modifiers):
            raise RuntimeError('Restore active groom modifiers first: '+obj.name)
    pose = rig.data.pose_position
    rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
    try:
        records = [_read(o) for o in objects+ties]
        for record in records[:2]:
            _tail(record,tail_volume,tail_length)
        _bangs(records[2],bang_width)
        for i, record in enumerate(records[3:]):
            _fit_tie(record,records[i//2],i%2)
        for record in records:
            _check(record)
            record['new_local'] = _to_local(record['new_world'],record['matrix'])
            if record.get('pin_points'):
                for start in record['starts']:
                    record['new_local'][start:start+3] = record['local'][start:start+3]
        report = {'parameters':{'tailVolume':tail_volume,'tailLength':tail_length,'bangWidth':bang_width},
                  'objects':[{'name':r['object'].name,
                              'maximumStepMm':r.get('maximum_step_mm'),
                              'tail':r.get('tail_summary'),'tie':r.get('tie_summary')}
                             for r in records],
                  'writesPerformed':not diagnostics_only,
                  'requires':'New static silhouette review and a fresh PBD/native cache before motion playback',
                  'validation':'Topology, fixed root points and bounded segment lengths only; no render or collision claim'}
        if diagnostics_only:
            print(json.dumps(report,ensure_ascii=False)); return report
        for record in records:
            obj = record['object']
            backup = obj.data.copy(); backup.name = obj.data.name+' pre-R05-groom'
            backup.use_fake_user = True
            points = obj.data.position_data if obj.type=='CURVES' else obj.data.vertices
            points.foreach_set('vector' if obj.type=='CURVES' else 'co',record['new_local'].ravel())
            if 'new_radius' in record:
                obj.data.attributes['radius'].data.foreach_set('value',record['new_radius'])
            obj[BACKUP] = backup.name; obj[MARKER] = True
            obj.data.update_tag(); obj.update_tag()
        rig['R05_groom_requires_fresh_hair_cache'] = True
        bpy.context.view_layer.update()
        print(json.dumps(report,ensure_ascii=False)); return report
    finally:
        rig.data.pose_position = pose
        bpy.context.view_layer.update()


def restore():
    restored = []
    for obj in list(bpy.context.scene.objects):
        if not obj.get(MARKER):
            continue
        collection = bpy.data.hair_curves if obj.type=='CURVES' else bpy.data.meshes
        backup = collection.get(obj.get(BACKUP,''))
        if backup is None:
            raise RuntimeError('Retained groom data missing: '+obj.name)
        obj.data.use_fake_user = True
        obj.data = backup; backup.use_fake_user = False
        del obj[MARKER]; del obj[BACKUP]
        restored.append(obj.name)
    bpy.context.view_layer.update()
    return restored


if __name__ == '__main__':
    run()

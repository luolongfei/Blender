"""Pure-array front exposure measurement for an actual lip-seal candidate.

This measures visible inner-lip surface, not an inferred missing-mouth hole.
It is deliberately independent of lip displacement generation. Real contact,
crossings, wall thickness and appearance still require separate checks.
"""
from pathlib import Path
import runpy
import numpy as np


def _triangles(faces):
    triangles=[];polygon_ids=[]
    for index,face in enumerate(faces):
        for k in range(1,len(face)-1):
            triangles.append((int(face[0]),int(face[k]),int(face[k+1])))
            polygon_ids.append(index)
    return np.asarray(triangles,np.int32),np.asarray(polygon_ids,np.int32)


def grid_for(world,faces,groups,x_bounds=(-.024,.024),x_step=.0005,z_step=.00005):
    """Create once from the baseline; reuse exactly for every A/B state."""
    world=np.asarray(world,float);groups=np.asarray(groups)
    vertices={}
    for group in (7,8,17,22):
        rows=[np.asarray(face,np.int32) for face,g in zip(faces,groups) if g==group]
        if not rows:raise ValueError('Actual inner and outer lip face groups required')
        vertices[group]=np.unique(np.concatenate(rows))
    opening=np.union1d(np.intersect1d(vertices[7],vertices[17]),np.intersect1d(vertices[8],vertices[22]))
    if not len(opening):raise ValueError('No shared inner/outer anatomical lip edge')
    points=world[opening]
    zlo=float(np.floor((points[:,2].min()-.002)/z_step)*z_step)
    zhi=float(np.ceil((points[:,2].max()+.002)/z_step)*z_step)
    if not .00002<=z_step<=.0001 or not .0001<=x_step<=.001:raise ValueError('Explicit bounded ray resolution required')
    return {'x':np.arange(float(x_bounds[0]),float(x_bounds[1])+x_step*.1,x_step),
            'z':np.arange(zlo,zhi+z_step*.1,z_step),
            'xStepMeters':float(x_step),'zStepMeters':float(z_step)}


def _runs(mask):
    active=np.flatnonzero(mask)
    return [] if not len(active) else np.split(active,np.flatnonzero(np.diff(active)>1)+1)


def union_grid(grids):
    """Cover baseline rest and smile opening rings with one fixed grid."""
    if not grids:raise ValueError('At least one baseline grid required')
    result={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in grids[0].items()}
    for grid in grids[1:]:
        if not np.array_equal(grid['x'],result['x']) or grid['zStepMeters']!=result['zStepMeters']:
            raise ValueError('Incompatible baseline review grids')
    low=min(float(grid['z'][0]) for grid in grids);high=max(float(grid['z'][-1]) for grid in grids)
    dz=result['zStepMeters'];result['z']=low+np.arange(round((high-low)/dz)+1)*dz
    return result


def measure(world,faces,groups,grid=None,triangles=None,triangle_polygon_ids=None):
    """Return (report,arrays) from first -Y head hits on a fixed X/Z grid.

    Optional actual loop triangles plus their polygon indices avoid choosing a
    diagonal for non-planar quads. Without them a declared polygon fan is used;
    that result must not be described as an exact Blender loop-triangle test.
    Polygon indices here address the provided faces/groups, not global IDs.
    """
    world=np.asarray(world,float);groups=np.asarray(groups,np.int32)
    if len(faces)!=len(groups):raise ValueError('Face-group alignment mismatch')
    if grid is None:grid=grid_for(world,faces,groups)
    if triangles is None:
        if triangle_polygon_ids is not None:raise ValueError('Triangle polygon IDs without triangles')
        triangles,polygon_ids=_triangles(faces);triangulation='Declared polygon fan; not Blender loop-triangle evidence'
    else:
        triangles=np.asarray(triangles,np.int32);polygon_ids=np.asarray(triangle_polygon_ids,np.int32)
        if polygon_ids.shape!=(len(triangles),):raise ValueError('Actual triangle/polygon mapping required')
        triangulation='Caller-supplied current surface triangles and polygon mapping'
    if np.any(polygon_ids<0) or np.any(polygon_ids>=len(groups)):raise ValueError('Invalid triangle polygon mapping')
    xs=np.asarray(grid['x'],float);zs=np.asarray(grid['z'],float)
    if len(xs)<2 or len(zs)<2 or len(xs)*len(zs)>150000:raise ValueError('Bounded nonempty ray grid required')
    dx=float(grid['xStepMeters']);dz=float(grid['zStepMeters'])
    if not np.allclose(np.diff(xs),dx,rtol=0,atol=1e-12) or not np.allclose(np.diff(zs),dz,rtol=0,atol=1e-12):
        raise ValueError('Uniform fixed ray-grid spacing required')
    pts=world[triangles];lo=pts[:,:,[0,2]].min(axis=1);hi=pts[:,:,[0,2]].max(axis=1)
    selected=(hi[:,0]>=xs[0])&(lo[:,0]<=xs[-1])&(hi[:,1]>=zs[0])&(lo[:,1]<=zs[-1])
    selected_triangle_ids=np.flatnonzero(selected)
    if not len(selected_triangle_ids):raise ValueError('No triangles in lip review window')
    api=runpy.run_path(str(Path(__file__).with_name('adult_blink_inner_thickness_support_r11.py')))
    index=api['FrontTriangleIndex'](world,triangles[selected])
    xx,zz=np.meshgrid(xs,zs,indexing='ij');points=np.column_stack((xx.ravel(),np.zeros(xx.size),zz.ravel()))
    y,local,tested=index.front(points);valid=local>=0
    triangle_rows=np.full(len(points),-1,np.int32);triangle_rows[valid]=selected_triangle_ids[local[valid]]
    first_polygons=np.full(len(points),-1,np.int32);first_polygons[valid]=polygon_ids[triangle_rows[valid]]
    first_groups=np.full(len(points),-1,np.int32);first_groups[valid]=groups[first_polygons[valid]]
    first_normals=np.full(len(points),np.nan);first_normals[valid]=index.normal_y[local[valid]]
    shape=(len(xs),len(zs));visible=np.isin(first_groups,[7,8]).reshape(shape)
    exposure=[]
    for i,x in enumerate(xs):
        runs=[]
        for active in _runs(visible[i]):
            flat=i*len(zs)+active
            runs.append({'firstSampleZ':float(zs[active[0]]),'lastSampleZ':float(zs[active[-1]]),
                         'sampleCentreSpanMeters':float(zs[active[-1]]-zs[active[0]]),
                         'sampleCellHeightMeters':float(len(active)*dz),'sampleCount':len(active),
                         'firstFaceGroups':sorted(map(int,np.unique(first_groups[flat]))),
                         'frontYRangeMeters':[float(y[flat].min()),float(y[flat].max())]})
        exposure.append({'x':float(x),'innerLipVisibleSamples':int(visible[i].sum()),
                         'totalSampleCellHeightMeters':float(visible[i].sum()*dz),'runs':runs})
    named=[]
    for target in (-.012,0.,.012):
        row=exposure[int(abs(xs-target).argmin())].copy();row['requestedX']=target;named.append(row)
    report={'method':'First actual supplied head triangle from negative world Y at fixed X/Z rays. First-hit face sets 7/8 are exposed inner lip; 17/22 are outer vermilion.',
            'triangulation':triangulation,'rayCount':len(points),'headHitCount':int(valid.sum()),
            'innerLipVisibleSampleCount':int(visible.sum()),
            'innerLipExposureTouchesZWindowBoundary':bool(visible[:,[0,-1]].any()),
            'innerLipVisibleProjectedSampleAreaSquareMeters':float(visible.sum()*dx*dz),
            'maximumColumnSampleCellHeightMeters':max(row['totalSampleCellHeightMeters'] for row in exposure),
            'namedColumns':named,'columns':exposure,
            'grid':{'xBounds':[float(xs[0]),float(xs[-1])],'zBounds':[float(zs[0]),float(zs[-1])],
                    'xCount':len(xs),'zCount':len(zs),'xStepMeters':dx,'zStepMeters':dz},
            'limits':['Projected ray sampling is not 3D contact, minimum separation, self-intersection or visual acceptance.',
                      'Sample-cell area is an estimate; boundaries are uncertain by the stated grid spacing.',
                      'Face sets classify source anatomy, not pigment. A small visible mucosal line can be legitimate.',
                      'Input must include all occluding outer head surfaces in the review window; a cropped oral-only mesh gives false exposure.',
                      'Reuse this exact grid for comparisons; recorded side columns are nearest grid columns.'],
            'appearanceAccepted':False}
    arrays={'x':xs,'z':zs,'frontY':y.reshape(shape),'firstTriangleRows':triangle_rows.reshape(shape),
            'firstPolygonRows':first_polygons.reshape(shape),'firstGroups':first_groups.reshape(shape),
            'firstNormalY':first_normals.reshape(shape),'visibleInnerLip':visible,
            'candidateTriangleCounts':tested.reshape(shape)}
    return report,arrays

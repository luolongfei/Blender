import bpy,json,numpy as np,hashlib
rows=[]
for o in bpy.data.objects:
    if o.type!='MESH' or not o.name.startswith('RT06 Face / '):continue
    for a in o.data.attributes:
        if a.data_type=='INT16_2D':
            d=a.data[0]
            props=[]
            for p in d.bl_rna.properties:
                if p.identifier=='rna_type':continue
                v=getattr(d,p.identifier)
                props.append({'id':p.identifier,'type':p.type,'array':p.is_array,'length':getattr(p,'array_length',None),'value':list(v) if hasattr(v,'__len__') else v})
            buffer=np.empty(len(a.data)*2,np.int32);a.data.foreach_get('value',buffer)
            rows.append({'object':o.name,'attribute':a.name,'domain':a.domain,'count':len(a.data),'properties':props,'int32Foreach':True,'min':int(buffer.min()),'max':int(buffer.max()),'sha256':hashlib.sha256(buffer.tobytes()).hexdigest()})
print('R08_INT16_ATTRIBUTE',json.dumps(rows))

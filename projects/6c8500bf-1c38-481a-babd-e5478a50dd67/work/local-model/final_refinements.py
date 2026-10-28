"""Observed fit corrections for the current original character mesh."""
import bpy
import math
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent


def apply():
    corrections=[]
    col=bpy.data.collections["CHARACTER · Original geometry"]
    for o in col.objects:
        if o.type!="MESH":
            continue
        if o.name.startswith("Face ·") and not o.get("chin_profile_refined"):
            # The reference's chin is anterior to the neck, not centred on its axis.
            table=[
                (1.318,-.050,.004,.004),(1.326,-.037,.040,.025),
                (1.340,-.015,.069,.041),(1.360,.000,.083,.061),
                (1.380,.005,.0875,.0775)
            ]
            def interpolate(z,i):
                for a,b in zip(table,table[1:]):
                    if a[0]<=z<=b[0]:
                        t=(z-a[0])/(b[0]-a[0])
                        return a[i]*(1-t)+b[i]*t
                return table[0 if z<table[0][0] else -1][i]
            for v in o.data.vertices:
                z=v.co.z
                if z>=1.38:
                    continue
                t=math.tau*(v.index%112)/112
                c=math.cos(t)
                cy,front,back=(interpolate(z,i) for i in (1,2,3))
                y=cy-front*max(c,0)**.57 if c>=0 else cy+back*(-c)
                if c>0:
                    x=v.co.x
                    nose=.014*math.exp(-(x/.012)**2-((z-1.390)/.015)**2)
                    bridge=.007*math.exp(-(x/.009)**2-((z-1.415)/.028)**2)
                    lips=.003*math.exp(-(x/.024)**4-((z-1.365)/.008)**2)
                    cheek=.003*(math.exp(-((x-.055)/.025)**2)+math.exp(-((x+.055)/.025)**2))*math.exp(-((z-1.387)/.020)**2)
                    y-=(nose+bridge+lips+cheek)*c**4
                v.co.y=y
            o["chin_profile_refined"]=True;o.data.update();corrections.append(o.name)
        if o.get("shoe_opening_fixed"):
            continue
        if o.name.startswith("Shoe_Leather_Upper_"):
            for v in o.data.vertices:
                t=max(0,min(1,(v.co.z-.080)/.028));v.co.y-=.018*t*t*(3-2*t)
            o["shoe_opening_fixed"]=True;o.data.update();corrections.append(o.name)
        elif o.name.startswith(("Shoe_Inner_Collar_","Shoe_Rolled_Collar_")):
            for v in o.data.vertices:v.co.y-=.018
            o["shoe_opening_fixed"]=True;o.data.update();corrections.append(o.name)
    return corrections

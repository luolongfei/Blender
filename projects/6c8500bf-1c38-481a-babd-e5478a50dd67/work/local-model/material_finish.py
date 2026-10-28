"""Character-specific gradual MToon shading for the natural-proportion model.

Texture paths, UVs, alpha and normals are retained. This intentionally lives in
the project; it does not modify the distribution kit's global material profile.
"""

def apply_runtime(meshes):
    materials={m.name:m for o in meshes for m in o.data.materials if m}
    report=[]
    for name,material in materials.items():
        gltf=material.vrm_addon_extension.mtoon1
        if not gltf.enabled:
            continue
        lower=name.lower()
        if any(w in lower for w in ("iris","sclera","catchlight","pupil")):
            kind,shade,toony,shift="eyes",.91,.08,0
        elif any(w in lower for w in ("skin","lip","ear","nostril")):
            kind,shade,toony,shift="skin",.76,.10,-.03
        elif "hair" in lower or "charcoal" in lower or "brow" in lower or "lash" in lower:
            kind,shade,toony,shift="hair",.69,.14,-.02
        elif any(w in lower for w in ("blouse","collar","cloth","skirt","sock","cuff","woven","garment","button")):
            kind,shade,toony,shift="garment",.76,.10,-.02
        else:
            kind,shade,toony,shift="detail",.74,.14,0
        mtoon=gltf.extensions.vrmc_materials_mtoon
        base=gltf.pbr_metallic_roughness.base_color_factor
        mtoon.shade_color_factor=tuple(channel*shade for channel in base[:3])
        mtoon.shading_toony_factor=toony
        mtoon.shading_shift_factor=shift
        mtoon.gi_intensity_factor=1
        mtoon.outline_width_mode="none"
        material["delivery_tone_profile"]="natural-gradual-shading-02"
        report.append({"material":name,"role":kind,"shadeMultiplier":shade,
                       "shadingToonyFactor":toony,"shadingShiftFactor":shift})
    return report

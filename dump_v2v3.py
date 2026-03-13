#!/usr/bin/env python3
"""Dump WFI SCA corner positions as V2/V3 arcsec (via planar idl_to_tel) for JS embedding."""
import pysiaf, json, numpy as np

rsiaf   = pysiaf.Siaf("Roman")
wfi_cen = rsiaf["WFI_CEN"]
V2Ref   = float(wfi_cen.V2Ref)
V3Ref   = float(wfi_cen.V3Ref)
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]

scas = []
for s in sensors:
    # Use pysiaf's IDL→tel transform (planar approximation, which is what corners() uses)
    xs = [float(getattr(s, f"XIdlVert{k}")) for k in range(1,5)]
    ys = [float(getattr(s, f"YIdlVert{k}")) for k in range(1,5)]
    v2s, v3s = s.idl_to_tel(np.array(xs), np.array(ys))
    scas.append({
        "v2": [round(float(v), 4) for v in v2s],
        "v3": [round(float(v), 4) for v in v3s],
    })

print(f"V2Ref={V2Ref:.6f}")
print(f"V3Ref={V3Ref:.6f}")
print(json.dumps(scas))

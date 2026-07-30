#!/usr/bin/env python3
import pysiaf, json, numpy as np, collections
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

ECSV = ("/Users/adam/repos/roman_notebooks/notebooks/"
        "footprint_visualization/aux_data/roman_gps.sim.ecsv")

_BP_WAVE = {"F062": 0.62, "F087": 0.87, "F106": 1.06, "F129": 1.29,
            "F146": 1.46, "F158": 1.58, "F184": 1.84, "F213": 2.13,
            "GRISM": 1.50, "PRISM": 1.50}

def unique_pointings(tbl, target):
    sub = tbl[tbl["TARGET_NAME"] == target]
    bps = sorted(set(str(b) for b in sub["BANDPASS"]),
                 key=lambda x: _BP_WAVE.get(x, 0.0))
    sub = sub[sub["BANDPASS"] == bps[-1]]
    pa_vals  = [round(float(r["PA"]), 1) for r in sub]
    modal_pa = collections.Counter(pa_vals).most_common(1)[0][0]
    sub      = [r for r, pa in zip(sub, pa_vals) if pa == modal_pa]
    pts = {}
    for r in sub:
        pts[(round(float(r["RA"]), 6), round(float(r["DEC"]), 6))] = True
    return list(sorted(pts))

def dedup_pointings(pts, sep=7.5):
    if len(pts) <= 1:
        return list(pts)
    ras  = np.radians([p[0] for p in pts])
    decs = np.radians([p[1] for p in pts])
    sd, cd = np.sin(decs), np.cos(decs)
    thresh = np.radians(sep / 60.0)
    kept = [0]
    for i in range(1, len(pts)):
        cs = sd[i]*sd[kept] + cd[i]*cd[kept]*np.cos(ras[i] - ras[kept])
        if np.all(np.arccos(np.clip(cs, -1, 1)) >= thresh):
            kept.append(i)
    return [pts[k] for k in kept]

def gal_coords(ra, dec):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    g = c.galactic
    return round(float(g.l.deg), 4), round(float(g.b.deg), 4)

def gal_plane_pa(ra, dec):
    c  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    c2 = SkyCoord(l=c.galactic.l + 0.2*u.deg, b=c.galactic.b, frame=Galactic()).icrs
    return round(c.position_angle(c2).deg, 3)

print("Reading ECSV...")
sim_table = Table.read(ECSV)
all_targets = sorted(set(str(x) for x in sim_table["TARGET_NAME"]))

result = {}
for tname in all_targets:
    all_pts = unique_pointings(sim_table, tname)
    pts     = dedup_pointings(all_pts, sep=7.5)
    tiles   = []
    for ra, dec in pts:
        l, b   = gal_coords(ra, dec)
        pa_gal = gal_plane_pa(ra, dec)
        tiles.append({"ra": ra, "dec": dec, "l": l, "b": b, "pa_gal": pa_gal})
    result[tname] = tiles
    print(f"  {tname}: {len(pts)} pts")

with open("/tmp/rgps_centers.json", "w") as f:
    json.dump(result, f, separators=(',', ':'))
print("done:", sum(len(v) for v in result.values()), "pointings")

#!/usr/bin/env python3
"""
Generate DS9 .reg files for the Roman GBTDS and RGPS footprints using the
actual WFI focal-plane geometry from the Roman SIAF (pysiaf).

SCA corner positions are computed with:
    attitude_matrix(V2Ref, V3Ref, ra, dec, pa_v3)
where V2Ref/V3Ref are the WFI_CEN offsets from the telescope boresight
(+1546.4", -892.8").  Using attitude(0, 0, ...) would displace every
footprint by ~0.496°.

GBTDS field centres come from:
    https://github.com/rachel3834/rgps/ config/rgps_survey_definitions.json
    (tiled to cover each survey region with non-overlapping WFI pointings)

RGPS pointings come from:
    roman_gps.sim.ecsv  (APT simulator output)
    — only the longest-wavelength bandpass per target is used, and
      pointings within 7.5' of each other are deduplicated.
"""

import json
import numpy as np
import requests
import pysiaf
from pysiaf.utils.rotations import attitude_matrix
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
from pathlib import Path

# ── pysiaf setup ──────────────────────────────────────────────────────────────
print("Loading Roman SIAF …")
rsiaf   = pysiaf.Siaf("Roman")
wfi_cen = rsiaf["WFI_CEN"]
V2Ref   = wfi_cen.V2Ref    # +1546.38"  WFI_CEN offset from V3 boresight
V3Ref   = wfi_cen.V3Ref    # -892.79"
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]
print(f"  WFI_CEN  V2={V2Ref:+.1f}\"  V3={V3Ref:+.1f}\"  "
      f"offset={(V2Ref**2+V3Ref**2)**0.5/3600:.4f}°")


def sca_corners_icrs(ra_cen, dec_cen, pa_v3):
    """
    Return list of 18 SCA polygons [(ra,dec)×4] for one WFI pointing.

    ra_cen, dec_cen : ICRS position of WFI_CEN (degrees)
    pa_v3           : V3-axis position angle, East of North, celestial (degrees)
                      For galactic-plane alignment: pa_v3 = gal_plane_pa(ra, dec)
    """
    attmat = attitude_matrix(V2Ref, V3Ref, ra_cen, dec_cen, pa_v3)
    polys = []
    for sensor in sensors:
        sensor.set_attitude_matrix(attmat)
        c = sensor.corners("sky")   # (ra_4pts, dec_4pts)
        polys.append([(round(float(c[0][i]), 5), round(float(c[1][i]), 5))
                      for i in range(4)])
    return polys   # 18 × [(ra,dec)×4]


# ── Galactic-plane orientation helper ─────────────────────────────────────────

def gal_plane_pa(ra_deg, dec_deg):
    """
    Position angle (East of North, degrees, celestial) of the Galactic
    longitude direction at the given ICRS position.
    Pass this as pa_v3 to align the WFI long axis with the Galactic plane.
    (V3 is the long dimension of WFI; V3IdlYAngle = -60° for all WFI apertures.)
    """
    c  = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    c2 = SkyCoord(l=c.galactic.l + 0.2*u.deg, b=c.galactic.b, frame=Galactic()).icrs
    return c.position_angle(c2).deg


# ── GBTDS: tile survey fields from config JSON ────────────────────────────────

# Tiling step — match pysiaf WFI footprint span in V3/V2 (incl. SCA gaps)
FPA_V3_SPAN = 0.59   # degrees along V3 (long axis, aligned with galactic plane)
FPA_V2_SPAN = 0.45   # degrees along V2 (short axis)


def tile_field(l_range, b_range):
    """
    Tile a rectangular Galactic field with non-overlapping WFI pointing centres.
    Long axis (V3) goes along l, short axis (V2) goes along b.
    Returns list of (l_cen, b_cen) tuples.
    """
    l_min, l_max = sorted(l_range)
    b_min, b_max = sorted(b_range)
    n_l = max(1, round((l_max - l_min) / FPA_V3_SPAN))
    n_b = max(1, round((b_max - b_min) / FPA_V2_SPAN))
    step_l = (l_max - l_min) / n_l
    step_b = (b_max - b_min) / n_b
    return [(l_min + (i + 0.5) * step_l, b_min + (j + 0.5) * step_b)
            for i in range(n_l) for j in range(n_b)]


def dedup_fields(section):
    """Return a name→entry dict from a survey section, deduplicating across filters."""
    seen = {}
    for filt_entries in section.values():
        if not isinstance(filt_entries, list):
            continue
        for entry in filt_entries:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            if entry["name"] not in seen:
                seen[entry["name"]] = entry
    return seen


def build_gbtds_entries(field_dict, color_map):
    """Build region_entries list for GBTDS fields using config tiling."""
    entries = []
    for name, entry in field_dict.items():
        color = color_map.get(name, "white")
        if "l" in entry and "b" in entry:
            centers_gal = tile_field(entry["l"], entry["b"])
        elif "pointing" in entry:
            centers_gal = [(entry["pointing"][0], entry["pointing"][1])]
        else:
            continue
        pointings = []
        for (lc, bc) in centers_gal:
            c = SkyCoord(l=lc*u.deg, b=bc*u.deg, frame=Galactic()).icrs
            pa_v3 = gal_plane_pa(c.ra.deg, c.dec.deg)
            pointings.append(sca_corners_icrs(c.ra.deg, c.dec.deg, pa_v3))
        n = len(centers_gal)
        print(f"  {name}: {n} pointing(s) × 18 SCAs = {n*18} polygons  [{color}]")
        entries.append((name, color, pointings))
    return entries


# ── RGPS: read actual pointings from roman_gps.sim.ecsv ────────────────────────

ECSV = ("/Users/adam/repos/roman_notebooks/notebooks/"
        "footprint_visualization/aux_data/roman_gps.sim.ecsv")

# Roman WFI bandpass central wavelengths (µm) — pick longest λ per target
_BP_WAVE = {"F062": 0.62, "F087": 0.87, "F106": 1.06, "F129": 1.29,
            "F146": 1.46, "F158": 1.58, "F184": 1.84, "F213": 2.13,
            "GRISM": 1.50, "PRISM": 1.50}


def unique_pointings_ecsv(table, target_name):
    """Return sorted (ra, dec, pa) tuples for a target, longest-λ bandpass only."""
    sub = table[table["TARGET_NAME"] == target_name]
    bps = sorted({str(b) for b in sub["BANDPASS"]}, key=lambda x: _BP_WAVE.get(x, 0.0))
    sub = sub[sub["BANDPASS"] == bps[-1]]
    pts = {}
    for r in sub:
        key = (round(float(r["RA"]), 6), round(float(r["DEC"]), 6),
               round(float(r["PA"]), 3))
        pts[key] = True
    return sorted(pts.keys())


def deduplicate_close_pointings(pointings, min_sep_arcmin=7.5):
    """
    Greedy spatial dedup: keep the first pointing in a sorted list and discard
    any subsequent one within min_sep_arcmin (great-circle) of a kept pointing.
    """
    if len(pointings) <= 1:
        return list(pointings)
    ras   = np.radians([p[0] for p in pointings])
    decs  = np.radians([p[1] for p in pointings])
    sin_d = np.sin(decs)
    cos_d = np.cos(decs)
    thresh = np.radians(min_sep_arcmin / 60.0)
    kept = [0]
    for i in range(1, len(pointings)):
        cos_sep = (sin_d[i] * sin_d[kept] +
                   cos_d[i] * cos_d[kept] * np.cos(ras[i] - ras[kept]))
        if np.all(np.arccos(np.clip(cos_sep, -1.0, 1.0)) >= thresh):
            kept.append(i)
    return [pointings[k] for k in kept]


def build_rgps_entries(table, target_names, color_map):
    """Build region_entries list for RGPS fields using ECSV pointings."""
    entries = []
    for tname in target_names:
        color = color_map.get(tname, "white")
        all_pts  = unique_pointings_ecsv(table, tname)
        pts      = deduplicate_close_pointings(all_pts, min_sep_arcmin=7.5)
        pointings = [sca_corners_icrs(ra, dec, pa) for ra, dec, pa in pts]
        print(f"  {tname}: {len(all_pts)} → {len(pts)} pointings (7.5' dedup) "
              f"× 18 SCAs = {len(pts)*18} polygons  [{color}]")
        entries.append((tname, color, pointings))
    return entries



# -- DS9 output helpers -------------------------------------------------------

def poly_to_ds9(corners, color, label):
    """Format 4 ICRS corners as a DS9 fk5 polygon line."""
    coords = " ".join(f"{ra:.6f},{dec:.6f}" for ra, dec in corners)
    return f"polygon({coords}) # color={color} tag={{{label}}}"


def write_reg(filepath, region_entries):
    """
    Write a DS9 .reg file.
    region_entries: list of (label, color, list_of_pointings)
    Each pointing is a list of 18 SCA corner lists (each 4 (ra,dec) tuples).
    """
    hdr = [
        "# Region file format: DS9 version 4.1",
        "# Roman WFI: 18 SCAs, pysiaf Roman SIAF geometry",
        "# SCA corners: attitude_matrix(V2Ref, V3Ref, ra, dec, pa_v3)",
        f"# WFI_CEN: V2={V2Ref:+.1f}\"  V3={V3Ref:+.1f}\"",
        "# GBTDS: pa_v3 = galactic-longitude PA (E of celestial N) at each pointing",
        "# RGPS:  pa_v3 from roman_gps.sim.ecsv (all PA=0.0 nominal, celestial)",
        "fk5",
    ]
    n_polys = 0
    polys = []
    for (label, color, pointings) in region_entries:
        for sca_list in pointings:
            for sca_corners in sca_list:
                polys.append(poly_to_ds9(sca_corners, color, label))
                n_polys += 1
    Path(filepath).write_text("\n".join(hdr + polys) + "\n")
    kb = Path(filepath).stat().st_size // 1024
    print(f"  {Path(filepath).name}: {n_polys} SCA polygons, {kb} KB")


# -- DS9 colour assignments ---------------------------------------------------
GBTDS_COLORS = {
    "TDS_Galactic_Center_Q4": "blue",
    "TDS_Galactic_Center_Q1": "orange",
    "TDS_NGC6334_6357":       "green",
    "TDS_Carina":             "red",
    "TDS_W43":                "magenta",
    "TDS_Serpens_W40":        "cyan",
}
RGPS_TARGET_COLORS = {
    "Disk1":         "yellow",
    "Disk2":         "yellow",
    "Disk3":         "yellow",
    "Disk4":         "yellow",
    "Disk5":         "yellow",
    "Disk6":         "yellow",
    "Disk7":         "yellow",
    "Deep+Spec_W51": "orange",
}

# -- Fetch GBTDS survey field definitions (GBTDS tiling) ----------------------
SURVEY_URL = (
    "https://raw.githubusercontent.com/rachel3834/rgps/"
    "6b45addc01368c68839fb5828cbea222b9898574/"
    "config/rgps_survey_definitions.json"
)
print("Fetching RGPS/GBTDS survey definitions ...")
resp = requests.get(SURVEY_URL, timeout=30)
resp.raise_for_status()
config = resp.json()

# -- Read RGPS ECSV -----------------------------------------------------------
print("Reading roman_gps.sim.ecsv ...")
sim_table    = Table.read(ECSV)
rgps_targets = sorted({str(t) for t in sim_table["TARGET_NAME"]})
print(f"  {len(sim_table):,} rows, {len(rgps_targets)} targets: {rgps_targets}")

# -- Build GBTDS .reg ---------------------------------------------------------
print("\nBuilding GBTDS footprint ...")
gbtds_fields  = dedup_fields(config["time_domain"])
gbtds_entries = build_gbtds_entries(gbtds_fields, GBTDS_COLORS)
print("Writing GBTDS reg ...")
write_reg("/Users/adam/Downloads/roman_gbtds_footprint.reg", gbtds_entries)

# -- Build RGPS .reg ----------------------------------------------------------
print("\nBuilding RGPS footprint ...")
rgps_entries = build_rgps_entries(sim_table, rgps_targets, RGPS_TARGET_COLORS)
print("Writing RGPS reg ...")
write_reg("/Users/adam/Downloads/roman_rgps_footprint.reg", rgps_entries)

# -- Combined .reg ------------------------------------------------------------
print("\nWriting combined reg ...")
write_reg("/Users/adam/Downloads/roman_gbtds_rgps_combined.reg",
          gbtds_entries + rgps_entries)

print("\nDone. Files in ~/Downloads/:")
for fname in ["roman_gbtds_footprint.reg",
              "roman_rgps_footprint.reg",
              "roman_gbtds_rgps_combined.reg"]:
    p = Path(f"/Users/adam/Downloads/{fname}")
    print(f"  {fname}: {p.read_text().count(chr(10))} lines, {p.stat().st_size//1024} KB")

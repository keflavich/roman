#!/usr/bin/env python3
"""
generate_gc_html.py

Generate roman_footprint_gc.html — Roman WFI footprints centred on the
Galactic Center, combining:

  · GBTDS  (Galactic Bulge Time Domain Survey) — 7 hardcoded tile centres from
             arXiv:2505.10574 Table 9, two seasons PA≈90.6° and PA≈270.6°
  · RGPS   (Roman Galactic Plane Survey) from roman_gps.sim.ecsv:
       - RGPS Time-Domain (TDS_* targets)
       - RGPS Bulge, Disk, Deep+Spec, …

PA fix: the ECSV contains a mix of PA=0° and PA=270° entries for some targets
(different sun-avoidance windows).  We select the sky positions from the modal PA,
then display RGPS tiles at gal_plane_pa() (galactic-plane orientation) for a
consistent footprint orientation.

SCA corners: attitude_matrix(V2Ref, V3Ref, ra, dec, pa) — pysiaf Roman SIAF.
"""

import json
import collections
import numpy as np
import pysiaf
from pysiaf.utils.rotations import attitude_matrix
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

ECSV = (
    "/Users/adam/repos/roman_notebooks/notebooks/"
    "footprint_visualization/aux_data/roman_gps.sim.ecsv"
)
OUT = "/Users/adam/work/roman/roman_footprint_gc.html"

# ── pysiaf ────────────────────────────────────────────────────────────────────
print("Loading Roman SIAF …")
rsiaf   = pysiaf.Siaf("Roman")
wfi_cen = rsiaf["WFI_CEN"]
V2Ref   = wfi_cen.V2Ref
V3Ref   = wfi_cen.V3Ref
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]
print(f"  WFI_CEN  V2={V2Ref:+.1f}\"  V3={V3Ref:+.1f}\"")


def sca_polygons(ra, dec, pa):
    attmat = attitude_matrix(V2Ref, V3Ref, ra, dec, pa)
    polys  = []
    for s in sensors:
        s.set_attitude_matrix(attmat)
        c = s.corners("sky")
        polys.append([[round(float(c[0][i]), 5), round(float(c[1][i]), 5)]
                      for i in range(4)])
    return polys


# ── Galactic-plane PA helper (for GBTDS tiling) ───────────────────────────────
def gal_plane_pa(ra, dec):
    """PA (E of celestial N) of galactic longitude direction at (ra, dec)."""
    c  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    c2 = SkyCoord(l=c.galactic.l + 0.2*u.deg, b=c.galactic.b,
                  frame=Galactic()).icrs
    return c.position_angle(c2).deg


# ── ECSV helpers ──────────────────────────────────────────────────────────────
_BP_WAVE = {"F062": 0.62, "F087": 0.87, "F106": 1.06, "F129": 1.29,
            "F146": 1.46, "F158": 1.58, "F184": 1.84, "F213": 2.13,
            "GRISM": 1.50, "PRISM": 1.50}


def unique_pointings(tbl, target):
    """
    Return sorted (ra, dec) tuples of unique sky positions.

    Strategy:
    1. Select the longest-wavelength imaging bandpass.
    2. Filter to the most common PA (modal PA) — avoids mixing positions from
       different sun-avoidance windows (which cover the same sky at different rolls).
    3. Return unique (ra, dec) positions; seasonal PAs (PA_SPRING / PA_AUTUMN)
       are applied at polygon-computation time.
    """
    sub = tbl[tbl["TARGET_NAME"] == target]
    # 1. Longest bandpass
    bps = sorted(set(str(b) for b in sub["BANDPASS"]),
                 key=lambda x: _BP_WAVE.get(x, 0.0))
    sub = sub[sub["BANDPASS"] == bps[-1]]
    # 2. Modal PA
    pa_vals  = [round(float(r["PA"]), 1) for r in sub]
    modal_pa = collections.Counter(pa_vals).most_common(1)[0][0]
    sub      = [r for r, pa in zip(sub, pa_vals) if pa == modal_pa]
    # 3. Unique (ra, dec) positions
    pts = {}
    for r in sub:
        pts[(round(float(r["RA"]),  6),
             round(float(r["DEC"]), 6))] = True
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


# ── Target metadata ───────────────────────────────────────────────────────────
# GBTDS — Galactic Bulge Time Domain Survey
# 7 tiles × 2 seasons (arXiv:2505.10574 Table 9)
# on=True → visible in the initial ~12° GC view
GBTDS_META = {
    "GBTDS_Spring": {"color": "#4488ff", "label": "GBTDS Spring (PA≈90°)",  "on": True},
    "GBTDS_Autumn": {"color": "#ff8844", "label": "GBTDS Autumn (PA≈271°)", "on": True},
}

# RGPS Time-Domain (TDS_* from roman_gps.sim.ecsv)
RGPS_TDS_META = {
    "TDS_Galactic_Center_Lneg":       {"color": "#ff7777", "label": "GC L<0",          "on": True},
    "TDS_Galactic_Center_Lpos":       {"color": "#ffbb55", "label": "GC L>0",          "on": True},
    "TDS_GC_Neg_High+Hourly-Cadence": {"color": "#ff9999", "label": "GC L<0 (h-cad)", "on": True},
    "TDS_GC_Pos_High+Hourly-Cadence": {"color": "#ffcc77", "label": "GC L>0 (h-cad)", "on": True},
    "TDS_NGC6334_NGC6357":            {"color": "#66dd66", "label": "NGC 6334/6357",   "on": False},
    "TDS_NGC6334_NGC6357_High-Cad":   {"color": "#99ff77", "label": "NGC 6334 (h-cad)","on": False},
    "TDS_W43":                        {"color": "#ff66ff", "label": "W43",             "on": False},
    "TDS_W43_High-Cadence":           {"color": "#ffaaff", "label": "W43 (h-cad)",     "on": False},
    "TDS_Carina_High-Cadence":        {"color": "#66aaff", "label": "Carina (h-cad)",  "on": False},
    "TDS_Carina_Region":              {"color": "#99ccff", "label": "Carina Region",   "on": False},
}

# RGPS — everything else in the ECSV
RGPS_META = {
    "Bulge1_Bpos":  {"color": "#ffe070", "label": "Bulge 1 b+",  "on": True,  "group": "bulge"},
    "Bulge2_Bpos":  {"color": "#ffd050", "label": "Bulge 2 b+",  "on": True,  "group": "bulge"},
    "Bulge3_Bpos":  {"color": "#ffc040", "label": "Bulge 3 b+",  "on": True,  "group": "bulge"},
    "Bulge4_Bneg":  {"color": "#ffb030", "label": "Bulge 4 b−",  "on": True,  "group": "bulge"},
    "Bulge5_Bneg":  {"color": "#ffa020", "label": "Bulge 5 b−",  "on": True,  "group": "bulge"},
    "Bulge6_Bneg":  {"color": "#ff9010", "label": "Bulge 6 b−",  "on": True,  "group": "bulge"},
    "Bulge7_Bpos":  {"color": "#ff7800", "label": "Bulge 7 b+",  "on": True,  "group": "bulge"},
    "Bulge8_BNeg":  {"color": "#ff6000", "label": "Bulge 8 b−",  "on": True,  "group": "bulge"},
    "Disk1_Carina": {"color": "#80ffee", "label": "Disk 1 Carina","on": False, "group": "disk"},
    "Disk2":        {"color": "#60eedd", "label": "Disk 2",        "on": False, "group": "disk"},
    "Disk3":        {"color": "#40ddcc", "label": "Disk 3",        "on": False, "group": "disk"},
    "Disk4":        {"color": "#20ccbb", "label": "Disk 4",        "on": False, "group": "disk"},
    "Disk5":        {"color": "#00bbaa", "label": "Disk 5",        "on": False, "group": "disk"},
    "Disk6":        {"color": "#00aa99", "label": "Disk 6",        "on": False, "group": "disk"},
    "Disk7":        {"color": "#009988", "label": "Disk 7",        "on": False, "group": "disk"},
    "Deep+Spec_ASSC_85":           {"color": "#aad4ff", "label": "Deep ASSC 85",      "on": False, "group": "deep"},
    "Deep+Spec_Acrux":             {"color": "#99c4ff", "label": "Deep Acrux",         "on": False, "group": "deep"},
    "Deep+Spec_G333":              {"color": "#88b4ff", "label": "Deep G333",          "on": False, "group": "deep"},
    "Deep+Spec_M17_Omega":         {"color": "#77a4ff", "label": "Deep M17/Omega",     "on": False, "group": "deep"},
    "Deep+Spec_NGC3324_Carina":    {"color": "#6694ff", "label": "Deep NGC 3324",      "on": False, "group": "deep"},
    "Deep+Spec_NGC5269+NGC5281":   {"color": "#5584ff", "label": "Deep NGC 5269/81",   "on": False, "group": "deep"},
    "Deep+Spec_NGC6357_Lobster":   {"color": "#bbdeff", "label": "Deep NGC 6357",      "on": False, "group": "deep"},
    "Deep+Spec_Teutsch_84":        {"color": "#ccdfff", "label": "Deep Teutsch 84",    "on": False, "group": "deep"},
    "Deep+Spec_Trumpler_35":       {"color": "#dde8ff", "label": "Deep Trumpler 35",   "on": False, "group": "deep"},
    "Deep+Spec_VVV_CL001_UKS_1":   {"color": "#ddf4ff", "label": "Deep VVV/UKS 1",    "on": False, "group": "deep"},
    "Deep+Spec_W40":               {"color": "#aaf0ee", "label": "Deep W40",           "on": False, "group": "deep"},
    "Deep+Spec_W44":               {"color": "#99eecc", "label": "Deep W44",           "on": False, "group": "deep"},
    "Deep+Spec_W51":               {"color": "#88eebb", "label": "Deep W51",           "on": False, "group": "deep"},
    "Deep+Spec_Window_319.5_-0.2": {"color": "#77ddaa", "label": "Deep Win 319.5",     "on": False, "group": "deep"},
    "Deep+Spec_Window_355_-0.3":   {"color": "#aaffdd", "label": "Deep Win 355",       "on": False, "group": "deep"},
    "Serpens_South": {"color": "#cc88ff", "label": "Serpens South", "on": False, "group": "other"},
}

# ── Compute GBTDS tiles (arXiv:2505.10574, Table 9) ─────────────────────────
# Exactly 7 tile centres, two observing seasons
_GBTDS_TILES = {
    'Tile 1':       (-0.62, -1.20, 267.2142, -30.0898),
    'Tile 2':       (-0.21, -1.20, 267.4568, -29.7391),
    'Tile 3 (ref)': ( 0.20, -1.20, 267.6974, -29.3884),
    'Tile 4':       ( 0.60, -1.20, 267.9364, -29.0373),
    'Tile 5':       ( 1.01, -1.20, 268.1742, -28.6854),
    'Tile 6':       ( 1.42, -1.20, 268.4104, -28.3331),
    'Tile 7 (GC)':  ( 0.00, -0.12, 266.5270, -29.0013),
}
PA_SPRING = 90.6   # V3 axis ~East; WFI long axis along galactic longitude
PA_AUTUMN = 270.6  # V3 axis ~West; mosaic rotated 180°

print("Computing GBTDS footprints …")
gbtds_data = {}
for season, pa in [("GBTDS_Spring", PA_SPRING), ("GBTDS_Autumn", PA_AUTUMN)]:
    tiles = []
    for tile_name, (l_deg, b_deg, ra_deg, dec_deg) in _GBTDS_TILES.items():
        tiles.append({
            "ra":   round(ra_deg, 5),
            "dec":  round(dec_deg, 5),
            "l":    round(l_deg, 4),
            "b":    round(b_deg, 4),
            "pa":   pa,
            "name": tile_name,
            "polygons": sca_polygons(ra_deg, dec_deg, pa),
        })
    gbtds_data[season] = tiles
    print(f"  [{season}] 7 tiles × 18 SCAs = {7*18} polygons")

# ── Read ECSV and compute RGPS polygons ───────────────────────────────────────
print("Reading roman_gps.sim.ecsv …")
sim_table   = Table.read(ECSV)
all_targets = sorted(set(str(x) for x in sim_table["TARGET_NAME"]))
print(f"  {len(sim_table):,} rows · {len(all_targets)} targets")

rgps_tds_targets = [t for t in all_targets if t.startswith("TDS_")]
rgps_targets     = [t for t in all_targets if not t.startswith("TDS_")]
print(f"  RGPS-TDS: {len(rgps_tds_targets)} targets   RGPS: {len(rgps_targets)} targets")


def compute_tiles(targets, label):
    """Compute SCA polygon tiles for a list of ECSV targets (both seasons)."""
    data = {}
    for tname in targets:
        all_pts = unique_pointings(sim_table, tname)
        pts     = dedup_pointings(all_pts, sep=7.5)
        tiles   = []
        for i, (ra, dec) in enumerate(pts):
            if (i + 1) % 100 == 0:
                print(f"  {tname}: {i+1}/{len(pts)} …")
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            gal   = coord.galactic
            tiles.append({
                "ra":     ra,  "dec": dec,
                "l":      round(float(gal.l.deg), 4),
                "b":      round(float(gal.b.deg), 4),
                "polys_s": sca_polygons(ra, dec, PA_SPRING),
                "polys_a": sca_polygons(ra, dec, PA_AUTUMN),
            })
        data[tname] = tiles
        print(f"  [{label}] {tname}: {len(all_pts)} → {len(pts)} pts "
              f"→ {len(pts)*18}×2 season polygons")
    return data


print("Computing RGPS Time-Domain footprints …")
rgps_tds_data = compute_tiles(rgps_tds_targets, "RGPS-TDS")

print("Computing RGPS footprints (may take a few minutes) …")
rgps_data = compute_tiles(rgps_targets, "RGPS")

# ── Totals ────────────────────────────────────────────────────────────────────
n_gbtds    = sum(len(v) for v in gbtds_data.values())
n_rgps_tds = sum(len(v) for v in rgps_tds_data.values())
n_rgps     = sum(len(v) for v in rgps_data.values())
n_total_polys = n_gbtds * 18 + (n_rgps_tds + n_rgps) * 36
print(f"\nGBTDS: {n_gbtds} ptgs  RGPS-TDS: {n_rgps_tds} ptgs  "
      f"RGPS: {n_rgps} ptgs  total: {n_total_polys:,} polygons (RGPS ×2 seasons)")

# ── Serialise ─────────────────────────────────────────────────────────────────
gbtds_js    = json.dumps(gbtds_data,    separators=(',', ':'))
rgps_tds_js = json.dumps(rgps_tds_data, separators=(',', ':'))
rgps_js     = json.dumps(rgps_data,     separators=(',', ':'))

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _id(name):
    return name.replace('+', '-').replace(' ', '-').replace('.', '_')


def make_buttons(data, meta_dict, btn_class):
    html = ""
    for name in data:
        m      = meta_dict.get(name, {"color": "#aaaaaa", "label": name, "on": False})
        active = " active" if m["on"] else ""
        n      = len(data[name])
        html  += (f'<button class="layer-btn {btn_class}{active}" '
                  f'data-layer="{name}" '
                  f'style="--bc:{m["color"]}">'
                  f'{m["label"]} ({n})</button>\n')
    return html


def make_legend(data, meta_dict):
    html = ""
    for name in data:
        m    = meta_dict.get(name, {"color": "#aaaaaa", "label": name})
        html += (f'<div class="swatch" style="background:{m["color"]}"></div>'
                 f'<div class="ll">{m["label"]}</div>\n')
    return html


def js_meta(meta_dict, keys):
    items = []
    for k in keys:
        m     = meta_dict.get(k, {})
        color = json.dumps(m.get("color", "#aaaaaa"))
        on    = "true" if m.get("on", False) else "false"
        items.append(f'  {json.dumps(k)}: {{color:{color},on:{on}}}')
    return "{\n" + ",\n".join(items) + "\n}"


gbtds_btn_html    = make_buttons(gbtds_data,    GBTDS_META,    "gbtds-btn")
rgps_tds_btn_html = make_buttons(rgps_tds_data, RGPS_TDS_META, "rgps-tds-btn")

# RGPS by group
def rgps_buttons_by_group(data, meta_dict):
    groups = {"bulge": [], "disk": [], "deep": [], "other": []}
    for name in data:
        g = meta_dict.get(name, {}).get("group", "other")
        groups[g].append(name)
    sections = []
    labels = {"bulge": "RGPS Bulge", "disk": "RGPS Disk",
              "deep": "RGPS Deep+Spec", "other": "RGPS Other"}
    for gkey in ("bulge", "disk", "deep", "other"):
        names = groups[gkey]
        if not names:
            continue
        btns = make_buttons({n: data[n] for n in names}, meta_dict,
                             f"rgps-btn rgps-{gkey}")
        sections.append((gkey, labels[gkey], btns))
    return sections

rgps_sections = rgps_buttons_by_group(rgps_data, RGPS_META)

gbtds_meta_js    = js_meta(GBTDS_META,    list(gbtds_data.keys()))
rgps_tds_meta_js = js_meta(RGPS_TDS_META, list(rgps_tds_data.keys()))
rgps_meta_js     = js_meta(RGPS_META,     list(rgps_data.keys()))

gbtds_legend    = make_legend(gbtds_data,    GBTDS_META)
rgps_tds_legend = make_legend(rgps_tds_data, RGPS_TDS_META)
rgps_legend     = make_legend(rgps_data,     RGPS_META)

rgps_panel_html = ""
for gkey, glabel, btns in rgps_sections:
    rgps_panel_html += f"""
    <div class="section">
      <div class="section-label">
        {glabel}
        <button class="mini-btn" data-grp="rgps-{gkey}" data-on="1">all</button>
        <button class="mini-btn" data-grp="rgps-{gkey}" data-on="0">none</button>
      </div>
      <div class="btn-row">
        {btns}
      </div>
    </div>"""

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roman WFI — Galactic Center — Aladin Lite</title>
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html,body{{height:100%;margin:0;font-family:system-ui,'Segoe UI',Roboto,sans-serif;background:#000}}
    #aladin{{position:absolute;inset:0}}
    #ui{{
      position:absolute;top:10px;right:10px;z-index:10;width:240px;
      max-height:calc(100vh - 20px);overflow-y:auto;
      background:rgba(12,12,18,.88);color:#e8e8e8;
      border:1px solid rgba(255,255,255,.12);border-radius:8px;font-size:12px;
      backdrop-filter:blur(6px);
    }}
    #ui h3{{
      margin:0;padding:8px 12px;font-size:13px;font-weight:600;
      background:rgba(255,255,255,.06);border-bottom:1px solid rgba(255,255,255,.1);
      position:sticky;top:0;z-index:1;
    }}
    .section{{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.07)}}
    .section-label{{
      font-size:10px;text-transform:uppercase;letter-spacing:.8px;
      color:#888;margin-bottom:5px;display:flex;align-items:center;gap:4px;flex-wrap:wrap;
    }}
    .btn-row{{display:flex;gap:4px;flex-wrap:wrap}}
    button{{
      cursor:pointer;padding:3px 7px;border:1px solid rgba(255,255,255,.18);
      border-radius:4px;background:rgba(255,255,255,.07);color:#ddd;
      font-size:11px;transition:background .15s,color .15s,border-color .15s;
    }}
    button:hover{{background:rgba(255,255,255,.16);color:#fff}}
    button.survey.active{{background:rgba(180,180,255,.2);color:#c0c0ff;border-color:#c0c0ff}}
    button.layer-btn.active{{
      border-color:var(--bc,rgba(255,255,255,.5));color:var(--bc,#fff);
      background:rgba(255,255,255,.1);font-weight:600;
    }}
    .mini-btn{{font-size:9px;padding:1px 5px;color:#999;border-color:rgba(255,255,255,.2)}}
    .mini-btn:hover{{color:#fff}}
    button.season-btn.active{{background:rgba(100,200,255,.2);color:#64c8ff;border-color:#64c8ff}}
    .legend-grid{{display:grid;grid-template-columns:14px 1fr;gap:3px 6px;align-items:center}}
    .swatch{{width:14px;height:3px;border-radius:2px}}
    .ll{{font-size:10px;color:#bbb}}
    #info{{
      position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
      z-index:10;pointer-events:none;background:rgba(10,10,15,.72);color:#aaa;
      padding:4px 14px;border-radius:4px;font-size:11px;
      border:1px solid rgba(255,255,255,.1);white-space:nowrap;
    }}
  </style>
</head>
<body>
  <div id="aladin"></div>

  <div id="ui">
    <h3>Roman WFI — Galactic Center</h3>

    <div class="section">
      <div class="section-label">Background survey</div>
      <div class="btn-row">
        <button class="survey active" data-survey="P/Spitzer/GLIMPSE360">GLIMPSE</button>
        <button class="survey" data-survey="P/2MASS/color">2MASS</button>
        <button class="survey" data-survey="P/allWISE/color">WISE</button>
        <button class="survey" data-survey="P/DSS/color">DSS</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">
        GBTDS (Galactic Bulge Time Domain Survey)
        <button class="mini-btn" data-grp="gbtds-btn" data-on="1">all</button>
        <button class="mini-btn" data-grp="gbtds-btn" data-on="0">none</button>
      </div>
      <div class="btn-row">
        {gbtds_btn_html}
      </div>
    </div>

    <div class="section">
      <div class="section-label">RGPS observing season</div>
      <div class="btn-row">
        <button class="season-btn active" data-season="spring">Spring PA≈90°</button>
        <button class="season-btn" data-season="autumn">Autumn PA≈271°</button>
        <button class="season-btn" data-season="both">Both</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">
        RGPS Time-Domain (TDS) fields
        <button class="mini-btn" data-grp="rgps-tds-btn" data-on="1">all</button>
        <button class="mini-btn" data-grp="rgps-tds-btn" data-on="0">none</button>
      </div>
      <div class="btn-row">
        {rgps_tds_btn_html}
      </div>
    </div>

    {rgps_panel_html}

    <div class="section">
      <div class="section-label">Legend — GBTDS</div>
      <div class="legend-grid">{gbtds_legend}</div>
    </div>

    <div class="section">
      <div class="section-label">JWST Region</div>
      <div class="btn-row">
        <button class="layer-btn active" data-layer="JWST_Target_Area" style="--bc:#ff4444">target_area.reg</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Legend — RGPS TDS</div>
      <div class="legend-grid">{rgps_tds_legend}</div>
    </div>

    <div class="section">
      <div class="section-label">Legend — RGPS</div>
      <div class="legend-grid">{rgps_legend}</div>
      <div style="margin-top:5px;font-size:9px;color:#555;line-height:1.5">
        GBTDS: arXiv:2505.10574 Table 9, 7 tiles × 2 seasons<br>
        RGPS: roman_gps.sim.ecsv (APT), same PA≈90.6°/270.6° as GBTDS<br>
        7.5’ spatial dedup · pysiaf attitude_matrix
      </div>
    </div>

    <div class="section" style="border:0;padding-bottom:9px">
      <button id="dl" style="width:100%;padding:5px">↓ Export pointing centres JSON</button>
    </div>
  </div>

  <div id="info">Roman WFI · GBTDS {n_gbtds} + RGPS-TDS {n_rgps_tds} + RGPS {n_rgps} pointings · {n_total_polys:,} polygons · pysiaf</div>

  <script>
  const GBTDS       = {gbtds_js};
  const RGPS_TDS    = {rgps_tds_js};
  const RGPS        = {rgps_js};
  const GBTDS_META    = {gbtds_meta_js};
  const RGPS_TDS_META = {rgps_tds_meta_js};
  const RGPS_META     = {rgps_meta_js};
  const CUSTOM_META   = {{
  "JWST_Target_Area": {{color:"#ff4444",on:true}}
}};
  const CUSTOM_REGIONS = {{
  "JWST_Target_Area": [
    [266.447262, -28.665625],
    [266.108552, -29.153986],
    [266.082376, -29.185502],
    [266.099365, -29.275829],
    [265.99966, -29.41689],
    [266.048406, -29.437154],
    [266.104287, -29.466578],
    [266.165553, -29.501565],
    [266.272546, -29.427271],
    [266.353777, -29.323741],
    [266.707721, -28.819941],
    [266.725371, -28.761043],
    [267.001371, -28.346998],
    [266.795062, -28.256913],
    [266.597259, -28.53203]
  ]
}};

  let curSeason = 'spring';  // 'spring' | 'autumn' | 'both'
  let aladin;
  const gbtdsOv   = {{}};  // name → single overlay  (GBTDS; tiles have .polygons)
  const customOv  = {{}};  // name → single overlay  (custom static polygons)
  const rgpsTdsOv = {{}};  // name → {{s, a, on}}    (RGPS-TDS; tiles have .polys_s/.polys_a)
  const rgpsOv    = {{}};  // name → {{s, a, on}}

  // GBTDS: single overlay per entry (Spring and Autumn are separate named entries)
  function buildOverlays(data, meta, store, suffix, lw) {{
    for (const [name, tiles] of Object.entries(data)) {{
      const color = (meta[name] || {{}}).color || '#ffffff';
      const ov = A.graphicOverlay({{ color, lineWidth: lw, name: name + suffix }});
      aladin.addOverlay(ov);
      tiles.forEach(tile => tile.polygons.forEach(c => ov.add(A.polygon(c))));
      store[name] = ov;
      if (!(meta[name] || {{}}).on) ov.hide();
    }}
  }}

  function buildCustomOverlays(data, meta, store, suffix, lw) {{
    for (const [name, poly] of Object.entries(data)) {{
      const color = (meta[name] || {{}}).color || '#ffffff';
      const ov = A.graphicOverlay({{ color, lineWidth: lw, name: name + suffix }});
      aladin.addOverlay(ov);
      ov.add(A.polygon(poly));
      store[name] = ov;
      if (!(meta[name] || {{}}).on) ov.hide();
    }}
  }}

  // RGPS/RGPS-TDS: spring + autumn overlays per entry (same PA as GBTDS seasons)
  function buildSeasonOverlays(data, meta, store, suffix, lw) {{
    for (const [name, tiles] of Object.entries(data)) {{
      const color = (meta[name] || {{}}).color || '#ffffff';
      const ovS = A.graphicOverlay({{ color, lineWidth: lw,        name: name + suffix }});
      const ovA = A.graphicOverlay({{ color, lineWidth: lw * 0.65, name: name + suffix + ' (A)' }});
      aladin.addOverlay(ovS);
      aladin.addOverlay(ovA);
      tiles.forEach(tile => {{
        tile.polys_s.forEach(c => ovS.add(A.polygon(c)));
        tile.polys_a.forEach(c => ovA.add(A.polygon(c)));
      }});
      const on = !!(meta[name] || {{}}).on;
      store[name] = {{s: ovS, a: ovA, on}};
      if (on) applySeason(ovS, ovA); else {{ ovS.hide(); ovA.hide(); }}
    }}
  }}

  function applySeason(ovS, ovA) {{
    if (curSeason === 'spring')      {{ ovS.show(); ovA.hide(); }}
    else if (curSeason === 'autumn') {{ ovS.hide(); ovA.show(); }}
    else                             {{ ovS.show(); ovA.show(); }}  // 'both'
  }}

  A.init.then(() => {{
    aladin = A.aladin('#aladin', {{
      survey:   'P/Spitzer/GLIMPSE360',
      target:   '0 0',
      fov:      12,
      cooFrame: 'galactic',
    }});

    buildOverlays(GBTDS,    GBTDS_META,    gbtdsOv,   ' (GBTDS)',    1.8);
    buildCustomOverlays(CUSTOM_REGIONS, CUSTOM_META, customOv, ' (JWST)', 2.2);
    buildSeasonOverlays(RGPS_TDS, RGPS_TDS_META, rgpsTdsOv, ' (RGPS-TDS)', 1.4);
    buildSeasonOverlays(RGPS,     RGPS_META,     rgpsOv,    ' (RGPS)',     1.0);

    // ── Individual layer toggles ──────────────────────────────────
    document.querySelectorAll('.layer-btn').forEach(btn => {{
      btn.addEventListener('click', function() {{
        const name  = this.dataset.layer;
        const gov   = gbtdsOv[name];
        const cov   = customOv[name];
        const rov   = rgpsTdsOv[name] ?? rgpsOv[name];
        const nowOn = !this.classList.contains('active');
        if (gov !== undefined) {{
          nowOn ? gov.show() : gov.hide();
        }} else if (cov !== undefined) {{
          nowOn ? cov.show() : cov.hide();
        }} else if (rov !== undefined) {{
          rov.on = nowOn;
          if (nowOn) applySeason(rov.s, rov.a);
          else {{ rov.s.hide(); rov.a.hide(); }}
        }}
        this.classList.toggle('active', nowOn);
      }});
    }});

    // ── Group all/none buttons ────────────────────────────────────
    document.querySelectorAll('.mini-btn[data-grp]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const grp = btn.dataset.grp;
        const on  = btn.dataset.on === '1';
        document.querySelectorAll(`.layer-btn.${{grp}}`).forEach(lb => {{
          const name = lb.dataset.layer;
          const gov  = gbtdsOv[name];
          const rov  = rgpsTdsOv[name] ?? rgpsOv[name];
          if (gov !== undefined) {{
            on ? gov.show() : gov.hide();
          }} else if (rov !== undefined) {{
            rov.on = on;
            if (on) applySeason(rov.s, rov.a);
            else {{ rov.s.hide(); rov.a.hide(); }}
          }}
          lb.classList.toggle('active', on);
        }});
      }});
    }});

    // ── Season toggle (RGPS) ─────────────────────────────────────
    document.querySelectorAll('.season-btn').forEach(btn => {{
      btn.addEventListener('click', function() {{
        curSeason = this.dataset.season;
        document.querySelectorAll('.season-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        for (const store of [rgpsTdsOv, rgpsOv]) {{
          for (const entry of Object.values(store)) {{
            if (entry.on) applySeason(entry.s, entry.a);
          }}
        }}
      }});
    }});

    // ── Background survey switcher ────────────────────────────────
    document.querySelectorAll('button.survey').forEach(btn => {{
      btn.onclick = () => {{
        document.querySelectorAll('button.survey').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        aladin.setImageSurvey(btn.dataset.survey);
      }};
    }});

    // ── Download ──────────────────────────────────────────────────
    document.getElementById('dl').onclick = () => {{
      const stripG = tiles => tiles.map(({{ra,dec,l,b,pa}}) => ({{ra,dec,l,b,pa}}));
      const stripR = tiles => tiles.map(({{ra,dec,l,b}}) => ({{ra,dec,l,b,pa_s:{PA_SPRING},pa_a:{PA_AUTUMN}}}));
      const out = {{
        gbtds:    Object.fromEntries(Object.entries(GBTDS)   .map(([n,t]) => [n, stripG(t)])),
        rgps_tds: Object.fromEntries(Object.entries(RGPS_TDS).map(([n,t]) => [n, stripR(t)])),
        rgps:     Object.fromEntries(Object.entries(RGPS)    .map(([n,t]) => [n, stripR(t)])),
      }};
      const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'roman_gc_pointings.json';
      document.body.appendChild(a); a.click(); a.remove();
    }};
  }});
  </script>
</body>
</html>
"""

with open(OUT, "w") as f:
    f.write(html)

print(f"\nWritten {OUT}")
kb = len(html.encode()) // 1024
print(f"  File size: {kb} KB")
print(f"  GBTDS:    {n_gbtds} pointing centres ({len(gbtds_data)} fields)")
print(f"  RGPS-TDS: {n_rgps_tds} pointing centres ({len(rgps_tds_data)} targets)")
print(f"  RGPS:     {n_rgps} pointing centres ({len(rgps_data)} targets)")
print(f"  Total:    {n_total_polys:,} SCA polygons")

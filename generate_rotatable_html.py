#!/usr/bin/env python3
"""
generate_rotatable_html.py

Generate roman_footprint_rotatable.html — Roman WFI footprints with a
user-controlled PA slider/text box.  All polygon geometry is computed in
the browser from boresight (ra, dec) + SCA IDL vertices, so the user can
type any PA and see the footprints update instantly.

The JS implements the same attitude_matrix transform as pysiaf:
  1. Build a 3x3 rotation matrix M from (V2Ref, V3Ref, ra, dec, pa)
  2. For each SCA corner (x_idl, y_idl) in arcsec:
       (v2, v3) from IDL → tangent-plane → rotate by M → sky (ra_c, dec_c)
"""

import json, numpy as np, pysiaf, collections
from pysiaf.utils.rotations import attitude_matrix
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

ECSV = (
    "/Users/adam/repos/roman_notebooks/notebooks/"
    "footprint_visualization/aux_data/roman_gps.sim.ecsv"
)
OUT = "/Users/adam/work/roman/roman_footprint_rotatable.html"

# ── pysiaf: extract SCA IDL vertices and reference point ─────────────────────
print("Loading Roman SIAF …")
rsiaf   = pysiaf.Siaf("Roman")
wfi_cen = rsiaf["WFI_CEN"]
V2REF   = float(wfi_cen.V2Ref)   # arcsec
V3REF   = float(wfi_cen.V3Ref)   # arcsec
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]

# Each SCA: 4 V2/V3 corners in telescope frame (arcsec), converted from IDL
SCA_VERTS = []
for s in sensors:
    v2s = []; v3s = []
    for k in range(1, 5):
        xi = float(getattr(s, f"XIdlVert{k}"))
        yi = float(getattr(s, f"YIdlVert{k}"))
        v2, v3 = s.idl_to_tel(xi, yi)
        v2s.append(round(float(v2), 4))
        v3s.append(round(float(v3), 4))
    SCA_VERTS.append({"v2": v2s, "v3": v3s})
print(f"  V2Ref={V2REF:+.1f}\"  V3Ref={V3REF:+.1f}\"  SCAs: {len(SCA_VERTS)}")

# ── ECSV helpers ──────────────────────────────────────────────────────────────
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

def gal_plane_pa(ra, dec):
    c  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    c2 = SkyCoord(l=c.galactic.l + 0.2*u.deg, b=c.galactic.b,
                  frame=Galactic()).icrs
    return round(c.position_angle(c2).deg, 3)

# ── Build pointing-centre dataset ─────────────────────────────────────────────
print("Reading roman_gps.sim.ecsv …")
sim_table   = Table.read(ECSV)
all_targets = sorted(set(str(x) for x in sim_table["TARGET_NAME"]))
print(f"  {len(sim_table):,} rows · {len(all_targets)} targets")

rgps_data = {}
for tname in all_targets:
    all_pts = unique_pointings(sim_table, tname)
    pts     = dedup_pointings(all_pts, sep=7.5)
    tiles   = []
    for ra, dec in pts:
        c     = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        gal   = c.galactic
        tiles.append({
            "ra":     ra,
            "dec":    dec,
            "l":      round(float(gal.l.deg), 4),
            "b":      round(float(gal.b.deg), 4),
            "pa_gal": gal_plane_pa(ra, dec),
        })
    rgps_data[tname] = tiles
    print(f"  {tname}: {len(pts)} pts")

# ── GBTDS hardcoded tiles ─────────────────────────────────────────────────────
_GBTDS_TILES = {
    'Tile 1':       (-0.62, -1.20, 267.2142, -30.0898),
    'Tile 2':       (-0.21, -1.20, 267.4568, -29.7391),
    'Tile 3 (ref)': ( 0.20, -1.20, 267.6974, -29.3884),
    'Tile 4':       ( 0.60, -1.20, 267.9364, -29.0373),
    'Tile 5':       ( 1.01, -1.20, 268.1742, -28.6854),
    'Tile 6':       ( 1.42, -1.20, 268.4104, -28.3331),
    'Tile 7 (GC)':  ( 0.00, -0.12, 266.5270, -29.0013),
}
PA_SPRING = 90.6
PA_AUTUMN = 270.6
gbtds_centers = [
    {"ra": ra, "dec": dec, "l": l, "b": b, "name": name,
     "pa_gal": gal_plane_pa(ra, dec)}
    for name, (l, b, ra, dec) in _GBTDS_TILES.items()
]

# ── Layer metadata ─────────────────────────────────────────────────────────────
GBTDS_META = {
    "color": "#4488ff",
    "label": "GBTDS (7 tiles)",
    "on": True,
}

RGPS_TDS_META = {
    "TDS_Galactic_Center_Lneg":       {"color": "#ff7777", "label": "GC L<0",           "on": False},
    "TDS_Galactic_Center_Lpos":       {"color": "#ffbb55", "label": "GC L>0",           "on": False},
    "TDS_GC_Neg_High+Hourly-Cadence": {"color": "#ff9999", "label": "GC L<0 (h-cad)",  "on": True},
    "TDS_GC_Pos_High+Hourly-Cadence": {"color": "#ffcc77", "label": "GC L>0 (h-cad)",  "on": True},
    "TDS_NGC6334_NGC6357":            {"color": "#66dd66", "label": "NGC 6334/6357",    "on": False},
    "TDS_NGC6334_NGC6357_High-Cad":   {"color": "#99ff77", "label": "NGC 6334 (h-cad)","on": False},
    "TDS_W43":                        {"color": "#ff66ff", "label": "W43",              "on": False},
    "TDS_W43_High-Cadence":           {"color": "#ffaaff", "label": "W43 (h-cad)",      "on": False},
    "TDS_Carina_High-Cadence":        {"color": "#66aaff", "label": "Carina (h-cad)",   "on": False},
    "TDS_Carina_Region":              {"color": "#99ccff", "label": "Carina Region",    "on": False},
}

RGPS_META = {
    "Bulge1_Bpos":  {"color": "#ffe070", "label": "Bulge 1 b+",  "on": False, "group": "bulge"},
    "Bulge2_Bpos":  {"color": "#ffd050", "label": "Bulge 2 b+",  "on": False, "group": "bulge"},
    "Bulge3_Bpos":  {"color": "#ffc040", "label": "Bulge 3 b+",  "on": False, "group": "bulge"},
    "Bulge4_Bneg":  {"color": "#ffb030", "label": "Bulge 4 b−",  "on": False, "group": "bulge"},
    "Bulge5_Bneg":  {"color": "#ffa020", "label": "Bulge 5 b−",  "on": False, "group": "bulge"},
    "Bulge6_Bneg":  {"color": "#ff9010", "label": "Bulge 6 b−",  "on": False, "group": "bulge"},
    "Bulge7_Bpos":  {"color": "#ff7800", "label": "Bulge 7 b+",  "on": False, "group": "bulge"},
    "Bulge8_BNeg":  {"color": "#ff6000", "label": "Bulge 8 b−",  "on": False, "group": "bulge"},
    "Disk1_Carina": {"color": "#80ffee", "label": "Disk 1 Carina","on": False, "group": "disk"},
    "Disk2":        {"color": "#60eedd", "label": "Disk 2",        "on": False, "group": "disk"},
    "Disk3":        {"color": "#40ddcc", "label": "Disk 3",        "on": False, "group": "disk"},
    "Disk4":        {"color": "#20ccbb", "label": "Disk 4",        "on": False, "group": "disk"},
    "Disk5":        {"color": "#00bbaa", "label": "Disk 5",        "on": False, "group": "disk"},
    "Disk6":        {"color": "#00aa99", "label": "Disk 6",        "on": False, "group": "disk"},
    "Disk7":        {"color": "#009988", "label": "Disk 7",        "on": False, "group": "disk"},
    "Deep+Spec_ASSC_85":           {"color": "#aad4ff", "label": "Deep ASSC 85",       "on": False, "group": "deep"},
    "Deep+Spec_Acrux":             {"color": "#99c4ff", "label": "Deep Acrux",          "on": False, "group": "deep"},
    "Deep+Spec_G333":              {"color": "#88b4ff", "label": "Deep G333",           "on": False, "group": "deep"},
    "Deep+Spec_M17_Omega":         {"color": "#77a4ff", "label": "Deep M17/Omega",      "on": False, "group": "deep"},
    "Deep+Spec_NGC3324_Carina":    {"color": "#6694ff", "label": "Deep NGC 3324",       "on": False, "group": "deep"},
    "Deep+Spec_NGC5269+NGC5281":   {"color": "#5584ff", "label": "Deep NGC 5269/81",    "on": False, "group": "deep"},
    "Deep+Spec_NGC6357_Lobster":   {"color": "#bbdeff", "label": "Deep NGC 6357",       "on": False, "group": "deep"},
    "Deep+Spec_Teutsch_84":        {"color": "#ccdfff", "label": "Deep Teutsch 84",     "on": False, "group": "deep"},
    "Deep+Spec_Trumpler_35":       {"color": "#dde8ff", "label": "Deep Trumpler 35",    "on": False, "group": "deep"},
    "Deep+Spec_VVV_CL001_UKS_1":   {"color": "#ddf4ff", "label": "Deep VVV/UKS 1",     "on": False, "group": "deep"},
    "Deep+Spec_W40":               {"color": "#aaf0ee", "label": "Deep W40",            "on": False, "group": "deep"},
    "Deep+Spec_W44":               {"color": "#99eecc", "label": "Deep W44",            "on": False, "group": "deep"},
    "Deep+Spec_W51":               {"color": "#88eebb", "label": "Deep W51",            "on": False, "group": "deep"},
    "Deep+Spec_Window_319.5_-0.2": {"color": "#77ddaa", "label": "Deep Win 319.5",      "on": False, "group": "deep"},
    "Deep+Spec_Window_355_-0.3":   {"color": "#aaffdd", "label": "Deep Win 355",        "on": False, "group": "deep"},
    "Serpens_South": {"color": "#cc88ff", "label": "Serpens South", "on": False, "group": "other"},
}

# ── Serialise ─────────────────────────────────────────────────────────────────
sca_verts_js    = json.dumps(SCA_VERTS,     separators=(',', ':'))
gbtds_js        = json.dumps(gbtds_centers, separators=(',', ':'))
rgps_js         = json.dumps(rgps_data,     separators=(',', ':'))

tds_names    = [t for t in all_targets if t.startswith("TDS_")]
rgps_names   = [t for t in all_targets if not t.startswith("TDS_")]

def js_meta(meta_dict, keys):
    items = []
    for k in keys:
        m     = meta_dict.get(k, {})
        color = json.dumps(m.get("color", "#aaaaaa"))
        on    = "true" if m.get("on", False) else "false"
        items.append(f'  {json.dumps(k)}: {{color:{color},on:{on}}}')
    return "{\n" + ",\n".join(items) + "\n}"

rgps_tds_meta_js = js_meta(RGPS_TDS_META, tds_names)
rgps_meta_js     = js_meta(RGPS_META,     rgps_names)

def make_buttons(names, meta_dict, btn_class):
    html = ""
    for name in names:
        m      = meta_dict.get(name, {"color": "#aaaaaa", "label": name, "on": False})
        active = " active" if m.get("on") else ""
        html  += (f'<button class="layer-btn {btn_class}{active}" '
                  f'data-layer="{name}" style="--bc:{m["color"]}">'
                  f'{m["label"]}</button>\n')
    return html

def rgps_buttons_by_group(names, meta_dict):
    groups = {"bulge": [], "disk": [], "deep": [], "other": []}
    for name in names:
        g = meta_dict.get(name, {}).get("group", "other")
        groups[g].append(name)
    sections = []
    labels = {"bulge": "RGPS Bulge", "disk": "RGPS Disk",
              "deep": "RGPS Deep+Spec", "other": "RGPS Other"}
    for gkey in ("bulge", "disk", "deep", "other"):
        ns = groups[gkey]
        if not ns:
            continue
        btns = make_buttons(ns, meta_dict, f"rgps-btn rgps-{gkey}")
        sections.append((gkey, labels[gkey], btns))
    return sections

tds_btn_html   = make_buttons(tds_names, RGPS_TDS_META, "rgps-tds-btn")
rgps_sections  = rgps_buttons_by_group(rgps_names, RGPS_META)

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

n_pts = sum(len(v) for v in rgps_data.values()) + len(gbtds_centers)

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roman WFI — Rotatable Footprints — Aladin Lite</title>
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html,body{{height:100%;margin:0;font-family:system-ui,'Segoe UI',Roboto,sans-serif;background:#000}}
    #aladin{{position:absolute;inset:0}}
    #ui{{
      position:absolute;top:10px;right:10px;z-index:10;width:256px;
      max-height:calc(100vh - 20px);overflow-y:auto;
      background:rgba(12,12,18,.90);color:#e8e8e8;
      border:1px solid rgba(255,255,255,.12);border-radius:8px;font-size:12px;
      backdrop-filter:blur(6px);
    }}
    #ui h3{{margin:0;padding:8px 12px;font-size:13px;font-weight:600;
      background:rgba(255,255,255,.06);border-bottom:1px solid rgba(255,255,255,.1);
      position:sticky;top:0;z-index:1}}
    .section{{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.07)}}
    .section-label{{font-size:10px;text-transform:uppercase;letter-spacing:.8px;
      color:#888;margin-bottom:5px;display:flex;align-items:center;gap:4px;flex-wrap:wrap}}
    .btn-row{{display:flex;gap:4px;flex-wrap:wrap}}
    button{{cursor:pointer;padding:3px 7px;border:1px solid rgba(255,255,255,.18);
      border-radius:4px;background:rgba(255,255,255,.07);color:#ddd;
      font-size:11px;transition:background .15s,color .15s,border-color .15s}}
    button:hover{{background:rgba(255,255,255,.16);color:#fff}}
    button.survey.active{{background:rgba(180,180,255,.2);color:#c0c0ff;border-color:#c0c0ff}}
    button.layer-btn.active{{border-color:var(--bc,rgba(255,255,255,.5));color:var(--bc,#fff);
      background:rgba(255,255,255,.1);font-weight:600}}
    .mini-btn{{font-size:9px;padding:1px 5px;color:#999;border-color:rgba(255,255,255,.2)}}
    .mini-btn:hover{{color:#fff}}

    /* PA control */
    .pa-grid{{display:grid;grid-template-columns:1fr auto auto;gap:4px;align-items:center}}
    .pa-grid label{{font-size:10px;color:#aaa}}
    #pa-val{{
      width:56px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);
      border-radius:4px;color:#e8e8e8;font-size:12px;padding:2px 5px;text-align:right
    }}
    #pa-val:focus{{outline:none;border-color:#64c8ff}}
    #pa-range{{width:100%;accent-color:#64c8ff;margin-top:3px}}
    .pa-presets{{display:flex;gap:3px;flex-wrap:wrap;margin-top:4px}}
    .pa-preset{{font-size:10px;padding:2px 6px;border-radius:3px;
      background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.15)}}
    .pa-preset:hover{{background:rgba(100,200,255,.15);border-color:#64c8ff;color:#64c8ff}}
    #apply-btn{{
      width:100%;padding:4px;margin-top:5px;
      background:rgba(100,200,255,.18);border-color:#64c8ff;
      color:#64c8ff;font-weight:600;border-radius:4px
    }}
    #apply-btn:hover{{background:rgba(100,200,255,.32)}}

    /* GBTDS toggle */
    #gbtds-row{{display:flex;gap:4px;align-items:center}}
    #gbtds-btn.active{{border-color:#4488ff;color:#4488ff;background:rgba(68,136,255,.12);font-weight:600}}

    #info{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
      z-index:10;pointer-events:none;background:rgba(10,10,15,.72);color:#aaa;
      padding:4px 14px;border-radius:4px;font-size:11px;
      border:1px solid rgba(255,255,255,.1);white-space:nowrap}}
    #status{{font-size:10px;color:#888;margin-top:3px;min-height:14px;text-align:center}}

    /* Edit mode */
    #edit-btn{{width:100%;padding:4px;margin-top:4px;font-weight:600;border-radius:4px;
      background:rgba(255,200,80,.10);border-color:rgba(255,200,80,.4);color:#ffc850}}
    #edit-btn:hover{{background:rgba(255,200,80,.22)}}
    #edit-btn.active{{background:rgba(255,200,80,.28);border-color:#ffc850;color:#ffe090}}
    #edit-panel{{display:none;margin-top:6px;padding-top:6px;
      border-top:1px solid rgba(255,200,80,.2)}}
    #edit-panel.visible{{display:block}}
    .edit-hint{{font-size:9px;color:#aaa;line-height:1.5;margin-bottom:6px}}
    #selected-info{{display:none;margin-bottom:6px;padding:6px;
      background:rgba(255,255,255,.05);border-radius:4px;border:1px solid rgba(255,200,80,.25)}}
    #selected-info.visible{{display:block}}
    .selected-lbl{{font-size:10px;font-weight:600;color:#ffc850;margin-bottom:5px}}
    .coord-row{{display:grid;grid-template-columns:28px 1fr auto;gap:3px;align-items:center;margin-bottom:3px}}
    .coord-row label{{font-size:10px;color:#aaa}}
    .coord-input{{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);
      border-radius:3px;color:#e8e8e8;font-size:11px;padding:2px 4px;width:100%;text-align:right}}
    .coord-input:focus{{outline:none;border-color:#ffe000}}
    .edit-btn-row{{display:flex;gap:4px;margin-top:4px}}
    #move-btn{{flex:1;padding:3px;font-size:11px;border-radius:3px;
      background:rgba(255,200,80,.15);border-color:rgba(255,200,80,.5);color:#ffc850}}
    #move-btn:hover{{background:rgba(255,200,80,.28)}}
    #reset-this-btn{{flex:1;padding:3px;font-size:11px;border-radius:3px;
      background:rgba(255,120,120,.10);border-color:rgba(255,120,120,.4);color:#ff9090}}
    #reset-this-btn:hover{{background:rgba(255,120,120,.22)}}
    #export-btn{{width:100%;padding:3px;font-size:11px;border-radius:4px;margin-bottom:4px;
      background:rgba(100,255,150,.10);border-color:rgba(100,255,150,.4);color:#64ff96}}
    #export-btn:hover{{background:rgba(100,255,150,.22)}}
    #reset-btn{{width:100%;padding:3px;font-size:11px;border-radius:4px;
      background:rgba(255,100,100,.10);border-color:rgba(255,100,100,.4);color:#ff6464}}
    #reset-btn:hover{{background:rgba(255,100,100,.22)}}    #load-btn{{width:100%;padding:3px;font-size:11px;border-radius:4px;margin-bottom:4px;
      background:rgba(255,200,80,.10);border-color:rgba(255,200,80,.4);color:#ffc850}}
    #load-btn:hover{{background:rgba(255,200,80,.22)}}    body.edit-mode #aladin{{cursor:crosshair}}
  </style>
</head>
<body>
<div id="aladin"></div>

<div id="ui">
  <h3>Roman WFI — Rotatable Footprints</h3>

  <!-- Background survey -->
  <div class="section">
    <div class="section-label">Background survey</div>
    <div class="btn-row">
      <button class="survey active" data-survey="P/GLIMPSE360">GLIMPSE</button>
      <button class="survey" data-survey="P/2MASS/color">2MASS</button>
      <button class="survey" data-survey="P/allWISE/color">WISE</button>
      <button class="survey" data-survey="P/DSS2/color">DSS</button>
    </div>
  </div>

  <!-- PA control -->
  <div class="section">
    <div class="section-label">Position angle (V3, deg E of N)</div>
    <div class="pa-grid">
      <label>PA =</label>
      <input id="pa-val" type="number" min="0" max="360" step="0.1" value="90.6">
      <span style="font-size:10px;color:#666">°</span>
    </div>
    <input id="pa-range" type="range" min="0" max="360" step="0.5" value="90.6">
    <div class="pa-presets">
      <button class="pa-preset" data-pa="90.6">Spring (90.6°)</button>
      <button class="pa-preset" data-pa="270.6">Autumn (270.6°)</button>
      <button class="pa-preset" data-pa="0">0°</button>
      <button class="pa-preset" data-pa="45">45°</button>
      <button class="pa-preset" data-pa="180">180°</button>
    </div>
    <button id="apply-btn">↺ Apply to all visible layers</button>
    <div id="status"></div>
  </div>

  <!-- GBTDS -->
  <div class="section">
    <div class="section-label">GBTDS (7 tiles)</div>
    <div id="gbtds-row">
      <button id="gbtds-btn" class="layer-btn active" style="--bc:#4488ff">GBTDS (7)</button>
      <span style="font-size:9px;color:#666;flex:1;text-align:right">uses current PA</span>
    </div>
  </div>

  <div class="section">
    <div class="section-label">JWST Region</div>
    <div class="btn-row">
      <button id="jwst-btn" class="layer-btn active" data-layer="JWST_Target_Area" style="--bc:#ff4444">target_area.reg</button>
    </div>
  </div>

  <!-- RGPS TDS -->
  <div class="section">
    <div class="section-label">
      RGPS Time-Domain (TDS) fields
      <button class="mini-btn" data-grp="rgps-tds-btn" data-on="1">all</button>
      <button class="mini-btn" data-grp="rgps-tds-btn" data-on="0">none</button>
    </div>
    <div class="btn-row">
      {tds_btn_html}
    </div>
  </div>

  {rgps_panel_html}

  <!-- Edit mode -->
  <div class="section" style="border:0;padding-bottom:9px">
    <button id="edit-btn">✎ Edit pointings</button>
    <div id="edit-panel">
      <div class="edit-hint">Click any WFI square to select that pointing.</div>
      <div id="selected-info">
        <div id="selected-label" class="selected-lbl">—</div>
        <div class="coord-row">
          <label>RA</label>
          <input id="edit-ra" class="coord-input" type="number" step="0.0001" min="0" max="360">
          <span style="font-size:9px;color:#666">°</span>
        </div>
        <div class="coord-row">
          <label>Dec</label>
          <input id="edit-dec" class="coord-input" type="number" step="0.0001" min="-90" max="90">
          <span style="font-size:9px;color:#666">°</span>
        </div>
        <div class="coord-row">
          <label>PA</label>
          <input id="edit-pa" class="coord-input" type="number" step="0.1" min="0" max="360">
          <span style="font-size:9px;color:#666">°</span>
        </div>
        <div class="edit-btn-row">
          <button id="move-btn">↵ Move</button>
          <button id="reset-this-btn">↺ Reset</button>
        </div>
      </div>
      <button id="load-btn" style="margin-top:4px">⬆ Load positions (JSON)</button>
      <input type="file" id="load-file" accept=".json" style="display:none">
      <button id="export-btn" style="margin-top:4px">⬇ Export positions (JSON)</button>
      <button id="reset-btn" style="margin-top:4px">↺ Reset all to defaults</button>
      <div id="edit-status" style="font-size:9px;color:#888;margin-top:3px;min-height:12px"></div>
    </div>
    <div style="font-size:9px;color:#555;line-height:1.6;margin-top:6px">
      {n_pts} pointing centres · pysiaf attitude_matrix in JS<br>
      PA slider rotates all active layers simultaneously<br>
      Each pointing: 18 SCA polygons recomputed on demand
    </div>
  </div>
</div>

<div id="info">Roman WFI · {n_pts} pointing centres · PA-rotatable · pysiaf geometry</div>

<script>
// ─── Embedded data ────────────────────────────────────────────────────────────
// WFI reference point (arcsec)
const V2REF = {V2REF};
const V3REF = {V3REF};

// 18 SCA IDL corner vertices (arcsec in the ideal focal plane)
const SCA_VERTS = {sca_verts_js};

// GBTDS boresights (ra, dec in deg) — 7 tiles
const GBTDS_CENTERS = {gbtds_js};

// RGPS + RGPS-TDS boresights keyed by target name
const RGPS = {rgps_js};

const RGPS_TDS_META = {rgps_tds_meta_js};
const RGPS_META     = {rgps_meta_js};
const JWST_TARGET_AREA = [[266.447262,-28.665625],[266.108552,-29.153986],[266.082376,-29.185502],[266.099365,-29.275829],[265.99966,-29.41689],[266.048406,-29.437154],[266.104287,-29.466578],[266.165553,-29.501565],[266.272546,-29.427271],[266.353777,-29.323741],[266.707721,-28.819941],[266.725371,-28.761043],[267.001371,-28.346998],[266.795062,-28.256913],[266.597259,-28.53203]];

// ─── pysiaf attitude_matrix reimplemented in JS ───────────────────────────────
// All angles in radians unless noted.
// Implements the same algorithm as pysiaf.utils.rotations.attitude_matrix:
// Attitude matrix: M = Rz(+ra) · Ry(-dec) · Rx(-pa) · Ry(+v3ref) · Rz(-v2ref)
// Verified to match pysiaf.utils.rotations.attitude_matrix exactly.
// Inputs: v2_as, v3_as in arcsec (reference point); ra, dec, pa in degrees.

function deg2rad(d) {{ return d * Math.PI / 180; }}
function rad2deg(r) {{ return r * 180 / Math.PI; }}

function Rx(a) {{
  const c=Math.cos(a), s=Math.sin(a);
  return [[1,0,0],[0,c,-s],[0,s,c]];
}}
function Ry(a) {{
  const c=Math.cos(a), s=Math.sin(a);
  return [[c,0,s],[0,1,0],[-s,0,c]];
}}
function Rz(a) {{
  const c=Math.cos(a), s=Math.sin(a);
  return [[c,-s,0],[s,c,0],[0,0,1]];
}}
function matmul(A, B) {{
  const R = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i=0;i<3;i++) for (let j=0;j<3;j++) for (let k=0;k<3;k++)
    R[i][j] += A[i][k]*B[k][j];
  return R;
}}
function matvec(M, v) {{
  return [
    M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
    M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
    M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2],
  ];
}}

// Build attitude matrix for a pointing (ra, dec, pa in deg; v2ref, v3ref in arcsec)
// Formula: M = Rz(+ra) · Ry(-dec) · Rx(-pa) · Ry(+v3r) · Rz(-v2r)
function attitudeMatrix(v2_as, v3_as, ra_deg, dec_deg, pa_deg) {{
  const v2 = deg2rad(v2_as / 3600);
  const v3 = deg2rad(v3_as / 3600);
  const ra  = deg2rad(ra_deg);
  const dec = deg2rad(dec_deg);
  const pa  = deg2rad(pa_deg);
  return matmul(
    matmul(Rz(ra), Ry(-dec)),
    matmul(matmul(Rx(-pa), Ry(v3)), Rz(-v2))
  );
}}

// Convert a V2/V3 telescope-frame corner (arcsec) to sky (ra, dec) in degrees
// using the precomputed attitude matrix M.
// Unit vector: u = [cos(v2)*cos(v3), sin(v2)*cos(v3), sin(v3)]
function telToSky(v2_as, v3_as, M) {{
  const v2 = deg2rad(v2_as / 3600);
  const v3 = deg2rad(v3_as / 3600);
  const u = [Math.cos(v2)*Math.cos(v3), Math.sin(v2)*Math.cos(v3), Math.sin(v3)];
  const [wx, wy, wz] = matvec(M, u);
  let ra = rad2deg(Math.atan2(wy, wx));
  if (ra < 0) ra += 360;
  const dec = rad2deg(Math.asin(Math.max(-1, Math.min(1, wz))));
  return [ra, dec];
}}

// Compute all 18×4-corner polygons for a pointing
function scaPolygons(ra, dec, pa) {{
  const M = attitudeMatrix(V2REF, V3REF, ra, dec, pa);
  return SCA_VERTS.map(sca => {{
    const corners = [];
    for (let k = 0; k < 4; k++) {{
      const [sky_ra, sky_dec] = telToSky(sca.v2[k], sca.v3[k], M);
      corners.push([sky_ra, sky_dec]);
    }}
    return corners;
  }});
}}

// ─── Aladin + overlay state ───────────────────────────────────────────────────
let aladin;
let curPA = 90.6;

// gbtdsOv  : single A.graphicOverlay (rebuilt on PA change)
// jwstOv   : single A.graphicOverlay (static region)
// rgpsOvs  : Map<name, A.graphicOverlay> (rebuilt on PA change)
// layerOn  : Map<name, bool>
let gbtdsOv  = null;
let jwstOv   = null;
let rgpsOvs  = new Map();
let layerOn  = new Map();

// Initialise layerOn from metadata
for (const [name, m] of Object.entries(RGPS_TDS_META)) layerOn.set(name, !!m.on);
for (const [name, m] of Object.entries(RGPS_META))     layerOn.set(name, !!m.on);
layerOn.set('GBTDS', true);
layerOn.set('JWST_Target_Area', true);

// ─── Edit-mode state ──────────────────────────────────────────────────────────
// customPos: Map<key, {{ra, dec}}> where key = "LAYERNAME:INDEX"
// selected : current selected key (or null)
// editMode : bool
let customPos   = new Map();
let editMode    = false;
let selected    = null;      // "LAYERNAME:INDEX"
let selectionOv = null;      // A.graphicOverlay for the selected pointing (yellow highlight)
let downPos     = null;      // {{x, y}} at mousedown, for click-vs-pan detection

// Return effective ra/dec for a pointing (custom override or original)
function getPos(layerName, idx, origRa, origDec) {{
  const key = `${{layerName}}:${{idx}}`;
  return customPos.has(key) ? customPos.get(key) : {{ ra: origRa, dec: origDec }};
}}

// Build a flat list of all {{layerName, idx, ra, dec}} for active layers
function allActivePointings() {{
  const list = [];
  if (layerOn.get('GBTDS')) {{
    GBTDS_CENTERS.forEach((t, i) => {{
      const p = getPos('GBTDS', i, t.ra, t.dec);
      list.push({{ layerName: 'GBTDS', idx: i, ra: p.ra, dec: p.dec, origRa: t.ra, origDec: t.dec }});
    }});
  }}
  for (const [name, tiles] of Object.entries(RGPS)) {{
    if (!layerOn.get(name)) continue;
    tiles.forEach((t, i) => {{
      const p = getPos(name, i, t.ra, t.dec);
      list.push({{ layerName: name, idx: i, ra: p.ra, dec: p.dec, origRa: t.ra, origDec: t.dec }});
    }});
  }}
  return list;
}}

// Background survey buttons – attached at parse time so they don't depend on
// A.init timing. Guard with `aladin &&` since Aladin may not be ready yet.
document.querySelectorAll('button.survey').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('button.survey').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (aladin) {{
      try {{ aladin.setBaseImageLayer(btn.dataset.survey); }}
      catch(e) {{ aladin.setImageSurvey(btn.dataset.survey); }}
    }}
  }});
}});

function getColor(name) {{
  return (RGPS_TDS_META[name] || RGPS_META[name] || {{}}).color || '#ffffff';
}}

// Rebuild just the selection overlay (cheap: 18 polygons)
function rebuildSelection(pa) {{
  if (selectionOv) {{ aladin.removeOverlay(selectionOv); selectionOv = null; }}
  if (!selected || !editMode) return;
  const [layerName, idxStr] = selected.split(':');
  const idx  = parseInt(idxStr);
  const tile = layerName === 'GBTDS' ? GBTDS_CENTERS[idx] : (RGPS[layerName] || [])[idx];
  if (!tile) return;
  const pos = getPos(layerName, idx, tile.ra, tile.dec);
  selectionOv = A.graphicOverlay({{ color: '#ffe000', lineWidth: 2.5, name: '__sel__' }});
  aladin.addOverlay(selectionOv);
  for (const poly of scaPolygons(pos.ra, pos.dec, pos.pa ?? pa)) {{
    selectionOv.add(A.polygon(poly));
  }}
}}

// Sync the RA/Dec inputs to the currently selected pointing
function updateEditPanel() {{
  const infoDiv = document.getElementById('selected-info');
  if (!selected || !editMode) {{ infoDiv.classList.remove('visible'); return; }}
  infoDiv.classList.add('visible');
  const [layerName, idxStr] = selected.split(':');
  const idx  = parseInt(idxStr);
  const tile = layerName === 'GBTDS' ? GBTDS_CENTERS[idx] : (RGPS[layerName] || [])[idx];
  if (!tile) return;
  const pos = getPos(layerName, idx, tile.ra, tile.dec);
  document.getElementById('selected-label').textContent =
    layerName === 'GBTDS' ? `GBTDS tile ${{idx + 1}}` : `${{layerName}} #${{idx + 1}}`;
  document.getElementById('edit-ra').value  = pos.ra.toFixed(6);
  document.getElementById('edit-dec').value = pos.dec.toFixed(6);
  document.getElementById('edit-pa').value  = (pos.pa ?? curPA).toFixed(1);
}}

// Rebuild all overlays from scratch at the given PA
function rebuildAll(pa) {{
  const t0 = performance.now();
  setStatus('Computing…');

  // Remove old overlays (including any selection highlight)
  if (selectionOv) {{ aladin.removeOverlay(selectionOv); selectionOv = null; }}
  if (gbtdsOv) aladin.removeOverlay(gbtdsOv);
  if (jwstOv)  aladin.removeOverlay(jwstOv);
  gbtdsOv = null;
  jwstOv  = null;
  for (const ov of rgpsOvs.values()) aladin.removeOverlay(ov);
  rgpsOvs.clear();

  // GBTDS
  if (layerOn.get('GBTDS')) {{
    gbtdsOv = A.graphicOverlay({{ color: '#4488ff', lineWidth: 1.8, name: 'GBTDS', selectable: false }});
    aladin.addOverlay(gbtdsOv);
    GBTDS_CENTERS.forEach((tile, i) => {{
      const p = getPos('GBTDS', i, tile.ra, tile.dec);
      for (const poly of scaPolygons(p.ra, p.dec, p.pa ?? pa)) {{
        gbtdsOv.add(A.polygon(poly));
      }}
    }});
  }}

  if (layerOn.get('JWST_Target_Area')) {{
    jwstOv = A.graphicOverlay({{ color: '#ff4444', lineWidth: 2.2, name: 'JWST target_area.reg', selectable: false }});
    aladin.addOverlay(jwstOv);
    jwstOv.add(A.polygon(JWST_TARGET_AREA));
  }}

  for (const [name, tiles] of Object.entries(RGPS)) {{
    if (!layerOn.get(name)) continue;
    const color = getColor(name);
    const lw    = name.startsWith('TDS_') ? 1.4 : 1.0;
    const ov    = A.graphicOverlay({{ color, lineWidth: lw, name, selectable: false }});
    aladin.addOverlay(ov);
    tiles.forEach((tile, i) => {{
      const p = getPos(name, i, tile.ra, tile.dec);
      for (const poly of scaPolygons(p.ra, p.dec, p.pa ?? pa)) {{
        ov.add(A.polygon(poly));
      }}
    }});
    rgpsOvs.set(name, ov);
  }}

  // Add selection highlight on top
  if (editMode) rebuildSelection(pa);

  const ms = Math.round(performance.now() - t0);
  setStatus(`PA = ${{pa.toFixed(1)}}°  ·  ${{ms}} ms`);
}}

function setStatus(msg) {{
  document.getElementById('status').textContent = msg;
}}

// Find nearest ORIGINAL pointing key to (ra, dec) ICRS — used when loading by coordinate
function findNearestKey(ra, dec) {{
  let bestKey = null, bestAng2 = Infinity;
  const sinD = Math.sin(deg2rad(dec)), cosD = Math.cos(deg2rad(dec));
  function check(t, key) {{
    const cosd = sinD * Math.sin(deg2rad(t.dec)) + cosD * Math.cos(deg2rad(t.dec)) * Math.cos(deg2rad(ra - t.ra));
    const ang2 = Math.acos(Math.max(-1, Math.min(1, cosd))) ** 2;
    if (ang2 < bestAng2) {{ bestAng2 = ang2; bestKey = key; }}
  }}
  GBTDS_CENTERS.forEach((t, i) => check(t, `GBTDS:${{i}}`));
  for (const [name, tiles] of Object.entries(RGPS))
    tiles.forEach((t, i) => check(t, `${{name}}:${{i}}`));
  return bestAng2 < deg2rad(0.125) ** 2 ? bestKey : null;  // ~7.5 arcmin
}}

// Load custom positions from a parsed JSON object.
// Accepted formats:
//   {{"pa": 90.6, "pointings": [...]}}         full export format
//   [...]                                      plain array
// Each pointing: {{layer, index, new_ra, new_dec [,pa]}} or {{ra, dec [,pa]}} (matched by coord)
function loadFromJSON(data) {{
  let newPA = null, pointings = null;
  if (Array.isArray(data)) {{
    pointings = data;
  }} else if (data && typeof data === 'object') {{
    if (data.pa != null) newPA = +data.pa;
    pointings = data.pointings || data.items || null;
  }}
  if (newPA != null && !isNaN(newPA)) {{
    newPA = ((newPA % 360) + 360) % 360;
    curPA = newPA;
    const el = document.getElementById('pa-val'), er = document.getElementById('pa-range');
    if (el) el.value = newPA.toFixed(1);
    if (er) er.value = newPA;
  }}
  let count = 0;
  if (pointings) {{
    for (const item of pointings) {{
      const ra  = +(item.new_ra  ?? item.ra  ?? NaN);
      const dec = +(item.new_dec ?? item.dec ?? NaN);
      if (isNaN(ra) || isNaN(dec)) continue;
      const itemPA = (item.pa != null && !isNaN(+item.pa)) ? +item.pa : undefined;
      const key   = (item.layer != null && item.index != null)
                  ? `${{item.layer}}:${{item.index}}`
                  : findNearestKey(ra, dec);
      if (!key) continue;
      const entry = {{ ra: ((ra%360)+360)%360, dec: Math.max(-90, Math.min(90, dec)) }};
      if (itemPA !== undefined) entry.pa = ((itemPA%360)+360)%360;
      customPos.set(key, entry);
      count++;
    }}
  }}
  rebuildAll(curPA);
  if (selected && editMode) {{ rebuildSelection(curPA); updateEditPanel(); }}
  return `Loaded ${{count}} pointing(s)${{newPA != null ? `, PA=${{newPA.toFixed(1)}}°` : ''}}.`;
}}

// ─── Initialise Aladin ────────────────────────────────────────────────────────
A.init.then(() => {{
  aladin = A.aladin('#aladin', {{
    survey:   'DSS2/color',
    target:   '0 0',
    fov:      12,
    cooFrame: 'galactic',
  }});

  rebuildAll(curPA);

  // ── PA text input ─────────────────────────────────────────────
  const paVal   = document.getElementById('pa-val');
  const paRange = document.getElementById('pa-range');

  function syncPA(pa) {{
    pa = ((pa % 360) + 360) % 360;
    curPA = pa;
    paVal.value   = pa.toFixed(1);
    paRange.value = pa;
  }}

  paVal.addEventListener('change', () => {{
    syncPA(parseFloat(paVal.value) || 0);
  }});
  paVal.addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{ syncPA(parseFloat(paVal.value) || 0); }}
  }});
  paRange.addEventListener('input', () => {{
    syncPA(parseFloat(paRange.value));
  }});

  // Preset buttons
  document.querySelectorAll('.pa-preset').forEach(btn => {{
    btn.addEventListener('click', () => {{
      syncPA(parseFloat(btn.dataset.pa));
    }});
  }});

  // Apply button
  document.getElementById('apply-btn').addEventListener('click', () => {{
    rebuildAll(curPA);
  }});

  // Also re-apply on Enter in the number box
  paVal.addEventListener('keydown', e => {{
    if (e.key === 'Enter') rebuildAll(curPA);
  }});

  // ── Layer toggle buttons ──────────────────────────────────────
  // GBTDS
  document.getElementById('gbtds-btn').addEventListener('click', function() {{
    const nowOn = !this.classList.contains('active');
    layerOn.set('GBTDS', nowOn);
    this.classList.toggle('active', nowOn);
    rebuildAll(curPA);
  }});

  // RGPS / RGPS-TDS
  document.querySelectorAll('.layer-btn:not(#gbtds-btn)').forEach(btn => {{
    btn.addEventListener('click', function() {{
      const name  = this.dataset.layer;
      const nowOn = !this.classList.contains('active');
      layerOn.set(name, nowOn);
      this.classList.toggle('active', nowOn);
      rebuildAll(curPA);
    }});
  }});

  // Group all/none
  document.querySelectorAll('.mini-btn[data-grp]').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const grp = btn.dataset.grp;
      const on  = btn.dataset.on === '1';
      document.querySelectorAll(`.layer-btn.${{grp}}`).forEach(lb => {{
        const name = lb.dataset.layer;
        layerOn.set(name, on);
        lb.classList.toggle('active', on);
      }});
      rebuildAll(curPA);
    }});
  }});

  // ── Edit mode ────────────────────────────────────────────────
  const editBtn    = document.getElementById('edit-btn');
  const editPanel  = document.getElementById('edit-panel');
  const editStatus = document.getElementById('edit-status');
  const aladinDiv  = document.getElementById('aladin');

  function setEditStatus(msg) {{
    editStatus.textContent = msg;
  }}

  editBtn.addEventListener('click', () => {{
    editMode = !editMode;
    editBtn.classList.toggle('active', editMode);
    editPanel.classList.toggle('visible', editMode);
    document.body.classList.toggle('edit-mode', editMode);
    if (!editMode) {{
      selected = null;
      if (selectionOv) {{ aladin.removeOverlay(selectionOv); selectionOv = null; }}
      updateEditPanel();
    }}
    rebuildAll(curPA);
    setEditStatus(editMode ? 'Click any WFI square to select that pointing.' : '');
  }});

  // Hit-test in PIXEL space — coordinate-frame independent.
  // aladin.world2pix(ra, dec) accepts ICRS J2000 regardless of the display cooFrame,
  // so this works correctly even when cooFrame='galactic'.
  function hitTestByPixel(clickX, clickY) {{
    const allPts = allActivePointings();
    let best = null, bestD2 = Infinity;
    for (const pt of allPts) {{
      const xy = aladin.world2pix(pt.ra, pt.dec);
      if (!xy || xy[0] == null) continue;
      const d2 = (clickX - xy[0])**2 + (clickY - xy[1])**2;
      if (d2 < bestD2) {{ bestD2 = d2; best = pt; }}
    }}
    // Accept nearest pointing within 300 px (scales naturally with zoom level)
    return bestD2 < 300 * 300 ? best : null;
  }}

  // Click detection: use capture phase (true) so we receive events before Aladin's
  // own handlers. No preventDefault → panning still works normally in edit mode.
  aladinDiv.addEventListener('mousedown', (e) => {{
    if (!editMode || e.button !== 0) return;
    downPos = {{ x: e.clientX, y: e.clientY }};
  }}, true);

  aladinDiv.addEventListener('mouseup', (e) => {{
    if (!editMode || !downPos || e.button !== 0) return;
    const dx = e.clientX - downPos.x, dy = e.clientY - downPos.y;
    downPos = null;
    if (Math.hypot(dx, dy) > 8) return;  // was a pan, not a click
    const rect = aladinDiv.getBoundingClientRect();
    // Use pixel-space hit test — world2pix accepts ICRS regardless of display cooFrame
    const pt = hitTestByPixel(e.clientX - rect.left, e.clientY - rect.top);
    if (!pt) {{
      selected = null;
      if (selectionOv) {{ aladin.removeOverlay(selectionOv); selectionOv = null; }}
      updateEditPanel();
      setEditStatus('No pointing nearby — try clicking closer to a WFI square.');
      return;
    }}
    selected = `${{pt.layerName}}:${{pt.idx}}`;
    rebuildSelection(curPA);
    updateEditPanel();
    const label = pt.layerName === 'GBTDS'
      ? `GBTDS tile ${{pt.idx + 1}}`
      : `${{pt.layerName}} #${{pt.idx + 1}}`;
    setEditStatus(`Selected: ${{label}}. Edit RA/Dec and click ↵ Move.`);
  }}, true);

  // Apply RA/Dec/PA inputs to move the selected pointing
  function applyEditCoords() {{
    if (!selected) return;
    let ra  = parseFloat(document.getElementById('edit-ra').value);
    let dec = parseFloat(document.getElementById('edit-dec').value);
    let pa  = parseFloat(document.getElementById('edit-pa').value);
    if (isNaN(ra) || isNaN(dec)) {{ setEditStatus('Invalid coordinates.'); return; }}
    ra  = ((ra % 360) + 360) % 360;
    dec = Math.max(-90, Math.min(90, dec));
    const entry = {{ ra, dec }};
    if (!isNaN(pa)) entry.pa = ((pa % 360) + 360) % 360;
    customPos.set(selected, entry);
    document.getElementById('edit-ra').value  = ra.toFixed(6);
    document.getElementById('edit-dec').value = dec.toFixed(6);
    if (!isNaN(pa)) document.getElementById('edit-pa').value = entry.pa.toFixed(1);
    rebuildAll(curPA);
    setEditStatus(`Moved · RA=${{ra.toFixed(4)}}°  Dec=${{dec.toFixed(4)}}° · ${{customPos.size}} modified`);
  }}

  document.getElementById('move-btn').addEventListener('click', applyEditCoords);
  document.getElementById('edit-ra').addEventListener('keydown',  e => {{ if (e.key==='Enter') applyEditCoords(); }});
  document.getElementById('edit-dec').addEventListener('keydown', e => {{ if (e.key==='Enter') applyEditCoords(); }});
  document.getElementById('edit-pa').addEventListener('keydown',  e => {{ if (e.key==='Enter') applyEditCoords(); }});

  // Reset just the selected pointing
  document.getElementById('reset-this-btn').addEventListener('click', () => {{
    if (!selected) return;
    customPos.delete(selected);
    rebuildAll(curPA);
    updateEditPanel();
    setEditStatus('Pointing reset to original position.');
  }});

  // Export all modified pointings as JSON — format: {{pa, pointings:[...]}}
  document.getElementById('export-btn').addEventListener('click', () => {{
    const items = [];
    for (const [key, pos] of customPos) {{
      const [layerName, idxStr] = key.split(':');
      const idx  = parseInt(idxStr);
      const tile = layerName === 'GBTDS' ? GBTDS_CENTERS[idx] : (RGPS[layerName] || [])[idx];
      const item = {{
        layer:    layerName,
        index:    idx,
        orig_ra:  tile ? tile.ra  : null,
        orig_dec: tile ? tile.dec : null,
        new_ra:   +pos.ra.toFixed(6),
        new_dec:  +pos.dec.toFixed(6),
      }};
      if (pos.pa !== undefined) item.pa = +pos.pa.toFixed(3);
      items.push(item);
    }}
    const out = {{ pa: +curPA.toFixed(3), pointings: items }};
    const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'roman_custom_pointings.json';
    a.click();
    setEditStatus(`Exported ${{items.length}} pointing(s) · PA=${{curPA.toFixed(1)}}°.`);
  }});

  // Load positions from JSON file
  document.getElementById('load-btn').addEventListener('click', () => {{
    document.getElementById('load-file').click();
  }});
  document.getElementById('load-file').addEventListener('change', (e) => {{
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {{
      try {{
        const msg = loadFromJSON(JSON.parse(ev.target.result));
        setEditStatus(msg);
      }} catch(err) {{
        setEditStatus(`Load error: ${{err.message}}`);
      }}
      e.target.value = '';  // allow re-loading same file
    }};
    reader.readAsText(file);
  }});

  // Reset all pointings
  document.getElementById('reset-btn').addEventListener('click', () => {{
    customPos.clear();
    selected = null;
    if (selectionOv) {{ aladin.removeOverlay(selectionOv); selectionOv = null; }}
    updateEditPanel();
    rebuildAll(curPA);
    setEditStatus('All pointings reset to defaults.');
  }});

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
print(f"  Pointing centres: {n_pts}")
print(f"  Polygons per PA change: {n_pts}×18 = {n_pts*18:,} SCA polygons (active layers only)")

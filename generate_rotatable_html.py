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
      <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/rgb_final_uncropped_hips/">CMZ RGB</button>
      <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/MUSTANG_12m_feather_noaxes_hips/">MUSTANG</button>
      <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/jwst_cmz_hips/">JWST CMZ</button>
      <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/Brick_RGB_444-356-200_transparent_hips/">Brick JWST</button>
      <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/SgrA_RGB_MIRI_1500-1000-560_transparent_hips/">Sgr A* MIRI</button>
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
    <div class="section-label" style="margin-top:6px">JWST actual footprint (GO 10678, 139 pointings)</div>
    <div class="btn-row">
      <button class="layer-btn" data-layer="JWST_GC_NIRCAM" style="--bc:#ffaa00">NIRCam</button>
      <button class="layer-btn" data-layer="JWST_GC_MIRI" style="--bc:#ff66cc">MIRI</button>
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

// Actual JWST/NIRCam Legacy Survey of the Galactic Center footprint (GO 10678, 139 planned pointings).
// One quad per pointing per instrument, from footprints.json (data.rc.ufl.edu/pub/adamginsburg/jwst-gc/).
const JWST_GC_NIRCAM = [[[266.141799,-29.528916],[266.135712,-29.429907],[266.182041,-29.427740],[266.188172,-29.526748]],[[266.106448,-29.510033],[266.100363,-29.411023],[266.146683,-29.408857],[266.152813,-29.507865]],[[266.071110,-29.491141],[266.065026,-29.392132],[266.111338,-29.389965],[266.117467,-29.488973]],[[266.035786,-29.472238],[266.029703,-29.373229],[266.076005,-29.371063],[266.082133,-29.470070]],[[266.000475,-29.453327],[265.994393,-29.354318],[266.040687,-29.352151],[266.046814,-29.451159]],[[266.215508,-29.493311],[266.209424,-29.394301],[266.255736,-29.392135],[266.261866,-29.491142]],[[266.180156,-29.474447],[266.174073,-29.375437],[266.220377,-29.373271],[266.226505,-29.472278]],[[266.144818,-29.455575],[266.138736,-29.356565],[266.185031,-29.354399],[266.191157,-29.453406]],[[266.109492,-29.436691],[266.103411,-29.337682],[266.149697,-29.335515],[266.155823,-29.434523]],[[266.074179,-29.417800],[266.068099,-29.318790],[266.114377,-29.316624],[266.120501,-29.415631]],[[266.038879,-29.398900],[266.032801,-29.299890],[266.079070,-29.297724],[266.085193,-29.396731]],[[266.253813,-29.438822],[266.247732,-29.339812],[266.294019,-29.337646],[266.300145,-29.436653]],[[266.218473,-29.419969],[266.212393,-29.320959],[266.258672,-29.318793],[266.264797,-29.417801]],[[266.183146,-29.401105],[266.177067,-29.302096],[266.223337,-29.299929],[266.229461,-29.398937]],[[266.147832,-29.382233],[266.141754,-29.283223],[266.188016,-29.281057],[266.194138,-29.380065]],[[266.112531,-29.363352],[266.106454,-29.264343],[266.152707,-29.262176],[266.158829,-29.361184]],[[266.077243,-29.344461],[266.071167,-29.245451],[266.117412,-29.243285],[266.123532,-29.342292]],[[266.292077,-29.384322],[266.285999,-29.285312],[266.332262,-29.283146],[266.338384,-29.382153]],[[266.256748,-29.365477],[266.250672,-29.266468],[266.296926,-29.264301],[266.303047,-29.363309]],[[266.221433,-29.346625],[266.215357,-29.247615],[266.261603,-29.245449],[266.267723,-29.344456]],[[266.186131,-29.327763],[266.180056,-29.228754],[266.226293,-29.226588],[266.232413,-29.325595]],[[266.150842,-29.308891],[266.144768,-29.209882],[266.190997,-29.207715],[266.197115,-29.306723]],[[266.115566,-29.290011],[266.109493,-29.191001],[266.155713,-29.188835],[266.161830,-29.287842]],[[266.080302,-29.271122],[266.074231,-29.172112],[266.120443,-29.169946],[266.126558,-29.268953]],[[266.330300,-29.329811],[266.324225,-29.230801],[266.370463,-29.228635],[266.376582,-29.327642]],[[266.294983,-29.310977],[266.288910,-29.211968],[266.335139,-29.209801],[266.341257,-29.308809]],[[266.259679,-29.292136],[266.253607,-29.193126],[266.299828,-29.190960],[266.305945,-29.289967]],[[266.224389,-29.273283],[266.218318,-29.174273],[266.264531,-29.172107],[266.270646,-29.271115]],[[266.189112,-29.254422],[266.183042,-29.155412],[266.229246,-29.153246],[266.235360,-29.252253]],[[266.153847,-29.235550],[266.147778,-29.136540],[266.193974,-29.134374],[266.200087,-29.233381]],[[266.118596,-29.216669],[266.112528,-29.117659],[266.158715,-29.115493],[266.164827,-29.214501]],[[266.368482,-29.275288],[266.362411,-29.176279],[266.408624,-29.174113],[266.414740,-29.273120]],[[266.333177,-29.256466],[266.327107,-29.157457],[266.373312,-29.155290],[266.379427,-29.254298]],[[266.297886,-29.237633],[266.291816,-29.138623],[266.338013,-29.136457],[266.344127,-29.235465]],[[266.262607,-29.218791],[266.256539,-29.119782],[266.302727,-29.117615],[266.308839,-29.216623]],[[266.227341,-29.199938],[266.221274,-29.100929],[266.267454,-29.098763],[266.273565,-29.197770]],[[266.192088,-29.181077],[266.186022,-29.082068],[266.232193,-29.079901],[266.238304,-29.178909]],[[266.156849,-29.162208],[266.150784,-29.063198],[266.196946,-29.061032],[266.203056,-29.160040]],[[266.406624,-29.220758],[266.400556,-29.121748],[266.446745,-29.119582],[266.452857,-29.218590]],[[266.371331,-29.201944],[266.365264,-29.102934],[266.411444,-29.100768],[266.417556,-29.199776]],[[266.336051,-29.183119],[266.329985,-29.084109],[266.376157,-29.081943],[266.382267,-29.180951]],[[266.300784,-29.164288],[266.294719,-29.065279],[266.340883,-29.063113],[266.346992,-29.162120]],[[266.265530,-29.145447],[266.259466,-29.046437],[266.305621,-29.044271],[266.311729,-29.143278]],[[266.230289,-29.126597],[266.224226,-29.027587],[266.270373,-29.025421],[266.276480,-29.124428]],[[266.195061,-29.107736],[266.188999,-29.008726],[266.235137,-29.006560],[266.241243,-29.105567]],[[266.444725,-29.166213],[266.438660,-29.067204],[266.484825,-29.065038],[266.490934,-29.164045]],[[266.409444,-29.147411],[266.403380,-29.048401],[266.449536,-29.046235],[266.455644,-29.145242]],[[266.374176,-29.128597],[266.368113,-29.029587],[266.414261,-29.027421],[266.420368,-29.126428]],[[266.338921,-29.109775],[266.332859,-29.010765],[266.378998,-29.008599],[266.385104,-29.107606]],[[266.303678,-29.090944],[266.297618,-28.991935],[266.343749,-28.989768],[266.349853,-29.088776]],[[266.268449,-29.072102],[266.262389,-28.973093],[266.308512,-28.970926],[266.314615,-29.069934]],[[266.233232,-29.053252],[266.227174,-28.954243],[266.273288,-28.952076],[266.279390,-29.051084]],[[266.482786,-29.111661],[266.476725,-29.012651],[266.522865,-29.010485],[266.528971,-29.109492]],[[266.447517,-29.092866],[266.441456,-28.993857],[266.487588,-28.991690],[266.493693,-29.090698]],[[266.412261,-29.074063],[266.406201,-28.975054],[266.452324,-28.972888],[266.458428,-29.071895]],[[266.377017,-29.055250],[266.370959,-28.956240],[266.417074,-28.954074],[266.423176,-29.053081]],[[266.341787,-29.036430],[266.335729,-28.937421],[266.381835,-28.935254],[266.387937,-29.034262]],[[266.306569,-29.017600],[266.300512,-28.918590],[266.346610,-28.916424],[266.352711,-29.015431]],[[266.271363,-28.998758],[266.265308,-28.899748],[266.311398,-28.897582],[266.317497,-28.996590]],[[266.520808,-29.057097],[266.514749,-28.958087],[266.560864,-28.955921],[266.566967,-29.054928]],[[266.485550,-29.038311],[266.479493,-28.939301],[266.525600,-28.937135],[266.531702,-29.036142]],[[266.450306,-29.019519],[266.444249,-28.920510],[266.490348,-28.918343],[266.496449,-29.017351]],[[266.415073,-29.000716],[266.409018,-28.901707],[266.455109,-28.899540],[266.461208,-28.998548]],[[266.379855,-28.981905],[266.373800,-28.882896],[266.419882,-28.880729],[266.425981,-28.979737]],[[266.344648,-28.963083],[266.338595,-28.864073],[266.384669,-28.861907],[266.390766,-28.960915]],[[266.309455,-28.944252],[266.303403,-28.845243],[266.349468,-28.843076],[266.355564,-28.942084]],[[266.558789,-29.002522],[266.552733,-28.903512],[266.598825,-28.901346],[266.604924,-29.000353]],[[266.523543,-28.983747],[266.517489,-28.884737],[266.563572,-28.882571],[266.569670,-28.981578]],[[266.488310,-28.964963],[266.482257,-28.865954],[266.528331,-28.863788],[266.534429,-28.962795]],[[266.453090,-28.946172],[266.447038,-28.847162],[266.493104,-28.844996],[266.499200,-28.944003]],[[266.417882,-28.927369],[266.411831,-28.828360],[266.457889,-28.826193],[266.463984,-28.925201]],[[266.382688,-28.908558],[266.376637,-28.809548],[266.422687,-28.807382],[266.428781,-28.906390]],[[266.347506,-28.889738],[266.341457,-28.790729],[266.387498,-28.788563],[266.393591,-28.887570]],[[266.596730,-28.947936],[266.590677,-28.848926],[266.636745,-28.846760],[266.642841,-28.945767]],[[266.561496,-28.929172],[266.555445,-28.830162],[266.601504,-28.827996],[266.607599,-28.927003]],[[266.526275,-28.910397],[266.520224,-28.811387],[266.566275,-28.809221],[266.572369,-28.908228]],[[266.491066,-28.891613],[266.485017,-28.792604],[266.531059,-28.790438],[266.537152,-28.889445]],[[266.455870,-28.872822],[266.449822,-28.773812],[266.495856,-28.771646],[266.501948,-28.870653]],[[266.420687,-28.854022],[266.414640,-28.755012],[266.460666,-28.752846],[266.466757,-28.851853]],[[266.385517,-28.835211],[266.379471,-28.736201],[266.425488,-28.734035],[266.431578,-28.833042]],[[266.634632,-28.893338],[266.628582,-28.794329],[266.674625,-28.792163],[266.680719,-28.891170]],[[266.599410,-28.874586],[266.593361,-28.775576],[266.639396,-28.773410],[266.645488,-28.872417]],[[266.564200,-28.855822],[266.558153,-28.756812],[266.604179,-28.754646],[266.610270,-28.853653]],[[266.529003,-28.837047],[266.522956,-28.738037],[266.568975,-28.735871],[266.575064,-28.834878]],[[266.493818,-28.818266],[266.487773,-28.719257],[266.533783,-28.717090],[266.539872,-28.816098]],[[266.458647,-28.799475],[266.452603,-28.700465],[266.498605,-28.698299],[266.504692,-28.797306]],[[266.423488,-28.780675],[266.417445,-28.681665],[266.463439,-28.679499],[266.469525,-28.778506]],[[266.672494,-28.838733],[266.666447,-28.739723],[266.712466,-28.737557],[266.718556,-28.836565]],[[266.637283,-28.819988],[266.631238,-28.720979],[266.677249,-28.718813],[266.683337,-28.817820]],[[266.602085,-28.801233],[266.596041,-28.702223],[266.642043,-28.700057],[266.648131,-28.799065]],[[266.566900,-28.782472],[266.560857,-28.683462],[266.606851,-28.681296],[266.612937,-28.780303]],[[266.531727,-28.763700],[266.525685,-28.664690],[266.571671,-28.662524],[266.577757,-28.761531]],[[266.496567,-28.744916],[266.490526,-28.645907],[266.536504,-28.643740],[266.542588,-28.742748]],[[266.461419,-28.726127],[266.455379,-28.627118],[266.501349,-28.624951],[266.507432,-28.723959]],[[266.710316,-28.784116],[266.704273,-28.685107],[266.750268,-28.682940],[266.756355,-28.781948]],[[266.675117,-28.765380],[266.669075,-28.666371],[266.715062,-28.664204],[266.721147,-28.763212]],[[266.639931,-28.746636],[266.633890,-28.647626],[266.679868,-28.645460],[266.685953,-28.744467]],[[266.604757,-28.727883],[266.598717,-28.628873],[266.644687,-28.626707],[266.650771,-28.725715]],[[266.569596,-28.709119],[266.563557,-28.610110],[266.609519,-28.607943],[266.615601,-28.706951]],[[266.534447,-28.690350],[266.528410,-28.591340],[266.574363,-28.589174],[266.580445,-28.688181]],[[266.499311,-28.671569],[266.493275,-28.572560],[266.539220,-28.570393],[266.545300,-28.669401]],[[266.748099,-28.729488],[266.742059,-28.630479],[266.788030,-28.628313],[266.794114,-28.727320]],[[266.712912,-28.710763],[266.706873,-28.611754],[266.752836,-28.609588],[266.758918,-28.708595]],[[266.677737,-28.692027],[266.671699,-28.593018],[266.717654,-28.590851],[266.723735,-28.689859]],[[266.642575,-28.673286],[266.636538,-28.574276],[266.682484,-28.572110],[266.688564,-28.671117]],[[266.607425,-28.654533],[266.601390,-28.555523],[266.647328,-28.553357],[266.653407,-28.652365]],[[266.572288,-28.635769],[266.566254,-28.536760],[266.612184,-28.534593],[266.618261,-28.633601]],[[266.785843,-28.674850],[266.779806,-28.575840],[266.825753,-28.573674],[266.831834,-28.672681]],[[266.750668,-28.656136],[266.744632,-28.557126],[266.790571,-28.554960],[266.796650,-28.653967]],[[266.715504,-28.637411],[266.709469,-28.538401],[266.755400,-28.536235],[266.761478,-28.635242]],[[266.680354,-28.618675],[266.674320,-28.519665],[266.720242,-28.517499],[266.726319,-28.616506]],[[266.645215,-28.599933],[266.639183,-28.500923],[266.685097,-28.498757],[266.691173,-28.597765]],[[266.610090,-28.581180],[266.604058,-28.482171],[266.649965,-28.480004],[266.656039,-28.579012]],[[266.823548,-28.620202],[266.817514,-28.521193],[266.863437,-28.519026],[266.869514,-28.618034]],[[266.788384,-28.601497],[266.782351,-28.502487],[266.828266,-28.500321],[266.834342,-28.599329]],[[266.753232,-28.582780],[266.747200,-28.483771],[266.793107,-28.481604],[266.799182,-28.580612]],[[266.718093,-28.564058],[266.712062,-28.465048],[266.757961,-28.462882],[266.764035,-28.561890]],[[266.682967,-28.545325],[266.676937,-28.446315],[266.722828,-28.444149],[266.728900,-28.543156]],[[266.647852,-28.526580],[266.641824,-28.427571],[266.687706,-28.425404],[266.693778,-28.524412]],[[266.861214,-28.565544],[266.855183,-28.466535],[266.901082,-28.464368],[266.907156,-28.563376]],[[266.826061,-28.546847],[266.820032,-28.447837],[266.865923,-28.445671],[266.871996,-28.544679]],[[266.790921,-28.528141],[266.784893,-28.429132],[266.830776,-28.426965],[266.836847,-28.525973]],[[266.755793,-28.509427],[266.749766,-28.410418],[266.795641,-28.408251],[266.801711,-28.507259]],[[266.720678,-28.490702],[266.714652,-28.391693],[266.760519,-28.389526],[266.766588,-28.488534]],[[266.685576,-28.471972],[266.679550,-28.372962],[266.725409,-28.370796],[266.731477,-28.469804]],[[266.898841,-28.510875],[266.892813,-28.411865],[266.938689,-28.409699],[266.944759,-28.508706]],[[266.863700,-28.492188],[266.857673,-28.393179],[266.903541,-28.391013],[266.909610,-28.490020]],[[266.828571,-28.473491],[266.822546,-28.374482],[266.868405,-28.372315],[266.874473,-28.471323]],[[266.793455,-28.454788],[266.787431,-28.355779],[266.833282,-28.353613],[266.839349,-28.452620]],[[266.758351,-28.436075],[266.752328,-28.337065],[266.798171,-28.334899],[266.804238,-28.433906]],[[266.723260,-28.417350],[266.717238,-28.318340],[266.763073,-28.316174],[266.769138,-28.415181]],[[266.936429,-28.456197],[266.930404,-28.357187],[266.976256,-28.355021],[266.982324,-28.454029]],[[266.901300,-28.437519],[266.895276,-28.338510],[266.941120,-28.336343],[266.947186,-28.435351]],[[266.866182,-28.418833],[266.860160,-28.319823],[266.905996,-28.317657],[266.912061,-28.416665]],[[266.831078,-28.400138],[266.825056,-28.301129],[266.870884,-28.298963],[266.876948,-28.397970]],[[266.795985,-28.381433],[266.789965,-28.282423],[266.835785,-28.280257],[266.841848,-28.379265]],[[266.760905,-28.362719],[266.754886,-28.263710],[266.800698,-28.261543],[266.806760,-28.360551]],[[265.800679,-28.721538],[265.794639,-28.622529],[265.840607,-28.620363],[265.846690,-28.719370]],[[265.762681,-28.774975],[265.756639,-28.675965],[265.802630,-28.673799],[265.808716,-28.772806]]];
const JWST_GC_MIRI = [[[266.172913,-29.370377],[266.170655,-29.333802],[266.211812,-29.331866],[266.214084,-29.368440]],[[266.137556,-29.351494],[266.135299,-29.314919],[266.176448,-29.312982],[266.178720,-29.349557]],[[266.102213,-29.332602],[266.099956,-29.296027],[266.141097,-29.294091],[266.143369,-29.330665]],[[266.066883,-29.313699],[266.064626,-29.277124],[266.105760,-29.275188],[266.108031,-29.311762]],[[266.031566,-29.294788],[266.029310,-29.258213],[266.070436,-29.256277],[266.072706,-29.292851]],[[266.246612,-29.334771],[266.244355,-29.298197],[266.285497,-29.296260],[266.287768,-29.332834]],[[266.211254,-29.315908],[266.208997,-29.279333],[266.250132,-29.277396],[266.252403,-29.313971]],[[266.175909,-29.297035],[266.173653,-29.260460],[266.214780,-29.258524],[266.217051,-29.295098]],[[266.140578,-29.278152],[266.138322,-29.241577],[266.179441,-29.239641],[266.181712,-29.276215]],[[266.105259,-29.259260],[266.103004,-29.222685],[266.144115,-29.220749],[266.146385,-29.257323]],[[266.069954,-29.240360],[266.067699,-29.203785],[266.108803,-29.201849],[266.111072,-29.238423]],[[266.284900,-29.280283],[266.282644,-29.243708],[266.323764,-29.241771],[266.326034,-29.278346]],[[266.249554,-29.261430],[266.247299,-29.224855],[266.288411,-29.222918],[266.290681,-29.259493]],[[266.214221,-29.242566],[266.211966,-29.205991],[266.253071,-29.204055],[266.255340,-29.240629]],[[266.178901,-29.223694],[266.176647,-29.187119],[266.217744,-29.185182],[266.220013,-29.221757]],[[266.143594,-29.204813],[266.141340,-29.168238],[266.182430,-29.166302],[266.184699,-29.202876]],[[266.108301,-29.185921],[266.106047,-29.149347],[266.147129,-29.147410],[266.149398,-29.183984]],[[266.323147,-29.225783],[266.320892,-29.189208],[266.361990,-29.187271],[266.364259,-29.223846]],[[266.287813,-29.206938],[266.285559,-29.170363],[266.326649,-29.168427],[266.328918,-29.205001]],[[266.252491,-29.188085],[266.250238,-29.151510],[266.291321,-29.149574],[266.293589,-29.186148]],[[266.217184,-29.169224],[266.214930,-29.132649],[266.256006,-29.130713],[266.258274,-29.167287]],[[266.181889,-29.150352],[266.179636,-29.113777],[266.220704,-29.111841],[266.222971,-29.148415]],[[266.146607,-29.131471],[266.144355,-29.094897],[266.185415,-29.092960],[266.187682,-29.129534]],[[266.111338,-29.112583],[266.109086,-29.076008],[266.150139,-29.074071],[266.152405,-29.110646]],[[266.361353,-29.171271],[266.359100,-29.134697],[266.400176,-29.132760],[266.402444,-29.169334]],[[266.326031,-29.152438],[266.323778,-29.115863],[266.364847,-29.113927],[266.367114,-29.150501]],[[266.290721,-29.133596],[266.288469,-29.097022],[266.329530,-29.095085],[266.331797,-29.131659]],[[266.255426,-29.114744],[266.253174,-29.078169],[266.294227,-29.076232],[266.296494,-29.112807]],[[266.220142,-29.095883],[266.217891,-29.059308],[266.258937,-29.057371],[266.261203,-29.093946]],[[266.184872,-29.077010],[266.182621,-29.040435],[266.223660,-29.038499],[266.225925,-29.075073]],[[266.149615,-29.058130],[266.147364,-29.021555],[266.188396,-29.019619],[266.190661,-29.056193]],[[266.399519,-29.116749],[266.397267,-29.080174],[266.438321,-29.078238],[266.440588,-29.114812]],[[266.364208,-29.097927],[266.361957,-29.061352],[266.403004,-29.059416],[266.405270,-29.095990]],[[266.328911,-29.079094],[266.326660,-29.042519],[266.367699,-29.040582],[266.369965,-29.077157]],[[266.293627,-29.060252],[266.291376,-29.023677],[266.332408,-29.021741],[266.334673,-29.058315]],[[266.258355,-29.041399],[266.256105,-29.004824],[266.297130,-29.002888],[266.299395,-29.039462]],[[266.223097,-29.022538],[266.220847,-28.985963],[266.261864,-28.984027],[266.264128,-29.020601]],[[266.187851,-29.003669],[266.185602,-28.967094],[266.226612,-28.965157],[266.228876,-29.001732]],[[266.437644,-29.062219],[266.435393,-29.025644],[266.476426,-29.023707],[266.478692,-29.060282]],[[266.402346,-29.043405],[266.400095,-29.006830],[266.441121,-29.004894],[266.443386,-29.041468]],[[266.367060,-29.024580],[266.364810,-28.988005],[266.405828,-28.986069],[266.408092,-29.022643]],[[266.331787,-29.005749],[266.329538,-28.969174],[266.370548,-28.967238],[266.372812,-29.003812]],[[266.296528,-28.986908],[266.294278,-28.950333],[266.335281,-28.948396],[266.337545,-28.984971]],[[266.261281,-28.968058],[266.259032,-28.931483],[266.300028,-28.929546],[266.302291,-28.966121]],[[266.226047,-28.949196],[266.223799,-28.912622],[266.264787,-28.910685],[266.267049,-28.947260]],[[266.475729,-29.007674],[266.473480,-28.971099],[266.514491,-28.969163],[266.516755,-29.005737]],[[266.440442,-28.988871],[266.438193,-28.952297],[266.479197,-28.950360],[266.481461,-28.986935]],[[266.405169,-28.970058],[266.402920,-28.933483],[266.443916,-28.931546],[266.446179,-28.968121]],[[266.369908,-28.951235],[266.367660,-28.914660],[266.408648,-28.912724],[266.410911,-28.949298]],[[266.334660,-28.932405],[266.332412,-28.895830],[266.373393,-28.893894],[266.375656,-28.930468]],[[266.299424,-28.913563],[266.297177,-28.876988],[266.338151,-28.875052],[266.340413,-28.911626]],[[266.264202,-28.894713],[266.261955,-28.858138],[266.302922,-28.856202],[266.305183,-28.892776]],[[266.513774,-28.953121],[266.511526,-28.916547],[266.552515,-28.914610],[266.554778,-28.951185]],[[266.478499,-28.934327],[266.476251,-28.897752],[266.517233,-28.895816],[266.519496,-28.932390]],[[266.443237,-28.915524],[266.440989,-28.878949],[266.481964,-28.877013],[266.484226,-28.913587]],[[266.407988,-28.896710],[266.405741,-28.860135],[266.446708,-28.858199],[266.448970,-28.894773]],[[266.372751,-28.877891],[266.370505,-28.841316],[266.411465,-28.839380],[266.413726,-28.875954]],[[266.337528,-28.859060],[266.335281,-28.822485],[266.376234,-28.820549],[266.378495,-28.857123]],[[266.302317,-28.840219],[266.300071,-28.803644],[266.341016,-28.801708],[266.343277,-28.838282]],[[266.551779,-28.898558],[266.549531,-28.861983],[266.590500,-28.860046],[266.592761,-28.896621]],[[266.516516,-28.879771],[266.514269,-28.843197],[266.555230,-28.841260],[266.557491,-28.877835]],[[266.481266,-28.860980],[266.479019,-28.824405],[266.519972,-28.822469],[266.522233,-28.859043]],[[266.446028,-28.842177],[266.443782,-28.805602],[266.484728,-28.803666],[266.486988,-28.840240]],[[266.410803,-28.823366],[266.408558,-28.786791],[266.449496,-28.784855],[266.451756,-28.821429]],[[266.375591,-28.804544],[266.373346,-28.767969],[266.414277,-28.766033],[266.416537,-28.802607]],[[266.340392,-28.785713],[266.338147,-28.749138],[266.379071,-28.747202],[266.381330,-28.783776]],[[266.589744,-28.843983],[266.587498,-28.807408],[266.628444,-28.805471],[266.630705,-28.842046]],[[266.554492,-28.825208],[266.552247,-28.788633],[266.593186,-28.786696],[266.595446,-28.823271]],[[266.519254,-28.806424],[266.517008,-28.769849],[266.557940,-28.767913],[266.560200,-28.804487]],[[266.484028,-28.787633],[266.481783,-28.751058],[266.522708,-28.749121],[266.524967,-28.785696]],[[266.448815,-28.768830],[266.446570,-28.732255],[266.487487,-28.730319],[266.489746,-28.766893]],[[266.413614,-28.750019],[266.411370,-28.713444],[266.452280,-28.711508],[266.454538,-28.748082]],[[266.378427,-28.731199],[266.376183,-28.694624],[266.417086,-28.692688],[266.419344,-28.729262]],[[266.627668,-28.789396],[266.625424,-28.752822],[266.666349,-28.750885],[266.668608,-28.787460]],[[266.592429,-28.770633],[266.590185,-28.734058],[266.631102,-28.732121],[266.633361,-28.768696]],[[266.557202,-28.751858],[266.554958,-28.715283],[266.595869,-28.713346],[266.598127,-28.749921]],[[266.521988,-28.733074],[266.519744,-28.696499],[266.560648,-28.694563],[266.562905,-28.731137]],[[266.486786,-28.714283],[266.484543,-28.677708],[266.525439,-28.675771],[266.527697,-28.712346]],[[266.451598,-28.695483],[266.449355,-28.658908],[266.490244,-28.656971],[266.492501,-28.693546]],[[266.416422,-28.676671],[266.414179,-28.640097],[266.455060,-28.638160],[266.457317,-28.674735]],[[266.665554,-28.734799],[266.663310,-28.698224],[266.704214,-28.696288],[266.706472,-28.732862]],[[266.630326,-28.716046],[266.628083,-28.679472],[266.668979,-28.677535],[266.671237,-28.714110]],[[266.595111,-28.697283],[266.592868,-28.660708],[266.633757,-28.658771],[266.636014,-28.695346]],[[266.559908,-28.678508],[266.557666,-28.641933],[266.598548,-28.639996],[266.600804,-28.676571]],[[266.524718,-28.659727],[266.522476,-28.623152],[266.563351,-28.621216],[266.565607,-28.657790]],[[266.489541,-28.640935],[266.487300,-28.604360],[266.528167,-28.602424],[266.530423,-28.638999]],[[266.454377,-28.622135],[266.452136,-28.585560],[266.492996,-28.583624],[266.495251,-28.620199]],[[266.703400,-28.680194],[266.701157,-28.643619],[266.742040,-28.641683],[266.744296,-28.678257]],[[266.668183,-28.661449],[266.665941,-28.624874],[266.706817,-28.622938],[266.709073,-28.659512]],[[266.632980,-28.642694],[266.630738,-28.606119],[266.671606,-28.604183],[266.673862,-28.640757]],[[266.597789,-28.623933],[266.595548,-28.587358],[266.636409,-28.585421],[266.638664,-28.621996]],[[266.562611,-28.605160],[266.560370,-28.568585],[266.601223,-28.566649],[266.603479,-28.603224]],[[266.527445,-28.586377],[266.525205,-28.549802],[266.566051,-28.547866],[266.568306,-28.584440]],[[266.492292,-28.567588],[266.490052,-28.531013],[266.530891,-28.529077],[266.533145,-28.565651]],[[266.741206,-28.625577],[266.738965,-28.589002],[266.779826,-28.587066],[266.782082,-28.623640]],[[266.706002,-28.606841],[266.703761,-28.570266],[266.744615,-28.568330],[266.746870,-28.604904]],[[266.670810,-28.588096],[266.668569,-28.551522],[266.709416,-28.549585],[266.711671,-28.586160]],[[266.635630,-28.569344],[266.633390,-28.532769],[266.674230,-28.530833],[266.676484,-28.567407]],[[266.600464,-28.550580],[266.598224,-28.514005],[266.639056,-28.512069],[266.641310,-28.548643]],[[266.565310,-28.531810],[266.563070,-28.495235],[266.603895,-28.493299],[266.606149,-28.529874]],[[266.530168,-28.513030],[266.527929,-28.476455],[266.568747,-28.474519],[266.571000,-28.511093]],[[266.778973,-28.570949],[266.776733,-28.534374],[266.817573,-28.532438],[266.819828,-28.569012]],[[266.743780,-28.552224],[266.741540,-28.515649],[266.782373,-28.513713],[266.784627,-28.550288]],[[266.708600,-28.533488],[266.706360,-28.496913],[266.747186,-28.494977],[266.749440,-28.531551]],[[266.673432,-28.514746],[266.671193,-28.478172],[266.712011,-28.476235],[266.714265,-28.512810]],[[266.638277,-28.495994],[266.636039,-28.459419],[266.676850,-28.457483],[266.679102,-28.494057]],[[266.603134,-28.477230],[266.600896,-28.440655],[266.641700,-28.438719],[266.643952,-28.475293]],[[266.816701,-28.516310],[266.814462,-28.479735],[266.855281,-28.477799],[266.857534,-28.514374]],[[266.781520,-28.497596],[266.779281,-28.461022],[266.820093,-28.459085],[266.822346,-28.495660]],[[266.746351,-28.478871],[266.744113,-28.442297],[266.784917,-28.440360],[266.787169,-28.476935]],[[266.711195,-28.460135],[266.708957,-28.423560],[266.749754,-28.421624],[266.752006,-28.458199]],[[266.676051,-28.441394],[266.673814,-28.404819],[266.714604,-28.402883],[266.716855,-28.439457]],[[266.640920,-28.422641],[266.638683,-28.386066],[266.679466,-28.384130],[266.681717,-28.420704]],[[266.854389,-28.461663],[266.852152,-28.425088],[266.892950,-28.423152],[266.895201,-28.459726]],[[266.819220,-28.442958],[266.816983,-28.406383],[266.857773,-28.404447],[266.860025,-28.441021]],[[266.784063,-28.424241],[266.781826,-28.387666],[266.822609,-28.385730],[266.824860,-28.422304]],[[266.748918,-28.405519],[266.746682,-28.368944],[266.787458,-28.367008],[266.789709,-28.403582]],[[266.713786,-28.386785],[266.711550,-28.350210],[266.752319,-28.348274],[266.754569,-28.384849]],[[266.678667,-28.368041],[266.676431,-28.331466],[266.717193,-28.329530],[266.719442,-28.366104]],[[266.892039,-28.407005],[266.889803,-28.370430],[266.930579,-28.368494],[266.932830,-28.405068]],[[266.856882,-28.388308],[266.854645,-28.351733],[266.895415,-28.349797],[266.897665,-28.386371]],[[266.821736,-28.369602],[266.819500,-28.333027],[266.860262,-28.331091],[266.862512,-28.367665]],[[266.786603,-28.350888],[266.784367,-28.314313],[266.825123,-28.312377],[266.827372,-28.348951]],[[266.751482,-28.332163],[266.749247,-28.295588],[266.789995,-28.293652],[266.792244,-28.330226]],[[266.716374,-28.313433],[266.714139,-28.276858],[266.754880,-28.274922],[266.757129,-28.311496]],[[266.929650,-28.352335],[266.927415,-28.315760],[266.968171,-28.313824],[266.970420,-28.350399]],[[266.894504,-28.333649],[266.892269,-28.297074],[266.933017,-28.295138],[266.935267,-28.331713]],[[266.859370,-28.314952],[266.857135,-28.278377],[266.897877,-28.276441],[266.900125,-28.313015]],[[266.824248,-28.296249],[266.822014,-28.259674],[266.862748,-28.257738],[266.864997,-28.294313]],[[266.789139,-28.277535],[266.786905,-28.240960],[266.827633,-28.239024],[266.829881,-28.275599]],[[266.754042,-28.258810],[266.751809,-28.222235],[266.792529,-28.220299],[266.794777,-28.256874]],[[266.967223,-28.297658],[266.964988,-28.261083],[267.005723,-28.259147],[267.007972,-28.295721]],[[266.932088,-28.278980],[266.929854,-28.242405],[266.970582,-28.240469],[266.972830,-28.277043]],[[266.896965,-28.260294],[266.894732,-28.223719],[266.935452,-28.221783],[266.937700,-28.258357]],[[266.861855,-28.241599],[266.859622,-28.205024],[266.900336,-28.203088],[266.902583,-28.239663]],[[266.826758,-28.222894],[266.824525,-28.186319],[266.865231,-28.184383],[266.867478,-28.220957]],[[266.791672,-28.204180],[266.789440,-28.167605],[266.830139,-28.165669],[266.832385,-28.202243]],[[265.831550,-28.562999],[265.829310,-28.526424],[265.870147,-28.524488],[265.872402,-28.561062]],[[265.793568,-28.616435],[265.791327,-28.579860],[265.832185,-28.577924],[265.834440,-28.614499]]];

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
let jwstNircamOv = null;
let jwstMiriOv   = null;
let rgpsOvs  = new Map();
let layerOn  = new Map();

// Initialise layerOn from metadata
for (const [name, m] of Object.entries(RGPS_TDS_META)) layerOn.set(name, !!m.on);
for (const [name, m] of Object.entries(RGPS_META))     layerOn.set(name, !!m.on);
layerOn.set('GBTDS', true);
layerOn.set('JWST_Target_Area', true);
layerOn.set('JWST_GC_NIRCAM', false);
layerOn.set('JWST_GC_MIRI', false);

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
      const tgt = btn.dataset.survey;
      const survey = tgt.startsWith('http') ? A.HiPS(tgt) : tgt;
      try {{ aladin.setBaseImageLayer(survey); }}
      catch(e) {{ aladin.setImageSurvey(survey); }}
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
  if (jwstNircamOv) aladin.removeOverlay(jwstNircamOv);
  if (jwstMiriOv)   aladin.removeOverlay(jwstMiriOv);
  gbtdsOv = null;
  jwstOv  = null;
  jwstNircamOv = null;
  jwstMiriOv   = null;
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

  if (layerOn.get('JWST_GC_NIRCAM')) {{
    jwstNircamOv = A.graphicOverlay({{ color: '#ffaa00', lineWidth: 1.2, name: 'JWST GC NIRCam (GO 10678)', selectable: false }});
    aladin.addOverlay(jwstNircamOv);
    for (const poly of JWST_GC_NIRCAM) jwstNircamOv.add(A.polygon(poly));
  }}

  if (layerOn.get('JWST_GC_MIRI')) {{
    jwstMiriOv = A.graphicOverlay({{ color: '#ff66cc', lineWidth: 1.2, name: 'JWST GC MIRI (GO 10678)', selectable: false }});
    aladin.addOverlay(jwstMiriOv);
    for (const poly of JWST_GC_MIRI) jwstMiriOv.add(A.polygon(poly));
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

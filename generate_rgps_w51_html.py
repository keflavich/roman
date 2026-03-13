#!/usr/bin/env python3
"""
Generate roman_footprint_rgps_w51.html using real WFI SCA polygons
computed via pysiaf from roman_gps.sim.ecsv.

Layers:
  • Deep+Spec_W51 — 8 unique WFI_CEN pointings (dedicated deep W51 field)
  • Disk7 (all)   — 632 unique WFI_CEN pointings covering l=29–50.5°
"""

import sys, json, math
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import pysiaf
from pysiaf.utils.rotations import attitude_matrix

# ── paths ────────────────────────────────────────────────────────────────────
ECSV = (
    "/Users/adam/repos/roman_notebooks/notebooks/"
    "footprint_visualization/aux_data/roman_gps.sim.ecsv"
)
OUT_HTML = "/Users/adam/Downloads/roman_footprint_rgps_w51.html"

# ── pysiaf setup ─────────────────────────────────────────────────────────────
print("Loading Roman SIAF …")
rsiaf = pysiaf.Siaf("Roman")
wfi_cen = rsiaf['WFI_CEN']
V2Ref   = wfi_cen.V2Ref   # +1546.38"  (WFI focal-plane offset from V3 boresight)
V3Ref   = wfi_cen.V3Ref   # -892.79"
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]


def sca_polygons(ra_cen, dec_cen, pa):
    """Return list of 18 SCA polygons, each [[ra,dec], …] × 4 corners."""
    att = attitude_matrix(V2Ref, V3Ref, ra_cen, dec_cen, pa)
    polys = []
    for sensor in sensors:
        sensor.set_attitude_matrix(att)
        c = sensor.corners("sky")          # [ra_4pts, dec_4pts]
        polys.append([[round(float(c[0][i]), 5), round(float(c[1][i]), 5)]
                      for i in range(4)])
    return polys


# ── read ECSV ────────────────────────────────────────────────────────────────
print("Reading roman_gps.sim.ecsv …")
t = Table.read(ECSV)


def unique_pointings(target_name):
    """Return sorted list of unique (ra, dec, pa) tuples for a TARGET_NAME."""
    sub = t[t["TARGET_NAME"] == target_name]
    pts = {}
    for r in sub:
        key = (round(float(r["RA"]), 6), round(float(r["DEC"]), 6),
               round(float(r["PA"]), 3))
        pts[key] = True
    return sorted(pts.keys())


# ── Deep+Spec_W51 ─────────────────────────────────────────────────────────────
print("Computing polygons for Deep+Spec_W51 …")
w51_pts = unique_pointings("Deep+Spec_W51")
print(f"  {len(w51_pts)} unique pointings → {len(w51_pts)*18} polygons")

w51_data = []
for ra, dec, pa in w51_pts:
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal = coord.galactic
    w51_data.append({
        "ra": ra, "dec": dec, "pa": pa,
        "l": round(float(gal.l.deg), 4),
        "b": round(float(gal.b.deg), 4),
        "polys": sca_polygons(ra, dec, pa),
    })
print(f"  Done.")

# ── Disk7 (all) ───────────────────────────────────────────────────────────────
print("Computing polygons for Disk7 (all 632 pointings) …")
d7_pts = unique_pointings("Disk7")
print(f"  {len(d7_pts)} unique pointings → {len(d7_pts)*18} polygons")
print("  (this may take ~30 s) …")

d7_data = []
for i, (ra, dec, pa) in enumerate(d7_pts):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(d7_pts)} …")
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal = coord.galactic
    d7_data.append({
        "ra": ra, "dec": dec, "pa": pa,
        "l": round(float(gal.l.deg), 4),
        "b": round(float(gal.b.deg), 4),
        "polys": sca_polygons(ra, dec, pa),
    })
print(f"  Done.")

# ── serialize to JSON ─────────────────────────────────────────────────────────
print("Serialising …")
w51_json = json.dumps(w51_data, separators=(",", ":"))
d7_json  = json.dumps(d7_data,  separators=(",", ":"))
print(f"  Deep+Spec_W51 JSON: {len(w51_json)//1024} KB")
print(f"  Disk7 JSON:         {len(d7_json)//1024} KB")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roman RGPS — W51 Footprint (pysiaf)</title>
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: system-ui, 'Segoe UI', Roboto, sans-serif; background: #000; }}
    #aladin {{ position: absolute; inset: 0; }}

    #ui {{
      position: absolute; top: 10px; right: 10px; z-index: 10;
      width: 230px;
      background: rgba(15,15,20,0.85); color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; font-size: 12px;
      backdrop-filter: blur(6px);
      overflow: hidden;
    }}
    #ui h3 {{
      margin: 0; padding: 8px 12px; font-size: 13px; font-weight: 600;
      background: rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.1);
      letter-spacing: .3px;
    }}
    .section {{ padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.07); }}
    .section-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .8px;
                      color: #888; margin-bottom: 5px; }}
    .btn-row {{ display: flex; gap: 4px; flex-wrap: wrap; }}
    button {{
      cursor: pointer; padding: 4px 9px; border: 1px solid rgba(255,255,255,0.18);
      border-radius: 4px; background: rgba(255,255,255,0.07); color: #ddd;
      font-size: 11px; transition: background .15s, color .15s;
    }}
    button:hover {{ background: rgba(255,255,255,0.18); color: #fff; }}
    button.active {{ border-color: currentColor; font-weight: 600; }}
    button.layer-w51.active  {{ background: rgba(255,200,30,0.25); color: #ffd040; border-color: #ffd040; }}
    button.layer-disk7.active {{ background: rgba(80,200,255,0.22); color: #50c8ff; border-color: #50c8ff; }}
    button.layer-jwst.active {{ background: rgba(255,80,80,0.2); color: #ff6666; border-color: #ff6666; }}
    button.overlay.active    {{ background: rgba(100,220,100,0.2); color: #7ddf7d; border-color: #7ddf7d; }}
    button.survey.active     {{ background: rgba(180,180,255,0.2); color: #c0c0ff; border-color: #c0c0ff; }}

    .legend-grid {{ display: grid; grid-template-columns: 14px 1fr; gap: 4px 6px; align-items: center; }}
    .swatch {{ width: 14px; height: 3px; border-radius: 2px; }}
    .legend-label {{ font-size: 11px; color: #ccc; }}

    #info {{
      position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
      z-index: 10; pointer-events: none;
      background: rgba(10,10,15,0.7); color: #aaa;
      padding: 4px 14px; border-radius: 4px; font-size: 11px;
      border: 1px solid rgba(255,255,255,0.1);
    }}
    #loading {{
      position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
      z-index: 20; color: #ccc; font-size: 14px;
      background: rgba(10,10,15,0.85); padding: 16px 28px; border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.15);
    }}
  </style>
</head>
<body>
  <div id="aladin"></div>
  <div id="loading">⏳ Building WFI overlay polygons…</div>

  <div id="ui">
    <h3>Roman RGPS — W51 WFI Footprint</h3>

    <div class="section">
      <div class="section-label">Background survey</div>
      <div class="btn-row">
        <button class="survey active" data-survey="P/DSS2/color">DSS</button>
        <button class="survey" data-survey="P/2MASS/color">2MASS</button>
        <button class="survey" data-survey="P/allWISE/color">WISE</button>
        <button class="survey" data-survey="P/Spitzer/GLIMPSE360">GLIMPSE</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Image overlay</div>
      <div class="btn-row">
        <button class="overlay" data-hips="CDS/P/HST/EPO">HST Outreach</button>
        <button class="overlay" data-hips="P/allWISE/W4">WISE W4</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">WFI footprint layers</div>
      <div class="btn-row">
        <button class="layer-disk7 active" id="toggle-disk7">Disk7 (all)</button>
        <button class="layer-w51  active" id="toggle-w51">Deep+Spec W51</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">JWST Region</div>
      <div class="btn-row">
        <button class="layer-jwst active" id="toggle-jwst-target">target_area.reg</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Legend</div>
      <div class="legend-grid">
        <div class="swatch" style="background:#50c8ff"></div>
        <div class="legend-label">Disk7 (l=29–50.5°, PA=0°)</div>
        <div class="swatch" style="background:#ffd040"></div>
        <div class="legend-label">Deep+Spec W51 (l=49.4°)</div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:#666;line-height:1.5;">
        18 SCAs/pointing · PA=0° (nominal)<br>
        Source: roman_gps.sim.ecsv (APT output)<br>
        Geometry: pysiaf Roman SIAF<br>
        WFI01_FULL … WFI18_FULL apertures
      </div>
    </div>
  </div>

  <div id="info">Roman RGPS · pysiaf WFI geometry · {len(d7_data)} Disk7 + {len(w51_data)} Deep W51 pointings · 18 SCAs each</div>

  <script>
  // ── Real WFI SCA polygon data from pysiaf + roman_gps.sim.ecsv ──────────
  // Each entry: {{ra, dec, pa, l, b, polys}}
  // polys: 18 arrays of 4 [RA, Dec] corner pairs (ICRS degrees)
  // Apertures: WFI01_FULL … WFI18_FULL via pysiaf Roman SIAF
  // PA=0 (nominal APT schedule)
  const DEEP_W51 = {w51_json};
  const DISK7    = {d7_json};

  const COLOR_W51   = '#ffd040';
  const COLOR_DISK7 = '#50c8ff';
  const COLOR_JWST  = '#ff4444';
  const JWST_TARGET_AREA = [[266.447262,-28.665625],[266.108552,-29.153986],[266.082376,-29.185502],[266.099365,-29.275829],[265.99966,-29.41689],[266.048406,-29.437154],[266.104287,-29.466578],[266.165553,-29.501565],[266.272546,-29.427271],[266.353777,-29.323741],[266.707721,-28.819941],[266.725371,-28.761043],[267.001371,-28.346998],[266.795062,-28.256913],[266.597259,-28.53203]];

  let aladinInst;
  let overlayW51   = null;
  let overlayDisk7 = null;
  let overlayJwst  = null;
  let showW51   = true;
  let showDisk7 = true;
  let showJwst  = true;
  const HipsLayers = {{}};
  let activeHips = null;

  A.init.then(() => {{
    aladinInst = A.aladin('#aladin', {{
      survey: 'P/DSS2/color',
      target: '49.4 -0.2',   // W51: l=49.4°, b=-0.2° (galactic)
      fov: 5,
      cooFrame: 'galactic',
    }});

    // Build overlays
    function buildPolygons(data, color, name) {{
      const ov = A.graphicOverlay({{ color, lineWidth: 1.2, name }});
      aladinInst.addOverlay(ov);
      data.forEach(pt => {{
        pt.polys.forEach(corners => ov.add(A.polygon(corners)));
      }});
      return ov;
    }}

    overlayDisk7 = buildPolygons(DISK7,    COLOR_DISK7, 'RGPS Disk7');
    overlayW51   = buildPolygons(DEEP_W51, COLOR_W51,   'RGPS Deep+Spec W51');
    overlayJwst  = A.graphicOverlay({{ color: COLOR_JWST, lineWidth: 2.2, name: 'JWST target_area.reg' }});
    aladinInst.addOverlay(overlayJwst);
    overlayJwst.add(A.polygon(JWST_TARGET_AREA));

    document.getElementById('loading').style.display = 'none';

    // ── survey buttons ──
    document.querySelectorAll('button[data-survey]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('button.survey').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        aladinInst.setImageSurvey(btn.dataset.survey);
      }});
    }});

    // ── overlay buttons ──
    document.querySelectorAll('button[data-hips]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const url = btn.dataset.hips;
        if (activeHips === url) {{
          if (HipsLayers[url]) {{ aladinInst.removeImageLayer(HipsLayers[url]); delete HipsLayers[url]; }}
          activeHips = null;
          btn.classList.remove('active');
        }} else {{
          if (activeHips && HipsLayers[activeHips]) {{
            aladinInst.removeImageLayer(HipsLayers[activeHips]);
            delete HipsLayers[activeHips];
            document.querySelector(`button[data-hips="${{activeHips}}"]`).classList.remove('active');
          }}
          const layer = aladinInst.createImageSurvey(url, url, url, 'ICRS', 3, {{imgFormat:'png'}});
          HipsLayers[url] = aladinInst.setOverlayImageLayer(layer);
          activeHips = url;
          btn.classList.add('active');
        }}
      }});
    }});

    // ── WFI layer toggles ──
    document.getElementById('toggle-disk7').addEventListener('click', function() {{
      showDisk7 = !showDisk7;
      showDisk7 ? overlayDisk7.show() : overlayDisk7.hide();
      this.classList.toggle('active', showDisk7);
    }});
    document.getElementById('toggle-w51').addEventListener('click', function() {{
      showW51 = !showW51;
      showW51 ? overlayW51.show() : overlayW51.hide();
      this.classList.toggle('active', showW51);
    }});
    document.getElementById('toggle-jwst-target').addEventListener('click', function() {{
      showJwst = !showJwst;
      showJwst ? overlayJwst.show() : overlayJwst.hide();
      this.classList.toggle('active', showJwst);
    }});
  }});
  </script>
</body>
</html>"""

with open(OUT_HTML, "w") as f:
    f.write(HTML)

size_kb = len(HTML) // 1024
print(f"\nWrote {OUT_HTML}")
print(f"File size: {size_kb} KB ({len(HTML):,} bytes)")

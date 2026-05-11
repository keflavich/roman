import json
import numpy as np
import pysiaf
from pysiaf.utils.rotations import attitude_matrix
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

ECSV = (
    "/Users/adam/repos/roman_notebooks/notebooks/"
    "footprint_visualization/aux_data/roman_gps.sim.ecsv"
)

print("Loading Roman SIAF …")
rsiaf = pysiaf.Siaf("Roman")
wfi_cen = rsiaf['WFI_CEN']
V2Ref   = wfi_cen.V2Ref   # +1546.38"  (WFI focal-plane offset from V3 boresight)
V3Ref   = wfi_cen.V3Ref   # -892.79"
sensors = [rsiaf[f"WFI{j:02d}_FULL"] for j in range(1, 19)]

print("Reading roman_gps.sim.ecsv …")
t = Table.read(ECSV)


def unique_pointings(target_name):
    sub = t[t["TARGET_NAME"] == target_name]
    pts = {}
    for r in sub:
        key = (round(float(r["RA"]), 6), round(float(r["DEC"]), 6),
               round(float(r["PA"]), 3))
        pts[key] = True
    return sorted(pts.keys())


def all_sca_polygons(pointings):
    """Return flat list of all SCA corner polygons for a list of pointings."""
    polys = []
    for ra, dec, pa in pointings:
        attmat = attitude_matrix(V2Ref, V3Ref, ra, dec, pa)
        for sensor in sensors:
            sensor.set_attitude_matrix(attmat)
            c = sensor.corners("sky")          # (ra_4pts, dec_4pts)
            polys.append([[round(float(c[0][i]), 5), round(float(c[1][i]), 5)]
                          for i in range(4)])
    return polys


print("Computing Deep+Spec_W51 …")
w51_pts = unique_pointings("Deep+Spec_W51")
deep_polys = all_sca_polygons(w51_pts)
print(f"  {len(deep_polys)} SCA polygons")

print("Computing Disk7 (may take ~30 s) …")
d7_pts = unique_pointings("Disk7")
disk7_polys = all_sca_polygons(d7_pts)
print(f"  {len(disk7_polys)} SCA polygons")

deep_poly_js  = json.dumps(deep_polys,  separators=(',', ':'))
disk7_poly_js = json.dumps(disk7_polys, separators=(',', ':'))

html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roman RGPS Footprint — W51 — Aladin Lite</title>
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body { height: 100%; margin: 0; font-family: system-ui, 'Segoe UI', Roboto, sans-serif; background: #000; }
    #aladin { position: absolute; inset: 0; }
    #ui {
      position: absolute; top: 10px; right: 10px; z-index: 10;
      width: 220px;
      background: rgba(15,15,20,0.82); color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; font-size: 12px;
      backdrop-filter: blur(6px);
      overflow: hidden;
    }
    #ui h3 {
      margin: 0; padding: 8px 12px; font-size: 13px; font-weight: 600;
      background: rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.1);
      letter-spacing: .3px;
    }
    .section { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.07); }
    .section-label { font-size: 10px; text-transform: uppercase; letter-spacing: .8px;
                     color: #888; margin-bottom: 5px; }
    .btn-row { display: flex; gap: 4px; flex-wrap: wrap; }
    button {
      cursor: pointer; padding: 4px 9px; border: 1px solid rgba(255,255,255,0.18);
      border-radius: 4px; background: rgba(255,255,255,0.07); color: #ddd;
      font-size: 11px; transition: background .15s, color .15s;
    }
    button:hover { background: rgba(255,255,255,0.18); color: #fff; }
    button.active { border-color: currentColor; font-weight: 600; }
    button.overlay.active { background: rgba(100,220,100,0.2); color: #7ddf7d; border-color: #7ddf7d; }
    button.survey.active { background: rgba(180,180,255,0.2); color: #c0c0ff; border-color: #c0c0ff; }
    button.layer-jwst.active { background: rgba(255,80,80,0.2); color: #ff6666; border-color: #ff6666; }
    .legend-grid { display: grid; grid-template-columns: 14px 1fr; gap: 4px 6px; align-items: center; }
    .swatch { width: 14px; height: 3px; border-radius: 2px; }
    .legend-label { font-size: 11px; color: #ccc; }
    #info {
      position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
      z-index: 10; pointer-events: none;
      background: rgba(10,10,15,0.7); color: #aaa;
      padding: 4px 14px; border-radius: 4px; font-size: 11px;
      border: 1px solid rgba(255,255,255,0.1);
    }
  </style>
</head>
<body>
  <div id="aladin"></div>

  <div id="ui">
    <h3>Roman RGPS — W51 Footprint</h3>

    <div class="section">
      <div class="section-label">Background survey</div>
      <div class="btn-row">
        <button class="survey active" data-survey="P/DSS2/color">DSS</button>
        <button class="survey" data-survey="P/2MASS/color">2MASS</button>
        <button class="survey" data-survey="P/allWISE/color">WISE</button>
        <button class="survey" data-survey="P/Spitzer/GLIMPSE360">GLIMPSE</button>
        <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/rgb_final_uncropped_hips/">CMZ RGB</button>
        <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/MUSTANG_12m_feather_noaxes_hips/">MUSTANG</button>
        <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/jwst_cmz_hips/">JWST CMZ</button>
        <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/Brick_RGB_444-356-200_transparent_hips/">Brick JWST</button>
        <button class="survey" data-survey="https://starformation.astro.ufl.edu/avm_images/SgrA_RGB_MIRI_1500-1000-560_transparent_hips/">Sgr A* MIRI</button>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Image overlay (click to toggle)</div>
      <div class="btn-row">
        <button class="overlay" data-hips="https://starformation.astro.ufl.edu/avm_images/w51_RGB_162-210-187_transparent_hips/">W51 RGB</button>
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
        <div class="swatch" style="background:#ffa040;height:2px"></div>
        <div class="legend-label">Deep_W51 (l=49.4°, b=−0.2°)</div>
        <div class="swatch" style="background:#4488ff;height:1px;opacity:0.6"></div>
        <div class="legend-label">Disk7 region (l=29–50.5°)</div>
        <div class="swatch" style="background:#7ddf7d;height:2px"></div>
        <div class="legend-label">Image overlay (active)</div>
      </div>
      <div style="margin-top:6px; font-size:10px; color:#666; line-height:1.6;">
        Source: Full_Roman_Target_list.csv<br>
        (github.com/rachel3834/rgps)<br>
        Real WFI SCA footprints via pysiaf<br>
        Source: roman_gps.sim.ecsv (APT)
      </div>
    </div>

    <div class="section" style="border:0; padding-bottom:9px;">
      <button id="dl" style="width:100%; padding:5px;">↓ Export survey regions JSON</button>
    </div>
  </div>

  <div id="info">Roman RGPS · pysiaf WFI geometry · Deep+Spec_W51 (8 pointings) · Disk7 (632 pointings) · Source: roman_gps.sim.ecsv</div>

  <script>
  // ─────────────────────────────────────────────────────────────────────────
  // RGPS survey regions near W51, from Full_Roman_Target_list.csv
  // (github.com/rachel3834/rgps)
  //
  // Deep_W51:  single WFI deep pointing, Lcen=49.4°, Bcen=−0.2°
  //            bounding box l=[48.6, 50.2°], b=[−1°, +0.6°]
  //
  // Disk7:     wide-area disk survey, l=[29°, 50.5°], b=[−2°, +2°]
  //            W51 lies near the eastern edge of this region.
  //
  // Polygons are the bounding-box outlines sampled along each galactic-coord
  // edge and converted to ICRS (20 points per side).
  // Per-SCA WFI tile positions are NOT in the official repo and are not shown.
  // ─────────────────────────────────────────────────────────────────────────

  const DEEP_W51_POLYGONS = """ + deep_poly_js + """;
  const DISK7_POLYGONS    = """ + disk7_poly_js + """;
  const JWST_TARGET_AREA = [[266.447262,-28.665625],[266.108552,-29.153986],[266.082376,-29.185502],[266.099365,-29.275829],[265.99966,-29.41689],[266.048406,-29.437154],[266.104287,-29.466578],[266.165553,-29.501565],[266.272546,-29.427271],[266.353777,-29.323741],[266.707721,-28.819941],[266.725371,-28.761043],[267.001371,-28.346998],[266.795062,-28.256913],[266.597259,-28.53203]];

  const REGIONS = {
    "Deep_W51": {
      lcen: 49.4, bcen: -0.2,
      lmin: 48.6, lmax: 50.2, bmin: -1.0, bmax: 0.6,
      polygons: DEEP_W51_POLYGONS,
      color: '#ffa040', lineWidth: 2.0
    },
    "Disk7": {
      lmin: 29.0, lmax: 50.5, bmin: -2.0, bmax: 2.0,
      polygons: DISK7_POLYGONS,
      color: '#4488ff', lineWidth: 1.0
    }
  };

  let aladinInstance;
  const OVERLAY_LAYERS = {};
  let activeOverlayUrl = null;
  let jwstOverlay = null;
  let showJwst = true;

  A.init.then(() => {
    aladinInstance = A.aladin('#aladin', {
      survey: 'P/DSS2/color',
      target: '49.4 -0.2',   // Galactic l, b — Deep_W51 centre
      fov: 4,
      cooFrame: 'galactic',
    });

    // Draw WFI SCA footprints
    Object.entries(REGIONS).forEach(([name, r]) => {
      const ov = A.graphicOverlay({ color: r.color, lineWidth: r.lineWidth, name: name });
      aladinInstance.addOverlay(ov);
      r.polygons.forEach(corners => ov.add(A.polygon(corners)));
    });

    jwstOverlay = A.graphicOverlay({ color: '#ff4444', lineWidth: 2.2, name: 'JWST target_area.reg' });
    aladinInstance.addOverlay(jwstOverlay);
    jwstOverlay.add(A.polygon(JWST_TARGET_AREA));

    // HiPS image overlay manager
    function showOverlay(url) {
      if (activeOverlayUrl && activeOverlayUrl !== url && OVERLAY_LAYERS[activeOverlayUrl]) {
        OVERLAY_LAYERS[activeOverlayUrl].setOpacity(0);
      }
      if (!OVERLAY_LAYERS[url]) {
        OVERLAY_LAYERS[url] = A.HiPS(url, { opacity: 1.0 });
        aladinInstance.setOverlayImageLayer(OVERLAY_LAYERS[url], 'image_overlay');
      } else {
        aladinInstance.setOverlayImageLayer(OVERLAY_LAYERS[url], 'image_overlay');
        OVERLAY_LAYERS[url].setOpacity(1.0);
      }
      activeOverlayUrl = url;
    }
    function hideOverlay() {
      if (activeOverlayUrl && OVERLAY_LAYERS[activeOverlayUrl]) {
        OVERLAY_LAYERS[activeOverlayUrl].setOpacity(0);
      }
      activeOverlayUrl = null;
    }

    document.querySelectorAll('button.survey').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('button.survey').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const _tgt = btn.dataset.survey;
        aladinInstance.setImageSurvey(_tgt.startsWith('http') ? A.HiPS(_tgt) : _tgt);
      };
    });

    document.querySelectorAll('button.overlay').forEach(btn => {
      btn.onclick = () => {
        const url = btn.dataset.hips;
        if (btn.classList.contains('active')) {
          hideOverlay();
          btn.classList.remove('active');
        } else {
          document.querySelectorAll('button.overlay').forEach(b => b.classList.remove('active'));
          showOverlay(url);
          btn.classList.add('active');
        }
      };
    });

    document.getElementById('toggle-jwst-target').addEventListener('click', function() {
      showJwst = !showJwst;
      showJwst ? jwstOverlay.show() : jwstOverlay.hide();
      this.classList.toggle('active', showJwst);
    });

    document.getElementById('dl').onclick = () => {
      const out = {};
      Object.entries(REGIONS).forEach(([name, r]) => {
        const entry = { lmin: r.lmin, lmax: r.lmax, bmin: r.bmin, bmax: r.bmax };
        if (r.lcen !== undefined) { entry.lcen = r.lcen; entry.bcen = r.bcen; }
        out[name] = entry;
      });
      const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'roman_rgps_w51_regions.json';
      document.body.appendChild(a); a.click(); a.remove();
    };
  });
  </script>
</body>
</html>
"""

with open("/Users/adam/Downloads/roman_footprint_rgps_w51.html", "w") as f:
    f.write(html)

print("Written roman_footprint_rgps_w51.html")

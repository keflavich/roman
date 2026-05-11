"""
Generate roman_footprint_rgps_w51.html from template + pysiaf WFI polygons.
Corrected to use attitude_matrix(V2Ref, V3Ref, ra, dec, pa) so footprints
are placed at the actual WFI pointing position (not displaced by ~0.496 deg
due to the WFI_CEN focal-plane offset from the V3 boresight).
"""
import json, math
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


# Roman WFI bandpass central wavelengths (µm) — for ordering
_BP_WAVE = {'F062': 0.62, 'F087': 0.87, 'F106': 1.06, 'F129': 1.29,
            'F146': 1.46, 'F158': 1.58, 'F184': 1.84, 'F213': 2.13,
            'GRISM': 1.50, 'PRISM': 1.50}


def unique_pointings(target_name):
    """Return sorted (ra, dec, pa) tuples, deduped on longest-wavelength bandpass."""
    sub = t[t["TARGET_NAME"] == target_name]
    # keep only the longest-wavelength imaging bandpass available for this target
    bps = sorted(set(str(b) for b in sub["BANDPASS"]),
                 key=lambda x: _BP_WAVE.get(x, 0.0))
    sub = sub[sub["BANDPASS"] == bps[-1]]
    pts = {}
    for r in sub:
        key = (round(float(r["RA"]), 6), round(float(r["DEC"]), 6),
               round(float(r["PA"]), 3))
        pts[key] = True
    return sorted(pts.keys())


def deduplicate_close_pointings(pointings, min_sep_arcmin=7.5):
    """
    Greedy spatial deduplication: discard any pointing whose centre lies within
    min_sep_arcmin of a pointing already in the kept set.  First occurrence wins.
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
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        if np.all(np.arccos(cos_sep) >= thresh):
            kept.append(i)
    return [pointings[i] for i in kept]


def sca_polygons(ra_cen, dec_cen, pa):
    """Return list of 18 SCA polygons using corrected attitude_matrix."""
    attmat = attitude_matrix(V2Ref, V3Ref, ra_cen, dec_cen, pa)
    polys = []
    for sensor in sensors:
        sensor.set_attitude_matrix(attmat)
        c = sensor.corners("sky")          # (ra_4pts, dec_4pts)
        polys.append([[round(float(c[0][i]), 5), round(float(c[1][i]), 5)]
                      for i in range(4)])
    return polys


print("Computing Deep+Spec_W51 polygons …")
_w51_all = unique_pointings("Deep+Spec_W51")
w51_pts  = deduplicate_close_pointings(_w51_all, min_sep_arcmin=7.5)
print(f"  {len(_w51_all)} → {len(w51_pts)} pointings after dedup (7.5\u2019 threshold) → {len(w51_pts)*18} polygons")

print("Computing Disk7 polygons (may take ~30 s) …")
_d7_all = unique_pointings("Disk7")
d7_pts  = deduplicate_close_pointings(_d7_all, min_sep_arcmin=7.5)
print(f"  {len(_d7_all)} → {len(d7_pts)} pointings after dedup (7.5\u2019 threshold) → {len(d7_pts)*18} polygons")

PA_ROT = 45.0   # extra roll angle to show as a second overlay

tiles = {}
for i, (ra, dec, pa) in enumerate(w51_pts):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    gal = coord.galactic
    tiles[f"Deep+Spec_W51_{i+1:03d}"] = {
        "ra": ra, "dec": dec, "pa": pa,
        "l": round(float(gal.l.deg), 4),
        "b": round(float(gal.b.deg), 4),
        "polygons": sca_polygons(ra, dec, pa),
    }
for i, (ra, dec, pa) in enumerate(d7_pts):
    if (i + 1) % 100 == 0:
        print(f"  Disk7: {i+1}/{len(d7_pts)} …")
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    gal = coord.galactic
    tiles[f"Disk7_{i+1:04d}"] = {
        "ra": ra, "dec": dec, "pa": pa,
        "l": round(float(gal.l.deg), 4),
        "b": round(float(gal.b.deg), 4),
        "polygons": sca_polygons(ra, dec, pa),
    }
n_sca = sum(len(v['polygons']) for v in tiles.values())
print(f"  {len(tiles)} tiles · {n_sca} SCA polygons")

# PA-rotated footprints (nominal PA + PA_ROT)
print(f"Computing PA+{PA_ROT:.0f}° rotated footprints …")
tiles_rot = {}
for key, tile in tiles.items():
    tiles_rot[key] = {
        "ra":  tile["ra"],  "dec": tile["dec"],
        "l":   tile["l"],   "b":   tile["b"],
        "polygons": sca_polygons(tile["ra"], tile["dec"], tile["pa"] + PA_ROT),
    }
print(f"  Done.")

# Compact JSON for embedding
rgps_js     = json.dumps(tiles,     separators=(',', ':'))
rgps_rot_js = json.dumps(tiles_rot, separators=(',', ':'))

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roman RGPS Footprint — W51 — Aladin Lite</title>
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: system-ui, 'Segoe UI', Roboto, sans-serif; background: #000; }}
    #aladin {{ position: absolute; inset: 0; }}

    /* ── control panel ───────────────────────────────────────────── */
    #ui {{
      position: absolute; top: 10px; right: 10px; z-index: 10;
      width: 220px;
      background: rgba(15,15,20,0.82); color: #e8e8e8;
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
    button.overlay.active {{ background: rgba(100,220,100,0.2); color: #7ddf7d; border-color: #7ddf7d; }}
    button.survey.active  {{ background: rgba(180,180,255,0.2); color: #c0c0ff; border-color: #c0c0ff; }}
    button.layer-pa0.active  {{ background: rgba(255,208,64,0.2);  color: #ffd040; border-color: #ffd040; }}
    button.layer-pa45.active {{ background: rgba(192,128,255,0.2); color: #c080ff; border-color: #c080ff; }}
    button.layer-jwst.active {{ background: rgba(255,80,80,0.2); color: #ff6666; border-color: #ff6666; }}

    /* ── legend ─────────────────────────────────────────────────── */
    .legend-grid {{ display: grid; grid-template-columns: 14px 1fr; gap: 4px 6px; align-items: center; }}
    .swatch {{ width: 14px; height: 3px; border-radius: 2px; }}
    .legend-label {{ font-size: 11px; color: #ccc; }}

    /* ── info bar ────────────────────────────────────────────────── */
    #info {{
      position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
      z-index: 10; pointer-events: none;
      background: rgba(10,10,15,0.7); color: #aaa;
      padding: 4px 14px; border-radius: 4px; font-size: 11px;
      border: 1px solid rgba(255,255,255,0.1);
    }}
  </style>
</head>
<body>
  <div id="aladin"></div>

  <div id="ui">
    <h3>Roman RGPS — W51 Footprint</h3>

    <div class="section">
      <div class="section-label">Background survey</div>
      <div class="btn-row">
        <button class="survey active" data-survey="P/2MASS/color">2MASS</button>
        <button class="survey" data-survey="P/DSS/color">DSS</button>
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
      <div class="section-label">WFI footprint layers</div>
      <div class="btn-row">
        <button class="layer-pa0 active" id="toggle-pa0">PA=0° (celestial N)</button>
        <button class="layer-pa45" id="toggle-pa45">PA=+45° (celestial)</button>
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
        <div class="swatch" style="background:#ffd040"></div>
        <div class="legend-label">Deep+Spec W51 (PA=0° cel.)</div>
        <div class="swatch" style="background:#50c8ff"></div>
        <div class="legend-label">Disk7 (PA=0° cel.)</div>
        <div class="swatch" style="background:#e080ff"></div>
        <div class="legend-label">PA=+45° (celestial) overlay</div>
        <div class="swatch" style="background:#7ddf7d"></div>
        <div class="legend-label">Image overlay (active)</div>
      </div>
      <div style="margin-top:6px; font-size:10px; color:#666; line-height:1.5;">
        {len(w51_pts)} W51 · {len(d7_pts)} Disk7 pointings · 18 SCAs each<br>
        PA = V3-axis angle E of <em>celestial</em> North<br>
        At W51: galactic N ≈ 298° E of cel. N, so PA=0° appears<br>
        tilted ∼62° from galactic N in a galactic frame.<br>
        Source: roman_gps.sim.ecsv (APT)
      </div>
    </div>

    <div class="section" style="border:0; padding-bottom:9px;">
      <button id="dl" style="width:100%; padding:5px;">↓ Export pointing centres JSON</button>
    </div>
  </div>

  <div id="info">Roman RGPS · pysiaf WFI geometry · {len(w51_pts)} W51 + {len(d7_pts)} Disk7 pointings · 18 SCAs each · F213 · roman_gps.sim.ecsv</div>

  <script>
  // ─────────────────────────────────────────────────────────────────────────
  // RGPS polygon data from roman_gps.sim.ecsv via pysiaf (F213 bandpass)
  // Nominal PA (from APT schedule, PA=0°) and PA+45° rotated overlay.
  // Each tile: ra, dec, l, b, polygons  (18 SCA corner arrays of 4 [RA,Dec] pairs)
  // ─────────────────────────────────────────────────────────────────────────
  const RGPS     = {rgps_js};
  const RGPS_ROT = {rgps_rot_js};

  const C_W51       = '#ffd040';   // Deep+Spec W51  nominal PA
  const C_DISK7     = '#50c8ff';   // Disk7           nominal PA
  const C_W51_ROT   = '#ff80c0';   // Deep+Spec W51  PA+45°
  const C_DISK7_ROT = '#c080ff';   // Disk7           PA+45°
  const C_JWST      = '#ff4444';
  const JWST_TARGET_AREA = [[266.447262,-28.665625],[266.108552,-29.153986],[266.082376,-29.185502],[266.099365,-29.275829],[265.99966,-29.41689],[266.048406,-29.437154],[266.104287,-29.466578],[266.165553,-29.501565],[266.272546,-29.427271],[266.353777,-29.323741],[266.707721,-28.819941],[266.725371,-28.761043],[267.001371,-28.346998],[266.795062,-28.256913],[266.597259,-28.53203]];

  let aladinInstance;
  let ovW51=null, ovDisk7=null, ovW51Rot=null, ovDisk7Rot=null, ovJwst=null;
  let showPa0=true, showPa45=false, showJwst=true;
  let activeOverlayUrl = null;

  function addPolygons(data, prefix, color, name) {{
    const ov = A.graphicOverlay({{ color, lineWidth: 1.2, name }});
    aladinInstance.addOverlay(ov);
    Object.entries(data).forEach(([k, tile]) => {{
      if (k.startsWith(prefix))
        tile.polygons.forEach(corners => ov.add(A.polygon(corners)));
    }});
    return ov;
  }}

  A.init.then(() => {{
    aladinInstance = A.aladin('#aladin', {{
      survey: 'P/2MASS/color',
      target: '49.4 -0.2',
      fov: 3,
      cooFrame: 'galactic',
    }});

    // ── Build all four overlay layers ────────────────────────────
    ovDisk7    = addPolygons(RGPS,     'Disk7',         C_DISK7,     'Disk7 (PA=0°)');
    ovW51      = addPolygons(RGPS,     'Deep+Spec_W51', C_W51,       'Deep+Spec W51 (PA=0°)');
    ovDisk7Rot = addPolygons(RGPS_ROT, 'Disk7',         C_DISK7_ROT, 'Disk7 (PA=+45°)');
    ovW51Rot   = addPolygons(RGPS_ROT, 'Deep+Spec_W51', C_W51_ROT,   'Deep+Spec W51 (PA=+45°)');
    ovJwst     = A.graphicOverlay({{ color: C_JWST, lineWidth: 2.2, name: 'JWST target_area.reg' }});
    aladinInstance.addOverlay(ovJwst);
    ovJwst.add(A.polygon(JWST_TARGET_AREA));
    ovDisk7Rot.hide();
    ovW51Rot.hide();

    // ── PA layer toggles ─────────────────────────────────────────
    document.getElementById('toggle-pa0').addEventListener('click', function() {{
      showPa0 = !showPa0;
      showPa0 ? ovDisk7.show()    : ovDisk7.hide();
      showPa0 ? ovW51.show()      : ovW51.hide();
      this.classList.toggle('active', showPa0);
    }});
    document.getElementById('toggle-pa45').addEventListener('click', function() {{
      showPa45 = !showPa45;
      showPa45 ? ovDisk7Rot.show() : ovDisk7Rot.hide();
      showPa45 ? ovW51Rot.show()   : ovW51Rot.hide();
      this.classList.toggle('active', showPa45);
    }});
    document.getElementById('toggle-jwst-target').addEventListener('click', function() {{
      showJwst = !showJwst;
      showJwst ? ovJwst.show() : ovJwst.hide();
      this.classList.toggle('active', showJwst);
    }});

    // ── HiPS image overlay manager ───────────────────────────────
    // Create a fresh HiPS layer when toggling on; remove by name when toggling off.
    // (A.HiPS objects do not reliably support setOpacity in Aladin v3)
    function showOverlay(url) {{
      aladinInstance.removeImageLayer('image_overlay');   // drop any existing
      const hips = A.HiPS(url, {{ opacity: 1.0 }});
      aladinInstance.setOverlayImageLayer(hips, 'image_overlay');
      activeOverlayUrl = url;
    }}
    function hideOverlay() {{
      aladinInstance.removeImageLayer('image_overlay');
      activeOverlayUrl = null;
    }}

    // ── Background survey switcher ───────────────────────────────
    document.querySelectorAll('button.survey').forEach(btn => {{
      btn.onclick = () => {{
        document.querySelectorAll('button.survey').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tgt = btn.dataset.survey;
        aladinInstance.setImageSurvey(tgt.startsWith('http') ? A.HiPS(tgt) : tgt);
      }};
    }});

    // ── Image overlay toggle ──────────────────────────────────────
    document.querySelectorAll('button.overlay').forEach(btn => {{
      btn.onclick = () => {{
        const url = btn.dataset.hips;
        if (btn.classList.contains('active')) {{
          hideOverlay();
          btn.classList.remove('active');
        }} else {{
          document.querySelectorAll('button.overlay').forEach(b => b.classList.remove('active'));
          showOverlay(url);
          btn.classList.add('active');
        }}
      }};
    }});

    // ── Download button ──────────────────────────────────────────
    document.getElementById('dl').onclick = () => {{
      const out = Object.fromEntries(
        Object.entries(RGPS).map(([name, t]) => [name, {{ra: t.ra, dec: t.dec, l: t.l, b: t.b}}])
      );
      const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'roman_rgps_w51_pointings.json';
      document.body.appendChild(a); a.click(); a.remove();
    }};
  }});
  </script>
</body>
</html>
"""

with open("/Users/adam/Downloads/roman_footprint_rgps_w51.html", "w") as f:
    f.write(html)

print(f"Written roman_footprint_rgps_w51.html")
print(f"  {len(tiles)} tiles, {sum(len(t['polygons']) for t in tiles.values())} total polygons")

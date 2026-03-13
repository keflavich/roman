import re, json, math, random
from astropy.coordinates import SkyCoord
import astropy.units as u

W51_RA, W51_DEC = 290.9583, 14.1
SEARCH_RADIUS = 2.0
SCAS_PER_TILE = 18

def sep(ra1,dec1,ra2,dec2):
    a,b,c,d = map(math.radians,[ra1,dec1,ra2,dec2])
    cs = math.sin(b)*math.sin(d)+math.cos(b)*math.cos(d)*math.cos(a-c)
    return math.degrees(math.acos(max(-1,min(1,cs))))

print("Loading...")
with open("/Users/adam/Downloads/roman_gbtds_rgps_combined.reg","r") as f:
    joined = " ".join(f.read().split("\n"))

matches = re.findall(r"polygon\(([^)]+)\)[^#]*# color=\w+ tag=\{([^}]+)\}", joined)
print(f"Total polygons: {len(matches)}")

polys = []
for cstr, tag in matches:
    if tag != "Disk": continue
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", cstr)]
    ras, decs = nums[::2], nums[1::2]
    cra = sum(ras)/len(ras)
    cdec = sum(decs)/len(decs)
    if sep(cra,cdec,W51_RA,W51_DEC) < SEARCH_RADIUS:
        polys.append({"cra":cra,"cdec":cdec,
                      "corners":[[ras[i],decs[i]] for i in range(len(ras))]})
print(f"Disk polygons within {SEARCH_RADIUS}deg: {len(polys)}")

random.seed(42)
k = round(len(polys)/SCAS_PER_TILE)
print(f"K-means k={k}")
spread = sorted(polys, key=lambda p:(p["cra"],p["cdec"]))
step = max(1, len(spread)//k)
centres = [(spread[min(i*step,len(spread)-1)]["cra"],
            spread[min(i*step,len(spread)-1)]["cdec"]) for i in range(k)]

for iteration in range(100):
    clusters = [[] for _ in range(k)]
    for p in polys:
        dists = [sep(p["cra"],p["cdec"],cx,cy) for cx,cy in centres]
        clusters[dists.index(min(dists))].append(p)
    new_centres = []
    for ci, cl in enumerate(clusters):
        if cl:
            new_centres.append((sum(p["cra"] for p in cl)/len(cl),
                                 sum(p["cdec"] for p in cl)/len(cl)))
        else:
            new_centres.append(centres[ci])
    if new_centres == centres:
        print(f"Converged after {iteration+1} iters")
        break
    centres = new_centres

sizes = [len(c) for c in clusters if c]
print(f"Sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

result_tiles = {}
for i, cl in enumerate(clusters):
    if not cl: continue
    cra  = sum(p["cra"]  for p in cl)/len(cl)
    cdec = sum(p["cdec"] for p in cl)/len(cl)
    coord = SkyCoord(ra=cra*u.deg, dec=cdec*u.deg)
    lval = coord.galactic.l.deg
    bval = coord.galactic.b.deg
    label = f"Tile {i+1}"
    result_tiles[label] = {"ra":round(cra,4),"dec":round(cdec,4),
                            "l":round(lval,2),"b":round(bval,2),
                            "polygons":[p["corners"] for p in cl]}
    print(f"  {label}: {len(cl):2d} SCAs, RA={cra:.3f}, Dec={cdec:.3f}, "
          f"l={lval:.2f}, b={bval:.2f}")

flat_corners = [p["corners"] for p in polys]
out = {"tiles": result_tiles, "all_corners": flat_corners}
with open("/Users/adam/Downloads/rgps_w51_tiles.json","w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {len(result_tiles)} tiles + {len(flat_corners)} polys")

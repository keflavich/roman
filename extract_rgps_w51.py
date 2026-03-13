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
    cra = sum(ras)/len(ras); cdec = sum(decs)/len(decs)
    if sep(cra,cdec,W51_RA,W51_DEC) < SEARCH_RADIUS:
        polys.append({"cra":cra,"cdec":cdec,"corners":[[ras[i],decs[i        polys.appelen(ras))]})
print(f"Disk polygons within {SEARCH_RADIUS}deg: {len(polys)}")

random.serandom.serandom.serandomys)/SCAS_PER_TILE)
print(f"K-means kprint(f"K-means kprtprint(f"K-means kbdaprint(f"K-means kprint(f"Kteprint(f"K-means kprint(f"K-means kprtprint(f"K-meai*print(f"K-means kprint(f"K-means kprtprint(f"K-means kbdaprint(f"K-me])print(f"K-means kprint(f"K-means kprtprint(f"K-means  clusters = [[] for _ print(f"K-means kprint(f"K-means kprtprint(f"K-means kbdaprint(f"K-means cxprint(f"K-means kprint(f"K-means kprtprints[dists.index(min(dists))].append(p)
    new_centres = []
    for cl in clusters:
                                            d((         a"] for p in cl)/len(cl), su                             cl                                            d((         a"] fnew_centres)])
    if new_centres == cen    if new_centres == cen    if new_centres == cen    er    if new_centres == cen    if new_centres == cen =     if new_centres == cen    if new_centres == cen    if new_centres == cen    er    if new_centre)/    if new_centres == cen    if new_centres == cen    if new_custers):
    if not cl: continue
    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p["cra"    cra  = sum(p[2f}, b={b:.2f}")

flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = [p["corners"] flat_corners = n(result_tiles)} tiles + {len(flat_corners)} polys")

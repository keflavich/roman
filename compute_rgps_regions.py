from astropy.coordinates import SkyCoord
import astropy.units as u
import json

# Deep_W51 from CSV: Lcen=49.4, Bcen=-0.2, Lmin=48.6, Lmax=50.2, Bmin=-1, Bmax=0.6
# Disk7: l=[29, 50.5], b=[-2, 2]

w51_gal = SkyCoord(l=49.4*u.deg, b=-0.2*u.deg, frame='galactic')
w51_icrs = w51_gal.icrs
print("Deep_W51 center: l=49.4, b=-0.2")
print("  ICRS: RA={:.4f}, Dec={:.4f}".format(w51_icrs.ra.deg, w51_icrs.dec.deg))

# Deep_W51 bounding box (sampled along each edge for curved projection)
def gal_edge(l0, l1, b0, b1, n=20):
    """Sample n points along each edge of a galactic rectangle, return ICRS corners."""
    pts = []
    # bottom edge (b=b0, l from l0 to l1)
    for l in [l0 + (l1-l0)*i/(n-1) for i in range(n)]:
        c = SkyCoord(l=l*u.deg, b=b0*u.deg, frame='galactic')
        pts.append([round(c.icrs.ra.deg, 4), round(c.icrs.dec.deg, 4)])
    # right edge (l=l1, b from b0 to b1)
    for b in [b0 + (b1-b0)*i/(n-1) for i in range(1, n)]:
        c = SkyCoord(l=l1*u.deg, b=b*u.deg, frame='galactic')
        pts.append([round(c.icrs.ra.deg, 4), round(c.icrs.dec.deg, 4)])
    # top edge (b=b1, l from l1 to l0)
    for l in [l1 + (l0-l1)*i/(n-1) for i in range(1, n)]:
        c = SkyCoord(l=l*u.deg, b=b1*u.deg, frame='galactic')
        pts.append([round(c.icrs.ra.deg, 4), round(c.icrs.dec.deg, 4)])
    # left edge (l=l0, b from b1 to b0)
    for b in [b1 + (b0-b1)*i/(n-1) for i in range(1, n)]:
        c = SkyCoord(l=l0*u.deg, b=b*u.deg, frame='galactic')
        pts.append([round(c.icrs.ra.deg, 4), round(c.icrs.dec.deg, 4)])
    return pts

deep_w51 = gal_edge(48.6, 50.2, -1.0, 0.6)
disk7    = gal_edge(29.0, 50.5, -2.0, 2.0)

print("\nDeep_W51 polygon has {} points".format(len(deep_w51)))
print("Disk7 polygon has {} points".format(len(disk7)))

out = {
    "Deep_W51": {
        "lcen": 49.4, "bcen": -0.2,
        "lmin": 48.6, "lmax": 50.2, "bmin": -1.0, "bmax": 0.6,
        "polygon": deep_w51
    },
    "Disk7": {
        "lmin": 29.0, "lmax": 50.5, "bmin": -2.0, "bmax": 2.0,
        "polygon": disk7
    }
}
with open("/Users/adam/Downloads/rgps_w51_regions.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved rgps_w51_regions.json")
print("Deep_W51 first few points:", deep_w51[:3])

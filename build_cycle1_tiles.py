#!/usr/bin/env python
"""
Roman Cycle 1 tile pointings for roman_footprints.html -> roman_cycle1_tiles.json

Inputs, per program, in APT_WORK:
    <id>.apt       APT XML from https://www.stsci.edu/roman-program-info/download/roman/apt/<id>/
    <id>.pointing  from Roman APT 2026.4.4:  APT -nogui -nostatus -export pointing -output . <id>.apt

The .pointing file lists, per survey step and observation, every exposure of
every mosaic tile with V2/V3 (arcsec): where the target sits in the telescope
frame.  Survey steps appear in the order of the XML's SurveyPlanStep elements;
each step names a PassPlan, whose TargetSelection is a fixed target or a
region target.  For region targets the exposures are a prototype repeated at
every "Region Point" the .pointing file lists for that region.

One row per unique tile:  [ra, dec, v2, v3, pa]
    ra, dec  target (fixed target, or region point), deg
    v2, v3   exposure 1 of the tile (dithers dropped), arcsec
    pa       pysiaf attitude PA = APT orient + 180 deg, where the APT orient
             is the midpoint of the step's first OrientRange or, for region
             targets without one, the region's planned orient; null when
             unconstrained (the page draws it at the PA slider value).  An
             unconstrained row is dropped when the same tile also appears
             with a constrained orient.
The page draws attitude(v2, v3, ra, dec, pa) applied to the WFI outline or
SCA corners (pysiaf conventions).

The +180: each region target's table in the .pointing file gives every region
point's V2/V3 while the telescope points at the region's reference position
at its planned orient.  pysiaf attitude(0, 0, ref_ra, ref_dec, orient + 180)
reproduces those positions (M31 halo region: 4" median); with orient alone
they miss by degrees.  check_orient_convention() repeats that test on every
run.
"""
import json
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

from astropy import units as u
from astropy.coordinates import SkyCoord

APT_WORK = "/blue/adamginsburg/adamginsburg/apt_roman/work"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "roman_cycle1_tiles.json")

# id: (short name, PI, hours or None, kind)
PROGRAMS = OrderedDict([
    (1400, ("HLWAS", None, None, "core")),
    (1410, ("HLTDS", None, None, "core")),
    (1420, ("GBTDS", None, None, "core")),
    (1430, ("GPS", None, None, "core")),
    (2000, ("Globular cluster census", "Bedin", 167.2, "gas")),
    (2001, ("RXDF", "Yan", 386.41, "gas")),
    (2002, ("M31/M33 legacy", "Gilbert", 310.7, "gas")),
    (2003, ("GRACE grism, CDFS", "Malhotra", 420.9, "gas")),
    (2004, ("Roman-Kepler", "Kraus", 170.8, "gas")),
])


def load_xml(path):
    root = ET.parse(path).getroot()
    for x in root.iter():
        x.tag = x.tag.split("}")[-1]
    return root


def coord(v):
    c = SkyCoord(v.replace(":", " "), unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def parse_apt(root):
    fixed = {t.findtext("Number"): (t.findtext("TargetName"),
                                    coord(t.find("EquatorialCoordinates").get("Value")))
             for t in root.iter("FixedTarget")}
    region = {t.get("Number"): t.get("Name") for t in root.iter("RegionTarget")}
    passes = {pp.get("Number"): pp.findtext("TargetSelection").strip()
              for pp in root.iter("PassPlan")
              if pp.findtext("TargetSelection") is not None}
    return fixed, region, passes


def parse_pointing(path):
    """[{obs: [[13 floats per exposure row], ...]}], {region: dict}"""
    steps, regions = [], {}
    rows, reg = None, None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if line.startswith("Survey Step"):
                steps.append([])
                reg = None
            elif line.startswith("Region Target:"):
                reg = dict(orient=None, ref=None, points=[], v2v3=[])
                regions[s.split(":", 1)[1].strip()] = reg
            elif reg is not None:
                p = s.split()
                if s.startswith("Reference Position:"):
                    reg["ref"] = tuple(float(x) for x in s.split(":")[1].split(","))
                elif s.startswith("Planned Orient:"):
                    reg["orient"] = float(s.split(":")[1].split()[0])
                elif len(p) == 5 and p[0].isdigit():
                    reg["points"].append((float(p[1]), float(p[2])))
                    reg["v2v3"].append((float(p[3]), float(p[4])))
            elif s.startswith("Observation") and steps:
                rows = []
                steps[-1].append(rows)
            else:
                p = s.split()
                if rows is not None and len(p) == 13 and p[0].isdigit():
                    rows.append([float(x) for x in p])
    return steps, regions


def step_orient(step):
    for o in step.iter("OrientRange"):
        lo = float(o.get("OrientMin").split()[0])
        hi = float(o.get("OrientMax").split()[0])
        if hi < lo:
            hi += 360
        return round(((lo + hi) / 2) % 360, 2)
    return None


def apt_pa(orient):
    return None if orient is None else round((orient + 180) % 360, 2)


def check_orient_convention(regions, prog):
    """Region points' V2/V3 must come back with pa = orient + 180.

    The residual grows with region size (APT's tables look planar: tens of
    arcsec for a few-degree region, a few arcmin for the largest), so the
    test is relative: pa = orient + 180 must fit at least 10x better than
    pa = orient.
    """
    import numpy as np
    from pysiaf.utils.rotations import attitude, pointing
    for name, reg in regions.items():
        truth = SkyCoord(*np.array(reg["points"]).T, unit="deg")
        meds = []
        for pa in (apt_pa(reg["orient"]), reg["orient"]):
            att = attitude(0, 0, *reg["ref"], pa)
            pts = SkyCoord(*np.array([pointing(att, v2, v3)
                                      for v2, v3 in reg["v2v3"]]).T, unit="deg")
            meds.append(np.median(pts.separation(truth).arcsec))
        print(f"  {prog} {name}: region points reproduced to {meds[0]:.0f}\" "
              f"median (pa = orient: {meds[1]:.0f}\")")
        if meds[0] * 10 > meds[1]:
            raise ValueError(f"{prog} {name}: orient convention check failed")


def build(prog):
    root = load_xml(f"{APT_WORK}/{prog}.apt")
    fixed, region, passes = parse_apt(root)
    steps_xml = list(root.iter("SurveyPlanStep"))
    steps, regions = parse_pointing(f"{APT_WORK}/{prog}.pointing")
    check_orient_convention(regions, prog)
    if len(steps) != len(steps_xml):
        raise ValueError(f"{prog}: {len(steps)} pointing steps vs "
                         f"{len(steps_xml)} SurveyPlanSteps")
    layers = OrderedDict()
    for obs, sx in zip(steps, steps_xml):
        sel = passes[sx.findtext("PassPlan")]
        orient = apt_pa(step_orient(sx))
        if sel.startswith("Fixed"):
            name, pt = fixed[sel.split(":")[1].strip()]
            pts = [pt]
        else:
            name = region[sel]
            pts = regions[name]["points"]
            if orient is None:
                orient = apt_pa(regions[name]["orient"])
        tiles = layers.setdefault(name, OrderedDict())
        for rows in obs:
            first = {}
            for row in rows:
                if row[0] not in first or row[1] < first[row[0]][1]:
                    first[row[0]] = row
            for row in first.values():
                for ra, dec in pts:
                    key = (round(ra, 6), round(dec, 6),
                           round(row[2], 2), round(row[3], 2))
                    tiles.setdefault(key, set()).add(orient)
    out = OrderedDict()
    for name, tiles in layers.items():
        rows = []
        for key, orients in tiles.items():
            if len(orients) > 1:
                orients.discard(None)
            rows += [list(key) + [o] for o in sorted(orients, key=str)]
        out[name] = rows
    title = root.findtext(".//Title")
    return title, out


def main():
    data = OrderedDict()
    for prog, (short, pi, hours, kind) in PROGRAMS.items():
        title, layers = build(prog)
        data[str(prog)] = dict(short=short, title=title, pi=pi, hours=hours,
                               kind=kind, layers=layers)
        n = sum(len(v) for v in layers.values())
        free = sum(r[4] is None for v in layers.values() for r in v)
        print(f"{prog} {short}: {len(layers)} targets, {n} tiles, "
              f"{free} unconstrained orient")
    with open(OUT, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()

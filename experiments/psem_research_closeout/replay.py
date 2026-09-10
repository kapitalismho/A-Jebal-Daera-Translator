import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
HERE = pathlib.Path(__file__).resolve().parent
SEC_TOL = 0.00005
MAP_TOL = 0.000001
def fail(msg):
    print("replay-FAIL " + msg)
    raise SystemExit(1)
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def load_evidence():
    return json.loads((HERE / "EVIDENCE.json").read_text(encoding="utf-8"))
def parse_numbers(text):
    out = {}
    for k, n in [("OVERALL SPEAKER DIARIZATION ERROR", "der"), ("MISSED SPEAKER TIME", "miss"), ("FALARM SPEAKER TIME", "fa"), ("SPEAKER ERROR TIME", "conf"), ("SCORED SPEAKER TIME", "scored")]:
        m = re.search(re.escape(k) + r"\s*=\s*([0-9.]+)", text)
        if not m:
            fail("missing " + k)
        out[n] = float(m.group(1))
    return out
def parse_map(text):
    rows = {}
    lines = text.splitlines()
    if not lines or lines[0] != "File,Channel,RefSpeaker,SysSpeaker,isMapped,timeOverlap":
        fail("bad map header")
    for line in lines[1:]:
        if not line.strip():
            continue
        p = line.split(",")
        rows.setdefault(p[2], []).append([p[3], p[4], float(p[5])])
    return rows
def maps_equal(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        ra, rb = a[k], b[k]
        if len(ra) != len(rb):
            return False
        for x, y in zip(sorted(ra), sorted(rb)):
            if x[0] != y[0] or x[1] != y[1] or abs(x[2] - y[2]) > MAP_TOL:
                return False
    return True
def score_once(scorer, ref_t, hyp_t, uem_t, collar, work, perl):
    ref = work / "ref.rttm"
    hyp = work / "hyp.rttm"
    uem = work / "run.uem"
    mp = work / "map.csv"
    ref.write_text(ref_t, encoding="utf-8")
    hyp.write_text(hyp_t, encoding="utf-8")
    uem.write_text(uem_t, encoding="utf-8")
    cmd = [perl, str(scorer), "-r", str(ref), "-s", str(hyp), "-u", str(uem), "-c", str(collar), "-m", "-M", str(mp)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        fail("scorer rc=%d %s" % (p.returncode, p.stderr[:300]))
    return p.stdout, mp.read_text(encoding="utf-8")
def iter_runs(ev):
    std = ev["standard_der"]
    for key, exp in std["expected"].items():
        arm, rest = key.split(".", 1)
        for cname, cval in std["collars"].items():
            if rest.startswith(cname + "."):
                scope = rest[len(cname) + 1:]
                collar = cval
                break
        yield key, std["refs"][scope], std["hyps"][arm][scope], std["uems"][scope], collar, exp
    np2 = ev["np2"]
    for arm, exp in np2["expected"].items():
        yield "NP2." + arm + ".c0", np2["ref"], np2["hyps"][arm], np2["uem"], 0.0, exp
def run_replay(ev):
    perl = shutil.which("perl")
    if not perl:
        fail("Perl not found; install Perl or add it to PATH (Windows Git usr/bin)")
    scorer = HERE / ev["scorer"]["file"]
    if sha_file(scorer) != ev["scorer"]["sha256"]:
        fail("scorer sha mismatch")
    n = 0
    with tempfile.TemporaryDirectory(prefix="psem_replay_") as td:
        work = pathlib.Path(td)
        for key, ref_t, hyp_t, uem_t, collar, exp in iter_runs(ev):
            out, mapt = score_once(scorer, ref_t, hyp_t, uem_t, collar, work, perl)
            got = parse_numbers(out)
            for f in ("miss", "fa", "conf", "scored"):
                if abs(got[f] - exp[f]) > SEC_TOL:
                    fail("%s %s got=%r exp=%r" % (key, f, got[f], exp[f]))
            if round(got["der"], 2) != exp["der"]:
                fail("%s der got=%r exp=%r" % (key, got["der"], exp["der"]))
            if not maps_equal(parse_map(mapt), {k: [list(r) for r in v] for k, v in exp["map"].items()}):
                fail("%s map mismatch" % key)
            n += 1
            print("replay-ok " + key + " der=%.2f" % got["der"])
    print("replay-ok %d/%d" % (n, n))
    return n
def verify_archive(ev):
    idx = json.loads((HERE / "ARCHIVE_INDEX.json").read_text(encoding="utf-8"))
    zp = pathlib.Path(idx["zip_path"])
    if not zp.exists():
        fail("archive missing " + str(zp))
    man_raw = None
    with zipfile.ZipFile(zp) as z:
        try:
            man_raw = z.read("ARCHIVE_MANIFEST.json")
        except KeyError:
            fail("archive manifest absent")
        if hashlib.sha256(man_raw).hexdigest() != idx["manifest_sha256"]:
            fail("archive manifest sha mismatch")
        man = json.loads(man_raw.decode())
        if len(man["members"]) != idx["member_count"]:
            fail("archive member count mismatch")
        for m in man["members"]:
            h = hashlib.sha256()
            with z.open(m["path"]) as f:
                for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(c)
            if h.hexdigest() != m["sha256"]:
                fail("archive member sha mismatch " + m["path"])
        bad = z.testzip()
        if bad:
            fail("archive crc error " + str(bad))
    if sha_file(zp) != idx["zip_sha256"]:
        fail("archive whole sha mismatch")
    print("archive-ok %s members=%d bytes=%d" % (idx["zip_name"], idx["member_count"], idx["zip_bytes"]))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-archive", action="store_true")
    args = ap.parse_args()
    ev = load_evidence()
    if ev.get("format") != "psem-compact-evidence.v1":
        fail("evidence format")
    run_replay(ev)
    if args.verify_archive:
        verify_archive(ev)
if __name__ == "__main__":
    main()

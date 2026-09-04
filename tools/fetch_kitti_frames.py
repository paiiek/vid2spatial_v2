#!/usr/bin/env python3
"""Fetch individual KITTI Tracking frames out of the remote image zip.

Why this exists
---------------
``data/kitti_tracking`` ships ``label_02`` only -- no images -- and the image
zip is 15.8 GB against roughly 18-20 GB free on this machine, so it cannot be
downloaded. The S3 bucket does honour HTTP range requests, so the few hundred
frames an evaluation actually needs can be pulled out of the remote archive
without ever materialising the whole thing. 200 frames is about 160 MB.

Method: read the zip's central directory over ranged reads, compute each
wanted member's byte range from its local-header offset and compressed size,
then fetch those ranges CONCURRENTLY and inflate locally. Reading members
serially through ``zipfile`` over a range-backed file object works but runs at
roughly two frames per minute, because each ``read`` becomes many small
requests; the parallel raw-range path does 197 frames in about seven minutes.

Which frames: by default, the first frame of every track in
``test/full_eval/depth_gt.json``, which is what the z0 estimator needs.

Usage:
    python tools/fetch_kitti_frames.py --out DIR [--max-frames 200]
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not support range requests")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = (off if whence == 0
                     else self._pos + off if whence == 1
                     else self.size + off)
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        data = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except Exception:
                time.sleep(2)
        if data is None:
            raise IOError(f"range read failed at {self._pos}")
        self._pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def wanted_frames(gt_path: Path, max_frames: int):
    """(seq, frame) of each track's first GT record, most-covering first."""
    recs = json.loads(gt_path.read_text())
    first = {}
    for r in recs:
        first.setdefault(r["track"], r)
    counts = Counter((t.split("_")[0], r["frame"]) for t, r in first.items())
    return [k for k, _ in counts.most_common(max_frames)]


_EXTRA_GUESS = 256   # slab allowance for the local extra field, usually enough


def _ranged_read(url: str, start: int, end: int) -> Optional[bytes]:
    """Bytes [start, end] inclusive, retried. None when the fetch failed."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception:
            time.sleep(2)
    return None


def fetch(url: str, zi: zipfile.ZipInfo, out: Path) -> bool:
    """One member, verified against its CRC and written atomically.

    The local header's extra field can be longer than the central directory's,
    so the slab allowance is a guess. When the guess is short the member data
    is TRUNCATED, and a stored (uncompressed) member would then be written out
    as a valid-looking but incomplete PNG with no error anywhere. So the real
    lengths are read back from the local header and the remainder re-fetched,
    and every member is checked against zi.CRC before it is kept.
    """
    start = zi.header_offset
    end = start + 30 + len(zi.filename) + _EXTRA_GUESS + zi.compress_size
    buf = _ranged_read(url, start, end)
    if buf is None or len(buf) < 30:
        return False
    nlen, elen = struct.unpack("<HH", buf[26:30])
    off = 30 + nlen + elen
    data = buf[off:off + zi.compress_size]
    if len(data) < zi.compress_size:          # the extra field overran the guess
        rest = _ranged_read(url, start + off + len(data),
                            start + off + zi.compress_size - 1)
        if rest is None:
            return False
        data += rest
    if len(data) != zi.compress_size:
        return False
    try:
        raw = data if zi.compress_type == 0 else zlib.decompress(data, -15)
    except zlib.error:
        return False                          # never let one member kill the run
    if zi.CRC and (zlib.crc32(raw) & 0xFFFFFFFF) != zi.CRC:
        return False
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(raw)
    tmp.replace(out)                          # atomic: readers see all or nothing
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", required=True, help="output dir for <seq>_<frame:06d>.png")
    ap.add_argument("--gt", default=str(repo / "test/full_eval/depth_gt.json"))
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-mb", type=float, default=2048.0,
                    help="refuse to exceed this much new disk (default 2 GB)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = wanted_frames(Path(a.gt), a.max_frames)
    print(f"want {len(sel)} frames", flush=True)

    t0 = time.time()
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20))
    info = {i.filename: i for i in z.infolist()}
    print(f"central directory read in {time.time() - t0:.0f}s", flush=True)

    jobs, budget = [], a.max_mb * 1024 * 1024
    have = sum(f.stat().st_size for f in out.glob("*.png"))
    for seq, frame in sel:
        dst = out / f"{seq}_{frame:06d}.png"
        zi = info.get(f"training/image_02/{seq}/{frame:06d}.png")
        if zi is None:
            continue
        # Idempotent on SIZE, not on "bigger than a threshold": a partial write
        # from an interrupted run is over 1000 bytes too, and would otherwise be
        # skipped forever and fed to the estimator as good data. Writes are
        # atomic now, so a size match means the member is complete.
        if dst.exists() and dst.stat().st_size == zi.file_size:
            continue
        if have + zi.file_size > budget:
            print(f"[stop] disk budget {a.max_mb:.0f} MB reached at {len(jobs)} frames")
            break
        have += zi.file_size
        jobs.append((zi, dst))
    print(f"fetching {len(jobs)} frames (~{have / 1e6:.0f} MB total on disk)", flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, good in enumerate(ex.map(lambda j: fetch(URL, j[0], j[1]), jobs)):
            ok += bool(good)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(jobs)} ok={ok} t={time.time()-t0:.0f}s", flush=True)

    total = sum(f.stat().st_size for f in out.glob("*.png"))
    print(f"done: {ok}/{len(jobs)} fetched, {len(list(out.glob('*.png')))} frames, "
          f"{total / 1e6:.0f} MB in {out}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())

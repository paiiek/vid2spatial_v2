"""Rasterise the official ISMAR 2026 teaser template (Teaser_Template_ISMAR_2026.pptx).

The template is one 16:9 slide: a media placeholder (x 91..1825, y 0..975 at 1080p),
a gradient footer banner at y 977..1080 carrying the ISMAR 2026 / IEEE / CS / VGTC
logos as vector freeforms, and two centred text placeholders in the banner
("Title of the work", 12 pt; "Authors of the work", 9 pt).  No PowerPoint here, so
the freeform paths are rendered directly with PIL (moveTo/lnTo/cubicBezTo/close,
even-odd fill) at 4x supersampling.

render(pptx) -> RGBA 1920x1080 overlay whose media area is transparent.
"""
import io
import re
import zipfile

from PIL import Image, ImageChops, ImageDraw

W, H = 1920, 1080
EMU_W, EMU_H = 12192000, 6858000
MEDIA = (91, 0, 1825, 975)          # media placeholder box in 1080p pixels
TITLE_BOX = (335, 985, 1585, 1043)  # 12 pt, centred
AUTHOR_BOX = (335, 1045, 1585, 1074)  # 9 pt, centred, anchored bottom
INK = (0x3D, 0x12, 0x09)            # template text colour


def _bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append((
            (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0],
            (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1],
        ))
    return out


def render(pptx_path, ss=4):
    z = zipfile.ZipFile(pptx_path)
    lay = z.read("ppt/slideLayouts/slideLayout1.xml").decode()
    banner = Image.open(io.BytesIO(z.read("ppt/media/image1.png"))).convert("RGBA")
    sx, sy = W * ss / EMU_W, H * ss / EMU_H
    canvas = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))

    pic = re.search(r"<p:pic>.*?</p:pic>", lay, re.S).group(0)
    bx, by, bw, bh = [int(v) for v in re.search(
        r'<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', pic).groups()]
    canvas.alpha_composite(banner.resize((round(bw * sx), round(bh * sy)), Image.LANCZOS),
                           (round(bx * sx), round(by * sy)))

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for sp in re.findall(r"<p:sp>.*?</p:sp>", lay, re.S):
        if "<a:custGeom>" not in sp:
            continue
        ox, oy, cx, cy = [int(v) for v in re.search(
            r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', sp).groups()]
        col = re.search(r'<a:solidFill><a:srgbClr val="([0-9A-Fa-f]{6})"', sp.split("<a:custGeom>")[1])
        rgb = tuple(int(col.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        for path in re.findall(r"<a:path\b[^>]*>.*?</a:path>", sp, re.S):
            pw = int(re.search(r'w="(\d+)"', path).group(1))
            ph = int(re.search(r'h="(\d+)"', path).group(1))
            fx, fy = cx / pw * sx, cy / ph * sy

            def P(m):
                return (ox * sx + int(m[0]) * fx, oy * sy + int(m[1]) * fy)

            polys, cur = [], []
            for tag, body in re.findall(
                    r"<a:(moveTo|lnTo|cubicBezTo|close)\b[^>]*?(?:/>|>(.*?)</a:\1>)", path, re.S):
                pts = [P(m) for m in re.findall(r'<a:pt x="(-?\d+)" y="(-?\d+)"/>', body)]
                if tag == "moveTo":
                    if len(cur) > 2:
                        polys.append(cur)
                    cur = [pts[0]]
                elif tag == "lnTo":
                    cur.append(pts[0])
                elif tag == "cubicBezTo":
                    cur += _bezier(cur[-1], *pts)
                elif tag == "close":
                    if len(cur) > 2:
                        polys.append(cur)
                    cur = []
            if len(cur) > 2:
                polys.append(cur)
            mask = Image.new("1", canvas.size, 0)
            for poly in polys:
                m2 = Image.new("1", canvas.size, 0)
                ImageDraw.Draw(m2).polygon(poly, fill=1)
                mask = ImageChops.logical_xor(mask, m2)
            layer.paste(Image.new("RGBA", canvas.size, rgb + (255,)), (0, 0), mask.convert("L"))
    canvas.alpha_composite(layer)
    return canvas.resize((W, H), Image.LANCZOS)

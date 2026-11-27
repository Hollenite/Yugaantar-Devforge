"""Builds brief/Yugaantar_DOOM_brief.pdf (exactly 2 pages, A4).

    python brief/make_brief.py

Every number in the copy is locked to the submitted model-selection run. Edit RESULTS and rebuild
only after a new selection run. The build fails (nonzero exit) if the PDF is not exactly 2 pages
or if any diagram label, table cell or the artifact strip would overflow.
"""

import math
import sys
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFontFile
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

OUT = Path(__file__).with_name("Yugaantar_DOOM_brief.pdf")
REPO_URL = "https://github.com/Hollenite/Yugaantar-Devforge"
TEAM = "Team We Who Remains — Yugaantar"
TOTAL_PAGES = 2

# --- layout constants ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
LIVE_W = PAGE_W - 2 * MARGIN  # 180 mm
LIVE_H = PAGE_H - 2 * MARGIN  # 267 mm
FOOTER_RULE_Y = 12 * mm
FOOTER_TEXT_Y = 8.2 * mm
TITLE_GAP = 4 * mm
SECTION_GAP_P1 = 2.8 * mm  # plus H2 space-before and paragraph space-after: about 5.5 mm block to block
SECTION_GAP_P2 = 3.5 * mm
MIN_FILL = 0.88

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#5b5b60")
ACCENT = colors.HexColor("#b3261e")
RULE = colors.HexColor("#d9d9de")
FILL = colors.HexColor("#f4f4f6")
TINT = colors.HexColor("#fbe9e7")

TITLE_SIZE, TITLE_LEAD = 18, 22
SUB_SIZE = 9
H2_SIZE, H2_LEAD, H2_BEFORE, H2_AFTER = 12, 15, 6, 3
BODY_SIZE, BODY_LEAD = 11, 14.5
BODY2_SIZE, BODY2_LEAD = 11, 14.5  # page 2 (underfill fix a)
SMALL_SIZE, SMALL_LEAD = 8.5, 10.8
SCORE_SIZE = 9
FOOT_SIZE = 8
STRIP_H, STRIP_SIZES = 8 * mm, (9, 8.5)
BULLET_INDENT = 4 * mm

# Architecture diagram (x in mm from the left margin)
DIA_HEADER_H = 6 * mm
DIA_ROW_H, DIA_ROW_GAP = 13 * mm, 3 * mm
COL_A = (0, 46)
COL_B = (50, 92)
COL_C = (96, 108)
MERGE_X, MERGE_W = 111.5, 1.5  # mm, pt
COL_D = (116, 141)
COL_E = (145, 180)
FUSION_H, FUSION_OFFSET = 12 * mm, 9 * mm  # concat sits above the row-area centre, GRU below
HEAD_H, HEAD_GAP = 15 * mm, 4 * mm

# EgoMap schematic beside the Spatial-memory bullets
EGO_TEXT_W, EGO_GAP, EGO_FIG_W, EGO_FIG_H, EGO_GRID = 120 * mm, 5 * mm, 55 * mm, 35 * mm, 24 * mm
BOX_PAD = 4
BOX_RADIUS = 3
DIA_LABEL, DIA_DETAIL, DIA_MIN = 8.5, 8, 8
ARROW_W, ARROW_HEAD = 0.8, 4

# Results table
TABLE_COLS_MM = [56, 14, 12, 12, 12, 12, 12, 50]
TABLE_SIZE, TABLE_SIZE_MIN = 9.5, 8.5
TABLE_PAD, TABLE_HEAD_PAD, TABLE_HPAD = 7, 8, 3

# Training-flow strip under the "Training pipeline" heading
FLOW_H, FLOW_BOX_H, FLOW_GAP = 28 * mm, 24 * mm, 6 * mm

# Limits callout
CALLOUT_PAD, CALLOUT_BAR = 8, 2

# --- results (exit rate per tier on held-out maps never seen in training) -----------------------
RESULTS = [
    # policy, score, T0..T4 exit rates, test set
    ("Random", "n/a", ["20%", "0%", "0%", "0%", "0%"], "25 held-out"),
    ("Privileged teacher (training only)", "n/a", ["100%", "40%", "80%", "80%", "40%"], "25 maps; sees true map"),
    ("Earlier v2, update 1600", "4.76", ["90%", "20%", "40%", "30%", "0%"], "50 fresh"),
    ("Best single ckpt v4-2800", "7.27", ["90%", "40%", "70%", "40%", "30%"], "same 50"),
    ("SUBMITTED: SWA v4 2600{EN}3000", "7.80", ["90%", "30%", "60%", "40%", "10%"], "same 50"),
]


# --- fonts -----------------------------------------------------------------------------------------
def _register_family(paths: dict, label: str):
    """paths: style -> (file, subfontIndex). Registers Body / Body-B / Body-I / Body-BI."""
    names = {"regular": "Body", "bold": "Body-B", "italic": "Body-I", "bolditalic": "Body-BI"}
    for style, (path, idx) in paths.items():
        pdfmetrics.registerFont(TTFont(names[style], str(path), subfontIndex=idx))
    bi = "Body-BI" if "bolditalic" in paths else "Body-B"
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic=bi)
    return "Body", "Body-B", "Body-I", label


def register_fonts():
    fira = Path(__file__).with_name("fonts")
    try:
        reg = fira / "FiraSans-Regular.ttf"
        bold = next(p for p in (fira / "FiraSans-SemiBold.ttf", fira / "FiraSans-Bold.ttf") if p.exists())
        ital = fira / "FiraSans-Italic.ttf"
        if reg.exists() and ital.exists():
            paths = {"regular": (reg, 0), "bold": (bold, 0), "italic": (ital, 0)}
            bi = fira / "FiraSans-SemiBoldItalic.ttf"
            if bi.exists():
                paths["bolditalic"] = (bi, 0)
            return _register_family(paths, "Fira Sans (brief/fonts)")
    except Exception:
        pass
    try:
        ttc = Path("/System/Library/Fonts/Helvetica.ttc")
        faces = {}
        for i in range(32):
            try:
                face = TTFontFile(str(ttc), subfontIndex=i)
            except Exception:
                break
            style = face.styleName.decode() if isinstance(face.styleName, bytes) else str(face.styleName)
            faces[style.strip().lower()] = i
        want = {"regular": "regular", "bold": "bold", "italic": "oblique", "bolditalic": "bold oblique"}
        if all(want[k] in faces for k in ("regular", "bold", "italic")):
            paths = {k: (ttc, faces[v]) for k, v in want.items() if v in faces}
            return _register_family(paths, "Apple Helvetica (embedded from /System/Library/Fonts/Helvetica.ttc)")
    except Exception:
        pass
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "base-14 Helvetica (not embedded)"


F, FB, FI, FONT_LABEL = register_fonts()
rl_config.canvas_basefontname = F  # otherwise every page also references unembedded base-14 Helvetica
print(f"font: {FONT_LABEL}")


def has_glyph(ch: str) -> bool:
    font = pdfmetrics.getFont(F)
    face = getattr(font, "face", None)
    if face is not None and hasattr(face, "charToGlyph"):
        return ord(ch) in face.charToGlyph
    return ch in "·—–×“”’"  # all present in WinAnsi for base-14 Helvetica


def glyph(ch: str, fallback: str) -> str:
    return ch if has_glyph(ch) else fallback


DOT, EM, EN, TIMES = glyph("·", "-"), glyph("—", "-"), glyph("–", "-"), glyph("×", "x")
LQ, RQ = glyph("“", '"'), glyph("”", '"')
TEAM = TEAM.replace("—", EM)
RESULTS = [(p.replace("{EN}", EN), s, t, ts) for p, s, t, ts in RESULTS]

# --- styles ------------------------------------------------------------------------------------------
body = ParagraphStyle("body", fontName=F, fontSize=BODY_SIZE, leading=BODY_LEAD, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=3)
body2 = ParagraphStyle("body2", parent=body, fontSize=BODY2_SIZE, leading=BODY2_LEAD)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=BULLET_INDENT, bulletIndent=0, spaceAfter=1.5,
                        bulletFontName=F, bulletFontSize=BODY_SIZE, bulletColor=ACCENT)
bullet2 = ParagraphStyle("bullet2", parent=bullet, fontSize=BODY2_SIZE, leading=BODY2_LEAD, bulletFontSize=BODY2_SIZE)
caption = ParagraphStyle("caption", parent=body, fontName=FI, fontSize=SMALL_SIZE, leading=SMALL_LEAD,
                         textColor=MUTED, spaceAfter=2)
score = ParagraphStyle("score", parent=body, fontSize=SCORE_SIZE, leading=SCORE_SIZE * 1.3, textColor=MUTED,
                       spaceAfter=0)
foot = ParagraphStyle("foot", parent=body, fontSize=SMALL_SIZE, leading=SMALL_LEAD, textColor=MUTED, spaceAfter=0)


def check_no_orphan(para, width):
    """Raise if the paragraph wraps and its last line holds a single word."""
    para.wrap(width, LIVE_H)
    bl = para.blPara
    if len(bl.lines) < 2:
        return para
    last = bl.lines[-1]
    text = " ".join(last[1]) if bl.kind == 0 else "".join(getattr(f, "text", "") for f in last.words)
    if len(text.split()) < 2:  # split() also breaks on &nbsp;, so it counts visible words
        raise ValueError(f"orphan word '{text.strip()}' at end of: {para.text[:60]}...")
    return para


def P(text, style=body, width=LIVE_W):
    return check_no_orphan(Paragraph(text, style), width)


def B(text, style=bullet, width=LIVE_W):
    return check_no_orphan(Paragraph(text, style, bulletText="•"), width)


# --- drawing helpers -----------------------------------------------------------------------------------
def arrow(c, x1, y1, x2, y2, color=INK):
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - ARROW_HEAD * math.cos(a), y2 - ARROW_HEAD * math.sin(a)
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(ARROW_W)
    c.line(x1, y1, bx, by)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - ARROW_HEAD * math.cos(a - 0.42), y2 - ARROW_HEAD * math.sin(a - 0.42))
    p.lineTo(x2 - ARROW_HEAD * math.cos(a + 0.42), y2 - ARROW_HEAD * math.sin(a + 0.42))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def box(c, x0, y0, x1, y1, lines, fill=FILL, stroke=RULE, width=0.6, dash=None, name="box", single_first=False):
    """lines: [(text, font, size, color)]. Wraps each to the inner width; raises if it does not fit.
    single_first: the first label must stay on one line."""
    inner = (x1 - x0) - 2 * BOX_PAD
    wrapped = []
    for li, (text, font, size, color) in enumerate(lines):
        if size < DIA_MIN:
            raise ValueError(f"{name}: font size {size} below {DIA_MIN} pt")
        split = simpleSplit(text, font, size, inner)
        if single_first and li == 0 and len(split) != 1:
            raise ValueError(f"{name}: '{text}' wraps to {len(split)} lines in {inner:.1f} pt")
        if len(split) > 1 and len(split[-1].split()) < 2:
            raise ValueError(f"{name}: orphan word '{split[-1]}' in '{text}'")
        for ln in split:
            if pdfmetrics.stringWidth(ln, font, size) > inner + 0.01:
                raise ValueError(f"{name}: '{ln}' is wider than {inner:.1f} pt")
            wrapped.append((ln, font, size, color, size * 1.2))
    need = sum(w[4] for w in wrapped)
    if need > (y1 - y0) - 3:
        raise ValueError(f"{name}: {len(wrapped)} lines need {need:.1f} pt, box is {y1 - y0:.1f} pt")
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    if dash:
        c.setDash(*dash)
    c.roundRect(x0, y0, x1 - x0, y1 - y0, BOX_RADIUS, stroke=1, fill=1)
    c.restoreState()
    y = (y0 + y1) / 2 + need / 2
    for ln, font, size, color, lead in wrapped:
        y -= lead
        c.setFont(font, size)
        c.setFillColor(color)
        c.drawCentredString((x0 + x1) / 2, y + (lead - size) / 2 + size * 0.18, ln)


# --- flowables ------------------------------------------------------------------------------------------
class TitleBlock(Flowable):
    def __init__(self, title, subtitle_parts):
        super().__init__()
        self.title, self.parts = title, subtitle_parts

    def wrap(self, aw, ah):
        self.width = aw
        self.sub_w = sum(pdfmetrics.stringWidth(t, F, SUB_SIZE) for t, _ in self.parts)
        if self.sub_w > aw:
            raise ValueError("subtitle does not fit on one line")
        self.height = TITLE_LEAD + 5 + SUB_SIZE + 5 + 2  # title, rule gap, subtitle, descent
        return aw, self.height

    def draw(self):
        c, h = self.canv, self.height
        c.setFont(FB, TITLE_SIZE)
        c.setFillColor(INK)
        c.drawString(0, h - TITLE_SIZE, self.title)
        rule_y = h - TITLE_LEAD - 5
        c.setStrokeColor(ACCENT)
        c.setLineWidth(1.2)
        c.line(0, rule_y, self.width, rule_y)
        x, y = 0, rule_y - SUB_SIZE - 5
        for text, link in self.parts:
            c.setFont(F, SUB_SIZE)
            c.setFillColor(MUTED)
            c.drawString(x, y, text)
            w = pdfmetrics.stringWidth(text, F, SUB_SIZE)
            if link:
                c.linkURL(link, (x, y - 2, x + w, y + SUB_SIZE), relative=1, thickness=0)
            x += w


class ArtifactStrip(Flowable):
    SEGMENTS = [("Submitted artifact: ", "b"), ("DoomAgent + policy.pt (SWA of five v4 snapshots)", "r"),
                (f" {DOT} ", "a"), ("Git LFS ~7 MB", "r"), (f" {DOT} ", "a"), ("validate.py passing", "r")]

    def _width(self, size):
        return sum(pdfmetrics.stringWidth(t, FB if k == "b" else F, size) for t, k in self.SEGMENTS)

    def wrap(self, aw, ah):
        self.width, self.height = aw, STRIP_H
        for size in STRIP_SIZES:
            if self._width(size) <= aw - 2 * 8:
                self.size = size
                return aw, self.height
        raise ValueError("artifact strip does not fit on one line at 8.5 pt")

    def draw(self):
        c = self.canv
        c.setFillColor(FILL)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.rect(0, 0, self.width, self.height, stroke=1, fill=1)
        x, y = 8, (self.height - self.size * 0.72) / 2
        for text, kind in self.SEGMENTS:
            font = FB if kind == "b" else F
            c.setFont(font, self.size)
            c.setFillColor(ACCENT if kind == "a" else INK)
            c.drawString(x, y, text)
            x += pdfmetrics.stringWidth(text, font, self.size)


class H2(Flowable):
    def __init__(self, text, keep=True):
        super().__init__()
        self.text = text
        self.keepWithNext = 1 if keep else 0
        self.spaceBefore, self.spaceAfter = H2_BEFORE, H2_AFTER

    def wrap(self, aw, ah):
        self.width, self.height = aw, H2_LEAD
        return aw, H2_LEAD

    def draw(self):
        c = self.canv
        c.setFillColor(ACCENT)
        c.rect(0, 2.2, 2, 10, stroke=0, fill=1)
        c.setFont(FB, H2_SIZE)
        c.drawString(7, 3.4, self.text)


class ArchDiagram(Flowable):
    ROWS = [
        (("64 depth rays", "distance + hit type"), "circular Conv1d", "192"),
        (("Visible entities (up to 32)", "dist, angle, velocity, type"), "MLP + attention pool", "128"),
        (("Vitals (17)", "health, ammo, keys, velocity"), "MLP", "64"),
        (("EgoMap crops", f"2 scales {TIMES} 8 ch, 32{TIMES}32"), "Conv2d", "256"),
        (("Previous action", "6 heads, one-hot"), "(direct)", "15"),
    ]

    def wrap(self, aw, ah):
        self.width = aw
        rows_h = len(self.ROWS) * DIA_ROW_H + (len(self.ROWS) - 1) * DIA_ROW_GAP
        self.rows_h = rows_h
        self.height = rows_h + DIA_HEADER_H
        return aw, self.height

    def draw(self):
        c = self.canv
        X = lambda v: v * mm  # noqa: E731
        top = self.rows_h
        c.setFont(FB, 8)
        c.setFillColor(MUTED)
        for (x0, _), label in ((COL_A, "SIGNAL"), (COL_B, "ENCODER"), (COL_C, "DIM")):
            c.drawString(X(x0), top + 2.2 * mm, label)
        mids = []
        for i, ((head, detail), enc, dim) in enumerate(self.ROWS):
            y1 = top - i * (DIA_ROW_H + DIA_ROW_GAP)
            y0 = y1 - DIA_ROW_H
            ym = (y0 + y1) / 2
            mids.append(ym)
            box(c, X(COL_A[0]), y0, X(COL_A[1]), y1,
                [(head, FB, DIA_LABEL, INK), (detail, F, DIA_DETAIL, MUTED)], name=f"signal {head}")
            box(c, X(COL_B[0]), y0, X(COL_B[1]), y1, [(enc, F, DIA_LABEL, INK)], fill=colors.white,
                name=f"encoder {enc}")
            cy0, cy1 = ym - 3.5 * mm, ym + 3.5 * mm
            box(c, X(COL_C[0]), cy0, X(COL_C[1]), cy1, [(dim, FB, DIA_LABEL, INK)], fill=colors.white,
                name=f"dim {dim}")
            arrow(c, X(COL_A[1]), ym, X(COL_B[0]), ym)
            arrow(c, X(COL_B[1]), ym, X(COL_C[0]), ym)
            arrow(c, X(COL_C[1]), ym, X(MERGE_X) - MERGE_W / 2, ym)
        # merge connector
        c.setStrokeColor(MUTED)
        c.setLineWidth(MERGE_W)
        c.line(X(MERGE_X), mids[-1], X(MERGE_X), mids[0])
        # fusion
        cx0, cx1 = X(COL_D[0]), X(COL_D[1])
        concat_mid, gru_mid = top / 2 + FUSION_OFFSET, top / 2 - FUSION_OFFSET
        box(c, cx0, concat_mid - FUSION_H / 2, cx1, concat_mid + FUSION_H / 2,
            [("concat", FB, DIA_LABEL, INK), ("Linear 384", F, DIA_DETAIL, MUTED)], fill=colors.white, name="concat")
        box(c, cx0, gru_mid - FUSION_H / 2, cx1, gru_mid + FUSION_H / 2, [("GRU 256", FB, 9, INK)],
            fill=TINT, stroke=ACCENT, width=1, name="gru")
        arrow(c, X(MERGE_X) + MERGE_W / 2, concat_mid, cx0, concat_mid)
        arrow(c, (cx0 + cx1) / 2, concat_mid - FUSION_H / 2, (cx0 + cx1) / 2, gru_mid + FUSION_H / 2)
        # heads
        heads = [
            ([("Move / Strafe / Turn", FB, DIA_LABEL, INK), ("(3+3+3)", F, DIA_DETAIL, MUTED)], {}),
            ([("Attack / Use / Weapon", FB, DIA_LABEL, INK), ("(2+2+2)", F, DIA_DETAIL, MUTED)], {}),
            ([("Value (critic)", FB, DIA_LABEL, MUTED), ("training only", FB, DIA_DETAIL, ACCENT)],
             dict(dash=(2, 2), fill=colors.white)),
        ]
        total = len(heads) * HEAD_H + (len(heads) - 1) * HEAD_GAP
        y_top = top / 2 + total / 2
        for i, (lines, kw) in enumerate(heads):
            y1 = y_top - i * (HEAD_H + HEAD_GAP)
            y0 = y1 - HEAD_H
            kw = {"fill": colors.white, **kw}
            box(c, X(COL_E[0]), y0, X(COL_E[1]), y1, lines, name=f"head {i}", single_first=True, **kw)
            arrow(c, cx1, gru_mid, X(COL_E[0]), (y0 + y1) / 2)


class EgoMapFigure(Flowable):
    """World-fixed 192x192 grid with the two heading-aligned crops drawn to scale (1.3k and 5.1k of 7.7k)."""

    def wrap(self, aw, ah):
        self.width, self.height = EGO_FIG_W, EGO_FIG_H
        return self.width, self.height

    @staticmethod
    def _square(c, cx, cy, side, angle, color, width):
        r = side / math.sqrt(2)
        pts = [(cx + r * math.cos(angle + math.pi / 4 + k * math.pi / 2),
                cy + r * math.sin(angle + math.pi / 4 + k * math.pi / 2)) for k in range(4)]
        p = c.beginPath()
        p.moveTo(*pts[0])
        for pt in pts[1:]:
            p.lineTo(*pt)
        p.close()
        c.setStrokeColor(color)
        c.setLineWidth(width)
        c.drawPath(p, stroke=1, fill=0)
        return pts

    def draw(self):
        c = self.canv
        s, y0 = EGO_GRID, 7 * mm
        cx, cy = s / 2, y0 + s / 2
        c.setStrokeColor(RULE)
        c.setLineWidth(0.8)
        c.rect(0, y0, s, s, stroke=1, fill=0)
        c.setFont(F, 8)
        c.setFillColor(MUTED)
        c.drawString(0, y0 + s + 2, f"192{TIMES}192")
        heading = math.radians(25)
        outer = self._square(c, cx, cy, s * 5.1 / 7.7, heading, ACCENT, 0.8)
        inner = self._square(c, cx, cy, s * 1.3 / 7.7, heading, MUTED, 0.8)
        # agent marker, pointing along the heading
        tri = [(cx + 1.6 * mm * math.cos(heading + a), cy + 1.6 * mm * math.sin(heading + a)) for a in (0, 2.5, -2.5)]
        p = c.beginPath()
        p.moveTo(*tri[0])
        p.lineTo(*tri[1])
        p.lineTo(*tri[2])
        p.close()
        c.setFillColor(INK)
        c.drawPath(p, stroke=0, fill=1)
        lx = s + 3 * mm
        right = lambda pts: max(pts, key=lambda q: q[0])  # noqa: E731
        for (ax, ay), y_text, units in ((right(outer), cy + 6 * mm, "5.1k"), (right(inner), cy - 9 * mm, "1.3k")):
            c.setStrokeColor(MUTED)
            c.setLineWidth(0.4)
            c.line(ax, ay, lx - 1, y_text + 3)
            c.setFont(F, 8)
            c.setFillColor(MUTED)
            c.drawString(lx, y_text + 1, f"32{TIMES}32 crop,")
            c.drawString(lx, y_text - 8.6, f"{units} units")
        arrow(c, 0, 2.5 * mm, 7 * mm, 2.5 * mm, color=ACCENT)
        c.setFont(F, 8)
        c.setFillColor(MUTED)
        c.drawString(8.5 * mm, 2.5 * mm - 2.8, "input only, never searched")


class TrainingFlow(Flowable):
    STAGES = [
        ("Procedural levels", "UDMF, 5 tiers"),
        ("Teacher + DAgger", "beta-mix, sees true map, training only"),
        ("PPO + CE to teacher", "32 envs, BPTT over GRU"),
        ("Solo eval\nevery 100 updates", f"SWA of v4 2600{EN}3000"),
    ]

    def wrap(self, aw, ah):
        self.width, self.height = aw, FLOW_H
        return aw, self.height

    def draw(self):
        c = self.canv
        n = len(self.STAGES)
        bw = (self.width - (n - 1) * FLOW_GAP) / n
        y0 = (self.height - FLOW_BOX_H) / 2
        y1, ym = y0 + FLOW_BOX_H, self.height / 2
        for i, (head, detail) in enumerate(self.STAGES):
            x0 = i * (bw + FLOW_GAP)
            kw = dict(fill=TINT, stroke=ACCENT, width=1) if i == n - 1 else {}
            box(c, x0, y0, x0 + bw, y1, [(head, FB, DIA_LABEL, INK), (detail, F, DIA_DETAIL, MUTED)],
                name=f"flow {head}", **kw)
            if i:
                arrow(c, x0 - FLOW_GAP, ym, x0, ym)


class Callout(Flowable):
    def __init__(self, para):
        super().__init__()
        self.para = para

    def wrap(self, aw, ah):
        self.width = aw
        _, h = self.para.wrap(aw - 2 * CALLOUT_PAD - CALLOUT_BAR, ah)
        self.height = h + 2 * CALLOUT_PAD
        return aw, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(FILL)
        c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        c.setFillColor(ACCENT)
        c.rect(0, 0, CALLOUT_BAR, self.height, stroke=0, fill=1)
        self.para.drawOn(c, CALLOUT_BAR + CALLOUT_PAD, CALLOUT_PAD)


def results_table():
    cols = [w * mm for w in TABLE_COLS_MM]
    rows = [["Policy", "Score", "T0", "T1", "T2", "T3", "T4", "Test set"]]
    for name, sc, tiers, test in RESULTS:
        rows.append([name, sc, *tiers, test])
    last = len(rows) - 1
    sub_size = TABLE_SIZE
    for r, row in enumerate(rows):
        font = FB if r in (0, last) else F
        size = TABLE_SIZE
        for ci, text in enumerate(row):
            avail = cols[ci] - 2 * TABLE_HPAD - (3 if (r == last and ci == 0) else 0)
            w = pdfmetrics.stringWidth(text, font, size)
            if w > avail and r == last and ci == 0 and pdfmetrics.stringWidth(text, font, TABLE_SIZE_MIN) <= avail:
                sub_size = TABLE_SIZE_MIN
            elif w > avail:
                raise ValueError(f"table cell '{text}' ({w:.1f} pt) wider than {avail:.1f} pt")
    t = Table(rows, colWidths=cols, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), F),
        ("FONTSIZE", (0, 0), (-1, -1), TABLE_SIZE),
        ("LEADING", (0, 0), (-1, -1), TABLE_SIZE * 1.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("FONTNAME", (0, 0), (-1, 0), FB),
        ("LINEBELOW", (0, 0), (-1, 0), 1, INK),
        ("TOPPADDING", (0, 0), (-1, 0), TABLE_HEAD_PAD),
        ("BOTTOMPADDING", (0, 0), (-1, 0), TABLE_HEAD_PAD),
        ("TOPPADDING", (0, 1), (-1, -1), TABLE_PAD),
        ("BOTTOMPADDING", (0, 1), (-1, -1), TABLE_PAD),
        ("LEFTPADDING", (0, 0), (-1, -1), TABLE_HPAD),
        ("RIGHTPADDING", (0, 0), (-1, -1), TABLE_HPAD),
        ("LINEBELOW", (0, 1), (-1, last - 1), 0.5, RULE),
        ("ALIGN", (1, 0), (6, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (0, 1), (-1, 2), MUTED),
        ("BACKGROUND", (0, last), (-1, last), TINT),
        ("FONTNAME", (0, last), (-1, last), FB),
        ("FONTSIZE", (0, last), (0, last), sub_size),
        ("LINEBEFORE", (0, last), (0, last), 2, ACCENT),
        ("LEFTPADDING", (0, last), (0, last), TABLE_HPAD + 3),
        ("TEXTCOLOR", (6, last), (6, last), ACCENT),
    ]
    t.setStyle(TableStyle(style))
    return t


def score_line():
    """Heuristic note: the claim on line 1, the formula on line 2, never a third line."""
    text = ("Score is our local heuristic, not the hidden judge weights:<br/>+10 exit, +1.5 key, +0.5 door, "
            "+0.3 kill, +0.03 area, -2 death, -0.01 damage; harder tiers weighted more.")
    for size in (SCORE_SIZE, 8.5):
        para = P(text, ParagraphStyle(f"score{size}", parent=score, fontSize=size, leading=size * 1.3))
        if len(para.blPara.lines) == 2:
            return para
    raise ValueError("score line does not fit on two lines at 8.5 pt")


class SquareItem(Flowable):
    """A paragraph preceded by a 3 pt accent square centred on its first line's x-height."""

    def __init__(self, para):
        super().__init__()
        self.para = para
        self.spaceAfter = para.style.spaceAfter

    def wrap(self, aw, ah):
        self.width = aw
        _, self.height = self.para.wrap(aw - BULLET_INDENT, ah)
        return aw, self.height

    def draw(self):
        c, size = self.canv, self.para.style.fontSize
        c.setFillColor(ACCENT)
        c.rect(0, self.height - size + 0.35 * size - 1.5, 3, 3, stroke=0, fill=1)
        self.para.drawOn(c, BULLET_INDENT, 0)


def rules_list():
    lines = [
        "<b>Two trained nets only.</b>",
        "EgoMap is deterministic bookkeeping and is never searched.",
        "Privileged info (true map, positions, monsters) is training only: teacher, rewards, odometry labels.",
        "Eval = vanilla engine + official observations. Weights in Git LFS, ~7 MB. All training code in training/.",
    ]
    return [SquareItem(P(t, body2, width=LIVE_W - BULLET_INDENT)) for t in lines]


# --- document --------------------------------------------------------------------------------------
class BriefDoc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fill = {}

    def afterFlowable(self, flowable):
        fr = self.frame
        used = (fr._y1 + fr._height) - fr._y
        self.fill[self.page] = max(self.fill.get(self.page, 0.0), used / fr._height)


def footer(c, doc):
    c.saveState()
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(MARGIN, FOOTER_RULE_Y, PAGE_W - MARGIN, FOOTER_RULE_Y)
    c.setFont(F, FOOT_SIZE)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, FOOTER_TEXT_Y, TEAM)
    c.drawRightString(PAGE_W - MARGIN, FOOTER_TEXT_Y, f"Yugaantar-Devforge {DOT} p. {doc.page}/{TOTAL_PAGES}")
    c.restoreState()


def story():
    s = []
    s += [TitleBlock("Yugaantar: a DOOM agent that maps as it moves",
                     [(f"DevForge Track 1, {LQ}Can It Play DOOM?{RQ} {DOT} {TEAM} {DOT} ", None),
                      ("github.com/Hollenite/Yugaantar-Devforge", REPO_URL)]),
          Spacer(1, TITLE_GAP), ArtifactStrip(), Spacer(1, SECTION_GAP_P1)]

    s += [H2("What makes it different"),
          P("Observations have no coordinates and no map. Hard levels need us to remember a keycard and find the "
            "locked door again. We <b>build the map while the agent plays</b>, and feed it to one net that "
            "makes&nbsp;every&nbsp;decision."),
          B("<b>Self-built spatial memory.</b> Deterministic Doom physics plus a learned odometry net (OdoNet, 86k) "
            "give pose. 64 depth rays are painted into a world-fixed grid: walls, doors, locked doors, items, "
            "enemies, keys, trail."),
          B("<b>Every action comes from trained weights.</b> No planner, path-finder or aim rule at test time. "
            "DoomNet 1.55M + OdoNet 86k, trained from scratch."),
          B("<b>A privileged teacher is used in training only.</b> It sees the true map. The student is distilled "
            "with DAgger, then PPO runs until it plays solo."),
          B("Small and fast: ~6.6 MB weights, ~1.8 ms/step on 1 CPU, validate.py passing."),
          Spacer(1, SECTION_GAP_P1)]

    sp = Spacer(1, 1.5 * mm)
    sp.keepWithNext = 1
    s += [H2("Architecture"), sp, ArchDiagram(), Spacer(1, 1.5 * mm),
          P("EgoMap is written each step from rays and entities at a pose from Doom physics + OdoNet.", caption),
          P(f"We use a circular 1-D conv so {LQ}behind me{RQ} wraps. We use attention over the variable entity set. "
            "We sample actions. We do <i>not</i> take argmax."),
          Spacer(1, SECTION_GAP_P1)]

    w = EGO_TEXT_W
    ego_bullets = [
        B("Odometry: thrust, friction 0.906, fixed turn. A residual is estimated from two consecutive ray scans.",
          width=w),
        B(f"Writing: 192{TIMES}192 grid, 40-unit cells, ~7.7k map units. The enemy channel fades.", width=w),
        B(f"Reading: two heading-aligned 32{TIMES}32 crops (1.3k and 5.1k units). "
          "<b>The&nbsp;map&nbsp;is&nbsp;an&nbsp;input. It&nbsp;is&nbsp;never searched.</b>", width=w),
    ]
    ego = Table([[ego_bullets, "", EgoMapFigure()]], colWidths=[EGO_TEXT_W, EGO_GAP, EGO_FIG_W], hAlign="LEFT")
    ego.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), F), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    s += [H2("Spatial memory (EgoMap)"), ego]

    s += [PageBreak()]

    sp3 = Spacer(1, 1.5 * mm)
    sp3.keepWithNext = 1
    s += [H2("Training pipeline"), sp3, TrainingFlow(), Spacer(1, 2 * mm),
          B(f"Procedural UDMF levels in 5 tiers that mirror the eval: empty, combat, multi-room, "
            f"keycard-to-lock-to-exit, gauntlet with 2{EN}3 locks. Curriculum by success, with floors and caps.",
            bullet2),
          B("Rewards mirror the scoring: exit (plus health/time), keys, doors, locks, kills, new areas, damage, death. "
            "A geodesic potential (training only). Curiosity = new EgoMap cells.", bullet2),
          B("32 parallel envs. Beta-mix teacher (DAgger), teacher roll-in into late states, and a growing solo "
            "share. PPO (clip, GAE, BPTT over the GRU) plus CE to the teacher, annealed.", bullet2),
          B(f"DrAC-style left-right mirror (observation + strafe/turn), LR anneal, 10% Freedoom MAP04{EN}10.",
            bullet2),
          B(f"Model selection: every 100 updates the student plays SOLO on held-out generated maps plus Freedoom "
            f"MAP01{EN}03 through the real DoomAgent. The submitted weights are the <b>SWA</b> of v4 eval snapshots "
            "2600, 2700, 2800, 2900, 3000 (Izmailov et al. 2018). It is still one trained net.", bullet2),
          Spacer(1, SECTION_GAP_P2)]

    sc = score_line()
    sc.keepWithNext = 1
    sp2 = Spacer(1, 2 * mm)
    sp2.keepWithNext = 1
    s += [H2("Results"), sc, sp2, results_table(), Spacer(1, 2 * mm),
          P(f"T0{EN}T4: exit rate per tier. Submitted row: 7.75 vs 6.49 on 60 more maps "
            "(SWA vs best single checkpoint).", foot),
          Spacer(1, SECTION_GAP_P2)]

    s += [H2("Limits"),
          Callout(P(f"<b>Submitted T4 exit = 10%</b> on this set. We are not a full gauntlet solver. The hidden eval "
                    f"is 6 unseen maps, tiers 1{EN}4 (T0 is training-only). Keys are found more often than the "
                    "matching lock is finished.", body2)),
          Spacer(1, SECTION_GAP_P2)]

    s += [H2("Rules compliance"), *rules_list()]
    return s


def page_count(path: Path, fallback: int) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return fallback


def build():
    doc = BriefDoc(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                   bottomMargin=MARGIN, title="Yugaantar: a DOOM agent that maps as it moves", author=TEAM)
    frame = Frame(MARGIN, MARGIN, LIVE_W, LIVE_H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                  id="live")
    doc.addPageTemplates([PageTemplate(id="brief", frames=[frame], onPage=footer)])
    doc.build(story())
    pages = page_count(OUT, doc.page)
    print(OUT)
    for p in sorted(doc.fill):
        print(f"page {p} fill: {doc.fill[p] * 100:.1f}% of live area")
    if pages != TOTAL_PAGES:
        print(f"ERROR: {pages} pages, expected {TOTAL_PAGES}", file=sys.stderr)
        sys.exit(1)
    for p, f in sorted(doc.fill.items()):
        if f < MIN_FILL:
            print(f"WARNING: page {p} fill {f * 100:.1f}% < {MIN_FILL * 100:.0f}%", file=sys.stderr)


if __name__ == "__main__":
    build()

"""Builds brief/Yugaantar_DOOM_brief.pdf (2 pages, A4).

    python brief/make_brief.py

Edit RESULTS below after a new model-selection run and rebuild.
"""

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path(__file__).with_name("Yugaantar_DOOM_brief.pdf")

# --- fonts (Fira Sans if present, else Helvetica) ------------------------------------------------
FIRA = Path("/usr/share/fonts/truetype/google-fonts/ofl/firasans")
if (FIRA / "FiraSans-Regular.ttf").exists():
    for name, f in [("Body", "Regular"), ("Body-B", "SemiBold"), ("Body-I", "Italic"), ("Body-BI", "SemiBoldItalic")]:
        pdfmetrics.registerFont(TTFont(name, str(FIRA / f"FiraSans-{f}.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")
    F, FB = "Body", "Body-B"
else:
    F, FB = "Helvetica", "Helvetica-Bold"

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#5b5b60")
ACCENT = colors.HexColor("#b3261e")
RULE = colors.HexColor("#d9d9de")
FILL = colors.HexColor("#f4f4f6")

body = ParagraphStyle("body", fontName=F, fontSize=11, leading=14.2, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=0, spaceAfter=2.2)
h1 = ParagraphStyle("h1", fontName=FB, fontSize=18, leading=22, textColor=INK, spaceAfter=1)
sub = ParagraphStyle("sub", fontName=F, fontSize=10.5, leading=13, textColor=MUTED, spaceAfter=7)
h2 = ParagraphStyle("h2", fontName=FB, fontSize=12, leading=15, textColor=ACCENT, spaceBefore=6, spaceAfter=3, keepWithNext=1)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=11.5, textColor=MUTED, spaceAfter=0)
cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName=FB)

# --- results (exit rate per tier on held-out maps never seen in training) -----------------------
RESULTS = [
    # name, score, tier exit rates (T0..T4), note
    ("Random actions", "n/a", ["20%", "0%", "0%", "0%", "0%"], "25 held-out maps"),
    ("Privileged teacher (training only)", "n/a", ["100%", "40%", "80%", "80%", "40%"], "25 held-out maps; sees the true map"),
    ("Earlier agent (v2, update 1600)", "4.76", ["90%", "20%", "40%", "30%", "0%"], "50 fresh held-out maps"),
    ("Submitted agent (v4, update 2800)", "7.27", ["90%", "40%", "70%", "40%", "30%"], "same 50 fresh held-out maps"),
]


def P(t, s=body):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, bullet, bulletText="•")


# --- architecture diagram ------------------------------------------------------------------------
def diagram(width):
    W, H = width, 66 * mm
    d = Drawing(W, H)

    def box(x, y, w, h, title, detail, fill=FILL, stroke=RULE, tc=INK):
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
        d.add(String(x + w / 2, y + h - 11.5, title, fontName=FB, fontSize=8.6, fillColor=tc, textAnchor="middle"))
        if detail:
            d.add(String(x + w / 2, y + 5, detail, fontName=F, fontSize=7.4, fillColor=MUTED, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.8))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        s = 4
        d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                       x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)], fillColor=MUTED, strokeColor=MUTED))

    rows = [
        ("64 depth rays", "distance + hit type", "Circular Conv1d", "192"),
        ("Visible entities (<=32)", "dist, angle, velocity, type", "MLP + attention pool", "128"),
        ("Vitals (17)", "health, ammo, keys, velocity", "MLP", "64"),
        ("EgoMap crops", "2 scales x 8 channels, 32x32", "Conv2d tower", "256"),
        ("Previous action", "6 action heads", "One-hot", "15"),
    ]
    bw1, bw2, bh, gap = 34 * mm, 29 * mm, 10.2 * mm, 2.3 * mm
    x1, x2 = 0, 38 * mm
    top = H - bh
    ys = [top - i * (bh + gap) for i in range(len(rows))]
    xc = x2 + bw2 + 8 * mm
    cw, ch = 20 * mm, 16 * mm
    cy = H / 2 - ch / 2
    for (t, dt, enc, dim), y in zip(rows, ys):
        box(x1, y, bw1, bh, t, dt)
        box(x2, y, bw2, bh, enc, f"-> {dim}", fill=colors.white)
        arrow(x1 + bw1, y + bh / 2, x2, y + bh / 2)
        arrow(x2 + bw2, y + bh / 2, xc, cy + ch / 2)
    box(xc, cy, cw, ch, "Concat", "Linear 384", fill=colors.white)
    xg = xc + cw + 7 * mm
    box(xg, cy, cw, ch, "GRU 256", "memory", fill=colors.HexColor("#fbe9e7"), stroke=ACCENT)
    arrow(xc + cw, cy + ch / 2, xg, cy + ch / 2)
    xh = xg + cw + 7 * mm
    hw = W - xh
    heads = [("Move / Strafe / Turn", "3 + 3 + 3 way"), ("Attack / Use / Weapon", "2 + 2 + 2 way"),
             ("Value", "critic, training only")]
    hh = 12 * mm
    for i, (t, dt) in enumerate(heads):
        y = cy + ch / 2 + hh * 1.0 - i * (hh + 2.5 * mm) + 2
        box(xh, y - hh / 2, hw, hh, t, dt, fill=colors.white)
        arrow(xg + cw, cy + ch / 2, xh, y)
    # odometry note under the map row
    d.add(String(x1, 1, "EgoMap is written every step from rays + entities at a pose dead-reckoned by Doom physics "
                        "+ OdoNet (86k params).", fontName=F, fontSize=7.4, fillColor=MUTED))
    return d


# --- document --------------------------------------------------------------------------------------
def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=14 * mm,
                            bottomMargin=13 * mm, title="Yugaantar: a DOOM agent that maps as it moves",
                            author="Team Yugaantar")
    W = A4[0] - 34 * mm
    s = []
    s += [P("Yugaantar: a DOOM agent that maps as it moves", h1),
          P("DevForge Track 1, “Can It Play DOOM?” · Team Yugaantar · "
            "github.com/Hollenite/Yugaantar-Devforge", sub)]

    s += [P("What makes it different", h2),
          P("The observations contain no coordinates and no map, yet the hardest levels require remembering where a "
            "keycard was and finding the locked door again. Our agent <b>builds its own map while it plays</b> and "
            "feeds it to one neural network that makes every decision."),
          B("<b>Self-built spatial memory.</b> Doom’s movement physics are deterministic, so replaying the "
            "agent’s own actions (corrected by a small learned odometry net) gives its pose; the 64 depth rays are "
            "painted into a world-fixed grid of walls, doors, locked doors, items, enemies, keys and its own trail."),
          B("<b>Every action comes from trained weights.</b> No planner, path-finder or aim rule runs at test time. "
            "One 1.55 M-parameter network (plus the 86 k-parameter odometry net) is trained from scratch."),
          B("<b>Learns from a privileged teacher, then on its own.</b> A teacher that sees the true map exists only "
            "during training; the student is distilled from it (DAgger) and improved with reinforcement learning (PPO) "
            "until it plays solo."),
          B("<b>Small and fast.</b> 6.6 MB of weights, about 1.8 ms per step on one CPU core, validator passing.")]

    s += [P("Architecture", h2), diagram(W), Spacer(1, 3),
          P("Each input stream has its own encoder: a circular 1-D convolution over the ray ring (so “behind "
            "me” wraps around), attention pooling over the variable-size set of visible entities, an MLP over "
            "vitals, and a CNN over two map crops. A GRU carries short-term memory between steps. Actions are "
            "factorised into six small heads whose every combination is a legal button press, which keeps "
            "exploration and imitation simple. Sampling (not argmax) avoids getting stuck in loops.")]

    s += [P("Spatial memory (EgoMap)", h2),
          B("<b>Odometry.</b> Each tic a held button adds fixed thrust, friction multiplies momentum by 0.906 and turns "
            "use a fixed angle. Replaying this from the agent’s previous action and the observed velocity gives a "
            "dead-reckoned pose relative to the start. OdoNet, trained on ground truth, predicts the residual from "
            "two consecutive ray scans (wall collisions, slides)."),
          B("<b>Writing.</b> A 192 x 192 grid of 40-unit cells (7.7 k map units across), fixed to the start pose. "
            "Free space is marked along each ray, the hit cell goes into its hit-type channel, entities are placed at "
            "their bearing and distance, and the enemy channel fades over time so stale threats disappear."),
          B("<b>Reading.</b> Two heading-aligned 32 x 32 crops are resampled each step: a fine local view "
            "(1.3 k units) and a coarse wide view (5.1 k units). Unexplored space, unopened doors and remembered keys "
            "are therefore visible to the policy as images. The map is only an input; it is never searched.")]

    s += [P("Training and exploration pipeline", h2),
          B("<b>Procedural levels.</b> A generator builds unlimited UDMF levels in five tiers that mirror the "
            "evaluation: empty rooms, combat arenas, multi-room mazes with doors, keycard → locked door → exit, "
            "and gauntlets with 2–3 locks. A curriculum samples tiers by current success, with floors and caps "
            "so no tier is forgotten."),
          B("<b>Rewards mirror the scoring.</b> Exit (plus health and time bonus), keys, doors, locked doors, kills, new "
            "areas, damage, death, and potential-based shaping along the true geodesic distance to the next goal "
            "(training only). Curiosity: a bonus for every cell the agent’s <i>own</i> EgoMap reveals."),
          B("<b>Teacher → student.</b> 32 parallel environments. With probability β the teacher acts (DAgger), "
            "some episodes start with a teacher “roll-in” to reach deep states (e.g. after the key), and a "
            "growing share of episodes are fully solo. The loss is PPO (clipped, GAE, BPTT over GRU chunks) plus "
            "a cross-entropy term towards the teacher’s action, annealed over training."),
          B("<b>Generalisation tricks from the literature.</b> Left–right mirror augmentation (rays, entities, "
            "map crops and strafe/turn actions mirrored, with a consistency loss, after DrAC), learning-rate "
            "annealing, and 10% of episodes on real Freedoom maps (MAP04–10) for visual/layout diversity."),
          B("<b>Honest model selection.</b> Every 100 updates a background job plays the student <i>solo</i> on "
            "held-out generated maps and on Freedoom MAP01–03 (never trained on) through the real "
            "<font face='Courier'>DoomAgent</font> code. The submitted weights are the head-to-head winner.")]

    rows = [[P("<b>Policy</b>", cellb), P("<b>Score</b>", cellb)]
            + [P(f"<b>T{i}</b>", cellb) for i in range(5)] + [P("<b>Test set</b>", cellb)]]
    for name, score, ex, note in RESULTS:
        rows.append([P(name, cell), P(score, cell)] + [P(e, cell) for e in ex] + [P(note, small)])
    t = Table(rows, colWidths=[54 * mm, 14 * mm] + [12 * mm] * 5 + [W - 128 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    s += [KeepTogether([P("Results", h2),
                        P("Exit rate per tier (T0 empty rooms → T4 gauntlet) on generated maps never seen in "
                          "training; score = 10 per exit + progression, kills and exploration − deaths, "
                          "harder tiers weighted more.", small), Spacer(1, 3), t])]
    s += [Spacer(1, 4),
          P("<b>Limits and next steps.</b> The keycard tier is the weakest, and full Freedoom campaign maps are still "
            "too long for a solo exit. Next: longer solo-only fine-tuning on keycard levels, more real-map "
            "training, and a wider memory crop for very large maps.")]
    s += [P("Rules compliance", h2),
          P("All decisions come from the two trained networks; the EgoMap is deterministic bookkeeping that the "
            "network reads, never searches. Privileged information (true map, positions, monster list) is used "
            "only by the training-time teacher, rewards and odometry labels. Evaluation runs the vanilla game "
            "through the official observations. Weights are in Git LFS (about 7 MB in total), and all training "
            "code is in <font face='Courier'>training/</font>.")]
    doc.build(s)
    print(OUT)


if __name__ == "__main__":
    build()

"""Build the findings deck for the MEAL team from the run outputs and the IndiKit metadata.

    .venv/bin/python make_findings_deck.py            -> outputs/indikit_findings_meal_2026-11-23.pptx

Order: the task (internship brief) -> the question -> how it was done (ESGReveal as the reference model, an LLM per
document x indicator pair, the three match types) -> findings -> what it means for the way of working -> caveats ->
next steps. Only indicators on the live website count as "IndiKit" in the sector and never-used views; the export-only
ones get their own slide.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "indikit_findings_meal_2026-11-24.pptx"
BLUE, ORANGE, AQUA, YELLOW = "2A78D6", "EB6834", "1BAF7A", "EDA100"
INK, MUTED, TILE = "1A1A19", "6B6A66", "F3F2EE"
FLOOR = 0.60

# ---------------------------------------------------------------- data
meta = {str(r["indikit_id"]): r for r in (json.loads(l) for l in open(ROOT / "pipeline/metadata/indikit_metadata.jsonl", encoding="utf-8") if l.strip())}
full = pd.DataFrame([{"id": i, "name": r["name"], "sector": r["aspect"][0]["sector"], "sub": r["aspect"][0].get("sub_sector") or "(no sub-sector)", "on_site": bool(r["on_site"])} for i, r in meta.items()])
d = pd.read_csv(ROOT / "outputs/latest/document_indicators.csv", dtype=str, keep_default_na=False)
d = d[d.is_indicator.str.lower() != "false"]
mx = pd.read_csv(ROOT / "outputs/latest/disclosure_matrix.csv", dtype=str, keep_default_na=False)
s = pd.read_csv(ROOT / "outputs/latest/document_summary.csv")
byc = pd.read_csv(ROOT / "outputs/latest/portfolio_by_country.csv")
mu = pd.read_csv(ROOT / "outputs/latest/portfolio_most_used_indikit.csv", dtype={"indikit_id": str})
clusters = pd.read_csv(ROOT / "outputs/latest/portfolio_novel_clusters.csv")
run_id = open(ROOT / "outputs/latest/LATEST_RUN.txt").read().split()[0]

used_ids = set(mx.indikit_id)
full["used"] = full.id.isin(used_ids)
cand = defaultdict(list)
for _, r in d.iterrows():
    for part in r.nearest_indikit.split(";"):
        part = part.strip()
        if part:
            i, _, sc = part.partition("(")
            cand[i.strip()].append((float(sc.rstrip(")")), r.indicator_text))
for k in cand:
    cand[k].sort(reverse=True)
full["best"] = full.id.map(lambda i: cand[i][0][0] if cand.get(i) else None)
full["closest"] = full.id.map(lambda i: cand[i][0][1] if cand.get(i) else "")
live = full[full.on_site].copy()
export_only = full[~full.on_site]
A = live[~live.used].copy()                                   # live, never used
A_present = A[A.best >= FLOOR].sort_values("best", ascending=False)
A_absent = A[~(A.best >= FLOOR)]
B = export_only[export_only.used]                             # used, not on the site
sub = live.groupby(["sector", "sub"]).agg(total=("id", "size"), used=("used", "sum")).reset_index()
unused_subs = sub[sub.used == 0].sort_values("total", ascending=False)
bm = mx[mx.indikit_id.isin(B.id)].groupby(["indikit_id", "indikit_name"]).agg(documents=("doc_id", "nunique")).reset_index().sort_values("documents", ascending=False)
B_docs = mx[mx.indikit_id.isin(B.id)].doc_id.nunique()
by_sector = live.groupby("sector").agg(total=("id", "size"), used=("used", "sum")).reset_index()
by_sector["never"] = by_sector.total - by_sector.used
by_sector = by_sector.sort_values("total", ascending=False)
total_ind, n_v, n_c, n_n = len(d), int((d.match_type == "verbatim").sum()), int((d.match_type == "close").sum()), int((d.match_type == "none").sum())
n_docs, n_skip = len(s), int(s.skipped_reason.notna().sum())
n_zero = int(((s.total_indicators == 0) & s.skipped_reason.isna()).sum())
dup_copies = int((s.doc_id.str[-8:].value_counts() - 1).clip(lower=0).sum())
median_doc = s.pct_matched.median()
country_names = {"drc": "DRC", "burkina": "Burkina Faso", "south_sudan": "South Sudan", "sudan": "Sudan", "ethiopia": "Ethiopia", "uganda": "Uganda", "burundi": "Burundi", "syria": "Syria", "yemen": "Yemen"}
byc["label"] = byc.country.str.lower().map(country_names).fillna(byc.country)
byc = byc.sort_values("pct_matched", ascending=False)

# ---------------------------------------------------------------- helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width
n = 0


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, s, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, line in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = rgb(color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=10):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        head, rest = it if isinstance(it, tuple) else (it, "")
        r = p.add_run(); r.text = "•  " + head; r.font.bold = bool(rest); r.font.size = Pt(size); r.font.color.rgb = rgb(color); r.font.name = "Calibri"
        if rest:
            r = p.add_run(); r.text = " " + rest; r.font.size = Pt(size); r.font.color.rgb = rgb(color); r.font.name = "Calibri"
    return tb


def header(slide, title, kicker=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(BLUE); bar.line.fill.background()
    if kicker:
        text(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.35), kicker.upper(), 11, True, MUTED)
    text(slide, Inches(0.6), Inches(0.6), Inches(12.1), Inches(0.9), title, 28, True, INK)


def footer(slide):
    text(slide, Inches(0.6), Inches(7.05), Inches(9), Inches(0.3), "IndiKit use in ZOA proposals · MEAL · 24 November 2026", 9, False, MUTED)
    text(slide, Inches(12.2), Inches(7.05), Inches(0.6), Inches(0.3), str(n), 9, False, MUTED, PP_ALIGN.RIGHT)


def section(title, subtitle, color):
    sl = new_slide()
    bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = rgb(color); bg.line.fill.background()
    text(sl, Inches(0.8), Inches(2.4), Inches(11.5), Inches(1.4), title, 38, True, "FFFFFF")
    text(sl, Inches(0.8), Inches(3.9), Inches(11.5), Inches(1.2), subtitle, 18, False, "FFFFFF")
    return sl


def tile(slide, x, y, w, h, number, label, color=BLUE):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid(); box.fill.fore_color.rgb = rgb(TILE); box.line.fill.background(); box.adjustments[0] = 0.08
    text(slide, x, y + Inches(0.15), w, Inches(0.9), number, 36, True, color, PP_ALIGN.CENTER)
    text(slide, x + Inches(0.1), y + Inches(1.05), w - Inches(0.2), h - Inches(1.1), label, 12, False, INK, PP_ALIGN.CENTER)


def table(slide, x, y, w, h, columns, rows, widths=None, size=11):
    t = slide.shapes.add_table(len(rows) + 1, len(columns), x, y, w, h).table
    if widths:
        for i, wd in enumerate(widths):
            t.columns[i].width = Inches(wd)
    for j, c in enumerate(columns):
        cell = t.cell(0, j); cell.text = str(c); cell.fill.solid(); cell.fill.fore_color.rgb = rgb(BLUE)
        r = cell.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb("FFFFFF"); r.font.name = "Calibri"
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = str(v); cell.fill.solid(); cell.fill.fore_color.rgb = rgb("FFFFFF" if i % 2 else TILE)
            cell.margin_top = cell.margin_bottom = Pt(3)
            r = cell.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(size); r.font.color.rgb = rgb(INK); r.font.name = "Calibri"
    return t


def bar_chart(slide, x, y, w, h, categories, series, colors, stacked=False, number_format="0", font=11):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series:
        cd.add_series(name, vals)
    ch = slide.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED if stacked else XL_CHART_TYPE.BAR_CLUSTERED, x, y, w, h, cd).chart
    ch.font.size, ch.font.name = Pt(font), "Calibri"
    ch.has_legend = len(series) > 1
    if ch.has_legend:
        ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
    plot = ch.plots[0]
    plot.gap_width = 60
    if stacked:
        plot.overlap = 100
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(font); plot.data_labels.number_format = number_format; plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.CENTER if stacked else XL_LABEL_POSITION.OUTSIDE_END
    for i, c in enumerate(colors):
        ser = plot.series[i]
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(c)
        ser.format.line.color.rgb = rgb("FFFFFF"); ser.format.line.width = Pt(1.5)
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_labels.font.size = Pt(font)
    ch.category_axis.format.line.color.rgb = rgb("CCCCCC")
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E6E6E6")
    ch.value_axis.tick_labels.font.size = Pt(font); ch.value_axis.tick_labels.font.color.rgb = rgb(MUTED)
    ch.value_axis.format.line.fill.background()
    return ch


def new_slide():
    global n
    n += 1
    return prs.slides.add_slide(BLANK)


# ================================================================ 1 title
sl = new_slide()
bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = rgb(BLUE); bg.line.fill.background()
text(sl, Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.4), "Which IndiKit indicators do ZOA proposals actually use?", 40, True, "FFFFFF")
text(sl, Inches(0.8), Inches(3.6), Inches(11.5), Inches(1.0), "Findings from reading 115 proposals and logframes with an AI-assisted pipeline", 20, False, "FFFFFF")
text(sl, Inches(0.8), Inches(5.6), Inches(11.5), Inches(0.8), ["MEAL internship · change management in working with data", "24 November 2026"], 14, False, "DCE8F8")

# ================================================================ 2 the task
sl = new_slide(); header(sl, "The task", "The problem")
text(sl, Inches(0.6), Inches(1.6), Inches(12), Inches(0.5), "From the internship brief:", 14, True, MUTED)
bullets(sl, Inches(0.6), Inches(2.1), Inches(12), Inches(4.8), [
    ("Country programmes track progress and impact with indicators from zoa.indikit.net.", ""),
    ("ZOA wants to know which of these indicators are most used in ZOA projects,", "and how labour-intensive the current way of tracking data is."),
    ("The goal:", "research the current way of working and recommend how to improve the processes that come before data visualisation."),
    ("In the long term:", "data can be visualised more effectively, country programmes can learn from each other through comparison, and data increasingly supports strategic decisions."),
], 17, gap=14)
footer(sl)

# ================================================================ 3 the question
sl = new_slide(); header(sl, "What this analysis answers", "The problem")
bullets(sl, Inches(0.6), Inches(1.7), Inches(7.2), Inches(5), [
    ("Which IndiKit indicators do proposals use,", "and how often?"),
    ("Which IndiKit indicators does nobody use?", ""),
    ("Which indicators do proposals use that IndiKit does not have?", ""),
    ("How much work is it to find this out by hand,", "and can a tool do it?"),
], 18, gap=16)
box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.2), Inches(1.7), Inches(4.6), Inches(4.6)); box.fill.solid(); box.fill.fore_color.rgb = rgb(TILE); box.line.fill.background(); box.adjustments[0] = 0.05
text(sl, Inches(8.4), Inches(1.9), Inches(4.2), Inches(0.5), "Why it is hard by hand", 14, True, BLUE)
bullets(sl, Inches(8.4), Inches(2.4), Inches(4.2), Inches(3.8), [
    f"{n_docs} documents in 9 countries, in Word, PDF and Excel, in English and French",
    "351 IndiKit indicators to check against every logframe row",
    "proposals adapt the wording, so a text search misses most matches",
    "the logframe is often in a separate annex",
], 13, gap=8)
footer(sl)

# ================================================================ 4 section method
section("How it was done", "A published method as the reference model, and a language model to do the reading", INK)

# ================================================================ 5 reference model
sl = new_slide(); header(sl, "The reference model: ESGReveal", "How it was done")
bullets(sl, Inches(0.6), Inches(1.6), Inches(6.2), Inches(5.3), [
    ("ESGReveal (Zou et al., 2023)", "is a published method that checks whether company reports disclose a list of sustainability (ESG) indicators."),
    ("Same problem, different words.", "Their indicator list is ESG; ours is IndiKit. Their documents are annual reports; ours are proposals and logframes."),
    ("Three building blocks, kept as in the paper:", "a structured library of the indicators, a layout-aware reading of the documents, and a language model that answers one question per document x indicator pair, with the evidence quoted."),
    ("One addition the paper does not have:", "a pass that collects the proposal indicators that match nothing in IndiKit."),
], 15, gap=12)
steps = [("1", "Library", "Every IndiKit indicator with its wording, purpose, sector and search terms"), ("2", "Documents", "Each proposal parsed into text, outline and tables; logframes detected"),
         ("3", "Pairing", "For each document, only the IndiKit indicators that look plausible are asked about"), ("4", "Answer", "The model says present / not present, with a quote and a match type"),
         ("5", "Check", "Unmatched proposal indicators get a second look; results go to a review file")]
y = Inches(1.6)
for num, h, b in steps:
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.2), y, Inches(0.5), Inches(0.5)); circ.fill.solid(); circ.fill.fore_color.rgb = rgb(BLUE); circ.line.fill.background()
    text(sl, Inches(7.2), y, Inches(0.5), Inches(0.5), num, 13, True, "FFFFFF", PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(sl, Inches(7.85), y - Inches(0.05), Inches(5), Inches(0.4), h, 14, True, INK)
    text(sl, Inches(7.85), y + Inches(0.3), Inches(5), Inches(0.6), b, 11, False, MUTED)
    y += Inches(1.0)
footer(sl)

# ================================================================ 6 the LLM comparison
sl = new_slide(); header(sl, "A language model reads each pair and gives a verdict", "How it was done")
bullets(sl, Inches(0.6), Inches(1.6), Inches(12), Inches(2.2), [
    ("The question asked, for each document and each plausible IndiKit indicator:", "'Does this document contain this indicator? Quote the evidence and say how close the wording is.'"),
    ("The model only sees the passages that retrieval found relevant,", "so it answers from the document, not from memory. Every answer carries a quote that a reviewer can check."),
    ("Models used:", "OpenAI gpt-6.1-terra for the answers, gpt-6-luna for extracting indicators from text sections. 8 086 calls for the whole portfolio, cost $24."),
], 15, gap=10)
text(sl, Inches(0.6), Inches(3.9), Inches(12), Inches(0.4), "The three verdicts", 15, True, BLUE)
cards = [("verbatim", BLUE, "The proposal uses the IndiKit indicator as is.", "Same measure, (nearly) identical wording. Only the target value, a placeholder or the word order differs.",
          "'Number of people having access to basic, safe and dignified shelter solutions'  →  5179 Access to Adequate Shelter"),
         ("close", ORANGE, "The proposal measures the same thing in its own words.", "Different wording, a different unit (number instead of %), a different population or a narrower / wider scope.",
          "'Number of people receiving multi-purpose cash'  →  5239 Number of Cash Transfer Recipients"),
         ("none", MUTED, "Nothing in the document measures the same thing.", "Same topic is not enough: an indicator about water points does not match one about handwashing. Activities ('5 VSLAs supported') never count.",
          "'5 VSLAs supported'  →  no IndiKit match (an activity, not an indicator)")]
for i, (name, col, one, two, ex) in enumerate(cards):
    x = Inches(0.6 + i * 4.1)
    box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(4.35), Inches(3.9), Inches(2.55)); box.fill.solid(); box.fill.fore_color.rgb = rgb(TILE); box.line.fill.background(); box.adjustments[0] = 0.05
    text(sl, x + Inches(0.15), Inches(4.4), Inches(3.6), Inches(0.4), name, 16, True, col)
    text(sl, x + Inches(0.15), Inches(4.8), Inches(3.6), Inches(0.5), one, 12, True, INK)
    text(sl, x + Inches(0.15), Inches(5.3), Inches(3.6), Inches(0.8), two, 10.5, False, INK)
    text(sl, x + Inches(0.15), Inches(6.15), Inches(3.6), Inches(0.7), ex, 9.5, False, MUTED)
footer(sl)

# ================================================================ 7 what went in
sl = new_slide(); header(sl, "What went in", "How it was done")
tile(sl, Inches(0.6), Inches(1.7), Inches(2.9), Inches(2.1), str(n_docs), "documents: proposals, logframes, annexes, trackers", INK)
tile(sl, Inches(3.7), Inches(1.7), Inches(2.9), Inches(2.1), "9", "countries: South Sudan, Uganda, Sudan, Burundi, Syria, Ethiopia, Yemen, DRC, Burkina Faso", INK)
tile(sl, Inches(6.8), Inches(1.7), Inches(2.9), Inches(2.1), "351", f"IndiKit indicators in the library ({len(live)} on the website today, {len(export_only)} not)", INK)
tile(sl, Inches(9.9), Inches(1.7), Inches(2.9), Inches(2.1), "7 016", "document x indicator pairs the model was asked about", INK)
bullets(sl, Inches(0.6), Inches(4.2), Inches(12), Inches(2.8), [
    ("75 Word, 33 PDF and 7 Excel files;", "112 in English, 3 in French. 107 parsed fully, 8 scanned PDFs only partially."),
    (f"{n_skip} documents only point to an annex for their logframe", "and were reported, not queried. 33 more have a logframe and also point to an annex."),
    (f"One proposal ('We are able') sits in six country folders.", f"The {dup_copies} extra copies are counted in the figures that follow unless stated otherwise."),
], 15, gap=12)
footer(sl)

# ================================================================ 8 section findings
section("Findings", "What the proposals use, what they never use, and what IndiKit does not have", BLUE)

# ================================================================ 9 headline
sl = new_slide(); header(sl, "Headline numbers", "Findings")
tiles = [(f"{total_ind:,}".replace(",", " "), "indicators found in the documents", INK), (f"{100 * (n_v + n_c) / total_ind:.0f} %", f"match an IndiKit indicator\n{n_v} verbatim · {n_c} close", BLUE),
         (f"{100 * n_n / total_ind:.0f} %", "have no IndiKit equivalent", ORANGE), (f"{int(live.used.sum())} of {len(live)}", "live IndiKit indicators are used at least once", BLUE),
         (str(len(A)), "live IndiKit indicators are used by nobody", ORANGE), (str(len(B)), "indicators are used but are not on the website", YELLOW)]
for i, (num, lab, col) in enumerate(tiles):
    tile(sl, Inches(0.6 + (i % 3) * 4.1), Inches(1.7 + (i // 3) * 2.5), Inches(3.9), Inches(2.2), num, lab, col)
footer(sl)

# ================================================================ 10 most matches are close
sl = new_slide(); header(sl, "Most matches are close, not verbatim", "Findings · match types")
cd = CategoryChartData(); cd.categories = ["verbatim", "close", "none"]; cd.add_series("document indicators", [n_v, n_c, n_n])
ch = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.6), Inches(5.6), Inches(5.2), cd).chart
ch.has_legend = False; ch.font.size = Pt(12); ch.font.name = "Calibri"
plot = ch.plots[0]; plot.gap_width = 80; plot.has_data_labels = True; plot.data_labels.font.size = Pt(12); plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for i, c in enumerate([BLUE, ORANGE, MUTED]):
    pt = plot.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
ch.value_axis.has_major_gridlines = True; ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E6E6E6"); ch.value_axis.format.line.fill.background()
bullets(sl, Inches(6.6), Inches(1.8), Inches(6.2), Inches(5), [
    (f"Only {n_v} indicators ({100 * n_v / total_ind:.0f} %) use IndiKit wording as is.", ""),
    (f"{n_c} ({100 * n_c / total_ind:.0f} %) measure the same thing in their own words.", "This is where most of the reuse of IndiKit actually happens, and it is invisible to a text search."),
    (f"{n_n} ({100 * n_n / total_ind:.0f} %) have no IndiKit equivalent.", "Most are project-specific output counts."),
    (f"Per document the picture is better:", f"the median document has {median_doc:.0f} % of its indicators matched; 14 documents reach 100 %."),
], 15, gap=12)
footer(sl)

# ================================================================ 11 by country
sl = new_slide(); header(sl, "Syria uses IndiKit most, DRC and Sudan least", "Findings · by country")
bar_chart(sl, Inches(0.6), Inches(1.6), Inches(7.2), Inches(5.3), byc.label.tolist(), [("% of document indicators matched", byc.pct_matched.round(1).tolist())], [BLUE], number_format='0"%"')
bullets(sl, Inches(8.1), Inches(1.8), Inches(4.8), Inches(5), [
    ("Syria", f"delivers {int(byc.set_index('label').loc['Syria', 'verbatim'])} of the {n_v} verbatim matches in the whole portfolio: its joint-response logframes use IndiKit wording."),
    ("DRC, Sudan, Uganda, Ethiopia, Burundi, South Sudan", "each hold a copy of the 'We are able' proposal (38 indicators, 5 matched), which lowers their share."),
    ("Burkina Faso", "has two documents and seven indicators; its share means little."),
], 14, gap=12)
footer(sl)

# ================================================================ 12 by sector
sl = new_slide(); header(sl, "Use differs a lot between sectors", "Findings · by IndiKit sector (website today)")
cats = [c.replace("cross-cutting: ", "") for c in by_sector.sector]
bar_chart(sl, Inches(0.6), Inches(1.6), Inches(7.6), Inches(5.3), cats, [("used", by_sector.used.tolist()), ("never used", by_sector.never.tolist())], [BLUE, ORANGE], stacked=True)
fs = by_sector.set_index("sector").loc["Food Security and Livelihoods"]
bullets(sl, Inches(8.4), Inches(1.8), Inches(4.5), Inches(5), [
    ("WASH has the most live indicators and uses few of them:", f"{int(by_sector.set_index('sector').loc['Water, Sanitation and Hygiene', 'used'])} of {int(by_sector.set_index('sector').loc['Water, Sanitation and Hygiene', 'total'])}."),
    ("Food Security and Livelihoods:", f"{int(fs.used)} of the {int(fs.total)} indicators still on the site are used; the sector lost most of its pages in November."),
    ("Shelter and Peacebuilding", "are used most intensively relative to their size."),
    ("Gender Diversity & Inclusion:", "6 of 23 used; its organisational and staff measures are untouched."),
], 14, gap=12)
footer(sl)

# ================================================================ 13 most used
sl = new_slide(); header(sl, "A small core of indicators carries most of the use", "Findings · most-used IndiKit indicators")
top = mu.head(10)
rows = [[r.indikit_id, r.indikit_name[:60], r.sector.replace("cross-cutting: ", ""), r.documents, r.verbatim, r.close, "yes" if meta[str(r.indikit_id)]["on_site"] else "no"] for r in top.itertuples()]
table(sl, Inches(0.5), Inches(1.6), Inches(8.6), Inches(4.4), ["ID", "IndiKit indicator", "sector", "documents", "verbatim", "close", "on site"], rows, [0.6, 3.9, 2.1, 0.8, 0.6, 0.5, 0.6], 11)
bullets(sl, Inches(9.4), Inches(1.7), Inches(3.6), Inches(5.2), [
    ("37 indicators", "appear in four or more documents."),
    ("68 of the used indicators", "appear in one document only."),
    ("The most-used one is not on the website", "(next slides)."),
], 14, gap=12)
footer(sl)

# ================================================================ 14 never used
sl = new_slide(); header(sl, f"{len(A)} live IndiKit indicators are used by no proposal", "Findings · never used")
asec = A.sector.value_counts().reindex([c for c in by_sector.sector if c in A.sector.values])
bar_chart(sl, Inches(0.6), Inches(1.6), Inches(6.2), Inches(3.3), [c.replace("cross-cutting: ", "") for c in asec.index], [("never used", asec.values.tolist())], [ORANGE])
text(sl, Inches(0.7), Inches(5.0), Inches(6.2), Inches(0.4), "Sub-sectors that nobody uses at all", 13, True, INK)
table(sl, Inches(0.7), Inches(5.4), Inches(6.1), Inches(1.4), ["sector", "sub-sector", "indicators"], [[r.sector.replace("cross-cutting: ", ""), r.sub, r.total] for r in unused_subs.itertuples()], [2.3, 3.0, 0.8], 10)
text(sl, Inches(7.3), Inches(1.6), Inches(5.6), Inches(0.4), "Two different reasons", 14, True, INK)
tile(sl, Inches(7.3), Inches(2.1), Inches(2.7), Inches(2.0), str(len(A_present)), "the concept is in a proposal, under other wording", AQUA)
tile(sl, Inches(10.2), Inches(2.1), Inches(2.7), Inches(2.0), str(len(A_absent)), "nothing in any proposal comes close", ORANGE)
bullets(sl, Inches(7.3), Inches(4.3), Inches(5.6), Inches(2.7), [
    ("First group:", "clearer wording or guidance would let proposals pick the IndiKit version."),
    ("Second group:", "either outside current programming, or outcome measures that proposals do not commit to."),
], 13, gap=10)
footer(sl)

# ================================================================ 15 concept present
sl = new_slide(); header(sl, "The concept is often there, the IndiKit wording is not", f"Findings · never used · {len(A_present)} indicators with a close proposal indicator")
rows = [[r.id, r.name[:55], f"{r.best:.2f}", r.closest[:100]] for r in A_present.head(9).itertuples()]
table(sl, Inches(0.5), Inches(1.6), Inches(12.3), Inches(4.2), ["ID", "IndiKit indicator (never used)", "similarity", "closest indicator found in a proposal"], rows, [0.6, 3.8, 0.9, 7.0], 10)
bullets(sl, Inches(0.5), Inches(6.0), Inches(12.3), Inches(1.0), [
    ("Example:", "four Climate Resilience indicators all sit next to the same proposal outcome 'communities with improved climate resilience'. The Household Hunger Score is used under its survey wording, not its IndiKit name."),
], 12)
footer(sl)

# ================================================================ 16 absent
sl = new_slide(); header(sl, "What no proposal measures at all", f"Findings · never used · {len(A_absent)} indicators with nothing close")
themes = [("Detailed WASH outcomes", "hygiene knowledge and practice, water quality and safe storage, queuing time at water points, latrine maintenance, menstrual hygiene facilities"),
          ("Cash: process and quality", "how cash and voucher assistance is delivered: timeliness, complaint handling, recipients' understanding of entitlements, post-distribution quality"),
          ("Organisational and staff measures", "staff understanding of gender, inclusion and protection mainstreaming; gender, climate and peace-nexus checks in evaluations; cash reaching local groups"),
          ("Gender-based violence and land rights", "all Peacebuilding GBV indicators; trust in justice providers"),
          ("Jobs and savings", "jobs created and retained, average savings in saving groups, functionality of self-help groups")]
y = Inches(1.7)
for h, b in themes:
    text(sl, Inches(0.7), y, Inches(3.4), Inches(0.8), h, 15, True, ORANGE)
    text(sl, Inches(4.1), y, Inches(8.7), Inches(0.8), b, 13)
    y += Inches(0.9)
text(sl, Inches(0.7), Inches(6.3), Inches(12), Inches(0.6), "Proposals commit to outputs (people reached, structures built, trainings held) far more than to the outcome and quality measures these indicators describe.", 13, False, MUTED)
footer(sl)

# ================================================================ 16b the website changed
sl = new_slide(); header(sl, "The website itself changed during the project", "Findings · zoa.indikit.net")
table(sl, Inches(0.6), Inches(1.6), Inches(5.6), Inches(2.0), ["what", "16 Nov 2026", "24 Nov 2026"],
      [["indicator pages that load", "310", str(len(live))], ["IDs in the site's sitemap", "214", "214"], ["library indicators not on the site", "41", str(len(export_only))]], [2.9, 1.35, 1.35], 12)
bullets(sl, Inches(6.6), Inches(1.6), Inches(6.3), Inches(5.3), [
    ("97 pages that loaded on 16 November return 'not found' today,", "94 of them Food Security and Livelihoods. The site's sitemap never listed them, so the removal looks planned."),
    ("Food Security and Livelihoods went from 172 to 33 live indicators.", "The live library is now led by WASH, Cash and Voucher Assistance, Gender, Peacebuilding and Shelter."),
    ("46 indicators that proposals use are among the removed or unpublished ones", "(next slide)."),
    ("The matches in this deck are unaffected:", "the pipeline checked all 351 library indicators. Only the 'on the website' flags were refreshed, with one command, after a re-scrape."),
], 14, gap=12)
text(sl, Inches(0.6), Inches(3.9), Inches(5.6), Inches(2.5), ["Examples of removed pages:", "5532 Accessing a Service (used in 9 proposals)", "5530 Active Participation in Activities (10 proposals)", "5092 Months of Adequate Household Food Provisioning", "5147 Number of Selling Points"], 12, False, MUTED)
footer(sl)

# ================================================================ 17 not on the site
sl = new_slide(); header(sl, "The most-used indicator of all is not on the website", "Findings · used in proposals, not on zoa.indikit.net today")
rows = [[r.indikit_id, r.indikit_name[:60], r.documents] for r in bm.head(13).itertuples()]
table(sl, Inches(0.5), Inches(1.6), Inches(6.4), Inches(4.9), ["ID", "IndiKit indicator (not on the website)", "proposals"], rows, [0.7, 4.6, 1.1], 11)
bullets(sl, Inches(7.3), Inches(1.7), Inches(5.6), Inches(5.2), [
    (f"{len(export_only)} of the 351 library indicators are not on the website today:", "41 never were (Excel export only), the rest disappeared in November (previous slide)."),
    (f"{len(B)} of them are used in proposals,", f"in {B_docs} documents. The table shows the most used; the full list is in the Analysis IndiKit export."),
    ("Gender Representation (5428)", "is used in 17 documents, 9 of them verbatim: the single most-used IndiKit indicator in the portfolio. Anyone browsing the website cannot find it."),
    (f"The other {len(export_only) - len(B)}", "are used by nobody."),
], 14, gap=12)
footer(sl)

# ================================================================ 18 not in IndiKit
sl = new_slide(); header(sl, "What proposals measure that IndiKit does not have", "Findings · proposal-only indicators")
bullets(sl, Inches(0.6), Inches(1.55), Inches(12.2), Inches(1.5), [
    (f"{n_n} proposal indicators have no IndiKit equivalent.", "Half of them are counts of activities (trainings held, sessions organised, people reached), which IndiKit does not define as indicators and should not."),
    ("The rest are outcomes written in the proposal's own words.", "Some appear in several proposals and countries. Those are the ones worth a look:"),
], 15, gap=8)
examples = [["% of households reporting improved resilience to shocks", "2", "DRC, South Sudan"],
            ["Number of people provided with resources to protect and start rebuilding livelihood assets", "3", "Syria, Yemen"],
            ["People's perception of stability and peace in the targeted areas", "2", "Ethiopia"],
            ["Number of conflict incidents in the targeted areas", "2", "Ethiopia"],
            ["Number of school hygiene clubs established and supported", "2", "Ethiopia, South Sudan"],
            ["Households testify that improved housing has a positive impact on their dignity and self-esteem", "2", "Burundi"],
            ["Number of advocacy initiatives carried out by civil-society organisations with their constituency", "1 (filed in 6 countries)", "We are able"],
            ["Number of civil-society organisations with increased lobby and advocacy capacity", "1 (filed in 6 countries)", "We are able"]]
table(sl, Inches(0.6), Inches(3.1), Inches(12.2), Inches(3.3), ["indicator, as written in the proposal", "proposals", "where"], examples, [8.2, 1.8, 2.2], 11)
text(sl, Inches(0.6), Inches(6.45), Inches(12.2), Inches(0.5), "Question for the team: should any of these become IndiKit indicators? (The 'We are able' proposal appears six times in the folders, so its indicators looked more common than they are.)", 13, True, BLUE)
footer(sl)

# ================================================================ 19 what it means
sl = new_slide(); header(sl, "What this means for the way of working", "Findings · back to the task")
bullets(sl, Inches(0.6), Inches(1.6), Inches(12.2), Inches(5.4), [
    ("Reuse of IndiKit is real but hidden.", "Four in five matches are 'close', so counting IndiKit use by searching for its wording undercounts it by a factor of four. Comparison across countries needs the measure, not the wording."),
    ("The labour is in the reading, and a tool can carry it.", "The pipeline read 115 documents and asked 7 016 questions for $24 and a day of machine time; the same by hand is weeks. What remains for people is the review of close matches, which the review file supports."),
    ("The logframes are the bottleneck for visualisation.", "Logframes live in annexes, result levels are missing in two thirds of rows, and the same proposal is filed six times. Fix the filing and the templates before fixing the dashboards."),
    ("IndiKit itself needs maintenance.", f"{len(A)} live indicators are unused, four sub-sectors entirely; {len(B)} used indicators are not on the site; the Food Consumption Score exists twice."),
], 15, gap=14)
footer(sl)

# ================================================================ 20 caveats
sl = new_slide(); header(sl, "Read the numbers with these in mind", "Caveats")
bullets(sl, Inches(0.6), Inches(1.6), Inches(12.2), Inches(5.4), [
    ("The matches have not been checked by a person yet.", "They are the model's answers with quotes. A verified sample is the next step; until then the figures are a first measurement."),
    ("'Never used' means never used in a logframe the pipeline could read.", f"{n_skip} documents keep their logframe in an annex that was not in the folder; {n_zero} narrative documents yielded no indicators."),
    ("Duplicates inflate the totals.", "Six copies of one proposal; 33 documents whose indicators also live in an annex."),
    ("'On the website' is a snapshot", "of zoa.indikit.net taken on 24 November 2026. The site lost a third of its pages the week before, so re-scrape before quoting these numbers (Site Comparison page, full re-scrape)."),
], 15, gap=14)
footer(sl)

# ================================================================ 21 next steps
sl = new_slide(); header(sl, "Recommended next steps", "Next steps")
recs = [("1", "Check a sample by hand", "Review the close matches of 8 to 11 documents in review.xlsx; that turns these figures into measured accuracy."),
        ("2", f"Ask the IndiKit owners about the {len(B)} used indicators that are not on the website", "97 pages disappeared in November, among them Active Participation in Activities and Accessing a Service; Gender Representation (5428) was never published. Restore them or tell the country teams what replaces them."),
        ("3", "Go through the unused live indicators with the sector leads", f"{len(A_present)} need clearer wording or guidance; {len(A_absent)} and the four empty sub-sectors need a keep-or-retire decision."),
        ("4", "Fix the inputs before the dashboards", "one copy per proposal, logframe in the proposal folder, result level on every row."),
        ("5", "Add the recurring proposal indicators to IndiKit", "advocacy and civil-society capacity, social inclusion, returnee reintegration."),
        ("6", "Keep the pipeline running per new proposal", "so the picture stays current and the review effort stays small.")]
y = Inches(1.55)
for num, h, b in recs:
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), y + Inches(0.05), Inches(0.5), Inches(0.5)); circ.fill.solid(); circ.fill.fore_color.rgb = rgb(BLUE); circ.line.fill.background()
    text(sl, Inches(0.6), y + Inches(0.05), Inches(0.5), Inches(0.5), num, 14, True, "FFFFFF", PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(sl, Inches(1.3), y, Inches(11.5), Inches(0.4), h, 15, True, INK)
    text(sl, Inches(1.3), y + Inches(0.4), Inches(11.5), Inches(0.5), b, 12, False, MUTED)
    y += Inches(0.9)
footer(sl)

prs.save(OUT)
print("saved", OUT, "slides:", n)

"""Synthetic document builders shared by the Phase 2 tests (created inside tmp_path, never read from proposals/)."""
from __future__ import annotations

from pathlib import Path


def make_docx(path: Path) -> Path:
    from docx import Document

    doc = Document()
    doc.add_heading("Project proposal", level=1)
    doc.add_paragraph("This project improves access to safe water for displaced households in Aleppo.")
    doc.add_heading("3.2 Logical framework", level=2)
    doc.add_paragraph("The logframe below lists the indicators.")
    table = doc.add_table(rows=4, cols=4)
    cells = [
        ["Intervention logic", "Indicator", "Baseline", "Target"],
        ["Outcome 1: Improved WASH", "% of households with handwashing facilities with soap and water", "12%", "60%"],
        ["", "Number of people receiving hygiene kits", "0", "2,500"],
        ["Output 1.1: Latrines built", "# of latrines constructed", "0", "40"],
    ]
    for i, row in enumerate(cells):
        for j, text in enumerate(row):
            table.cell(i, j).text = text
    # merged cell across the two result rows of outcome 1 (vertical) and a nested table in the last cell
    table.cell(1, 0).merge(table.cell(2, 0))
    nested = table.cell(3, 3).add_table(rows=1, cols=2)
    nested.cell(0, 0).text = "Q1"
    nested.cell(0, 1).text = "20"
    doc.add_paragraph("Assumptions: access to the area remains possible. See Annex 3 for the budget.")
    doc.add_heading("4 Monitoring", level=2)
    doc.add_paragraph("Detailed indicator definitions are provided in Annex C - Logical Framework.")
    doc.save(str(path))
    return path


def make_xlsx(path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Logframe"
    ws["A1"] = "Cadre logique de l'action"
    ws.merge_cells("A1:E1")
    ws.append([])
    ws.append(["Logique d'intervention", "Indicateurs objectivement vérifiables", "Valeur de référence", "Cible", "Sources et moyens de vérification"])
    ws.append(["Objectif spécifique 1", "% de ménages ayant accès à l'eau potable", "20%", "80%", "Enquête ménages"])
    ws.append([None, "Nombre de points d'eau réhabilités", 0, 15, "Rapports"])
    ws.append(["Résultat 1.1", "Taux de satisfaction des usagers", "n/a", "75%", "Enquête"])
    ws.merge_cells("A4:A5")
    ws2 = wb.create_sheet("Budget")
    ws2.append(["Item", "Unit", "Qty", "Cost"])
    ws2.append(["Pipes", "m", 100, 2500])
    ws2.append([])
    ws2.append([])
    ws2.append([])
    ws2.append(["Notes"])
    ws2.append(["Prices from 2024 quotation"])
    wb.save(str(path))
    return path


def make_pdf(path: Path, two_page_table: bool = True) -> Path:
    """A small PDF: a heading, two paragraphs, a ruled table that continues on page 2 (repeated header), a footer."""
    import pymupdf

    doc = pymupdf.open()
    width, height = 595, 842

    def draw_table(page, y0, rows, col_w=(100, 280, 55, 55)):
        x0 = 60
        y = y0
        for r, row in enumerate(rows):
            x = x0
            for j, cell in enumerate(row):
                rect = pymupdf.Rect(x, y, x + col_w[j], y + 24)
                page.draw_rect(rect, color=(0, 0, 0), width=0.7)
                fitted = page.insert_textbox(pymupdf.Rect(x + 2, y + 2, x + col_w[j] - 2, y + 22), cell, fontsize=7, fontname="helv")
                assert fitted >= 0, f"cell text did not fit: {cell!r}"
                x += col_w[j]
            y += 24
        return y

    header = ["Result", "Indicator", "Baseline", "Target"]
    page = doc.new_page(width=width, height=height)
    page.insert_text((60, 70), "2. Results framework", fontsize=16, fontname="hebo")
    page.insert_text((60, 110), "The project will reach 2,000 households in the target wereda with water trucking.", fontsize=10, fontname="helv")
    page.insert_text((60, 126), "Percent of households with soap and water at a handwashing station on premises is tracked.", fontsize=10, fontname="helv")
    rows1 = [header, ["Outcome 1", "% of households with handwashing facilities with soap and water", "10%", "50%"], ["", "Number of boreholes rehabilitated", "0", "12"]]
    draw_table(page, 700, rows1)
    page.insert_text((60, 820), "Proposal ZOA - page 1", fontsize=8, fontname="helv")
    page2 = doc.new_page(width=width, height=height)
    if two_page_table:
        rows2 = [header, ["Output 1.1", "% of water samples with free residual chlorine above 0.2 mg/l", "n/a", "90%"]]
        draw_table(page2, 50, rows2)
    page2.insert_text((60, 300), "3. Monitoring", fontsize=16, fontname="hebo")
    page2.insert_text((60, 330), "Indicators are reported quarterly in the Indicator Tracking Table (Annex O).", fontsize=10, fontname="helv")
    page2.insert_text((60, 820), "Proposal ZOA - page 2", fontsize=8, fontname="helv")
    page3 = doc.new_page(width=width, height=height)
    page3.insert_text((60, 70), "Annex A: Budget", fontsize=16, fontname="hebo")
    page3.insert_text((60, 100), "Budget lines are listed in the attached spreadsheet.", fontsize=10, fontname="helv")
    page3.insert_text((60, 820), "Proposal ZOA - page 3", fontsize=8, fontname="helv")
    doc.save(str(path))
    return path

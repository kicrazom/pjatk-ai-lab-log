from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def build_summary_pdf(summary: dict) -> bytes:
    buffer = BytesIO()
    doc = canvas.Canvas(buffer, pagesize=A4)
    y = 800
    doc.setFont("Helvetica-Bold", 16)
    doc.drawString(40, y, "Podsumowanie Kompleksowej Oceny Geriatrycznej")
    y -= 30
    doc.setFont("Helvetica", 11)
    for key, value in summary.items():
        doc.drawString(40, y, f"{key}: {value}")
        y -= 20
        if y < 60:
            doc.showPage()
            y = 800
    doc.save()
    return buffer.getvalue()

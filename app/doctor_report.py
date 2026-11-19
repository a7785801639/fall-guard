"""
Отчёт для врача в PDF: падения и ночная активность за выбранный период.

Шрифт — DejaVu Sans из app/fonts (полный, с кириллицей и цифрами), чтобы отчёт одинаково собирался
и в Docker, и на любой машине. Графики рисуются средствами reportlab, без внешних программ.
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT = "FGSans"
FONT_BOLD = "FGSans-Bold"
INK = colors.HexColor("#1B1F27")
MUTED = colors.HexColor("#5F6673")
LINE = colors.HexColor("#D9DCE2")
BAR = colors.HexColor("#3E6FD8")
BAR_OLD = colors.HexColor("#A9BDEB")
ALARM = colors.HexColor("#C62F36")
WARN_BG = colors.HexColor("#FFF4E0")
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

_fonts_ready = False


def _register_fonts() -> None:
    global _fonts_ready
    if _fonts_ready:
        return
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    _fonts_ready = True


def missing_glyphs(text: str) -> set[str]:
    """Символы, которых нет в шрифте отчёта (проверка в тестах: в PDF не должно быть «квадратиков»)."""
    _register_fonts()
    face = pdfmetrics.getFont(FONT).face
    cmap = getattr(face, "charToGlyph", None) or {}
    return {ch for ch in set(text) if ch.strip() and ord(ch) not in cmap}


def _d(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y")


def _hm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _num(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _utc_offset(stats: dict[str, Any]) -> str:
    m = int(stats.get("server_tz_offset_min", 0) or 0)
    sign = "+" if m >= 0 else "−"
    h, mm_ = divmod(abs(m), 60)
    return f"UTC{sign}{h}" + (f":{mm_:02d}" if mm_ else "")


def _night_name(day: str) -> str:
    d = datetime.strptime(day, "%Y-%m-%d")
    return f"{d.day} {MONTHS[d.month - 1]}, {WD[d.weekday()]}"


def _bar_chart(items: list[dict[str, Any]], key: Any, title: str, unit_fmt: Any, base: float | None, width: float) -> Drawing:
    """Столбики по ночам, одна ось. Последняя неделя — ярче; ночи с малым покрытием — светлые."""
    h = 46 * mm
    d = Drawing(width, h)
    left, bottom, top_pad = 22, 16, 18
    pw, ph = width - left - 4, h - bottom - top_pad
    d.add(String(0, h - 11, title, fontName=FONT_BOLD, fontSize=9.5, fillColor=INK))
    vals = [float(key(it)) for it in items]
    vmax = max([1.0, *vals, base or 0.0])
    # Верх оси — чётное целое: средняя отметка тогда тоже целая и честная
    vmax = float(int(vmax) + (1 if vmax > int(vmax) else 0))
    if vmax % 2:
        vmax += 1
    for t in (0, vmax / 2, vmax):
        y = bottom + ph * t / vmax
        d.add(Line(left, y, left + pw, y, strokeColor=LINE, strokeWidth=0.5))
        d.add(String(left - 4, y - 3, unit_fmt(t), fontName=FONT, fontSize=7, fillColor=MUTED, textAnchor="end"))
    n = max(len(items), 7)
    gap = 1.2
    bw = max(1.5, (pw - gap * (n - 1)) / n)
    recent_from = len(items) - 7
    for i, it in enumerate(items):
        v = vals[i]
        x = left + i * (bw + gap)
        bh = ph * v / vmax
        fill = BAR if (i >= recent_from and it.get("valid")) else BAR_OLD
        if not it.get("valid"):
            fill = colors.HexColor("#E1E6F1")
        if bh > 0:
            d.add(Rect(x, bottom, bw, bh, fillColor=fill, strokeColor=None))
        dt = datetime.strptime(it["day"], "%Y-%m-%d")
        if dt.weekday() == 0 or i == 0:
            d.add(String(x, 4, f"{dt.day}.{dt.month:02d}", fontName=FONT, fontSize=6.5, fillColor=MUTED))
    if base is not None:
        y = bottom + ph * base / vmax
        d.add(Line(left, y, left + pw, y, strokeColor=INK, strokeWidth=0.7, strokeDashArray=[3, 2]))
        d.add(String(left + 3, y + 3, f"обычно {_num(base)}", fontName=FONT, fontSize=7, fillColor=INK))
    return d


def build_report(stats: dict[str, Any], days: int, now_ts: float | None = None) -> bytes:
    """stats — то, что отдаёт engine.get_care_stats(days)."""
    _register_fonts()
    now = float(time.time() if now_ts is None else now_ts)
    nights = [n for n in stats.get("nights", []) if not n.get("partial")]
    falls = stats.get("falls", [])
    real_falls = [f for f in falls if not f.get("false_alarm")]
    false_count = len(falls) - len(real_falls)
    tr = stats.get("trend", {}) or {}
    ns, ne = stats.get("night_hours", [22, 7])
    valid = [n for n in nights if n.get("valid")]
    period_from = now - days * 86400.0

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
        title="FallGuard — отчёт для врача", author="FallGuard",
    )
    width = A4[0] - 32 * mm
    st = {
        "h1": ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=17, leading=21, textColor=INK, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontName=FONT, fontSize=9.5, leading=13, textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5),
        "p": ParagraphStyle("p", fontName=FONT, fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=8, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, textColor=INK),
    }
    story: list[Any] = []
    story.append(Paragraph("Отчёт для врача: падения и ночная активность", st["h1"]))
    story.append(Paragraph(
        f"Период: {_d(period_from)} — {_d(now)} ({days} дн.) · Ночь считается с {ns:02d}:00 до {ne:02d}:00 · "
        f"Сформирован {_d(now)} в {_hm(now)} · Время в отчёте — по часам системы ({_utc_offset(stats)})", st["sub"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Пациент: ______________________________ &nbsp;&nbsp; Дата рождения: ______________", st["p"]))

    # --- Главное в цифрах
    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    getups_avg = avg([n["getups"] for n in valid])
    bed_avg = avg([n["in_bed_sec"] / 3600.0 for n in valid])
    cells = [
        ("Падений", str(len(real_falls)), f"ложных тревог исключено: {false_count}" if false_count else "за период"),
        ("Вставала за ночь", _num(getups_avg) if valid else "—", "в среднем, раз"),
        ("В кровати ночью", _num(bed_avg) if valid else "—", "в среднем, часов"),
        ("Ночей с данными", f"{len(valid)} из {len(nights)}" if nights else "0", "камера видела ≥ 60% ночи"),
    ]
    k_label = ParagraphStyle("kl", fontName=FONT, fontSize=8, leading=10, textColor=MUTED)
    k_value = ParagraphStyle("kv", fontName=FONT_BOLD, fontSize=16, leading=21, textColor=INK)
    k_note = ParagraphStyle("kn", fontName=FONT, fontSize=7.5, leading=9.5, textColor=MUTED)
    kpi = Table(
        [[[Paragraph(a, k_label), Paragraph(b, k_value), Paragraph(c, k_note)] for a, b, c in cells]],
        colWidths=[width / 4] * 4,
    )
    kpi.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 5 * mm))
    story.append(kpi)

    # --- Изменения
    if tr.get("ready") and tr.get("items"):
        warn = Table([[Paragraph("<b>Замечено изменение.</b> " + " ".join(i["text"] + "." for i in tr["items"]), st["p"])]], colWidths=[width])
        warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WARN_BG), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#F0C878")),
                                  ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
        story.append(Spacer(1, 4 * mm))
        story.append(warn)
    elif tr.get("ready"):
        story.append(Spacer(1, 3 * mm))
        b = tr.get("baseline", {})
        story.append(Paragraph(f"Последняя неделя без заметных изменений относительно предыдущих ночей (обычно вставала {_num(b.get('getups', 0))} раза за ночь, в кровати {_num(b.get('in_bed_h', 0))} ч).", st["p"]))

    # --- Графики
    if nights:
        base = tr.get("baseline") if tr.get("ready") else None
        story.append(Paragraph("Ночи по дням", st["h2"]))
        story.append(_bar_chart(nights, lambda n: n["getups"], "Сколько раз вставала за ночь", lambda v: str(int(round(v))), base["getups"] if base else None, width))
        story.append(Spacer(1, 3 * mm))
        story.append(_bar_chart(nights, lambda n: n["in_bed_sec"] / 3600.0, "Сколько часов была в кровати ночью", lambda v: str(int(round(v))), base["in_bed_h"] if base else None, width))
        story.append(Paragraph("Тёмные столбики — последние 7 ночей; бледные — ночи, где камера видела меньше 60% времени (в средние значения не входят).", st["small"]))

    if not nights:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Сводки ночей пока не накоплены: система сохраняет их каждое утро, графики и таблица появятся в следующих отчётах.", st["p"]))

    # --- Падения
    story.append(Paragraph("Журнал падений", st["h2"]))
    if real_falls:
        rows = [["Дата и время", "Что зафиксировано"]] + [
            [f"{_d(f['ts'])}, {_hm(f['ts'])}", Paragraph(str(f.get("description") or f.get("title") or "Падение"), st["cell"])] for f in sorted(real_falls, key=lambda x: x["ts"])
        ]
        t = Table(rows, colWidths=[36 * mm, width - 36 * mm], repeatRows=1)
        t.setStyle(_table_style())
        story.append(t)
    else:
        story.append(Paragraph("За период падений не зафиксировано.", st["p"]))
    if false_count:
        word = "тревогу" if false_count % 10 == 1 and false_count % 100 != 11 else "тревоги" if false_count % 10 in (2, 3, 4) and false_count % 100 not in (12, 13, 14) else "тревог"
        story.append(Paragraph(f"Ещё {false_count} {word} семья отметила как ложные — они не включены.", st["small"]))

    # --- Ночи таблицей
    if nights:
        story.append(Paragraph("Ночная активность по ночам", st["h2"]))
        rows = [["Ночь", "Вставала", "Во сколько", "В кровати", "Садилась", "Данные"]]
        for n in sorted(nights, key=lambda x: x["day"], reverse=True):
            times = ", ".join(_hm(t) for t in n.get("getup_times", [])[:8]) + ("…" if len(n.get("getup_times", [])) > 8 else "")
            rows.append([_night_name(n["day"]), str(n["getups"]), Paragraph(times or "—", st["cell"]), f"{_num(n['in_bed_sec'] / 3600.0)} ч",
                         str(n.get("sit_ups", 0)), f"{int(round(n.get('coverage', 0) * 100))}%"])
        t = Table(rows, colWidths=[34 * mm, 18 * mm, width - 34 * mm - 18 * mm - 22 * mm - 18 * mm - 16 * mm, 22 * mm, 18 * mm, 16 * mm], repeatRows=1)
        t.setStyle(_table_style())
        for i, n in enumerate(sorted(nights, key=lambda x: x["day"], reverse=True), start=1):
            if not n.get("valid"):
                t.setStyle(TableStyle([("TEXTCOLOR", (0, i), (-1, i), MUTED)]))
        story.append(t)

    story.append(Spacer(1, 6 * mm))
    story.append(KeepTogether([Paragraph(
        "Как получены данные. Сведения собраны автоматически домашней системой FallGuard по видеокамере у кровати "
        "(распознавание позы человека). «Вставала» — выходы из кровати; возвраты в течение минуты считаются одним выходом. "
        "«В кровати» — время, когда человек лежал в кровати, — это не то же самое, что сон. "
        "Возможны пропуски (камера выключена, пауза наблюдения) и ошибки распознавания; столбец «Данные» показывает, "
        "какую часть ночи камера действительно видела. Отчёт не является медицинским заключением.", st["small"])]))

    def footer(canvas: Any, doc_: Any) -> None:
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 10 * mm, "FallGuard · отчёт для врача")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Стр. {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _table_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

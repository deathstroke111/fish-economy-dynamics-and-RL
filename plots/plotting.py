"""Minimal SVG plot generation utilities.

This keeps the project runnable without requiring matplotlib in the local
environment. The output files are plain SVGs that can be opened directly.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ColorSeries = Sequence[Tuple[str, Sequence[float]]]

COLORS = (
    "#0f766e",
    "#b45309",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#059669",
    "#d97706",
    "#1d4ed8",
)


def _ensure_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _chart_bounds(series: ColorSeries) -> Tuple[float, float]:
    values: List[float] = []
    for _, data in series:
        values.extend(data)
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def _scale_x(index: int, count: int, left: int, width: int) -> float:
    if count <= 1:
        return left + width / 2.0
    return left + (index / float(count - 1)) * width


def _scale_y(value: float, minimum: float, maximum: float, top: int, height: int) -> float:
    ratio = (value - minimum) / (maximum - minimum)
    return top + height - (ratio * height)


def line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: ColorSeries,
    output_path: Path,
) -> None:
    """Render a simple multi-line SVG chart."""

    _ensure_output(output_path)
    width = 1100
    height = 700
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    min_y, max_y = _chart_bounds(series)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fcfcfc"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="24" fill="#111827">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="18" fill="#111827">{x_label}</text>',
        f'<text x="25" y="{height / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 25 {height / 2})" fill="#111827">{y_label}</text>',
    ]

    for tick in range(6):
        ratio = tick / 5.0
        y_value = min_y + ((max_y - min_y) * (1.0 - ratio))
        y = margin_top + (plot_height * ratio)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" font-size="14" fill="#4b5563">{y_value:.1f}</text>'
        )

    legend_y = 60
    legend_x = margin_left
    for index, (label, data) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = []
        for value_index, value in enumerate(data):
            x = _scale_x(value_index, len(data), margin_left, plot_width)
            y = _scale_y(value, min_y, max_y, margin_top, plot_height)
            points.append(f"{x:.2f},{y:.2f}")
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )
        svg_parts.append(
            f'<rect x="{legend_x + index * 170}" y="{legend_y}" width="18" height="18" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 24 + index * 170}" y="{legend_y + 14}" font-size="14" fill="#111827">{label}</text>'
        )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def bar_chart(
    title: str,
    x_label: str,
    y_label: str,
    values: Dict[str, float],
    output_path: Path,
) -> None:
    """Render a simple single-series bar chart."""

    _ensure_output(output_path)
    width = 1100
    height = 700
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if values:
        min_value = min(min(values.values()), 0.0)
        max_value = max(max(values.values()), 0.0)
    else:
        min_value = 0.0
        max_value = 1.0
    if min_value == max_value:
        max_value = min_value + 1.0
    value_range = max_value - min_value
    zero_y = margin_top + plot_height - ((0.0 - min_value) / value_range) * plot_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fcfcfc"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-size="24" fill="#111827">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{zero_y}" x2="{margin_left + plot_width}" y2="{zero_y}" stroke="#374151" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-size="18" fill="#111827">{x_label}</text>',
        f'<text x="25" y="{height / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 25 {height / 2})" fill="#111827">{y_label}</text>',
    ]

    for tick in range(6):
        ratio = tick / 5.0
        y_value = max_value - ratio * value_range
        y = margin_top + ratio * plot_height
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" font-size="14" fill="#4b5563">{y_value:.1f}</text>'
        )

    count = len(values)
    bar_width = plot_width / max(count * 1.5, 1)
    spacing = bar_width / 2
    x = margin_left + spacing
    for index, (label, value) in enumerate(values.items()):
        color = COLORS[index % len(COLORS)]
        value_y = margin_top + plot_height - ((value - min_value) / value_range) * plot_height
        bar_height = abs(zero_y - value_y)
        y = min(zero_y, value_y)
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2}" y="{y - 8 if value >= 0 else y + bar_height + 16}" text-anchor="middle" font-size="12" fill="#111827">{value:.2f}</text>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2}" y="{margin_top + plot_height + 35}" text-anchor="end" font-size="12" fill="#111827" transform="rotate(-35 {x + bar_width / 2} {margin_top + plot_height + 35})">{label}</text>'
        )
        x += bar_width + spacing

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")

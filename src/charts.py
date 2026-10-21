"""Apache ECharts helpers for the notebook.

Charts are plain ECharts option dictionaries rendered inside an iframe, which
keeps scripts isolated from the notebook front end. The ECharts library is
loaded from a CDN, so an internet connection is needed when a chart is shown.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

ECHARTS_JS = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"

BLUE = "#3b82f6"
RED = "#ef4444"
FONT = "Inter, Segoe UI, system-ui, sans-serif"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: #ffffff; font-family: {font}; }}
  #bar {{ display: {bar_display}; padding: 6px 12px 0; text-align: right; }}
  #bar button {{ border: 1px solid #d1d5db; background: #f9fafb; border-radius: 6px;
                 padding: 4px 10px; margin-left: 6px; cursor: pointer; font-size: 12px; }}
  #bar button.active {{ background: #1f2937; color: #ffffff; border-color: #1f2937; }}
  #chart {{ width: 100%; height: {height}px; }}
</style>
<script src="{js}"></script></head>
<body>
<div id="bar">
  <button id="lin" class="active">Linear axes</button>
  <button id="log">Log axes</button>
</div>
<div id="chart"></div>
<script>
  const chart = echarts.init(document.getElementById('chart'), null, {{renderer: 'canvas'}});
  const opt = {option};
  chart.setOption(opt);
  window.addEventListener('resize', () => chart.resize());
  function setAxes(type) {{
    opt.xAxis.type = type;
    opt.yAxis.type = type;
    chart.setOption(opt, true);
    document.getElementById('lin').className = type === 'value' ? 'active' : '';
    document.getElementById('log').className = type === 'log' ? 'active' : '';
  }}
  document.getElementById('lin').onclick = () => setAxes('value');
  document.getElementById('log').onclick = () => setAxes('log');
</script></body></html>"""


def _dumps(option: dict) -> str:
    """Turn an option dict into JavaScript source text for the page."""
    text = json.dumps(option, allow_nan=False, separators=(",", ":"))
    # JSON cannot hold functions, so tooltip formatters are written as strings that
    # start with "function(". Strip their quotes so the browser sees real JavaScript.
    text = re.sub(r'"(function\([^"]*\})"', r"\1", text)
    # "</" would end the inline script early, so escape it.
    return text.replace("</", "<\\/")


def page_html(option: dict, height: int = 560, axis_toggle: bool = False) -> str:
    """Return a complete HTML document that renders the ECharts option."""
    return _PAGE.format(
        font=FONT,
        height=height,
        js=ECHARTS_JS,
        option=_dumps(option),
        bar_display="block" if axis_toggle else "none",
    )


def save_html(option: dict, path: str | Path, height: int = 560, axis_toggle: bool = False) -> Path:
    """Write a chart to a standalone HTML file."""
    path = Path(path)
    path.write_text(page_html(option, height, axis_toggle), encoding="utf-8")
    return path


class _Chart:
    """Notebook display object that renders as an iframe."""

    def __init__(self, markup: str):
        self._markup = markup

    def _repr_html_(self) -> str:
        return self._markup


def show(option: dict, height: int = 560, axis_toggle: bool = False) -> _Chart:
    """Display a chart inside a notebook."""
    srcdoc = html.escape(page_html(option, height, axis_toggle), quote=True)
    # Room for the Linear/Log button bar above the chart.
    extra = 44 if axis_toggle else 0
    return _Chart(
        f'<iframe srcdoc="{srcdoc}" width="100%" height="{height + extra + 8}" '
        'style="border:0" sandbox="allow-scripts"></iframe>'
    )


def _title(text: str, subtext: str) -> dict:
    return {
        "text": text,
        "subtext": subtext,
        "left": 16,
        "top": 8,
        "textStyle": {"fontSize": 18, "fontWeight": 600, "color": "#111827"},
        "subtextStyle": {"fontSize": 12, "color": "#6b7280"},
    }


def _axis(name: str) -> dict:
    return {
        "type": "value",
        "name": name,
        "nameLocation": "middle",
        "nameGap": 34,
        "nameTextStyle": {"fontSize": 13, "color": "#374151"},
        "axisLine": {"show": False},
        "axisTick": {"show": False},
        "axisLabel": {"color": "#6b7280"},
        "splitLine": {"lineStyle": {"color": "#e5e7eb", "type": "dashed"}},
    }


def _base(title: dict, x_label: str, y_label: str) -> dict:
    """Shared layout: axes, zoom controls and toolbox used by every chart."""
    y_axis = _axis(y_label)
    y_axis["nameGap"] = 62
    return {
        "animation": False,
        "title": title,
        "grid": {"left": 90, "right": 28, "top": 84, "bottom": 96},
        "xAxis": _axis(x_label),
        "yAxis": y_axis,
        "dataZoom": [
            {"type": "inside", "xAxisIndex": 0, "filterMode": "none"},
            {"type": "inside", "yAxisIndex": 0, "filterMode": "none"},
            {"type": "slider", "xAxisIndex": 0, "height": 18, "bottom": 18, "filterMode": "none"},
        ],
        "toolbox": {
            "right": 20,
            "top": 8,
            "feature": {
                "dataZoom": {"yAxisIndex": "none", "title": {"zoom": "Zoom", "back": "Undo"}},
                "restore": {"title": "Reset"},
                "saveAsImage": {"title": "Save PNG", "pixelRatio": 2},
            },
        },
    }


def scatter_option(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[Sequence],
    *,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
) -> dict:
    """Interactive scatter of all points; `labels` rows are (road, km_from, km_to)."""
    rows = [
        [round(float(a), 4), round(float(b), 3), str(r), k0, k1]
        for a, b, (r, k0, k1) in zip(x, y, labels)
    ]
    option = _base(_title(title, subtitle), x_label, y_label)
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "borderColor": "#e5e7eb",
        "textStyle": {"color": "#111827", "fontSize": 12},
        "formatter": (
            "function(p){var d=p.data;return '<b>'+d[2]+'</b><br/>km '+d[3]+' to '+d[4]"
            "+'<br/>Length: '+d[0]+' km<br/>GIS length: '+d[1].toLocaleString();}"
        ),
    }
    option["series"] = [
        {
            "name": "Road segments",
            "type": "scatter",
            "data": rows,
            "large": True,  # ECharts' fast path for tens of thousands of points
            "largeThreshold": 1000,
            "symbolSize": 5,
            "itemStyle": {"color": BLUE, "opacity": 0.35},
            "emphasis": {"itemStyle": {"color": RED, "opacity": 1}},
        }
    ]
    return option


def residual_option(
    x: Sequence[float],
    residuals: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
) -> dict:
    """Scatter of residuals (actual minus predicted) with a zero reference line."""
    points = np.column_stack([np.round(x, 4), np.round(residuals, 3)]).tolist()
    option = _base(_title(title, subtitle), x_label, y_label)
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "borderColor": "#e5e7eb",
        "textStyle": {"color": "#111827", "fontSize": 12},
        "formatter": (
            "function(p){var d=p.data;return 'Length: '+d[0]"
            "+' km<br/>Residual: '+d[1].toLocaleString();}"
        ),
    }
    option["series"] = [
        {
            "name": "Residuals",
            "type": "scatter",
            "data": points,
            "large": True,
            "largeThreshold": 1000,
            "symbolSize": 5,
            "itemStyle": {"color": BLUE, "opacity": 0.4},
            "markLine": {
                "silent": True,
                "symbol": "none",
                "label": {"show": False},
                "lineStyle": {"color": RED, "width": 2},
                "data": [{"yAxis": 0}],
            },
        }
    ]
    return option


def actual_predicted_option(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
) -> dict:
    """Scatter of actual against predicted values with a perfect-prediction line."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    points = np.column_stack(
        [
            np.round(actual_values, 3),
            np.round(predicted_values, 3),
            np.round(actual_values - predicted_values, 3),
        ]
    ).tolist()
    lower = float(min(actual_values.min(), predicted_values.min()))
    upper = float(max(actual_values.max(), predicted_values.max()))

    option = _base(_title(title, subtitle), x_label, y_label)
    option["xAxis"]["min"] = lower
    option["xAxis"]["max"] = upper
    option["yAxis"]["min"] = lower
    option["yAxis"]["max"] = upper
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "borderColor": "#e5e7eb",
        "textStyle": {"color": "#111827", "fontSize": 12},
        "formatter": (
            "function(p){var d=p.data;return 'Actual: '+d[0].toLocaleString()"
            "+'<br/>Predicted: '+d[1].toLocaleString()"
            "+'<br/>Difference (actual - predicted): '+d[2].toLocaleString();}"
        ),
    }
    option["series"] = [
        {
            "name": "Test segments",
            "type": "scatter",
            "data": points,
            "large": True,
            "largeThreshold": 1000,
            "symbolSize": 6,
            "itemStyle": {"color": BLUE, "opacity": 0.45},
            "markLine": {
                "silent": True,
                "symbol": "none",
                "label": {
                    "show": True,
                    "formatter": "Perfect prediction",
                    "position": "insideEndTop",
                },
                "lineStyle": {"color": RED, "width": 2, "type": "dashed"},
                "data": [[{"coord": [lower, lower]}, {"coord": [upper, upper]}]],
            },
        }
    ]
    return option


def fit_option(
    x: Sequence[float],
    y: Sequence[float],
    x_line: Sequence[float],
    y_line: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    equation: str,
) -> dict:
    """Scatter plus fitted line, with the equation shown as the line's label."""
    points = np.column_stack([np.round(x, 4), np.round(y, 3)]).tolist()
    line = np.column_stack([np.round(x_line, 4), np.round(y_line, 3)]).tolist()
    option = _base(_title(title, subtitle), x_label, y_label)
    # The fit line is negative near x = 0 (negative intercept); start at 0 to avoid empty space.
    option["yAxis"]["min"] = 0
    option["legend"] = {
        "right": 150,
        "top": 12,
        "textStyle": {"color": "#374151"},
        "itemGap": 18,
    }
    option["tooltip"] = {
        "trigger": "item",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "borderColor": "#e5e7eb",
        "textStyle": {"color": "#111827", "fontSize": 12},
        "formatter": (
            "function(p){var d=p.data;return p.seriesName+'<br/>Length: '+d[0]"
            "+' km<br/>GIS length: '+d[1].toLocaleString();}"
        ),
    }
    option["series"] = [
        {
            "name": "Road segments",
            "type": "scatter",
            "data": points,
            "large": True,
            "largeThreshold": 1000,
            "symbolSize": 5,
            "itemStyle": {"color": BLUE, "opacity": 0.25},
        },
        {
            "name": equation,
            "type": "line",
            "data": line,
            "showSymbol": False,
            "silent": True,
            "lineStyle": {"color": RED, "width": 3},
            "itemStyle": {"color": RED},
            "z": 10,
        },
    ]
    return option

"""Utilities for creating stacked 3D disaggregation bar charts with Plotly.

Workflow overview
-----------------
The module is intended to be used in three steps:

1. Prepare the input data with :func:`prepare_disagg`.
   This validates the requested columns, aggregates duplicate bins, optionally
   removes zero-valued rows, and optionally normalises the plotted values.

2. Create either a single chart with :func:`make_disagg_figure` or multiple
   charts with :func:`make_disagg_grid`.
   A chart can be based on any pair of columns for the horizontal axes and an
   optional third categorical column for stacked subdivisions.

3. Export the result with :func:`save_figure`.
   Interactive HTML and common static image formats are supported.

The plotting approach uses Plotly ``Mesh3d`` traces to draw cuboids for the
bars and ``Scatter3d`` line traces to draw the heavy black outlines required by
publication-style figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# -----------------------------
# Geometry helpers (cuboids)
# -----------------------------

def _cuboid_vertices(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    """Return the 8 vertices of a cuboid.

    Parameters are the lower and upper bounds in x, y, and z. The returned
    array has shape ``(8, 3)`` and is ordered to match the triangulation and
    edge path definitions used elsewhere in the module.
    """
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


_CUBOID_TRIS = np.array(
    [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 6, 2], [1, 5, 6],
    ],
    dtype=int,
)


_CUBOID_EDGE_PATHS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _new_mesh_acc() -> Dict[str, list]:
    """Create an accumulator for ``Mesh3d`` vertex and triangle arrays."""
    return {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}


def _new_line_acc() -> Dict[str, list]:
    """Create an accumulator for cuboid outline line segments."""
    return {"x": [], "y": [], "z": []}


def _append_cuboid(mesh_acc: Dict[str, list], bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Append one cuboid to a mesh accumulator and return its vertices.

    Parameters
    ----------
    mesh_acc:
        Mutable accumulator built by :func:`_new_mesh_acc`.
    bounds:
        ``(x0, x1, y0, y1, z0, z1)`` bounds for the cuboid.

    Returns
    -------
    np.ndarray
        The generated cuboid vertices. The caller uses these vertices to add a
        separate outline trace.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    v = _cuboid_vertices(x0, x1, y0, y1, z0, z1)
    base = len(mesh_acc["x"])

    mesh_acc["x"].extend(v[:, 0].tolist())
    mesh_acc["y"].extend(v[:, 1].tolist())
    mesh_acc["z"].extend(v[:, 2].tolist())

    tris = _CUBOID_TRIS + base
    mesh_acc["i"].extend(tris[:, 0].tolist())
    mesh_acc["j"].extend(tris[:, 1].tolist())
    mesh_acc["k"].extend(tris[:, 2].tolist())
    return v


def _append_cuboid_edges(line_acc: Dict[str, list], vertices: np.ndarray) -> None:
    """Append the 12 outline edges for a cuboid to a line accumulator."""
    for a, b in _CUBOID_EDGE_PATHS:
        line_acc["x"].extend([float(vertices[a, 0]), float(vertices[b, 0]), None])
        line_acc["y"].extend([float(vertices[a, 1]), float(vertices[b, 1]), None])
        line_acc["z"].extend([float(vertices[a, 2]), float(vertices[b, 2]), None])


# -----------------------------
# Styling
# -----------------------------

@dataclass
class AxisTitleStyle:
    """Style definition for one axis title.

    Attributes
    ----------
    text:
        Axis title string shown in the figure.
    font_size:
        Title font size in points.
    font_color:
        Title font colour accepted by Plotly.
    font_family:
        Title font family name.
    """

    text: str
    font_size: int = 14
    font_color: str = "black"
    font_family: str = "Arial"


@dataclass
class Disagg3DStyle:
    """Plot-wide style settings for a disaggregation chart.

    This dataclass groups the parameters that are usually shared across many
    plots, such as axis styling, camera position, figure dimensions, bar
    spacing, and grid appearance.
    """

    template: str = "plotly_white"
    bar_opacity: float = 1.0
    bar_gap: float = 0.08
    outline_color: str = "black"
    outline_width: float = 4.0

    z_scale: float = 1.0
    z_exaggeration: float = 8.0

    x_title: AxisTitleStyle = field(default_factory=lambda: AxisTitleStyle("Magnitude, M"))
    y_title: AxisTitleStyle = field(default_factory=lambda: AxisTitleStyle("Distance, R (km)"))
    z_title: AxisTitleStyle = field(default_factory=lambda: AxisTitleStyle("Contribution"))
    tick_font_size: int = 12
    tick_font_color: str = "black"
    tick_font_family: str = "Arial"
    round_xaxis_ticks: int = 0
    round_yaxis_ticks: int = 0

    camera_eye: Tuple[float, float, float] = (1.45, 1.25, 0.95)
    projection: str = "perspective"

    show_grid: bool = True
    show_axes_background: bool = True
    axis_color: str = "black"
    axis_background_color: str = "rgba(240,240,240,0.35)"
    grid_color: str = "grey"
    grid_width: float = 2.0
    grid_dash: str = "dashdot"

    width: int = 1100
    height: int = 650


def _coerce_axis_title(value: AxisTitleStyle | str) -> AxisTitleStyle:
    """Return an :class:`AxisTitleStyle` from either a string or style object."""
    if isinstance(value, AxisTitleStyle):
        return value
    return AxisTitleStyle(text=str(value))


# -----------------------------
# Data preparation
# -----------------------------

def prepare_disagg(
    df: pd.DataFrame,
    x_col: str = "Mag",
    y_col: str = "Dist",
    value_col: str = "P(m|X>x)",
    stack_col: Optional[str] = "Eps",
    *,
    drop_zeros: bool = True,
    zero_tol: float = 0.0,
    normalize: bool = False,
    normalize_to: float = 1.0,
) -> pd.DataFrame:
    """Validate and aggregate the input data for plotting.

    Parameters
    ----------
    df:
        Source table containing the plotting columns.
    x_col, y_col:
        Columns used for the two horizontal axes.
    value_col:
        Numeric column used as the bar height.
    stack_col:
        Optional categorical column used to subdivide each bar into stacked
        segments. Pass ``None`` to produce unstacked bars.
    drop_zeros:
        If ``True``, remove rows with values at or below ``zero_tol``.
    zero_tol:
        Absolute tolerance used when dropping zero-valued rows.
    normalize:
        If ``True``, scale the full table so that the sum of ``value_col`` is
        ``normalize_to``.
    normalize_to:
        Total used when ``normalize=True``.

    Returns
    -------
    pd.DataFrame
        A grouped table ready for plotting.

    Notes
    -----
    Use this function first in the workflow. It ensures that duplicate bin
    combinations are summed before the plotting code constructs cuboids.
    """
    cols = [x_col, y_col] + ([stack_col] if stack_col else []) + [value_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[cols].copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)

    if drop_zeros:
        out = out.loc[out[value_col].abs() > float(zero_tol)].copy()

    group_cols = [x_col, y_col] + ([stack_col] if stack_col else [])
    out = out.groupby(group_cols, as_index=False)[value_col].sum()

    if normalize:
        total = float(out[value_col].sum())
        if total > 0:
            out[value_col] = out[value_col] / total * float(normalize_to)

    return out


# -----------------------------
# Plotting
# -----------------------------

def make_disagg_figure(
    df: pd.DataFrame,
    *,
    x_col: str = "Mag",
    y_col: str = "Dist",
    value_col: str = "P(m|X>x)",
    z_col: Optional[str] = None,
    stack_col: Optional[str] = "Eps",
    stack_order: Optional[Sequence] = None,
    stack_label_map: Optional[Dict] = None,
    cmap: Optional[Dict] = None,
    legend_labels: Optional[Dict] = None,
    style: Optional[Disagg3DStyle] = None,
    title: Optional[str] = None,
    color_sequence: Optional[Sequence[str]] = None,
    coord_mode: str = "index",
    annotation: Optional[str] = None,
    annotation_x: float = 0.01,
    annotation_y: float = 0.99,
) -> go.Figure:
    """Create one stacked or unstacked 3D disaggregation figure.

    Parameters
    ----------
    df:
        Prepared plotting table. This is usually the output from
        :func:`prepare_disagg`.
    x_col, y_col:
        Columns used for the horizontal axes.
    value_col:
        Numeric column used for bar heights.
    z_col:
        Alias for ``stack_col``. This name is included because it matches the
        requirement wording from the project brief.
    stack_col:
        Categorical column used to subdivide each bar into stacked segments.
        Pass ``None`` to draw one bar per ``(x_col, y_col)`` bin.
    stack_order:
        Optional explicit ordering of stack categories in the bar and legend.
    stack_label_map:
        Optional mapping from category value to legend label.
    cmap:
        Optional mapping from category value to a discrete colour.
    legend_labels:
        Alias for ``stack_label_map``.
    style:
        A :class:`Disagg3DStyle` instance controlling appearance.
    title:
        Figure title.
    color_sequence:
        Fallback colour sequence used for categories not covered by ``cmap``.
    coord_mode:
        ``"index"`` uses evenly spaced square-like bins regardless of the raw
        numeric spacing in the data. ``"data"`` uses the data values directly.
    annotation:
        Optional figure-level annotation such as ``"(A)"``.
    annotation_x, annotation_y:
        Paper coordinates for the annotation.

    Returns
    -------
    go.Figure
        A Plotly figure ready for display or export.

    Workflow notes
    --------------
    The function groups data by x/y bin, builds one cuboid per stack segment,
    merges cuboids that share a colour into a single ``Mesh3d`` trace, and adds
    a second ``Scatter3d`` trace containing thick black outlines.
    """
    if style is None:
        style = Disagg3DStyle()

    style.x_title = _coerce_axis_title(style.x_title)
    style.y_title = _coerce_axis_title(style.y_title)
    style.z_title = _coerce_axis_title(style.z_title)

    if z_col is not None:
        stack_col = z_col
    if legend_labels is not None and stack_label_map is None:
        stack_label_map = legend_labels

    if coord_mode not in {"index", "data"}:
        raise ValueError("coord_mode must be 'index' or 'data'")

    if color_sequence is None:
        color_sequence = px.colors.qualitative.Set2

    required_cols = [x_col, y_col, value_col] + ([stack_col] if stack_col else [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x_vals = np.round(np.array(sorted(pd.unique(df[x_col]))), style.round_xaxis_ticks)
    y_vals = np.round(np.array(sorted(pd.unique(df[y_col]))), style.round_yaxis_ticks)

    if coord_mode == "index":
        x_pos = {v: i for i, v in enumerate(x_vals)}
        y_pos = {v: j for j, v in enumerate(y_vals)}
        x_plot = df[x_col].map(x_pos).astype(float)
        y_plot = df[y_col].map(y_pos).astype(float)
        dx = dy = 1.0
        x_tickvals = list(range(len(x_vals)))
        y_tickvals = list(range(len(y_vals)))
        x_ticktext = [str(v) for v in x_vals]
        y_ticktext = [str(v) for v in y_vals]
    else:
        x_plot = df[x_col].astype(float)
        y_plot = df[y_col].astype(float)
        dx = float(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 1.0
        dy = float(np.median(np.diff(y_vals))) if len(y_vals) > 1 else 1.0
        x_tickvals = y_tickvals = None
        x_ticktext = y_ticktext = None

    dx_eff = dx * (1.0 - float(style.bar_gap))
    dy_eff = dy * (1.0 - float(style.bar_gap))

    if stack_col:
        cats = list(pd.unique(df[stack_col]))
        if stack_order is not None:
            present = set(cats)
            cats = [c for c in stack_order if c in present] + [c for c in cats if c not in set(stack_order)]
        else:
            try:
                cats = sorted(cats)
            except Exception:
                pass
    else:
        cats = [None]

    meshes = {}
    line_accs = {}
    for idx, c in enumerate(cats):
        color = cmap.get(c) if cmap and c in cmap else color_sequence[idx % len(color_sequence)]
        meshes[c] = {"acc": _new_mesh_acc(), "color": color}
        line_accs[c] = _new_line_acc()

    df2 = df.copy()
    df2["__xplot__"] = x_plot.values
    df2["__yplot__"] = y_plot.values

    bin_cols = ["__xplot__", "__yplot__", x_col, y_col]
    for (xp, yp, _xv, _yv), g in df2.groupby(bin_cols, sort=False):
        z0 = 0.0
        if stack_col:
            g = g.set_index(stack_col)
            for c in cats:
                if c not in g.index:
                    continue
                val = float(g.loc[c, value_col])
                if np.isscalar(val) is False:
                    val = float(np.sum(val))
                if val <= 0:
                    continue

                z1 = z0 + val * float(style.z_scale)
                z0g = z0 * float(style.z_exaggeration)
                z1g = z1 * float(style.z_exaggeration)
                x0, x1 = float(xp) - dx_eff / 2, float(xp) + dx_eff / 2
                y0b, y1b = float(yp) - dy_eff / 2, float(yp) + dy_eff / 2
                verts = _append_cuboid(meshes[c]["acc"], (x0, x1, y0b, y1b, z0g, z1g))
                _append_cuboid_edges(line_accs[c], verts)
                z0 = z1
        else:
            val = float(g[value_col].sum())
            if val <= 0:
                continue
            z1 = z0 + val * float(style.z_scale)
            z0g = z0 * float(style.z_exaggeration)
            z1g = z1 * float(style.z_exaggeration)
            x0, x1 = float(xp) - dx_eff / 2, float(xp) + dx_eff / 2
            y0b, y1b = float(yp) - dy_eff / 2, float(yp) + dy_eff / 2
            verts = _append_cuboid(meshes[None]["acc"], (x0, x1, y0b, y1b, z0g, z1g))
            _append_cuboid_edges(line_accs[None], verts)

    fig = go.Figure()
    for c in cats:
        acc = meshes[c]["acc"]
        if len(acc["x"]) == 0:
            continue
        name = "Total" if c is None else (stack_label_map.get(c, c) if stack_label_map else c)
        fig.add_trace(
            go.Mesh3d(
                x=acc["x"], y=acc["y"], z=acc["z"],
                i=acc["i"], j=acc["j"], k=acc["k"],
                color=meshes[c]["color"],
                opacity=float(style.bar_opacity),
                flatshading=True,
                name=str(name),
                legendgroup=str(name),
                showlegend=True,
                hoverinfo="skip",
                showscale=False,
                lighting=dict(ambient=0.35, diffuse=0.8, specular=0.2, roughness=0.6, fresnel=0.05),
            )
        )
        lines = line_accs[c]
        fig.add_trace(
            go.Scatter3d(
                x=lines["x"], y=lines["y"], z=lines["z"],
                mode="lines",
                line=dict(color=style.outline_color, width=style.outline_width),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if coord_mode == "index":
        x_range = (-0.75, len(x_vals) - 0.25)
        y_range = (-0.75, len(y_vals) - 0.25)
    else:
        x_range = (float(x_vals.min()) - dx, float(x_vals.max()) + dx)
        y_range = (float(y_vals.min()) - dy, float(y_vals.max()) + dy)

    z_max_display = float(df.groupby([x_col, y_col])[value_col].sum().max()) * float(style.z_scale)
    z_range = (0.0, z_max_display * float(style.z_exaggeration) * 1.05)

    nzt = 6
    z_ticks_display = np.linspace(0.0, max(z_max_display, 1e-9), nzt)
    z_tickvals = (z_ticks_display * float(style.z_exaggeration)).tolist()
    z_ticktext = [f"{v:.3g}" for v in z_ticks_display]

    common_axis = dict(
        showgrid=style.show_grid,
        gridcolor=style.grid_color,
        gridwidth=style.grid_width,
        showline=True,
        linecolor=style.axis_color,
        tickcolor=style.axis_color,
        tickfont=dict(size=style.tick_font_size, color=style.tick_font_color, family=style.tick_font_family),
        color=style.axis_color,
        zeroline=False,
        showbackground=style.show_axes_background,
        backgroundcolor=style.axis_background_color if style.show_axes_background else "rgba(0,0,0,0)",
    )

    scene = dict(
        xaxis=dict(
            **common_axis,
            title=dict(text=style.x_title.text, font=dict(size=style.x_title.font_size, color=style.x_title.font_color, family=style.x_title.font_family)),
            range=list(x_range),
            tickmode="array" if x_tickvals is not None else "auto",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
        ),
        yaxis=dict(
            **common_axis,
            title=dict(text=style.y_title.text, font=dict(size=style.y_title.font_size, color=style.y_title.font_color, family=style.y_title.font_family)),
            range=list(y_range),
            tickmode="array" if y_tickvals is not None else "auto",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
        ),
        zaxis=dict(
            **common_axis,
            title=dict(text=style.z_title.text, font=dict(size=style.z_title.font_size, color=style.z_title.font_color, family=style.z_title.font_family)),
            range=list(z_range),
            tickmode="array",
            tickvals=z_tickvals,
            ticktext=z_ticktext,
        ),
        aspectmode="manual",
        aspectratio=dict(x=1.0, y=1.0, z=0.7),
        camera=dict(eye=dict(x=style.camera_eye[0], y=style.camera_eye[1], z=style.camera_eye[2]), projection=dict(type=style.projection)),
    )

    fig.update_layout(
        template=style.template,
        title=title,
        width=style.width,
        height=style.height,
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
        legend=dict(itemsizing="constant"),
        scene=scene,
    )

    
    try:
        fig.update_scenes(xaxis_griddash=style.grid_dash, yaxis_griddash=style.grid_dash, zaxis_griddash=style.grid_dash)
    except Exception:
        pass

    if annotation:
        fig.add_annotation(
            text=annotation,
            x=annotation_x,
            y=annotation_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="black", family="Arial"),
        )

    return fig


def make_disagg_grid(
    figures: Sequence[go.Figure],
    *,
    ncols: int = 2,
    titles: Optional[Sequence[str]] = None,
    width: int = 1200,
    height: int = 700,
    annotations: Optional[Sequence[str]] = None,
) -> go.Figure:
    """Arrange multiple 3D disaggregation figures into one subplot grid.

    Parameters
    ----------
    figures:
        Sequence of figures created by :func:`make_disagg_figure`.
    ncols:
        Number of columns in the subplot grid.
    titles:
        Optional subplot titles.
    width, height:
        Final output figure size in pixels.
    annotations:
        Optional per-panel labels such as ``["(A)", "(B)"]``.

    Returns
    -------
    go.Figure
        One Plotly figure containing all scenes.

    Notes
    -----
    The function copies traces and scene settings from each input figure. This
    keeps the workflow simple: build each panel independently, then assemble the
    final publication layout at the end.
    """
    n = len(figures)
    nrows = int(np.ceil(n / ncols))
    subplot_titles = list(titles) if titles is not None else [f"Plot {i + 1}" for i in range(n)]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for idx, f in enumerate(figures):
        r = idx // ncols + 1
        c = idx % ncols + 1
        for tr in f.data:
            fig.add_trace(tr, row=r, col=c)
        scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
        fig.layout[scene_name] = f.layout.scene

    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
    )

    if annotations:
        for idx, label in enumerate(annotations[:n]):
            r = idx // ncols
            c = idx % ncols
            x_domain = ((c) / ncols, (c + 1) / ncols)
            y_domain = (1 - (r + 1) / nrows, 1 - r / nrows)
            fig.add_annotation(
                text=label,
                x=x_domain[0] + 0.01,
                y=y_domain[1] - 0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="black", family="Arial"),
            )

    return fig


def save_figure(fig: go.Figure, path: str) -> None:
    """Save a figure to HTML or to a common static image format.

    Parameters
    ----------
    fig:
        Plotly figure to save.
    path:
        Output file path. Supported suffixes are ``.html``, ``.png``, ``.jpg``,
        ``.jpeg``, ``.pdf``, ``.svg``, and ``.webp``.

    Notes
    -----
    Static image export requires the Plotly Kaleido backend to be installed in
    the Python environment.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".html":
        fig.write_html(path, include_plotlyjs="cdn")
        return

    supported = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}
    if suffix not in supported:
        raise ValueError(f"Unsupported file extension: {suffix}. Use one of {sorted(supported | {'.html'})}")
    fig.write_image(path)

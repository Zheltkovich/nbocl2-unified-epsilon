#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise crystal → graph → pretrained GNN inference (ALIGNN / GNNOpt).

We do **not** train a network on NbOI₂ bulk here. We only:
  1. Read your structure file (nboi2.xyz, film extxyz, …)
  2. Build a neighbour graph (nodes = atoms, edges = bonds within cutoff)
  3. Run **frozen** JARVIS ALIGNN or GNNOpt weights (Figshare / models/)

``ml_dgl_bootstrap.py`` is a Windows import fix for DGL — not model training.

UI ink rules: see ``dielectric_ui_contract.json`` (pale-callout tables → compare-ink-light).

MANUSCRIPT_PICKUP — GNN graph: cite ALIGNN/GNNOpt DOI in Methods; no «we proposed a GNN».
SI figure ID: ``mortazavi_dielectric_shared.MANUSCRIPT_GNN_VIZ_FIGURE_ID`` (button G 2D + axonometric + PyG directed).
Run ``python -m py_compile ml_gnn_pipeline_viz.py`` after edits (orphan ``)`` regression tracked in contract).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DEFAULT_STRUCTURE = ROOT / "nboi2.xyz"
OUT_DIR = ROOT / "generated_films_GPT_38" / "ItoCl" / "dielectric_compare"
GNN_TEACHING_PNG_DIR = ROOT / "generated_films_GPT_38"
GNNOPT_CUTOFF_ANG = 6.0
ALIGNN_CUTOFF_ANG = 8.0
# 3D axonometry sphere size (matplotlib scatter s ∝ scale²); bumped for thick ItoCl films.
GNN_3D_NODE_MARKER_SCALE = 12.8

Lang = Literal["ru", "en"]
Theme = Literal["light", "dark"]

_ELEMENT_COLOR = {"Nb": "#6366f1", "O": "#ef4444", "I": "#14b8a6", "Cl": "#22c55e", "Br": "#a855f7"}

# Strict ink: light surface → #000000 only; dark surface → #ffffff only (Modulus rule).
_INK_ON_LIGHT = "#000000"
_INK_ON_DARK = "#ffffff"
_GNN_IO_SIDE_BOX_W = 0.69  # was 0.66 — ~4% wider INPUT/OUTPUT panels
_GNN_IO_SIDE_BOX_X = 0.155  # centered for wider box
_GNN_THUMB_INSET_SHRINK = 0.05  # 5% margin inside pipeline graph thumbs

# ipywidgets.HTML: pale panels need explicit ink (see dielectric_ui_contract.json).
_GNN_PANEL_INK = f"color:{_INK_ON_LIGHT} !important;"
_GNN_PANEL_CLASS = "pale-callout compare-ink-light gnn-pipeline-panel"
_GNN_TH = (
    "text-align:left;padding:4px 6px;border:1px solid #cbd5e1;background:#e2e8f0;"
    + _GNN_PANEL_INK
)
_GNN_TD = f"padding:4px 6px;border:1px solid #e2e8f0;{_GNN_PANEL_INK}"


def _l(lang: Lang, ru: str, en: str) -> str:
    return ru if lang == "ru" else en


def _fig_ink(theme: Theme, *, on_light_surface: bool = False) -> str:
    """Matplotlib/HTML ink on known surface type."""
    if on_light_surface:
        return _INK_ON_LIGHT
    return _INK_ON_DARK if theme == "dark" else _INK_ON_LIGHT


def _draw_glass_node_2d(
    ax,
    u: float,
    v: float,
    sym: str | None,
    col: str,
    *,
    ms: float,
    fs: float,
    lw_node: float,
    clip_on: bool,
) -> None:
    """Semi-transparent sphere; optional centered element label."""
    ax.scatter(
        u,
        v,
        s=ms,
        c=col,
        alpha=0.72,
        edgecolors="white",
        linewidths=max(lw_node * 1.15, 0.5),
        zorder=2,
        clip_on=clip_on,
    )
    ax.scatter(
        u,
        v,
        s=ms * 0.20,
        c="white",
        alpha=0.42,
        edgecolors="none",
        zorder=3,
        clip_on=clip_on,
    )
    if sym:
        ax.text(
            u,
            v,
            sym,
            ha="center",
            va="center",
            fontsize=fs,
            color=_INK_ON_DARK,
            fontweight="bold",
            zorder=4,
            clip_on=clip_on,
        )


def _camera_pull_vector(elev_deg: float, azim_deg: float) -> tuple[float, float, float]:
    """Unit vector from scene toward camera (for label pull-forward in 3D)."""
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    return (
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    )


def _draw_glass_nodes_3d(
    ax,
    pos: np.ndarray,
    syms: list[str],
    *,
    ms: float,
    label_fs: float,
    ec_lw: float,
    show_labels: bool,
    elev: float,
    azim: float,
    spans: tuple[float, float, float],
) -> None:
    """Glass-like spheres; labels drawn in a second pass toward the camera."""
    pull = max(spans) * 0.028
    vx, vy, vz = _camera_pull_vector(elev, azim)
    for i, sym in enumerate(syms):
        col = _ELEMENT_COLOR.get(sym, "#64748b")
        x, y, z = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
        ax.scatter(
            x,
            y,
            z,
            s=ms,
            c=col,
            alpha=0.68,
            edgecolors="white",
            linewidths=max(ec_lw * 1.2, 0.55),
            depthshade=False,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            z,
            s=ms * 0.18,
            c="white",
            alpha=0.38,
            edgecolors="none",
            depthshade=False,
            zorder=3,
        )
    if not show_labels:
        return
    for i, sym in enumerate(syms):
        x, y, z = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
        tx = x + vx * pull
        ty = y + vy * pull
        tz = z + vz * pull
        ax.text(
            tx,
            ty,
            tz,
            sym,
            color=_INK_ON_DARK,
            fontsize=label_fs,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )


def _count_undirected_edges(src: np.ndarray, dst: np.ndarray) -> int:
    return int(sum(1 for i, j in zip(src, dst) if int(i) < int(j)))


def _try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _load_ml_inference_cache(structure: Path) -> dict[str, Any]:
    """Cached JSON from widget buttons A / A′ / B for the same structure file."""
    stem = structure.stem.replace(".", "_")
    struct_name = structure.name
    buckets: dict[str, list[Path]] = {
        "alignn_optb88": [
            OUT_DIR / f"alignn_eg_{stem}.json",
            OUT_DIR / f"alignn_eg_{stem}_bulk.json",
            OUT_DIR / "alignn_eg_nboi2_bulk.json",
        ],
        "alignn_mbj": [
            OUT_DIR / f"alignn_eg_{stem}_mbj.json",
            OUT_DIR / f"alignn_eg_{stem}_bulk_mbj.json",
            OUT_DIR / "alignn_eg_nboi2_bulk_mbj.json",
        ],
        "gnnopt": [
            OUT_DIR / f"gnnopt_spectrum_{stem}.json",
            OUT_DIR / f"gnnopt_spectrum_{stem}_bulk.json",
            OUT_DIR / "gnnopt_spectrum_nboi2_bulk.json",
        ],
    }
    out: dict[str, Any] = {}
    for key, paths in buckets.items():
        for path in paths:
            data = _try_load_json(path)
            if not data:
                continue
            saved_struct = Path(str(data.get("structure", ""))).name
            if saved_struct and saved_struct != struct_name and stem not in path.stem:
                continue
            out[key] = data
            break
    return out


def alignn_graph_stats(structure: Path, *, cutoff: float = 8.0, max_neighbors: int = 12) -> dict[str, Any]:
    from ml_dgl_bootstrap import ensure_dgl_importable
    from ml_eg_bandgap_runner import _jarvis_atoms_from_file

    ensure_dgl_importable()
    from alignn.graphs import Graph

    atoms = _jarvis_atoms_from_file(structure)
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=cutoff, max_neighbors=max_neighbors)
    feat_dim = int(g.ndata["atom_features"].shape[1]) if "atom_features" in g.ndata else None
    return {
        "backend": "DGL (ALIGNN)",
        "n_atoms": int(g.num_nodes()),
        "n_edges_atom_graph": int(g.num_edges()),
        "n_nodes_line_graph": int(lg.num_nodes()),
        "n_edges_line_graph": int(lg.num_edges()),
        "cutoff_ang": cutoff,
        "max_neighbors": max_neighbors,
        "atom_feature_dim": feat_dim,
        "weights": "jv_optb88vdw_bandgap_alignn / jv_mbj_bandgap_alignn (Figshare, frozen)",
        "output": "scalar Eg (eV)",
        "training": "none — inference only",
    }


def gnnopt_graph_stats(structure: Path, *, cutoff: float = GNNOPT_CUTOFF_ANG) -> dict[str, Any]:
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms = read(str(structure))
    src_d, dst_d, _ = neighbor_list("ijS", a=atoms, cutoff=cutoff, self_interaction=True)
    src_u, dst_u, _ = neighbor_list("ijS", a=atoms, cutoff=cutoff, self_interaction=False)
    n_directed = int(len(src_d))
    n_undirected = _count_undirected_edges(src_u, dst_u)
    return {
        "backend": "PyTorch Geometric (GNNOpt)",
        "n_atoms": len(atoms),
        "n_edges": n_directed,
        "n_edges_directed": n_directed,
        "n_edges_undirected": n_undirected,
        "cutoff_ang": cutoff,
        "node_features": "Z one-hot + mass/dipole/radius (Mendeleev)",
        "edge_features": "edge_vec + periodic shift",
        "weights": "models/gnnopt/model_eps1_240406.torch + eps2 (frozen)",
        "output": "Tr(Re ε(ω)), Tr(Im ε(ω)) — 251 pt, 0–50 eV",
        "training": "none — inference only",
    }


def build_pipeline_summary(structure: Path | None = None) -> dict[str, Any]:
    structure = structure or DEFAULT_STRUCTURE
    out = {
        "structure": str(structure.resolve()),
        "alignn": alignn_graph_stats(structure),
        "gnnopt": gnnopt_graph_stats(structure),
        "dgl_bootstrap_note": (
            "ml_dgl_bootstrap.py stubs GraphBolt on Windows so DGL imports; "
            "does not change graph topology or weights."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = structure.stem.replace(".", "_")
    path = OUT_DIR / f"ml_gnn_graph_summary_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["saved_json"] = str(path)
    return out


def gnn_graph_teaser_html(lang: Lang = "ru") -> str:
    if lang == "en":
        return (
            "<p style='margin:10px 0 0;font-size:12px;color:#000000'>"
            "<b>GNN graph (button G):</b> shows how <code>nboi2.xyz</code> becomes a neighbour graph "
            "for ALIGNN/GNNOpt — <b>not</b> training a new network. "
            "<b>Full step-by-step (human language):</b> open guide chapter <b>§2b</b> in the accordion above."
            "</p>"
        )
    return (
        "<p style='margin:10px 0 0;font-size:12px;color:#000000'>"
        "<b>Граф GNN (кнопка G):</b> как <code>nboi2.xyz</code> превращается в граф соседства для ALIGNN/GNNOpt — "
        "<b>не</b> обучение новой сети. "
        "<b>Подробный алгоритм простым языком:</b> глава <b>§2b</b> в аккордеоне «§0–§9» выше."
        "</p>"
    )


def pyg_directed_edges_schematic_html(lang: Lang = "ru") -> str:
    """Inline SVG: one undirected bond vs two PyG directed edges."""
    ink = "color:#000000 !important;"
    if lang == "en":
        title = "One chemical bond → two directed edges in PyG (GNNOpt)"
        left = "Drawing (button G)"
        mid = "PyG edge_index"
        note = (
            "Distance Nb–O &lt; cutoff ⇒ <b>one</b> grey line on our 2D plot, but GNNOpt stores "
            "<b>two</b> directed edges: message i→j and j→i (MPNN passes features both ways). "
            "Table in button G: «N unique (plot) / 2N directed (inference)». "
            "<b>Not a Markov chain</b> — no P(s′|s); these are MPNN channels in GNNOpt."
        )
        e0, e1 = "edge 0→1", "edge 1→0"
        bond_lbl = "1 bond"
    else:
        title = "Одна химическая связь → два направленных ребра в PyG (GNNOpt)"
        left = "Рисунок (кнопка G)"
        mid = "PyG edge_index"
        note = (
            "Расстояние Nb–O &lt; cutoff ⇒ на нашем 2D-рисунке <b>одна</b> серая линия, "
            "а GNNOpt в памяти хранит <b>два</b> направленных ребра: сообщение i→j и j→i "
            "(MPNN передаёт признаки в обе стороны). "
            "В таблице кнопки G: «N уник. (рисунок) / 2N directed (inference)». "
            "<b>Не цепь Маркова</b> — нет P(s′|s); это каналы MPNN в GNNOpt."
        )
        e0, e1 = "ребро 0→1", "ребро 1→0"
        bond_lbl = "1 связь"
    return f"""
<h4 style="margin:16px 0 8px 0;color:#000000 !important">{title}</h4>
<div class="pale-callout compare-ink-light" style="margin:0 0 10px;padding:10px 12px;border:1px solid #6366f1;border-radius:8px;background:#eef2ff;{ink}">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 200" style="width:100%;max-width:720px;height:auto;display:block;margin:0 auto 8px">
  <text x="120" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#000000">{left}</text>
  <text x="360" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#000000">{mid}</text>
  <text x="600" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#000000">edge_index</text>
  <circle cx="60" cy="110" r="22" fill="#6366f1" stroke="#312e81" stroke-width="2"/>
  <text x="60" y="116" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Nb</text>
  <text x="60" y="150" text-anchor="middle" font-size="11" fill="#000000">i = 0</text>
  <circle cx="180" cy="110" r="22" fill="#ef4444" stroke="#991b1b" stroke-width="2"/>
  <text x="180" y="116" text-anchor="middle" font-size="14" font-weight="bold" fill="white">O</text>
  <text x="180" y="150" text-anchor="middle" font-size="11" fill="#000000">j = 1</text>
  <line x1="82" y1="110" x2="158" y2="110" stroke="#64748b" stroke-width="4" stroke-linecap="round"/>
  <text x="120" y="95" text-anchor="middle" font-size="10" fill="#000000">{bond_lbl}</text>
  <circle cx="300" cy="110" r="22" fill="#6366f1" stroke="#312e81" stroke-width="2"/>
  <text x="300" y="116" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Nb</text>
  <text x="300" y="150" text-anchor="middle" font-size="11" fill="#000000">i = 0</text>
  <circle cx="420" cy="110" r="22" fill="#ef4444" stroke="#991b1b" stroke-width="2"/>
  <text x="420" y="116" text-anchor="middle" font-size="14" font-weight="bold" fill="white">O</text>
  <text x="420" y="150" text-anchor="middle" font-size="11" fill="#000000">j = 1</text>
  <defs>
    <marker id="arrB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0284c7"/></marker>
    <marker id="arrR" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/></marker>
  </defs>
  <path d="M 322 92 Q 360 55 398 92" fill="none" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrB)"/>
  <text x="360" y="58" text-anchor="middle" font-size="9" fill="#0284c7">{e0}</text>
  <path d="M 398 128 Q 360 165 322 128" fill="none" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrR)"/>
  <text x="360" y="178" text-anchor="middle" font-size="9" fill="#dc2626">{e1}</text>
  <rect x="510" y="70" width="180" height="88" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="600" y="95" text-anchor="middle" font-family="monospace" font-size="12" fill="#000000">src = [0, 1]</text>
  <text x="600" y="118" text-anchor="middle" font-family="monospace" font-size="12" fill="#000000">dst = [1, 0]</text>
  <text x="600" y="145" text-anchor="middle" font-size="10" fill="#000000">2 directed edges</text>
</svg>
<p style="margin:0;font-size:11.5px;line-height:1.6;{ink}">{note}</p>
<p style="margin:8px 0 0;font-size:10px;line-height:1.5;{ink}"><b>MANUSCRIPT_PICKUP — SI PNG:</b>
<code>generated_films_GPT_38/gnn_pyg_directed_edges_schematic_{lang}_light.png</code>
(run <code>python ml_gnn_pipeline_viz.py --export-teaching</code>).</p>
</div>"""


def _pyg_directed_edge_arrows(
    x0: float,
    x1: float,
    y_c: float,
    r: float,
    *,
    attach_deg: float = 50.0,
    bow: float = 0.20,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
]:
    """Mirror Bézier anchors: inner upper arc 0→1, inner lower arc 1→0 (same chord length)."""
    mid = (x0 + x1) / 2
    top = (
        _circle_anchor(x0, y_c, r, attach_deg),
        _circle_anchor(x1, y_c, r, 180 - attach_deg),
        (mid, y_c + bow),
    )
    bot = (
        _circle_anchor(x1, y_c, r, -(180 - attach_deg)),
        _circle_anchor(x0, y_c, r, -attach_deg),
        (mid, y_c - bow),
    )
    return top, bot


def plot_pyg_directed_edges_schematic(
    *,
    lang: Lang = "ru",
    theme: Theme = "light",
    n_undirected: int | None = None,
    n_directed: int | None = None,
) -> plt.Figure:
    """Large teaching figure: undirected bond vs PyG directed edge_index (3 panels)."""
    c = _theme_colors(theme)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), facecolor=c["bg"])
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.03, right=0.99, wspace=0.18)

    fig.suptitle(
        _l(
            lang,
            "Одна химическая связь → два направленных ребра в PyG (GNNOpt)",
            "One chemical bond → two directed edges in PyG (GNNOpt)",
        ),
        fontsize=13.5,
        fontweight="bold",
        color=c["text"],
        y=0.97,
    )

    col_titles = (
        _l(lang, "Рисунок (кнопка G)", "Drawing (button G)"),
        "PyG edge_index",
        "edge_index",
    )
    x0, x1, y_c = 0.20, 0.80, 0.48
    r_node = 0.13
    nb_c, o_c = "#6366f1", "#ef4444"

    def _node(ax, x: float, label: str, sub: str, color: str) -> None:
        ax.add_patch(
            plt.Circle((x, y_c), r_node, color=color, alpha=0.72, ec="white", lw=2.2, zorder=3)
        )
        ax.add_patch(
            plt.Circle(
                (x - r_node * 0.22, y_c + r_node * 0.22),
                r_node * 0.20,
                color="white",
                alpha=0.45,
                ec="none",
                zorder=4,
            )
        )
        ax.text(
            x,
            y_c,
            label,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=_INK_ON_DARK,
            zorder=5,
        )
        ax.text(
            x,
            y_c - 0.30,
            sub,
            ha="center",
            va="top",
            fontsize=10.5,
            color=c["text"],
            zorder=5,
        )

    e0 = _l(lang, "ребро 0→1", "edge 0→1")
    e1 = _l(lang, "ребро 1→0", "edge 1→0")
    bond_lbl = _l(lang, "1 связь", "1 bond")

    for ax, col_title in zip(axes, col_titles):
        ax.set_facecolor(c["box"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.03, 0.06),
                0.94,
                0.86,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor=c["box"],
                edgecolor=c["edge"],
                linewidth=1.2,
                zorder=0,
            )
        )
        ax.set_title(col_title, fontsize=12, fontweight="bold", color=c["text"], pad=14)

    _node(axes[0], x0, "Nb", "i = 0", nb_c)
    _node(axes[0], x1, "O", "j = 1", o_c)
    axes[0].plot(
        [_circle_anchor(x0, y_c, r_node, 0)[0], _circle_anchor(x1, y_c, r_node, 180)[0]],
        [y_c, y_c],
        color=c["edge"],
        lw=4.0,
        solid_capstyle="round",
        zorder=2,
    )
    axes[0].text(0.5, y_c + 0.22, bond_lbl, ha="center", fontsize=11, color=c["muted"])

    _node(axes[1], x0, "Nb", "i = 0", nb_c)
    _node(axes[1], x1, "O", "j = 1", o_c)
    (p0_b, p1_b, c_b), (p0_r, p1_r, c_r) = _pyg_directed_edge_arrows(x0, x1, y_c, r_node)
    _draw_quadratic_arrow(axes[1], p0_b, p1_b, c_b, color="#0284c7")
    _draw_quadratic_arrow(axes[1], p0_r, p1_r, c_r, color="#dc2626")
    axes[1].text(0.5, y_c + 0.30, e0, ha="center", fontsize=10, color=c["text"], fontweight="bold")
    axes[1].text(0.5, y_c - 0.36, e1, ha="center", fontsize=10, color=c["text"], style="italic")

    code_lines = ["src = [0, 1]", "dst = [1, 0]"]
    if n_undirected is not None and n_directed is not None:
        footer = _l(
            lang,
            f"вся структура: {n_undirected} уник. / {n_directed} directed",
            f"full structure: {n_undirected} unique / {n_directed} directed",
        )
    else:
        footer = _l(lang, "2 directed edges", "2 directed edges")
    box_w, box_h = 0.74, 0.62
    axes[2].add_patch(
        FancyBboxPatch(
            (0.13, 0.24),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f8fafc" if theme == "light" else "#1a2433",
            edgecolor=c["edge"],
            linewidth=1.4,
            zorder=1,
        )
    )
    code_ink = c["text"]
    axes[2].text(
        0.5,
        0.58,
        "\n".join(code_lines),
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        color=code_ink,
        zorder=2,
    )
    axes[2].text(0.5, 0.30, footer, ha="center", va="center", fontsize=10, color=c["muted"], zorder=2)

    note = _l(
        lang,
        "Расстояние Nb–O < cutoff ⇒ на нашем 2D-рисунке одна серая линия, "
        "а GNNOpt хранит два направленных ребра: i→j и j→i (MPNN, не цепь Маркова). "
        "Таблица кнопки G: «N уник. (рисунок) / 2N directed (inference)».",
        "Distance Nb–O < cutoff ⇒ one grey line on our 2D plot, but GNNOpt stores "
        "two directed edges: message i→j and j→i (MPNN passes features both ways). "
        "Table in button G: «N unique (plot) / 2N directed (inference)». "
        "Not a Markov chain — no P(s′|s); these are MPNN channels in GNNOpt.",
    )
    fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=9.2, color=c["muted"], wrap=True)
    return fig


def plot_line_graph_alignn_schematic(
    *,
    lang: Lang = "ru",
    theme: Theme = "light",
) -> plt.Figure:
    """Publication figure: atom graph vs ALIGNN line graph (bond angles at B)."""
    c = _theme_colors(theme)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor=c["bg"])
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.04, right=0.98, wspace=0.12)

    supt = _l(
        lang,
        "ALIGNN — граф атомов + line graph (Choudhary & DeCost 2021)",
        "ALIGNN — atom graph + line graph (from Choudhary & DeCost 2021)",
    )
    fig.suptitle(supt, fontsize=13, fontweight="bold", color=c["text"], y=0.96)

    left_t = _l(lang, "граф атомов", "atom graph")
    right_t = _l(lang, "line graph", "line graph")
    angle_lbl = _l(lang, "угол при B", "angle at B")
    nodes_lbl = _l(lang, "узлы = связи, не атомы", "nodes = bonds, not atoms")
    angle_3d = "∠ABC in 3D"

    for ax, title in zip(axes, (left_t, right_t)):
        ax.set_facecolor("#eef2ff" if theme == "light" else c["box"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.04, 0.08),
                0.92,
                0.84,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor="#f5f3ff" if theme == "light" else c["box"],
                edgecolor="#7c3aed" if theme == "light" else c["edge"],
                linewidth=1.3,
                zorder=0,
            )
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color=c["text"], pad=10)

    ax_l, ax_r = axes
    # Atom graph — A–B–C
    pos = {"A": (0.22, 0.38), "B": (0.50, 0.72), "C": (0.78, 0.38)}
    colors = {"A": "#6366f1", "B": "#14b8a6", "C": "#ef4444"}
    for sym, (x, y) in pos.items():
        ax_l.add_patch(plt.Circle((x, y), 0.09, color=colors[sym], ec="white", lw=2.2, zorder=3))
        ax_l.text(x, y, sym, ha="center", va="center", fontsize=14, fontweight="bold", color=_INK_ON_DARK, zorder=4)
    ax_l.plot([pos["A"][0], pos["B"][0]], [pos["A"][1], pos["B"][1]], color="#64748b", lw=3.5, zorder=2)
    ax_l.plot([pos["B"][0], pos["C"][0]], [pos["B"][1], pos["C"][1]], color="#64748b", lw=3.5, zorder=2)
    ax_l.text(0.32, 0.58, "AB", ha="center", fontsize=10, color=c["text"])
    ax_l.text(0.68, 0.58, "BC", ha="center", fontsize=10, color=c["text"])
    ax_l.text(0.50, 0.22, angle_3d, ha="center", fontsize=10.5, color="#7c3aed", fontweight="bold")

    # Line graph — bond nodes
    bx, by = 0.32, 0.62
    cx, cy = 0.68, 0.62
    for x, lbl in ((bx, "AB"), (cx, "BC")):
        ax_r.add_patch(plt.Circle((x, by), 0.075, color="#f97316", ec="#c2410c", lw=2, zorder=3))
        ax_r.text(x, by, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=_INK_ON_DARK, zorder=4)
    ax_r.annotate(
        "",
        xy=(cx - 0.08, by),
        xytext=(bx + 0.08, by),
        arrowprops=dict(arrowstyle="-|>", color="#a855f7", lw=2.5, shrinkA=8, shrinkB=8),
        zorder=2,
    )
    ax_r.text(0.50, 0.82, angle_lbl, ha="center", fontsize=9.5, color="#7c3aed")
    ax_r.text(0.50, 0.22, nodes_lbl, ha="center", fontsize=10, color=c["text"])

    cap = _l(
        lang,
        "У центрального атома B соседи A и C → две связи AB и BC в графе атомов. "
        "В line graph каждая связь — узел (оранжевый); рёбра line graph соединяют связи, сходящиеся в B, "
        "и несут 3D-угол ∠ABC. Поэтому «рёбер line graph» в таблице ≫ рёбер atom graph.",
        "Central atom B has neighbours A and C → two bonds AB and BC in the atom graph. "
        "In the line graph, each bond is a node (orange); line-graph edges connect bonds that meet at B "
        "and carry the 3D bond angle ∠ABC. That is why «line graph edges» ≫ atom edges in the table.",
    )
    fig.text(0.5, 0.06, cap, ha="center", va="bottom", fontsize=9.5, color=c["muted"], wrap=True)
    return fig


def plot_gnn_training_vs_viz_table_schematic(
    *,
    lang: Lang = "ru",
    theme: Theme = "light",
) -> plt.Figure:
    """Publication table: authors' 3D graph vs button-G teaching plot (Methods / SI)."""
    c = _theme_colors(theme)
    fig, ax = plt.subplots(figsize=(11.2, 2.8), facecolor=c["bg"])
    ax.axis("off")

    if lang == "en":
        title = "3D graph in training (authors) vs our button-G plot — for Methods / SI"
        cols = ("Model", "Graph in training (authors)", "Our button G plot")
        rows = (
            ("ALIGNN", "3D atom graph + 3D bond-angle line graph (DGL)", "2D projection or axonometric — illustration only"),
            ("GNNOpt", "3D PyG graph, edge vectors r_ij in Å", "2D/3D teach plot, not model input"),
        )
        foot = (
            "Both models consume full 3D coordinates + PBC; button G only visualises neighbours — "
            "does not change inference tensors."
        )
    else:
        title = "3D-граф в обучении (авторы) vs рисунок кнопки G — для Methods / SI"
        cols = ("Модель", "Граф в обучении (авторы)", "Рисунок кнопки G")
        rows = (
            ("ALIGNN", "3D atom graph + line graph с 3D углами связей (DGL)", "2D-проекция или аксонометрия — только иллюстрация"),
            ("GNNOpt", "3D PyG, векторы r_ij в Å", "учебный 2D/3D, не вход модели"),
        )
        foot = (
            "Обе модели используют полные 3D-координаты + PBC; кнопка G только визуализирует соседей — "
            "тензоры inference не меняет."
        )

    ax.set_title(title, fontsize=12, fontweight="bold", color=c["text"], pad=14, loc="left", x=0.02)
    tbl = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=list(cols),
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.22, 0.96, 0.62],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    header_bg = "#e2e8f0" if theme == "light" else "#334155"
    for j in range(3):
        tbl[(0, j)].set_facecolor(header_bg)
        tbl[(0, j)].set_text_props(fontweight="bold", color=c["text"])
    for i in range(1, 3):
        for j in range(3):
            tbl[(i, j)].set_facecolor("#ffffff" if theme == "light" else "#1e293b")
            tbl[(i, j)].set_text_props(color=c["text"])
    fig.text(0.02, 0.04, foot, ha="left", va="bottom", fontsize=9.5, color=c["muted"], wrap=True)
    return fig


def save_gnn_teaching_schematics_png(
    out_dir: Path | str | None = None,
    *,
    langs: tuple[Lang, ...] = ("en", "ru"),
    themes: tuple[Theme, ...] = ("light", "dark"),
    dpi: int = 300,
) -> list[Path]:
    """Export §2b teaching figures for the paper → generated_films_GPT_38/."""
    dest = Path(out_dir) if out_dir is not None else GNN_TEACHING_PNG_DIR
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    exporters: tuple[tuple[str, Any], ...] = (
        ("gnn_alignn_line_graph_schematic", plot_line_graph_alignn_schematic),
        ("gnn_pyg_directed_edges_schematic", plot_pyg_directed_edges_schematic),
        ("gnn_training_vs_viz_table", plot_gnn_training_vs_viz_table_schematic),
    )
    for lang in langs:
        for theme in themes:
            tag = f"{lang}_{theme}"
            for stem, plot_fn in exporters:
                fig = plot_fn(lang=lang, theme=theme)
                path = dest / f"{stem}_{tag}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)
                saved.append(path)
    return saved


def pyg_directed_edges_figure_caption_html(lang: Lang = "ru") -> str:
    ink = "color:#000000 !important;"
    if lang == "en":
        body = (
            f'<p style="margin:6px 0 10px;font-size:11px;line-height:1.55;{ink}">'
            "<b>PyG directed edges:</b> GNNOpt message passing uses <code>edge_index</code> with "
            "both (i→j) and (j→i). Our neighbour plot collapses them to one line — counts in the table "
            "show <b>unique</b> vs <b>directed</b>.</p>"
        )
    else:
        body = (
            f'<p style="margin:6px 0 10px;font-size:11px;line-height:1.55;{ink}">'
            "<b>Directed в PyG:</b> GNNOpt в message passing использует <code>edge_index</code> "
            "с парами (i→j) и (j→i). На соседнем рисунке графа мы схлопываем их в одну линию — "
            "в таблице указаны <b>уникальные</b> и <b>directed</b> рёбра.</p>"
        )
    return body + pyg_mpnn_markov_note_html(lang)


def line_graph_alignn_schematic_html(lang: Lang = "ru") -> str:
    """Mini SVG: atom graph vs ALIGNN line graph (bond angles)."""
    ink = "color:#000000 !important;"
    if lang == "en":
        title = "ALIGNN — atom graph + line graph (from Choudhary &amp; DeCost 2021)"
        cap = (
            "Central atom <b>B</b> has neighbours <b>A</b> and <b>C</b> → two bonds AB and BC in the atom graph. "
            "In the <b>line graph</b>, each bond is a node (orange); line-graph edges connect bonds that meet at B "
            "and carry the <b>3D bond angle</b> ∠ABC. That is why «line graph edges» ≫ atom edges in the table."
        )
        lg = "line graph"
        ag = "atom graph"
    else:
        title = "ALIGNN — граф атомов + line graph (Choudhary &amp; DeCost 2021)"
        cap = (
            "У центрального атома <b>B</b> соседи <b>A</b> и <b>C</b> → две связи AB и BC в графе атомов. "
            "В <b>line graph</b> каждая связь — узел (оранжевый); рёбра line graph соединяют связи, сходящиеся в B, "
            "и несут <b>3D-угол</b> ∠ABC. Поэтому «рёбер line graph» в таблице ≫ рёбер atom graph."
        )
        lg = "line graph"
        ag = "граф атомов"
    return f"""
<h4 style="margin:16px 0 8px 0;color:#000000 !important">{title}</h4>
<div class="pale-callout compare-ink-light" style="margin:0 0 10px;padding:10px 12px;border:1px solid #7c3aed;border-radius:8px;background:#f5f3ff;{ink}">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 210" style="width:100%;max-width:640px;height:auto;display:block;margin:0 auto 8px">
  <text x="150" y="20" text-anchor="middle" font-size="12" font-weight="600" fill="#000000">{ag}</text>
  <text x="490" y="20" text-anchor="middle" font-size="12" font-weight="600" fill="#000000">{lg}</text>
  <circle cx="80" cy="120" r="20" fill="#6366f1" stroke="#312e81" stroke-width="2"/><text x="80" y="126" text-anchor="middle" fill="white" font-weight="bold">A</text>
  <circle cx="150" cy="80" r="22" fill="#14b8a6" stroke="#0f766e" stroke-width="2"/><text x="150" y="86" text-anchor="middle" fill="white" font-weight="bold">B</text>
  <circle cx="220" cy="120" r="20" fill="#ef4444" stroke="#991b1b" stroke-width="2"/><text x="220" y="126" text-anchor="middle" fill="white" font-weight="bold">C</text>
  <line x1="96" y1="108" x2="134" y2="92" stroke="#64748b" stroke-width="3"/>
  <line x1="166" y1="92" x2="204" y2="108" stroke="#64748b" stroke-width="3"/>
  <text x="115" y="88" font-size="10" fill="#000000">AB</text>
  <text x="195" y="88" font-size="10" fill="#000000">BC</text>
  <text x="150" y="145" text-anchor="middle" font-size="10" fill="#7c3aed">∠ABC in 3D</text>
  <circle cx="430" cy="95" r="16" fill="#f97316" stroke="#c2410c" stroke-width="2"/><text x="430" y="100" text-anchor="middle" fill="white" font-size="10" font-weight="bold">AB</text>
  <circle cx="550" cy="95" r="16" fill="#f97316" stroke="#c2410c" stroke-width="2"/><text x="550" y="100" text-anchor="middle" fill="white" font-size="10" font-weight="bold">BC</text>
  <line x1="446" y1="95" x2="534" y2="95" stroke="#a855f7" stroke-width="2.5" marker-end="url(#lgArr)"/>
  <defs><marker id="lgArr" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L5,3 L0,6 Z" fill="#a855f7"/></marker></defs>
  <text x="490" y="78" text-anchor="middle" font-size="9" fill="#7c3aed">angle at B</text>
  <text x="490" y="140" text-anchor="middle" font-size="10" fill="#000000">nodes = bonds, not atoms</text>
</svg>
<p style="margin:0;font-size:11.5px;line-height:1.62;{ink}">{cap}</p>
<p style="margin:8px 0 0;font-size:10px;line-height:1.5;{ink}"><b>MANUSCRIPT_PICKUP — SI PNG:</b>
<code>generated_films_GPT_38/gnn_alignn_line_graph_schematic_{lang}_light.png</code>
(run <code>python ml_gnn_pipeline_viz.py --export-teaching</code>).</p>
</div>"""


def gnn_graph_technical_depth_html(lang: Lang = "ru") -> str:
    """Expanded §2b — edges, node features, line graph, 2D vs 3D, manuscript novelty."""
    from mortazavi_dielectric_shared import (
        MANUSCRIPT_GNN_VIZ_FIGURE_ID,
        manuscript_gnn_graph_teaching_html,
    )

    ink = "color:#000000 !important;"
    fig_id = MANUSCRIPT_GNN_VIZ_FIGURE_ID
    if lang == "en":
        return f"""
<h4 style="margin:16px 0 8px 0;color:#000000 !important">In more detail — edges, nodes, line graph</h4>

<p style="margin:0 0 8px 0;{ink}"><b>3. Building edges.</b> ASE / JARVIS scan all atom pairs (with <b>periodic boundary conditions</b> if the cell is set).
If 3D distance d<sub>ij</sub> &lt; r<sub>cut</sub>, atoms i and j are neighbours. On our <b>teaching plot</b> (button G) we draw <b>one grey line</b> per pair.
In <b>PyG (GNNOpt)</b> the same pair is stored twice: (i→j) and (j→i) in <code>edge_index</code> — see schematic below.
In <b>DGL (ALIGNN)</b> the atom graph is also directed for message passing; the table counts atom edges + separate line-graph edges.</p>

<p style="margin:0 0 8px 0;{ink}"><b>4. Nodes = atoms + features.</b>
<b>ALIGNN</b> (CGCNN lineage, <a href="https://doi.org/10.1038/s41524-021-00650-1">Choudhary &amp; DeCost 2021</a>):
each node gets a fixed-length <b>CGCNN embedding</b> of the element (Z) — not just one-hot, a learned 92-dim style vector shipped with JARVIS.
<b>GNNOpt</b> (<a href="https://doi.org/10.1002/adma.202409175">Hung et al. 2024</a>):
one-hot Z plus Mendeleev mass / radius / dipole; on each directed edge the model sees the <b>3D bond vector</b> r<sub>ij</sub> (and periodic shift if the neighbour is an image cell atom).</p>

<p style="margin:0 0 8px 0;{ink}"><b>5. ALIGNN line graph only.</b> After the atom graph, ALIGNN builds a <b>second graph</b> whose nodes are <b>chemical bonds</b> (not atoms).
Two bond-nodes are linked if they share a central atom (e.g. bonds AB and BC at atom B); the edge carries the <b>bond angle in 3D</b>.
The network alternates updates on atom graph and line graph — that is the published ALIGNN architecture, not our extension.</p>

<h4 style="margin:16px 0 8px 0;color:#000000 !important">2D or 3D graphs in the original papers?</h4>
<div class="pale-callout compare-ink-light dielectric-manuscript-ok" style="margin:0 0 10px;padding:10px 12px;border:2px dashed #b45309;border-radius:8px;background:#fffbeb;{ink}">
<p style="margin:0 0 8px 0;{ink}"><b>MANUSCRIPT_PICKUP — cite DOI, do not claim a new graph:</b>
Both ALIGNN and GNNOpt were trained on <b>3D crystal structures</b> (Materials Project / JARVIS unit cells): full (x,y,z), lattice vectors, neighbours via <b>3D distances</b> and PBC.
They are <b>not</b> 2D image graphs (pixels) and <b>not</b> «only xy» chemistry drawings.
Your ItoCl <code>.extxyz</code> films are still <b>3D periodic slabs</b> (thin along one axis, but ML sees the same 3D neighbour search).</p>
<p style="margin:0;{ink}"><b>What is only ours:</b> button G <b>2D projections</b> (auto xy/xz/yz for flat films) and optional <b>3D axonometric sketch</b> — pedagogy for the paper SI, <b>not</b> the tensor the network consumes.
<b>No methodological novelty</b> in graph construction on NbOI₂/ItoCl: frozen inference + documentation. In Methods: cite
<a href="https://doi.org/10.1038/s41524-021-00650-1">ALIGNN DOI</a>,
<a href="https://doi.org/10.1038/s41524-020-00440-1">JARVIS-DFT DOI</a>,
<a href="https://doi.org/10.1002/adma.202409175">GNNOpt DOI</a>.
<b>SI (agreed):</b> export button-G panels as <i>{fig_id}</i> — «how we visualise the same 3D neighbour graph» (2D projection + axonometric + PyG directed schematic); <b>do not</b> write «we proposed a graph neural network» in Methods.</p>
</div>
<table style="width:100%;border-collapse:collapse;font-size:11px;margin:0 0 10px 0;{ink}">
<tr style="background:#e2e8f0"><th style="padding:5px 7px;border:1px solid #cbd5e1">Model</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Graph in training (authors)</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Our button G plot</th></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">ALIGNN</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">3D atom graph + 3D bond-angle line graph (DGL)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">2D projection or axonometric — <b>illustration only</b></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">GNNOpt</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">3D PyG graph, edge vectors r<sub>ij</sub> in Å</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">same — 2D/3D teach plot, not model input</td></tr>
</table>
{manuscript_gnn_graph_teaching_html(lang)}
{line_graph_alignn_schematic_html("en")}"""

    return f"""
<h4 style="margin:16px 0 8px 0;color:#000000 !important">Подробнее — рёбра, узлы, line graph</h4>

<p style="margin:0 0 8px 0;{ink}"><b>3. Строим рёбра.</b> ASE / JARVIS перебирают пары атомов (с <b>периодическими границами</b>, если задана ячейка).
Если 3D-расстояние d<sub>ij</sub> &lt; r<sub>cut</sub>, i и j — соседи. На <b>учебном рисунке</b> (кнопка G) рисуем <b>одну серую линию</b> на пару.
В <b>PyG (GNNOpt)</b> та же пара хранится дважды: (i→j) и (j→i) в <code>edge_index</code> — схема ниже.
В <b>DGL (ALIGNN)</b> граф атомов тоже ориентирован для message passing; в таблице отдельно считаются рёбра atom graph и line graph.</p>

<p style="margin:0 0 8px 0;{ink}"><b>4. Узлы = атомы + признаки.</b>
<b>ALIGNN</b> (линия CGCNN, <a href="https://doi.org/10.1038/s41524-021-00650-1">Choudhary &amp; DeCost 2021</a>):
у каждого узла — <b>CGCNN-вектор</b> элемента (фиксированная размерность из JARVIS), не просто one-hot Z.
<b>GNNOpt</b> (<a href="https://doi.org/10.1002/adma.202409175">Hung et al. 2024</a>):
one-hot Z + масса / радиус / диполь по Менделееву; на каждом направленном ребре — <b>3D-вектор связи</b> r<sub>ij</sub> (и сдвиг ячейки, если сосед из image).</p>

<p style="margin:0 0 8px 0;{ink}"><b>5. Только ALIGNN — line graph.</b> После графа атомов ALIGNN строит <b>второй граф</b>, узлы которого — <b>химические связи</b> (не атомы).
Два узла-line-graph связаны, если связи имеют общий центральный атом (например AB и BC у атома B); на ребре — <b>угол между связями в 3D</b>.
Сеть чередует обновления на графе атомов и line graph — это опубликованная архитектура ALIGNN, не наше расширение.</p>

<h4 style="margin:16px 0 8px 0;color:#000000 !important">2D или 3D графы у авторов?</h4>
<div class="pale-callout compare-ink-light dielectric-manuscript-ok" style="margin:0 0 10px;padding:10px 12px;border:2px dashed #b45309;border-radius:8px;background:#fffbeb;{ink}">
<p style="margin:0 0 8px 0;{ink}"><b>MANUSCRIPT_PICKUP — DOI в Methods, не «новый граф»:</b>
И ALIGNN, и GNNOpt обучены на <b>3D кристаллических структурах</b> (ячейки MP / JARVIS): полные (x,y,z), векторы решётки, соседи через <b>3D-расстояния</b> и PBC.
Это <b>не</b> 2D-картинки (пиксели) и <b>не</b> «только плоскость xy».
Ваши плёнки ItoCl в <code>.extxyz</code> — <b>3D periodic slab</b> (тонкий слой по одной оси, но ML делает тот же 3D neighbour search).</p>
<p style="margin:0;{ink}"><b>Что только у нас:</b> <b>2D-проекции</b> кнопки G (авто xy/xz/yz для плоских плёнок) и опциональная <b>3D-аксонометрия</b> — педагогика для SI статьи, <b>не</b> тензор на входе сети.
<b>Методической новизны</b> в построении графа на NbOI₂/ItoCl нет: frozen inference + пояснения. В Methods — ссылки
<a href="https://doi.org/10.1038/s41524-021-00650-1">ALIGNN DOI</a>,
<a href="https://doi.org/10.1038/s41524-020-00440-1">JARVIS-DFT DOI</a>,
<a href="https://doi.org/10.1002/adma.202409175">GNNOpt DOI</a>.
<b>SI (решено):</b> панели кнопки G → <i>{fig_id}</i> — «как визуализируем тот же 3D-граф соседства» (2D-проекция + аксонометрия + схема PyG directed); в Methods <b>не</b> писать «мы предложили графовую сеть».</p>
</div>
<table style="width:100%;border-collapse:collapse;font-size:11px;margin:0 0 10px 0;{ink}">
<tr style="background:#e2e8f0"><th style="padding:5px 7px;border:1px solid #cbd5e1">Модель</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Граф в обучении (авторы)</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Рисунок кнопки G</th></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">ALIGNN</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">3D atom graph + line graph с 3D углами связей (DGL)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">2D-проекция или аксонометрия — <b>только иллюстрация</b></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">GNNOpt</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">3D PyG, векторы r<sub>ij</sub> в Å</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">то же — учебный 2D/3D, не вход модели</td></tr>
</table>
{manuscript_gnn_graph_teaching_html(lang)}
{line_graph_alignn_schematic_html("ru")}"""


def gnn_neighbor_graph_algorithm_chapter_html(lang: Lang = "ru") -> str:
    """§2b — how neighbour graphs are built (honest: standard ML libs + our viz only)."""
    ink = "color:#000000 !important;"
    # No outer conc-i18n-section — wrap_dielectric_chapter_html (§10 pattern) provides the shell;
    # nested conc-i18n-section breaks finalize_callouts regex (first </div> truncates the chapter).
    if lang == "en":
        return f"""
<p style="margin:0 0 10px 0;{ink}"><b>§2b — Neighbour graphs for ALIGNN / GNNOpt (step by step)</b></p>

<div class="pale-callout compare-ink-light dielectric-manuscript-ok" style="margin:0 0 12px;padding:10px 12px;border:2px dashed #b45309;border-radius:8px;background:#fffbeb;{ink}">
<p style="margin:0;{ink}"><b>MANUSCRIPT_PICKUP — be honest in Methods:</b> We did <b>not</b> invent graph neural networks or a new graph construction.
The <b>physics models</b> are published ALIGNN (JARVIS) and GNNOpt; we only run <b>frozen inference</b> on your extxyz.
What we added in this project is <b>documentation + pictures</b> (button G, §2b, Phases A/B/C) so you see what those libraries do internally.</p>
</div>

<p style="margin:0 0 8px 0;{ink}"><b>In one sentence.</b> A crystal file is a list of atoms with coordinates; a <b>graph</b> connects each atom to nearby neighbours within a radius —
that graph is the input to a pretrained neural network (like a sentence of words for an NLP model).</p>

<h4 style="margin:14px 0 6px 0;color:#000000 !important">What is ours vs what is standard</h4>
<table style="width:100%;border-collapse:collapse;font-size:11.5px;margin:0 0 12px 0;{ink}">
<tr style="background:#e2e8f0"><th style="padding:5px 7px;border:1px solid #cbd5e1">Piece</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Who built it</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">In this repo</th></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">ALIGNN graph + line graph</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">JARVIS / Choudhary et al. (library <code>alignn</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Buttons A, A′ — <code>Graph.atom_dgl_multigraph</code></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">GNNOpt PyG graph</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">GNNOpt authors (local <code>models/gnnopt</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Button B — ASE <code>neighbor_list</code></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">MACEField α</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">MACE equivariant MP (not DGL line graph)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Button C — same atoms, different head</td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0"><b>2D / 3D pictures, pipeline schematic</b></td>
<td style="padding:5px 7px;border:1px solid #e2e8f0"><b>This widget</b> (<code>ml_gnn_pipeline_viz.py</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Button G — <b>visualisation only</b>, does not change ML</td></tr>
</table>

<h4 style="margin:14px 0 6px 0;color:#000000 !important">Algorithm — neighbour graph (human steps)</h4>
<ol style="margin:0 0 12px 20px;{ink}">
<li><b>Read the structure.</b> Open your <code>.xyz</code> / <code>.extxyz</code> with ASE. Each row → one atom: element + position (x,y,z) in Å, plus periodic cell if present.</li>
<li><b>Choose a cutoff radius.</b> «Who is neighbour of whom?» — all pairs closer than <b>r</b>. GNNOpt picture uses <b>6 Å</b>; ALIGNN uses <b>8 Å</b> and at most 12 neighbours per atom (JARVIS default).</li>
<li><b>Build edges.</b> For every pair (i,j) with distance &lt; r (and i≠j), add an edge. Undirected bond → one line in the plot, two directed edges in PyG.</li>
<li><b>Nodes = atoms.</b> Node i carries features: atomic number, CGCNN embedding (ALIGNN), or one-hot Z + mass (GNNOpt).</li>
<li><b>ALIGNN extra step — line graph.</b> Each bond becomes a node in a <b>second</b> graph; bond angles are encoded there. That is why ALIGNN edge counts look huge in the table.</li>
<li><b>Forward pass (inference).</b> Load frozen weights from Figshare / <code>models/gnnopt</code>. One pass → scalar Eg (ALIGNN) or Tr ε(ω) curve (GNNOpt). <b>No training</b> on NbOI₂ here.</li>
<li><b>Our drawings (button G).</b> We re-use step 2–3 with ASE <code>neighbor_list</code> only to <b>draw</b> nodes and edges:
<ul style="margin:6px 0 0 16px">
<li><b>2D panel</b> — auto-pick the least-squashed plane (xy / xz / yz) for thin films;</li>
<li><b>3D axonometric</b> — experimental extra view (elev≈28°, azim≈−52°), thickness along c slightly boosted for visibility.</li>
</ul></li>
<li><b>Save metadata.</b> <code>build_pipeline_summary()</code> writes JSON with node/edge counts — for tables in the article, not for retraining.</li>
</ol>

{gnn_graph_technical_depth_html("en")}
{pyg_directed_edges_schematic_html("en")}

<h4 style="margin:14px 0 6px 0;color:#000000 !important">Pseudocode (same logic as the code)</h4>
<pre class="dielectric-manuscript-ok" style="margin:0 0 12px;padding:10px 12px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;font-size:11px;line-height:1.5;overflow-x:auto;{ink}"><code>atoms = read("ItoCl_....extxyz")
for cutoff in (6.0, 8.0):                    # GNNOpt vs ALIGNN
    pairs = all atom pairs with distance &lt; cutoff
    graph.nodes = atoms
    graph.edges = pairs
    if ALIGNN:
        line_graph = bonds_as_nodes(bond_angles)
        Eg = frozen_ALIGNN(graph, line_graph)   # buttons A / A′
    if GNNOpt:
        eps_spectrum = frozen_GNNOpt(graph)     # button B
# Button G: matplotlib only — plot(graph) for teaching, no new weights</code></pre>

<p style="margin:0;{ink}"><b>Where to click:</b> ML tab → <b>G) Graph</b> for pictures; <b>A/A′/B</b> for real inference; Phases A/B/C accordion for calibration vs validation story.</p>"""

    return f"""
<p style="margin:0 0 10px 0;{ink}"><b>§2b — Граф соседства для ALIGNN / GNNOpt (пошагово)</b></p>

<div class="pale-callout compare-ink-light dielectric-manuscript-ok" style="margin:0 0 12px;padding:10px 12px;border:2px dashed #b45309;border-radius:8px;background:#fffbeb;{ink}">
<p style="margin:0;{ink}"><b>MANUSCRIPT_PICKUP — честно в Methods:</b> Мы <b>не изобретали</b> графовые нейросети и не придумывали новую топологию графа.
<b>Физические модели</b> — опубликованные ALIGNN (JARVIS) и GNNOpt; у нас только <b>готовый inference</b> на вашем extxyz.
В проекте добавлены <b>пояснения и рисунки</b> (кнопка G, §2b, фазы A/B/C), чтобы было видно, что делают библиотеки внутри.</p>
</div>

<p style="margin:0 0 8px 0;{ink}"><b>Одной фразой.</b> Файл кристалла — список атомов с координатами; <b>граф</b> соединяет каждый атом с ближайшими соседями в радиусе —
этот граф подаётся на вход уже обученной нейросети (как предложение из слов для языковой модели).</p>

<h4 style="margin:14px 0 6px 0;color:#000000 !important">Что чужое, что наше</h4>
<table style="width:100%;border-collapse:collapse;font-size:11.5px;margin:0 0 12px 0;{ink}">
<tr style="background:#e2e8f0"><th style="padding:5px 7px;border:1px solid #cbd5e1">Часть</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">Кто сделал</th>
<th style="padding:5px 7px;border:1px solid #cbd5e1">У нас в репо</th></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">Граф + line graph ALIGNN</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">JARVIS / Choudhary et al. (библиотека <code>alignn</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Кнопки A, A′ — <code>Graph.atom_dgl_multigraph</code></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">Граф GNNOpt (PyG)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Авторы GNNOpt (каталог <code>models/gnnopt</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Кнопка B — ASE <code>neighbor_list</code></td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0">MACEField α</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">MACE equivariant MP (не DGL line graph)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Кнопка C — те же атомы, другая голова</td></tr>
<tr><td style="padding:5px 7px;border:1px solid #e2e8f0"><b>Рисунки 2D/3D, схема pipeline</b></td>
<td style="padding:5px 7px;border:1px solid #e2e8f0"><b>Этот виджет</b> (<code>ml_gnn_pipeline_viz.py</code>)</td>
<td style="padding:5px 7px;border:1px solid #e2e8f0">Кнопка G — <b>только визуализация</b>, ML не меняет</td></tr>
</table>

<h4 style="margin:14px 0 6px 0;color:#000000 !important">Алгоритм — граф соседства (человеческим языком)</h4>
<ol style="margin:0 0 12px 20px;{ink}">
<li><b>Читаем структуру.</b> Файл <code>.xyz</code> / <code>.extxyz</code> через ASE. Каждая строка — атом: элемент + координаты (x,y,z) в Å, при необходимости periodic cell.</li>
<li><b>Выбираем радиус cutoff.</b> «Кто чей сосед?» — все пары ближе <b>r</b>. На рисунке GNNOpt — <b>6 Å</b>; у ALIGNN — <b>8 Å</b> и не больше 12 соседей на атом (дефолт JARVIS).</li>
<li><b>Строим рёбра.</b> Для каждой пары (i,j) с расстоянием &lt; r (i≠j) — ребро. Неориентированная связь на рисунке — одна линия; в PyG часто два направленных ребра.</li>
<li><b>Узлы = атомы.</b> У узла i — признаки: номер элемента, CGCNN-вектор (ALIGNN) или one-hot Z + масса (GNNOpt).</li>
<li><b>Доп. шаг ALIGNN — line graph.</b> Каждая связь становится узлом <b>второго</b> графа; углы между связями кодируются там. Поэтому в таблице у ALIGNN так много «рёбер line graph».</li>
<li><b>Прямой проход (inference).</b> Загружаем frozen-веса с Figshare / из <code>models/gnnopt</code>. Один проход → скаляр Eg (ALIGNN) или кривая Tr ε(ω) (GNNOpt). <b>Обучения</b> на NbOI₂ здесь нет.</li>
<li><b>Наши рисунки (кнопка G).</b> Шаги 2–3 повторяем через ASE <code>neighbor_list</code> только чтобы <b>нарисовать</b> узлы и рёбра:
<ul style="margin:6px 0 0 16px">
<li><b>2D</b> — авто-выбор наименее «сплюснутой» плоскости (xy / xz / yz) для тонких плёнок;</li>
<li><b>3D аксонометрия</b> — доп. вид (elev≈28°, azim≈−52°), толщина вдоль c слегка усилена для наглядности.</li>
</ul></li>
<li><b>Сохраняем метаданные.</b> <code>build_pipeline_summary()</code> пишет JSON с числом узлов/рёбер — для таблиц в статье, не для переобучения.</li>
</ol>

{gnn_graph_technical_depth_html("ru")}
{pyg_directed_edges_schematic_html("ru")}

<h4 style="margin:14px 0 6px 0;color:#000000 !important">Псевдокод (та же логика, что в коде)</h4>
<pre class="dielectric-manuscript-ok" style="margin:0 0 12px;padding:10px 12px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;font-size:11px;line-height:1.5;overflow-x:auto;{ink}"><code>atoms = read("ItoCl_....extxyz")
for cutoff in (6.0, 8.0):                    # GNNOpt vs ALIGNN
    pairs = все пары атомов с distance &lt; cutoff
    graph.nodes = atoms
    graph.edges = pairs
    if ALIGNN:
        line_graph = связи_как_узлы(углы_связей)
        Eg = frozen_ALIGNN(graph, line_graph)   # кнопки A / A′
    if GNNOpt:
        eps_spectrum = frozen_GNNOpt(graph)     # кнопка B
# Кнопка G: только matplotlib — рисуем graph для обучения глаз, веса не трогаем</code></pre>

<p style="margin:0;{ink}"><b>Где нажимать:</b> вкладка ML → <b>G) Граф</b> для картинок; <b>A/A′/B</b> для реального inference; аккордеон «Фазы A/B/C» — история калибровки vs валидации.</p>"""


def build_gnn_pipeline_html(summary: dict[str, Any], lang: Lang = "ru") -> str:
    from mortazavi_dielectric_shared import (
        ALIGNN_EG_MBJ_EV,
        ALIGNN_EG_OPTB88_EV,
        LIT_EG_HSE06_EV,
    )

    a = summary["alignn"]
    g = summary["gnnopt"]
    struct_path = Path(summary["structure"])
    struct_name = struct_path.name
    cache = _load_ml_inference_cache(struct_path)

    hdr = _l(lang, "Граф из структуры → готовая GNN (не обучение)", "Structure graph → pretrained GNN (not training)")
    lead = _l(
        lang,
        f"<b>Мы не собирали и не обучали нейросеть под ваш bulk.</b> Из <code>{struct_name}</code> "
        "автоматически строится <b>граф соседства</b> (узлы = атомы, рёбра = пары в радиусе cutoff). "
        "Дальше один forward pass через <b>чужие готовые веса</b> (JARVIS ALIGNN A/A′, GNNOpt B).",
        f"<b>We did not assemble or train a network on your bulk.</b> From <code>{struct_name}</code> "
        "a <b>neighbour graph</b> is built (nodes = atoms, edges = pairs within cutoff). "
        "Then one forward pass through <b>pretrained weights</b> (JARVIS ALIGNN A/A′, GNNOpt B).",
    )
    dgl = _l(
        lang,
        "<code>ml_dgl_bootstrap.py</code> — только починка импорта DGL на Windows (GraphBolt stub), "
        "не меняет топологию графа и не трогает веса.",
        "<code>ml_dgl_bootstrap.py</code> — Windows DGL import fix (GraphBolt stub) only; "
        "does not change graph topology or weights.",
    )

    n_gnn_undir = g.get("n_edges_undirected", g["n_edges"])
    n_gnn_dir = g.get("n_edges_directed", g["n_edges"])
    edges_alignn = _l(
        lang,
        f"{a['n_edges_atom_graph']} atom + line {a['n_edges_line_graph']}",
        f"{a['n_edges_atom_graph']} atom + line {a['n_edges_line_graph']}",
    )
    edges_gnn = _l(
        lang,
        f"{n_gnn_undir} уник. (рисунок) / {n_gnn_dir} directed (inference)",
        f"{n_gnn_undir} unique (plot) / {n_gnn_dir} directed (inference)",
    )

    def _eg_cell(data: dict[str, Any] | None, fallback: float) -> str:
        if data and data.get("Eg_eV_ML") is not None:
            return f"<b>{float(data['Eg_eV_ML']):.3f} eV</b>"
        return f"≈{fallback:.2f} eV ({_l(lang, 'тип.', 'typ.')})"

    eg_a = _eg_cell(cache.get("alignn_optb88"), ALIGNN_EG_OPTB88_EV)
    eg_ap = _eg_cell(cache.get("alignn_mbj"), ALIGNN_EG_MBJ_EV)

    gnn_inf = "—"
    if cache.get("gnnopt"):
        cmp_d = (cache["gnnopt"].get("compare_hse06") or {}).get("Re_eps_Tr") or {}
        if cmp_d:
            gnn_inf = (
                f"Tr(Re ε): MAE≈{cmp_d.get('MAE', 0):.1f}, "
                f"r≈{cmp_d.get('Pearson_r', 0):.2f} "
                f"({cmp_d.get('energy_eV_range', [0, 15])[0]:.0f}–"
                f"{cmp_d.get('energy_eV_range', [0, 15])[1]:.0f} eV vs HSE06 CSV)"
            )
        elif cache["gnnopt"].get("Tr_Re_eps_at_0eV") is not None:
            gnn_inf = f"Tr(Re ε)@0 eV ≈ {float(cache['gnnopt']['Tr_Re_eps_at_0eV']):.1f}"

    cache_note = _l(
        lang,
        "строки inference — из JSON после кнопок A/A′/B" if cache else "запустите A / A′ / B для свежих JSON в dielectric_compare/",
        "inference rows — from JSON after buttons A/A′/B" if cache else "run A / A′ / B for fresh JSON in dielectric_compare/",
    )

    topo_hdr = _l(lang, "1. Топология графа", "1. Graph topology")
    inf_hdr = _l(lang, "2. Веса, выход и inference", "2. Weights, output & inference")

    return f"""
<div class="{_GNN_PANEL_CLASS}" style="margin:8px 0;padding:10px 12px;border:1px solid #6366f1;border-radius:6px;background:#eef2ff;font-size:12px;line-height:1.58;{_GNN_PANEL_INK}">
<p style="margin:0 0 8px 0;{_GNN_PANEL_INK}"><b>{hdr}</b></p>
<p style="margin:0 0 8px 0;{_GNN_PANEL_INK}">{lead}</p>
<p style="margin:0 0 8px 0;{_GNN_PANEL_INK}"><b>{_l(lang, "Файл", "File")}:</b> <code style="color:#000000 !important;background:#f8fafc">{struct_name}</code> · {a['n_atoms']} {_l(lang, "ат.", "at.")}</p>

<p style="margin:10px 0 4px 0;font-weight:600;{_GNN_PANEL_INK}">{topo_hdr}</p>
<table style="width:100%;border-collapse:collapse;font-size:10.5px;margin:4px 0 10px 0;{_GNN_PANEL_INK}">
<tr><th style="{_GNN_TH}"></th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">ALIGNN (DGL)</th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">GNNOpt (PyG)</th></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Backend", "Backend")}</td>
<td style="{_GNN_TD}">{a['backend']}</td>
<td style="{_GNN_TD}">{g['backend']}</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Узлы", "Nodes")}</td>
<td style="{_GNN_TD}">{a['n_atoms']}</td>
<td style="{_GNN_TD}">{g['n_atoms']}</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Рёбра", "Edges")}</td>
<td style="{_GNN_TD}">{edges_alignn}<br>
<small style="{_GNN_PANEL_INK}">line: {a['n_nodes_line_graph']} {_l(lang, 'узлов', 'nodes')}</small></td>
<td style="{_GNN_TD}">{edges_gnn}</td></tr>
<tr><td style="{_GNN_TD}">cutoff</td>
<td style="{_GNN_TD}">{a['cutoff_ang']} Å, max_nb={a['max_neighbors']}</td>
<td style="{_GNN_TD}">{g['cutoff_ang']} Å</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Признаки", "Features")}</td>
<td style="{_GNN_TD}">CGCNN, dim={a.get('atom_feature_dim', '—')}</td>
<td style="{_GNN_TD}">{g['node_features']}<br><small style="{_GNN_PANEL_INK}">{g['edge_features']}</small></td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Line graph", "Line graph")}</td>
<td style="{_GNN_TD}">{_l(lang, 'да (bond angles)', 'yes (bond angles)')}</td>
<td style="{_GNN_TD}">{_l(lang, 'нет', 'no')}</td></tr>
</table>

<p style="margin:10px 0 4px 0;font-weight:600;{_GNN_PANEL_INK}">{inf_hdr}</p>
<p style="margin:0 0 6px 0;font-size:10px;{_GNN_PANEL_INK}">{cache_note}</p>
<table style="width:100%;border-collapse:collapse;font-size:10.5px;margin:4px 0 8px 0;{_GNN_PANEL_INK}">
<tr><th style="{_GNN_TH}"></th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">A) OptB88</th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">A′) mBJ</th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">B) GNNOpt</th>
<th style="{_GNN_TH.replace('text-align:left', 'text-align:center')}">C) MACEField</th></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Виджет", "Widget")}</td>
<td style="{_GNN_TD}"><b>A)</b></td>
<td style="{_GNN_TD}"><b>A′)</b></td>
<td style="{_GNN_TD}"><b>B)</b></td>
<td style="{_GNN_TD}"><b>C)</b> α</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Веса", "Weights")}</td>
<td style="{_GNN_TD}"><code style="color:#000000 !important;background:#f8fafc">jv_optb88vdw_bandgap_alignn</code></td>
<td style="{_GNN_TD}"><code style="color:#000000 !important;background:#f8fafc">jv_mbj_bandgap_alignn</code></td>
<td style="{_GNN_TD}"><code style="color:#000000 !important;background:#f8fafc">gnnopt/model_eps1_240406.torch</code> + eps2</td>
<td style="{_GNN_TD}"><code style="color:#000000 !important;background:#f8fafc">MACEField-MH-0-omat-dielectric</code></td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Выход", "Output")}</td>
<td style="{_GNN_TD}">Eg (eV)</td>
<td style="{_GNN_TD}">Eg (eV)</td>
<td style="{_GNN_TD}">{g['output']}</td>
<td style="{_GNN_TD}">α<sub>ii</sub>, P, BEC</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Inference", "Inference")}</td>
<td style="{_GNN_TD}">{eg_a}</td>
<td style="{_GNN_TD}">{eg_ap}</td>
<td style="{_GNN_TD}">{gnn_inf}</td>
<td style="{_GNN_TD}">{_l(lang, 'не граф', 'not a graph')}</td></tr>
<tr><td style="{_GNN_TD}">vs HSE06</td>
<td style="{_GNN_TD}" colspan="2">Eg≈{LIT_EG_HSE06_EV} eV ({_l(lang, 'лит./docx', 'lit./docx')}) — ≠ OptB88/mBJ XC</td>
<td style="{_GNN_TD}">CSV LOPTICS bulk (View Docx)</td>
<td style="{_GNN_TD}">—</td></tr>
<tr><td style="{_GNN_TD}">{_l(lang, "Обучение", "Training")}</td>
<td colspan="4" style="{_GNN_TD}"><b>{_l(lang, 'нет', 'none')}</b> — frozen Figshare / models/; {_l(lang, 'кнопка G только строит граф', 'button G only builds the graph')}</td></tr>
</table>

<p style="margin:8px 0 0;font-size:11px;{_GNN_PANEL_INK}">{dgl}</p>
<p style="margin:6px 0 0;font-size:11px;{_GNN_PANEL_INK}">MACEField (C/D) — equivariant MP {_l(lang, 'на тех же атомах', 'on the same atoms')}, <b>{_l(lang, 'без', 'without')}</b> DGL line graph; Unified ε — {_l(lang, 'режим Compare', 'Compare mode')}.
<b>{_l(lang, 'Пошаговый алгоритм графа (что наше, что JARVIS/GNNOpt):', 'Step-by-step graph algorithm (ours vs JARVIS/GNNOpt):')}</b> {_l(lang, 'глава', 'chapter')} <b>§2b</b>.</p>
</div>"""


def _theme_colors(theme: Theme) -> dict[str, str]:
    if theme == "dark":
        return {
            "bg": "#1a2433",
            "text": _INK_ON_DARK,
            "muted": _INK_ON_DARK,
            "box": "#1a2433",
            "edge": "#cbd5e1",
            "accent": _INK_ON_DARK,
            "arrow": _INK_ON_DARK,
            "box_ink": _INK_ON_LIGHT,
        }
    return {
        "bg": "#ffffff",
        "text": _INK_ON_LIGHT,
        "muted": _INK_ON_LIGHT,
        "box": "#f8fafc",
        "edge": "#64748b",
        "accent": _INK_ON_LIGHT,
        "arrow": _INK_ON_LIGHT,
        "box_ink": _INK_ON_LIGHT,
    }


# Pastel fills (always light) + black ink inside boxes only.
_SCHEMATIC_BOX_STYLES: dict[str, tuple[str, str, str]] = {
    "input": ("#dbeafe", "#1d4ed8", _INK_ON_LIGHT),
    "graph": ("#e0e7ff", "#4338ca", _INK_ON_LIGHT),
    "weights": ("#fef3c7", "#b45309", _INK_ON_LIGHT),
    "output": ("#dcfce7", "#15803d", _INK_ON_LIGHT),
}


def _wrap_schematic_lines(lines: list[str], max_chars: int = 24) -> list[str]:
    wrapped: list[str] = []

    def _flush_chunks(text: str) -> None:
        while len(text) > max_chars:
            wrapped.append(text[:max_chars])
            text = text[max_chars:]
        if text:
            wrapped.append(text)

    for raw in lines:
        text = str(raw)
        if len(text) <= max_chars:
            wrapped.append(text)
            continue
        if "_" in text:
            parts = text.split("_")
            chunk = parts[0]
            for part in parts[1:]:
                candidate = f"{chunk}_{part}"
                if len(candidate) <= max_chars:
                    chunk = candidate
                else:
                    _flush_chunks(chunk)
                    chunk = part
            _flush_chunks(chunk)
        else:
            _flush_chunks(text)
    return wrapped or [""]


def _schematic_box_dims(
    title: str,
    lines: list[str],
    *,
    min_w: float = 1.55,
    min_h: float = 1.35,
    max_chars: int = 24,
) -> tuple[float, float, list[str]]:
    body = _wrap_schematic_lines(lines, max_chars=max_chars)
    max_len = max(len(title), max((len(x) for x in body), default=0))
    w = max(min_w, 0.052 * max_len + 0.42)
    h = max(min_h, 0.38 + 0.21 * len(body) + 0.34)
    return w, h, body


def _schematic_text_layout(
    h_use: float,
    w_use: float,
    body: list[str],
    title: str,
) -> tuple[float, float, float, float, float, float]:
    """Padding + font sizes that fill fixed boxes without overlap (data or 0–1 axes)."""
    n_body = max(len(body), 1)
    normalized = h_use <= 1.5
    pad_top = h_use * (0.07 if normalized else 0.08)
    title_band = h_use * (0.17 if normalized else 0.18)
    pad_bottom = h_use * (0.06 if normalized else 0.07)
    body_region = max(h_use * 0.22, h_use - title_band - pad_bottom - pad_top * 0.25)
    line_gap = body_region / n_body

    if normalized:
        title_fs = min(16.5, max(11.0, h_use * 19.0))
        body_fs = min(14.5, max(9.5, line_gap * 50.0))
        width_budget = w_use * 34.0
    else:
        title_fs = min(17.5, max(11.5, h_use * 3.15))
        body_fs = min(15.5, max(10.0, line_gap * 25.5))
        width_budget = w_use * 6.2

    max_len = max(len(title), max((len(x) for x in body), default=0))
    if max_len > 0:
        width_cap = width_budget / max_len * 10.5
        body_fs = min(body_fs, width_cap)
        title_fs = min(title_fs, max(body_fs * 1.08, width_cap * 1.05))

    return pad_top, title_band, pad_bottom, title_fs, body_fs, line_gap


def _draw_schematic_label_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    style: str = "graph",
    max_chars: int | None = None,
    fixed_size: bool = False,
    preserve_text_size: bool = False,
) -> tuple[float, float, float, float]:
    """Rounded box with wrapped body lines; returns (x, y, w, h) actually used."""
    mc = max_chars or max(14, int(w * (32 if h <= 1.5 else 6.5)))
    w_auto, h_auto, body = _schematic_box_dims(title, lines, min_w=w, min_h=h, max_chars=mc)
    w_use = w if fixed_size else w_auto
    h_use = h if fixed_size else h_auto
    fc, ec, ink = _SCHEMATIC_BOX_STYLES.get(style, _SCHEMATIC_BOX_STYLES["graph"])
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w_use,
            h_use,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
    )
    text_h = h_use
    text_w = w_use
    pad_top, title_band, pad_bottom, title_fs, body_fs, _line_gap = _schematic_text_layout(
        text_h, text_w, body, title
    )
    if preserve_text_size:
        title_fs = min(title_fs * 1.22, 17.5)
        body_fs = min(body_fs * 1.22, 15.5)
    title_y = y + h_use - pad_top
    ax.text(x + w_use / 2, title_y, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color=ink)
    body_top = y + h_use - title_band
    body_bottom = y + pad_bottom
    n_body = len(body)
    if n_body == 1:
        ys = [(body_top + body_bottom) / 2]
    else:
        ys = [
            body_bottom + (body_top - body_bottom) * (i + 0.5) / n_body
            for i in range(n_body)
        ]
    for line, y_line in zip(body, ys):
        ax.text(
            x + w_use / 2,
            y_line,
            line,
            ha="center",
            va="center",
            fontsize=body_fs,
            color=ink,
        )
    return x, y, w_use, h_use


def _rect_anchor(x: float, y: float, w: float, h: float, side: str) -> tuple[float, float]:
    cy = y + h / 2
    if side == "right":
        return x + w, cy
    if side == "left":
        return x, cy
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    raise ValueError(f"unknown side: {side}")


def _flow_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
    lw: float = 1.65,
    mutation_scale: float = 13.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def _circle_anchor(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _draw_quadratic_arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    *,
    color: str,
    zorder: int = 5,
) -> None:
    """Quadratic Bézier arrow — reliable heads vs arc3,rad (matches inline SVG)."""
    path = MplPath([p0, control, p1], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.8,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def pyg_mpnn_markov_note_html(lang: Lang = "ru") -> str:
    """Honest MPNN vs Markov disclaimer for widget + manuscript."""
    ink = "color:#000000 !important;"
    if lang == "en":
        return (
            f'<p style="margin:8px 0 0;font-size:11px;line-height:1.58;{ink}">'
            "<b>Not a Markov chain.</b> Two directed edges are <b>not</b> transition probabilities "
            "<i>P(s′|s)</i> on discrete states. They are the standard PyG storage of one bond as "
            "<b>message passing</b> channels (i→j and j→i) in an MPNN / GNNOpt inference graph. "
            "A loose analogy: repeated GNN layers diffuse information along edges — but without "
            "a stochastic kernel or stationary distribution. For Methods cite ALIGNN/GNNOpt DOI, "
            "not Markov-chain textbooks.</p>"
        )
    return (
        f'<p style="margin:8px 0 0;font-size:11px;line-height:1.58;{ink}">'
        "<b>Это не цепь Маркова.</b> Два directed ребра — <b>не</b> вероятности перехода "
        "<i>P(s′|s)</i> между дискретными состояниями. В PyG одна связь Nb–O хранится как "
        "<b>два канала message passing</b> (i→j и j→i) в MPNN / GNNOpt. Слабая аналогия: "
        "несколько слоёв GNN «разносят» признаки по графу, но без стохастического ядра и "
        "стационарного распределения. В статье — DOI ALIGNN/GNNOpt, не учебник по Маркову.</p>"
    )


def normalize_gnn_plane(value: str | None, *, fallback: str = "xy") -> str:
    """Map dropdown value / label to xy | xz | yz."""
    if value in {"xy", "xz", "yz"}:
        return value
    s = str(value or "").lower().strip()
    for p in ("xy", "xz", "yz"):
        if s == p or s.startswith(p):
            return p
    return fallback


def plane_short_label(plane: str, lang: Lang = "ru") -> str:
    labels = {
        "xy": _l(lang, "xy — a×b", "xy — a×b"),
        "xz": _l(lang, "xz — a×c", "xz — a×c"),
        "yz": _l(lang, "yz — b×c", "yz — b×c"),
    }
    return labels.get(plane, labels["xy"])


def default_graph_plane(structure: Path, lang: Lang = "ru") -> str:
    from ase.io import read

    pos = read(str(structure)).get_positions()
    plane, _, _, _ = _pick_graph_plane(pos, lang)
    return plane


def plane_axis_labels(plane: str, lang: Lang = "ru") -> tuple[str, str]:
    labels = {
        "xy": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "y (crystal b)", "y (crystal b)")),
        "xz": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "z (crystal c)", "z (crystal c)")),
        "yz": (_l(lang, "y (crystal b)", "y (crystal b)"), _l(lang, "z (crystal c)", "z (crystal c)")),
    }
    return labels.get(plane, labels["xy"])


def _position_spans(pos: np.ndarray) -> tuple[float, float, float]:
    spans = [float(np.ptp(pos[:, k])) or 1.0 for k in range(3)]
    return spans[0], spans[1], spans[2]


def _pick_graph_plane(pos: np.ndarray, lang: Lang = "ru") -> tuple[str, str, str, float]:
    """Best 2D projection: avoid slab squashing (films often thin along z)."""
    sx, sy, sz = _position_spans(pos)
    spans = {"xy": (sx, sy), "xz": (sx, sz), "yz": (sy, sz)}
    plane = max(spans, key=lambda k: min(spans[k]))
    labels = {
        "xy": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "y (crystal b)", "y (crystal b)")),
        "xz": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "z (crystal c)", "z (crystal c)")),
        "yz": (_l(lang, "y (crystal b)", "y (crystal b)"), _l(lang, "z (crystal c)", "z (crystal c)")),
    }
    xlab, ylab = labels[plane]
    aspect = min(spans[plane]) / max(spans[plane])
    return plane, xlab, ylab, aspect


def _project_positions(pos: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
    if plane == "xz":
        return pos[:, 0], pos[:, 2]
    if plane == "yz":
        return pos[:, 1], pos[:, 2]
    return pos[:, 0], pos[:, 1]


def _spread_overlapping_uv(
    u: np.ndarray,
    v: np.ndarray,
    *,
    min_dist_frac: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fan out nearly coincident 2D nodes for display (GNN topology unchanged)."""
    u = np.asarray(u, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    span = max(float(np.ptp(u)), float(np.ptp(v)), 1e-6)
    min_d = min_dist_frac * span
    n = len(u)
    moved = False
    for _ in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = u[j] - u[i], v[j] - v[i]
                d = math.hypot(dx, dy)
                if d >= min_d:
                    continue
                moved = True
                if d > 1e-12:
                    push = (min_d - d) / 2
                    ux, uy = dx / d, dy / d
                    u[i] -= ux * push
                    u[j] += ux * push
                    v[i] -= uy * push
                    v[j] += uy * push
                else:
                    ang = 2 * math.pi * j / max(n, 1)
                    u[j] += min_d * math.cos(ang)
                    v[j] += min_d * math.sin(ang)
    return u, v, moved


def _is_thin_slab(pos: np.ndarray, *, ratio_thresh: float = 0.22) -> bool:
    sx, sy, sz = _position_spans(pos)
    return min(sx, sy, sz) / max(sx, sy, sz) < ratio_thresh


def _neighbor_figsize(n_atoms: int, aspect: float, *, combined: bool = False) -> tuple[float, float]:
    wide = 14.0 if n_atoms > 32 else 11.0
    if combined:
        tall = 5.2 if n_atoms <= 32 else 7.5
        return wide, tall + (3.0 if n_atoms > 80 else 1.5)
    if aspect < 0.25:
        return wide, max(7.0, min(14.0, 4.5 + n_atoms * 0.025))
    return wide, max(6.0, min(10.0, 4.8 + n_atoms * 0.012))


def _draw_neighbor_graph_on_ax(
    ax,
    atoms,
    *,
    plane: str,
    cutoff: float,
    theme: Theme,
    lang: Lang,
    show_labels: bool = True,
    max_edge_lines: int = 2500,
    ink_override: dict[str, str] | None = None,
    graph_context: Literal["panel", "thumb", "thumb_fill"] = "panel",
) -> int:
    """Draw neighbour graph; return undirected edge count."""
    from ase.neighborlist import neighbor_list

    c = {**_theme_colors(theme), **(ink_override or {})}
    pos = atoms.get_positions()
    u, v = _project_positions(pos, plane)
    spread_note = ""
    if graph_context in ("panel", "thumb_fill"):
        u, v, spread_moved = _spread_overlapping_uv(u, v)
        if spread_moved:
            spread_note = _l(
                lang,
                " · узлы с близкой проекцией разведены для наглядности",
                " · overlapping projections spread for clarity",
            )
    src, dst, _ = neighbor_list("ijS", a=atoms, cutoff=cutoff, self_interaction=False)
    n_undirected = _count_undirected_edges(src, dst)
    syms = atoms.get_chemical_symbols()
    edge_alpha = 0.55 if n_undirected < 600 else 0.28
    edge_lw = 0.75 if n_undirected < 600 else 0.35
    if graph_context == "thumb":
        edge_lw *= 0.85
    elif graph_context == "thumb_fill":
        edge_lw *= 1.05
    drawn = 0
    for i, j in zip(src, dst):
        if i < j:
            if drawn >= max_edge_lines:
                break
            ax.plot([u[i], u[j]], [v[i], v[j]], color=c["edge"], lw=edge_lw, alpha=edge_alpha, zorder=1)
            drawn += 1
    if graph_context == "thumb_fill":
        ms = 155 if len(atoms) <= 32 else (110 if len(atoms) <= 120 else 75)
        fs = 9 if len(atoms) <= 32 else 7
        pad = 0.045
        lw_node = 0.55
        clip_nodes = True
    elif graph_context == "thumb":
        ms = 48 if len(atoms) > 120 else 64
        fs = 5 if len(atoms) > 120 else 6
        pad = 0.26
        lw_node = 0.45
        clip_nodes = False
    else:
        ms = 360 if len(atoms) > 120 else 480
        fs = 10 if len(atoms) > 120 else 14
        pad = 0.14
        lw_node = 1.0
        clip_nodes = False
    label_cap = 32 if graph_context in ("thumb", "thumb_fill") else 48
    for i, sym in enumerate(syms):
        col = _ELEMENT_COLOR.get(sym, "#64748b")
        label = sym if show_labels and len(atoms) <= label_cap else None
        _draw_glass_node_2d(
            ax,
            float(u[i]),
            float(v[i]),
            label,
            col,
            ms=ms,
            fs=fs,
            lw_node=lw_node,
            clip_on=clip_nodes,
        )
    xr = float(np.ptp(u)) or 1.0
    yr = float(np.ptp(v)) or 1.0
    ax.set_xlim(float(u.min()) - xr * pad, float(u.max()) + xr * pad)
    ax.set_ylim(float(v.min()) - yr * pad, float(v.max()) + yr * pad)
    if graph_context == "thumb_fill":
        ax.set_aspect("equal", adjustable="box")
    elif _is_thin_slab(pos):
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax._gnn_spread_note = spread_note  # noqa: SLF001 — consumed by panel title helpers
    return n_undirected


def plot_structure_neighbor_graph(
    structure: Path | None = None,
    *,
    theme: Theme = "light",
    lang: Lang = "ru",
    plane: str | None = None,
    ax=None,
) -> plt.Figure:
    """2D projection of atoms + neighbour edges (GNNOpt cutoff 6 Å)."""
    from ase.io import read

    structure = structure or DEFAULT_STRUCTURE
    c = _theme_colors(theme)
    atoms = read(str(structure))
    pos = atoms.get_positions()
    plane_auto, xlab_auto, ylab_auto, aspect = _pick_graph_plane(pos, lang)
    if plane is None:
        plane, xlab, ylab = plane_auto, xlab_auto, ylab_auto
    else:
        labels = {
            "xy": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "y (crystal b)", "y (crystal b)")),
            "xz": (_l(lang, "x (crystal a)", "x (crystal a)"), _l(lang, "z (crystal c)", "z (crystal c)")),
            "yz": (_l(lang, "y (crystal b)", "y (crystal b)"), _l(lang, "z (crystal c)", "z (crystal c)")),
        }
        xlab, ylab = labels.get(plane, labels[plane_auto])
        sx, sy, sz = _position_spans(pos)
        spans = {"xy": (sx, sy), "xz": (sx, sz), "yz": (sy, sz)}
        aspect = min(spans[plane]) / max(spans[plane])

    if ax is None:
        fig, ax = plt.subplots(figsize=_neighbor_figsize(len(atoms), aspect), facecolor=c["bg"])
    else:
        fig = ax.figure

    n_undirected = _draw_neighbor_graph_on_ax(
        ax, atoms, plane=plane, cutoff=GNNOPT_CUTOFF_ANG, theme=theme, lang=lang
    )
    ax.set_xlabel(xlab, color=c["text"])
    ax.set_ylabel(ylab, color=c["text"])
    ax.tick_params(colors=c["text"])
    for spine in ax.spines.values():
        spine.set_color(c["edge"])
    ax.set_title(
        _l(
            lang,
            f"Граф соседства: {len(atoms)} узлов, {n_undirected} рёбер (r<{GNNOPT_CUTOFF_ANG} Å, {plane})"
            + getattr(ax, "_gnn_spread_note", ""),
            f"Neighbour graph: {len(atoms)} nodes, {n_undirected} edges (r<{GNNOPT_CUTOFF_ANG} Å, {plane})"
            + getattr(ax, "_gnn_spread_note", ""),
        ),
        color=c["text"],
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# Axonometric 3D view (extra figure G) — dimetric-style: z kept true scale, a/b foreshortened equally.
_AXON_ELEV_DEG = 28.0
_AXON_AZIM_DEG = -52.0


def _neighbor_figsize_3d(n_atoms: int) -> tuple[float, float]:
    side = 12.0 if n_atoms > 32 else 10.0
    return side, side * 0.88


def _style_3d_axon_panes(ax, c: dict[str, str], *, pane_alpha: float = 0.14) -> None:
    """Semi-transparent 3D panes (visible frame, not solid walls)."""
    pane_rgb = (0.55, 0.62, 0.78, pane_alpha) if c["bg"] != "#ffffff" else (0.72, 0.78, 0.90, pane_alpha)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_rgb)
        axis.pane.set_edgecolor(c["accent"])
        axis.pane.set_alpha(pane_alpha)
        axis._axinfo["grid"]["color"] = (c["edge"], 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.55


def _draw_3d_dimension_frame(
    ax,
    hx: float,
    hy: float,
    hz: float,
    spans: tuple[float, float, float],
    *,
    lang: Lang,
    c: dict[str, str],
) -> None:
    """Wire bounding box + a/b/c dimension lines outside the cell (Å)."""
    sx, sy, sz = spans
    corners = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    frame_col = c["accent"]
    for i, j in edges:
        p, q = corners[i], corners[j]
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=frame_col,
            alpha=0.42,
            lw=1.15,
            ls="--",
            zorder=0,
        )
    dim_col = c["text"]
    la = _l(lang, "a", "a")
    lb = _l(lang, "b", "b")
    lc = _l(lang, "c", "c")
    # a — along x, front-bottom
    ya, za = -hy * 1.16, -hz * 1.04
    ax.plot([-hx, hx], [ya, ya], [za, za], color=dim_col, alpha=0.82, lw=1.35, zorder=0)
    for x in (-hx, hx):
        ax.plot([x, x], [ya, hy * 0.06 - hy], [za, za], color=dim_col, alpha=0.65, lw=1.0, zorder=0)
    ax.text(0, ya, za, f"{la} ≈ {sx:.1f} Å", color=dim_col, fontsize=8.5, ha="center", va="top")
    # b — along y, left-bottom
    xa, za = -hx * 1.14, -hz * 1.04
    ax.plot([xa, xa], [-hy, hy], [za, za], color=dim_col, alpha=0.82, lw=1.35, zorder=0)
    for y in (-hy, hy):
        ax.plot([xa, hx * 0.06 - hx], [y, y], [za, za], color=dim_col, alpha=0.65, lw=1.0, zorder=0)
    ax.text(xa, 0, za, f"{lb} ≈ {sy:.1f} Å", color=dim_col, fontsize=8.5, ha="right", va="center")
    # c — vertical at front-left
    xa, ya = -hx * 1.10, -hy * 1.10
    ax.plot([xa, xa], [ya, ya], [-hz, hz], color=dim_col, alpha=0.82, lw=1.35, zorder=0)
    for z in (-hz, hz):
        ax.plot([xa, hx * 0.06 - hx], [ya, ya], [z, z], color=dim_col, alpha=0.65, lw=1.0, zorder=0)
    ax.text(xa, ya, 0, f"{lc} ≈ {sz:.1f} Å", color=dim_col, fontsize=8.5, ha="right", va="center")


def _draw_neighbor_graph_3d_on_ax(
    ax,
    atoms,
    *,
    cutoff: float,
    theme: Theme,
    lang: Lang,
    show_labels: bool = True,
    max_edge_lines: int = 2500,
    elev: float = _AXON_ELEV_DEG,
    azim: float = _AXON_AZIM_DEG,
    visual_scale: float = 1.38,
    node_marker_scale: float = 1.0,
    draw_dim_frame: bool = True,
    show_axis_labels: bool = True,
) -> tuple[int, tuple[float, float, float]]:
    """3D neighbour graph with axonometric camera — complements 2D auto-plane plot."""
    from ase.neighborlist import neighbor_list

    c = _theme_colors(theme)
    pos = atoms.get_positions().astype(float)
    pos = pos - pos.mean(axis=0)
    src, dst, _ = neighbor_list("ijS", a=atoms, cutoff=cutoff, self_interaction=False)
    n_undirected = _count_undirected_edges(src, dst)
    syms = atoms.get_chemical_symbols()
    edge_alpha = 0.72 if n_undirected < 600 else 0.48
    edge_lw = (0.78 if n_undirected < 600 else 0.42) * visual_scale
    edge_rgb = "#cbd5e1" if theme == "dark" else "#475569"
    drawn = 0
    for i, j in zip(src, dst):
        if i < j:
            if drawn >= max_edge_lines:
                break
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [pos[i, 2], pos[j, 2]],
                color=edge_rgb,
                lw=edge_lw,
                alpha=edge_alpha,
                zorder=1,
            )
            drawn += 1
    ms_base = 52 if len(atoms) > 120 else 80
    ms = ms_base * visual_scale * node_marker_scale
    ec_lw = 0.65 * visual_scale
    label_fs = max(10.0, min(15.0, math.sqrt(max(ms, 1.0)) * 0.42))
    sx, sy, sz = _position_spans(pos)
    spans = (sx, sy, sz)
    ax.view_init(elev=elev, azim=azim)
    if show_labels and len(atoms) <= 48:
        _draw_glass_nodes_3d(
            ax,
            pos,
            syms,
            ms=ms,
            label_fs=label_fs,
            ec_lw=ec_lw,
            show_labels=True,
            elev=elev,
            azim=azim,
            spans=spans,
        )
    else:
        _draw_glass_nodes_3d(
            ax,
            pos,
            syms,
            ms=ms,
            label_fs=label_fs,
            ec_lw=ec_lw,
            show_labels=False,
            elev=elev,
            azim=azim,
            spans=spans,
        )
    z_vis = max(sz, 0.14 * max(sx, sy, sz))
    try:
        ax.set_box_aspect((sx, sy, z_vis))
    except Exception:
        pass
    pad = 0.14
    hx, hy, hz = sx / 2 * (1 + pad), sy / 2 * (1 + pad), z_vis / 2 * (1 + pad)
    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)
    ax.set_zlim(-hz, hz)
    if draw_dim_frame:
        _draw_3d_dimension_frame(ax, hx, hy, hz, spans, lang=lang, c=c)
    if show_axis_labels:
        ax.set_xlabel(_l(lang, "a (Å)", "a (Å)"), color=c["text"], labelpad=8)
        ax.set_ylabel(_l(lang, "b (Å)", "b (Å)"), color=c["text"], labelpad=8)
        ax.set_zlabel(_l(lang, "c (Å)", "c (Å)"), color=c["text"], labelpad=8)
        ax.tick_params(colors=c["text"], labelsize=7.5)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(length=0)
    _style_3d_axon_panes(ax, c, pane_alpha=0.13)
    ax.grid(True, color=c["edge"], alpha=0.32, linewidth=0.5)
    return n_undirected, spans


def _gnn_io_input_lines(structure: Path, atoms, *, lang: Lang) -> list[str]:
    n_atoms = len(atoms)
    struct_name = structure.name
    is_bulk = n_atoms <= 32 and not _is_thin_slab(atoms.get_positions())
    in_lines = [
        struct_name,
        f"{n_atoms} " + _l(lang, "узлов (атомы)", "nodes (atoms)"),
        _l(lang, "pos (Å), Z, species", "pos (Å), Z, species"),
        _l(lang, "cell / PBC", "cell / PBC"),
        _l(
            lang,
            f"dist < {GNNOPT_CUTOFF_ANG} Å → рёбра",
            f"dist < {GNNOPT_CUTOFF_ANG} Å → edges",
        ),
        _l(lang, "→ DGL / PyG graph", "→ DGL / PyG graph"),
    ]
    if is_bulk:
        in_lines.insert(2, _l(lang, "16 at. bulk", "16 at. bulk"))
    return in_lines


def _prepare_gnn_io_side_axes(ax_in, ax_out, *, c: dict[str, str]) -> None:
    for ax_side in (ax_in, ax_out):
        ax_side.set_facecolor(c["bg"])
        ax_side.set_xlim(0, 1)
        ax_side.set_ylim(0, 1)
        ax_side.axis("off")


def _draw_gnn_io_input_panel(
    ax_in,
    *,
    in_lines: list[str],
    lang: Lang,
    c: dict[str, str],
    box_y: float = 0.23,
    box_h: float = 0.54,
    arrow_y: float = 0.06,
) -> None:
    _draw_schematic_label_box(
        ax_in,
        _GNN_IO_SIDE_BOX_X,
        box_y,
        _GNN_IO_SIDE_BOX_W,
        box_h,
        _l(lang, "ВХОД", "INPUT"),
        in_lines,
        style="input",
        max_chars=22,
        fixed_size=True,
        preserve_text_size=True,
    )
    ax_in.text(
        0.5,
        arrow_y,
        "→",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=c["arrow"],
    )


def _draw_gnn_io_output_panel(
    ax_out,
    *,
    summary: dict[str, Any],
    lang: Lang,
    c: dict[str, str],
    alignn_y: float = 0.58,
    alignn_h: float = 0.30,
    gnnopt_y: float = 0.10,
    gnnopt_h: float = 0.30,
    arrow_x: float = 0.04,
    arrow_y: float = 0.48,
) -> None:
    a = summary["alignn"]
    g = summary["gnnopt"]
    n_gnn = g.get("n_edges_undirected", g.get("n_edges", "—"))
    _draw_schematic_label_box(
        ax_out,
        _GNN_IO_SIDE_BOX_X,
        alignn_y,
        _GNN_IO_SIDE_BOX_W,
        alignn_h,
        _l(lang, "ВЫХОД ALIGNN", "OUTPUT ALIGNN"),
        [
            _l(lang, "Eg (eV)", "Eg (eV)"),
            "OptB88 / mBJ",
            _l(lang, "1 scalar · A / A′", "1 scalar · A / A′"),
            f"line graph {a['n_edges_line_graph']}",
        ],
        style="output",
        max_chars=20,
        fixed_size=True,
        preserve_text_size=True,
    )
    _draw_schematic_label_box(
        ax_out,
        _GNN_IO_SIDE_BOX_X,
        gnnopt_y,
        _GNN_IO_SIDE_BOX_W,
        gnnopt_h,
        _l(lang, "ВЫХОД GNNOpt", "OUTPUT GNNOpt"),
        [
            "Tr Re/Im ε(ω)",
            "251 pt, 0–50 eV",
            _l(lang, "спектр · B", "spectrum · B"),
            f"{n_gnn} " + _l(lang, "рёбер", "edges"),
        ],
        style="output",
        max_chars=20,
        fixed_size=True,
        preserve_text_size=True,
    )
    ax_out.text(
        arrow_x,
        arrow_y,
        "←",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=c["arrow"],
    )


def plot_gnn_axonometric_io_panel(
    structure: Path | None = None,
    *,
    theme: Theme = "light",
    lang: Lang = "ru",
    summary: dict[str, Any] | None = None,
    save_png: bool = True,
) -> plt.Figure:
    """Standalone 3D axonometry + left INPUT / right OUTPUT panels (separate Save/Copy in notebook)."""
    from ase.io import read
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    structure = structure or DEFAULT_STRUCTURE
    c = _theme_colors(theme)
    atoms = read(str(structure))
    n_atoms = len(atoms)
    if summary is None:
        summary = build_pipeline_summary(structure)

    fw = 20.0 if n_atoms > 32 else 17.0
    fig = plt.figure(figsize=(fw, 11.5), facecolor=c["bg"])
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.76, 3.0, 0.76], wspace=0.05)
    ax_in = fig.add_subplot(gs[0])
    ax_3d = fig.add_subplot(gs[1], projection="3d")
    ax_out = fig.add_subplot(gs[2])
    _prepare_gnn_io_side_axes(ax_in, ax_out, c=c)
    in_lines = _gnn_io_input_lines(structure, atoms, lang=lang)
    _draw_gnn_io_input_panel(ax_in, in_lines=in_lines, lang=lang, c=c)

    ax_3d.set_facecolor(c["bg"])
    n_undirected, _spans = _draw_neighbor_graph_3d_on_ax(
        ax_3d,
        atoms,
        cutoff=GNNOPT_CUTOFF_ANG,
        theme=theme,
        lang=lang,
        show_labels=True,
        visual_scale=1.48 if n_atoms <= 32 else 1.32,
        node_marker_scale=GNN_3D_NODE_MARKER_SCALE,
        draw_dim_frame=False,
        show_axis_labels=False,
    )
    slab = _is_thin_slab(atoms.get_positions())
    slab_note = _l(lang, " · тонкий slab", " · thin slab") if slab else ""
    ax_3d.set_title(
        _l(
            lang,
            f"3D аксонометрия · {n_atoms} узлов · {n_undirected} рёбер{slab_note}",
            f"3D axonometric · {n_atoms} nodes · {n_undirected} edges{slab_note}",
        ),
        color=c["text"],
        fontsize=11.5,
        pad=12,
    )

    _draw_gnn_io_output_panel(ax_out, summary=summary, lang=lang, c=c)

    fig.suptitle(
        _l(
            lang,
            "Inference only — frozen ALIGNN / GNNOpt (не обучение на вашей ячейке)",
            "Inference only — frozen ALIGNN / GNNOpt (not trained on your cell)",
        ),
        fontsize=12.5,
        fontweight="bold",
        color=c["text"],
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)
    if save_png:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = structure.stem.replace(".", "_")
        png_path = OUT_DIR / f"ml_gnn_axonometric_io_{tag}.png"
        fig.savefig(png_path, dpi=165, bbox_inches="tight", facecolor=c["bg"])
        fig._ml_axonometric_io_png = str(png_path)  # type: ignore[attr-defined]
    return fig


def plot_structure_neighbor_graph_3d_axonometric(
    structure: Path | None = None,
    *,
    theme: Theme = "light",
    lang: Lang = "ru",
) -> plt.Figure:
    """Backward-compatible alias → standalone IO axonometry panel."""
    return plot_gnn_axonometric_io_panel(structure, theme=theme, lang=lang, save_png=False)


def gnn_axonometric_io_save_hint_html(structure: Path, lang: Lang = "ru", *, theme: Theme = "dark") -> str:
    tag = structure.stem.replace(".", "_")
    rel = f"dielectric_compare/ml_gnn_axonometric_io_{tag}.png"
    ink = _fig_ink(theme)
    if lang == "en":
        return (
            f'<p style="margin:6px 0 0;font-size:10.5px;color:{ink} !important;">'
            f"Axonometry saved separately: <code>{rel}</code> — «Save PNG» / «Copy PNG» above the figure.</p>"
        )
    return (
        f'<p style="margin:6px 0 0;font-size:10.5px;color:{ink} !important;">'
        f"Аксонометрия сохранена отдельно: <code>{rel}</code> — «Сохранить PNG» / «Скопировать PNG» над рисунком.</p>"
    )


def gnn_graph_3d_axonometric_caption_html(lang: Lang = "ru") -> str:
    ink = "color:#000000 !important;"
    if lang == "en":
        return (
            f'<p style="margin:14px 0 6px;font-size:12px;font-weight:700;{ink}">'
            "3D axonometric — separate figure (INPUT | graph | OUTPUT)</p>"
            f'<p style="margin:0 0 8px;font-size:11px;line-height:1.55;{ink}">'
            "Atoms and bonds boosted for contrast (no dimension lines on the 3D view). "
            "Left: what enters the graph (pos, Z, cutoff); right: ALIGNN scalar Eg (A/A′) and GNNOpt ε(ω) spectrum (B). "
            "Pedagogical SI view — not the exact tensor layout inside DGL/PyG.</p>"
        )
    return (
        f'<p style="margin:14px 0 6px;font-size:12px;font-weight:700;{ink}">'
        "3D аксонометрия — отдельный рисунок (ВХОД | граф | ВЫХОД)</p>"
        f'<p style="margin:0 0 8px;font-size:11px;line-height:1.55;{ink}">'
        "Атомы и связи контрастнее (без размерных линий на 3D-виде). "
        "Слева — что подаётся на граф (pos, Z, cutoff); справа — скаляр Eg ALIGNN (A/A′) и спектр GNNOpt ε(ω) (B). "
        "Педагогика для SI — не точная раскладка тензора в DGL/PyG.</p>"
    )


def build_pipeline_schematic_legend_html(summary: dict[str, Any], lang: Lang = "ru") -> str:
    """Line-by-line legend for pipeline schematic boxes (display after figure G)."""
    a = summary["alignn"]
    g = summary["gnnopt"]
    struct_name = Path(summary["structure"]).name
    n_gnn = g.get("n_edges_undirected", g.get("n_edges", "—"))

    if lang == "en":
        rows = [
            ("Input · structure file", struct_name, "Your crystal file (ASE read): path to nboi2.xyz or film extxyz."),
            ("", f"{a['n_atoms']} at.", "Number of atoms = graph nodes."),
            ("ALIGNN · graph", "DGL atom + line graph", "Neighbour graph in DGL: atoms → nodes; bonds → edges; line graph encodes bond angles."),
            ("", f"r<{a['cutoff_ang']} Å", f"Cutoff radius {a['cutoff_ang']} Å, max {a['max_neighbors']} neighbours per atom."),
            (
                "Atomistic Line Graph Neural Network (ALIGNN)",
                "jv_optb88vdw_bandgap_alignn",
                "Pretrained JARVIS weights for band gap (OptB88vdW training labels) — button A).",
            ),
            (
                "",
                "jv_mbj_bandgap_alignn · Figshare",
                "TB-mBJ variant — button A′); downloaded from Figshare; weights frozen (not trained here).",
            ),
            ("GNNOpt · graph", "PyG neighbour edges", "PyTorch Geometric graph: nodes = atoms, undirected edges within cutoff."),
            ("", f"r<{g['cutoff_ang']} Å", f"Cutoff {g['cutoff_ang']} Å; {n_gnn} unique edges on this structure."),
            (
                "Graph Neural Network Optical Properties (GNNOpt)",
                "model_eps1_240406.torch",
                "Pretrained optical dielectric head (Re ε₁) — local models/gnnopt/.",
            ),
            ("", "model_eps2 · models/gnnopt", "Companion head for Im ε₂; frozen inference only — button B)."),
            ("ALIGNN output", "Scalar Eg (eV)", "Single number: predicted band gap from graph embedding."),
            ("", "A / A′", "Widget buttons: OptB88 (A) or mBJ (A′) checkpoint."),
            ("GNNOpt output", "Tr Re/Im ε(ω)", "Trace of dielectric tensor vs photon energy (251 pts, 0–50 eV)."),
            ("", "B", "Widget button B — compare Tr(Re ε) to HSE06 CSV in View Docx."),
        ]
        hdr = "Pipeline schematic — what each line in the boxes means"
    else:
        rows = [
            ("Вход · файл структуры", struct_name, "Файл кристалла (ASE): nboi2.xyz или extxyz плёнки."),
            ("", f"{a['n_atoms']} ат.", "Число атомов = узлов графа."),
            ("ALIGNN · граф", "DGL atom + line graph", "Граф соседства в DGL: атомы → узлы; связи → рёбра; line graph кодирует углы связей."),
            ("", f"r<{a['cutoff_ang']} Å", f"Радиус cutoff {a['cutoff_ang']} Å, max {a['max_neighbors']} соседей на атом."),
            (
                "Atomistic Line Graph Neural Network (ALIGNN)",
                "jv_optb88vdw_bandgap_alignn",
                "Готовые веса JARVIS для Eg (метки OptB88vdW) — кнопка A).",
            ),
            (
                "",
                "jv_mbj_bandgap_alignn · Figshare",
                "Вариант TB-mBJ — кнопка A′); скачано с Figshare; веса frozen (здесь не обучаются).",
            ),
            ("GNNOpt · граф", "PyG рёбра соседства", "Граф PyTorch Geometric: узлы = атомы, рёбра в радиусе cutoff."),
            ("", f"r<{g['cutoff_ang']} Å", f"Cutoff {g['cutoff_ang']} Å; {n_gnn} уник. рёбер на этой структуре."),
            (
                "Graph Neural Network Optical Properties (GNNOpt)",
                "model_eps1_240406.torch",
                "Готовая голова Re ε₁(ω) — каталог models/gnnopt/.",
            ),
            ("", "model_eps2 · models/gnnopt", "Парная голова Im ε₂; только inference — кнопка B)."),
            ("Выход ALIGNN", "Скаляр Eg (eV)", "Одно число: предсказанная ширина зоны по embedding графа."),
            ("", "A / A′", "Кнопки виджета: чекпоинт OptB88 (A) или mBJ (A′)."),
            ("Выход GNNOpt", "Tr Re/Im ε(ω)", "След тензора ε по энергии фотона (251 точка, 0–50 eV)."),
            ("", "B", "Кнопка B — сравнение Tr(Re ε) с HSE06 CSV (View Docx)."),
        ]
        hdr = "Схема pipeline — что означает каждая строка в боксах"

    lines: list[str] = []
    last_box = ""
    for box_title, inner, meaning in rows:
        label = box_title if box_title else last_box
        if box_title:
            last_box = box_title
        lines.append(
            f"<p style='margin:0 0 4px 0;line-height:1.55'>"
            f"<b>{label}</b> · <code style='font-size:11px'>{inner}</code><br>"
            f"<span style='color:#000000'>{meaning}</span></p>"
        )

    return (
        f"<div class='pale-callout conc-i18n-section compare-ink-light' "
        f"style='margin:10px 0 0;padding:10px 12px;border:1px solid #6366f1;border-radius:6px;"
        f"background:#f8fafc;font-size:12px;color:#000000'>"
        f"<p style='margin:0 0 8px 0;font-weight:700'>{hdr}</p>"
        f"{''.join(lines)}</div>"
    )


def plot_ml_gnn_pipeline_schematic(
    structure: Path | None = None,
    *,
    theme: Theme = "light",
    lang: Lang = "ru",
    ax=None,
    summary: dict[str, Any] | None = None,
    plane: str | None = None,
    save_png: bool = True,
) -> plt.Figure:
    """Standalone IO flowchart (save separately from neighbour-graph figure G)."""
    from ase.io import read

    structure = structure or DEFAULT_STRUCTURE
    atoms = read(str(structure))
    pos = atoms.get_positions()
    if plane is None:
        plane, _, _, _ = _pick_graph_plane(pos, lang)
    else:
        plane = normalize_gnn_plane(plane)

    if summary is None:
        try:
            summary = build_pipeline_summary(structure)
            a = summary["alignn"]
            g = summary["gnnopt"]
        except Exception:
            a = {"n_atoms": len(atoms), "n_edges_atom_graph": 224, "n_edges_line_graph": 3096, "cutoff_ang": 8.0, "max_neighbors": 12}
            g = {"n_atoms": len(atoms), "n_edges": 472, "cutoff_ang": 6.0}
    else:
        a = summary["alignn"]
        g = summary["gnnopt"]

    c = _theme_colors(theme)
    x_max, y_max = 22.0, 13.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(20.0, 11.0), facecolor=c["bg"])
    else:
        fig = ax.figure
    ax.set_facecolor(c["bg"])
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.axis("off")

    def graph_thumb(x: float, y: float, w: float, h: float, title: str, cutoff: float) -> tuple[float, float, float, float]:
        fc, ec, ink = _SCHEMATIC_BOX_STYLES["graph"]
        title_frac = 0.10
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=fc, edgecolor=ec, linewidth=1.8,
            )
        )
        title_fs = min(16.5, max(12.5, h * 2.55))
        ax.text(
            x + w / 2, y + h - h * 0.04, title,
            ha="center", va="top", fontsize=title_fs, fontweight="bold", color=ink,
        )
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xs, ys = x1 - x0, y1 - y0
        mx, my = 0.003, 0.005
        iw_full = w / xs - 2 * mx
        ih_full = (h / ys) * (1.0 - title_frac) - 2 * my
        sh = _GNN_THUMB_INSET_SHRINK
        iw = iw_full * (1.0 - 2 * sh)
        ih = ih_full * (1.0 - 2 * sh)
        ix = (x - x0) / xs + mx + iw_full * sh
        iy = (y - y0) / ys + my + ih_full * sh
        inset = ax.inset_axes([ix, iy, iw, ih])
        inset.set_facecolor(fc)
        inset.set_clip_on(True)
        _draw_neighbor_graph_on_ax(
            inset, atoms, plane=plane, cutoff=cutoff, theme=theme, lang=lang,
            show_labels=len(atoms) <= 32, max_edge_lines=1200,
            ink_override={"text": ink, "edge": "#64748b"},
            graph_context="thumb_fill",
        )
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)
        return x, y, w, h

    inp = structure.name
    is_bulk = len(atoms) <= 32 and not _is_thin_slab(pos)
    in_lines = [
        inp,
        _l(lang, "positions (Å)", "positions (Å)"),
        _l(lang, "cell / PBC", "cell / PBC"),
        _l(lang, "Z, species", "Z, species"),
    ]
    if is_bulk:
        in_lines.append(_l(lang, "16 at. bulk", "16 at. bulk"))
    else:
        in_lines.append(_l(lang, f"{len(atoms)} at. film", f"{len(atoms)} at. film"))

    graph_w, graph_h = 5.6, 5.1
    mid_y = y_max * 0.52
    row_gap = 0.55
    algn_y = mid_y + graph_h / 2 + row_gap / 2
    gnn_y = mid_y - graph_h / 2 - row_gap / 2

    inp_box = _draw_schematic_label_box(
        ax, 0.35, mid_y - 2.0, 3.33, 4.0,
        _l(lang, "ВХОД", "INPUT"),
        in_lines,
        style="input",
        max_chars=22,
        fixed_size=True,
    )
    algn_graph = graph_thumb(4.0, algn_y - graph_h / 2, graph_w, graph_h, f"ALIGNN · r<{a['cutoff_ang']} Å", float(a["cutoff_ang"]))
    gnn_graph = graph_thumb(4.0, gnn_y - graph_h / 2, graph_w, graph_h, f"GNNOpt · r<{g['cutoff_ang']} Å", float(g["cutoff_ang"]))

    algn_w = _draw_schematic_label_box(
        ax, 10.3, algn_y - 2.1, 5.4, 4.2,
        "ALIGNN weights",
        [
            "jv_optb88vdw_",
            "bandgap_alignn",
            "jv_mbj_bandgap_",
            "alignn",
            _l(lang, "frozen · Figshare", "frozen · Figshare"),
        ],
        style="weights",
        max_chars=22,
        fixed_size=True,
    )
    gnn_w = _draw_schematic_label_box(
        ax, 10.3, gnn_y - 2.1, 5.4, 4.2,
        "GNNOpt weights",
        [
            "model_eps1_240406.torch",
            "model_eps2",
            _l(lang, "models/gnnopt", "models/gnnopt"),
        ],
        style="weights",
        max_chars=22,
        fixed_size=True,
    )
    out_a = _draw_schematic_label_box(
        ax, 16.4, algn_y - 1.95, 4.78, 3.9,
        _l(lang, "ВЫХОД A/A′", "OUTPUT A/A′"),
        [_l(lang, "Eg (eV)", "Eg (eV)"), "OptB88 / mBJ", _l(lang, "1 scalar", "1 scalar")],
        style="output",
        fixed_size=True,
    )
    out_b = _draw_schematic_label_box(
        ax, 16.4, gnn_y - 1.95, 4.78, 3.9,
        _l(lang, "ВЫХОД B", "OUTPUT B"),
        ["Tr Re/Im ε(ω)", "251 pt, 0–50 eV", _l(lang, "спектр", "spectrum")],
        style="output",
        fixed_size=True,
    )

    accent = c["arrow"]
    _flow_arrow(ax, _rect_anchor(*inp_box, "right"), _rect_anchor(*algn_graph, "left"), color=accent, rad=0.22, mutation_scale=18)
    _flow_arrow(ax, _rect_anchor(*inp_box, "right"), _rect_anchor(*gnn_graph, "left"), color=accent, rad=-0.22, mutation_scale=18)
    _flow_arrow(ax, _rect_anchor(*algn_graph, "right"), _rect_anchor(*algn_w, "left"), color=accent, rad=0.06, mutation_scale=16)
    _flow_arrow(ax, _rect_anchor(*gnn_graph, "right"), _rect_anchor(*gnn_w, "left"), color=accent, rad=-0.06, mutation_scale=16)
    _flow_arrow(ax, _rect_anchor(*algn_w, "right"), _rect_anchor(*out_a, "left"), color=accent, rad=0.06, mutation_scale=16)
    _flow_arrow(ax, _rect_anchor(*gnn_w, "right"), _rect_anchor(*out_b, "left"), color=accent, rad=-0.06, mutation_scale=16)

    ax.text(
        x_max / 2, y_max - 0.55,
        _l(lang, "Inference only — сеть НЕ обучается на вашей структуре", "Inference only — network NOT trained on your structure"),
        ha="center", fontsize=13.5, fontweight="bold", color=c["text"],
    )
    detail = _l(
        lang,
        f"ALIGNN: line graph {a['n_edges_line_graph']} узлов  |  GNNOpt: {g.get('n_edges_undirected', g.get('n_edges', '—'))} рёбер  |  "
        f"мини-графы: {plane_short_label(normalize_gnn_plane(plane))}",
        f"ALIGNN: line graph {a['n_edges_line_graph']} nodes  |  GNNOpt: {g.get('n_edges_undirected', g.get('n_edges', '—'))} edges  |  "
        f"thumb plane: {plane_short_label(normalize_gnn_plane(plane), 'en')}",
    )
    ax.text(x_max / 2, 0.35, detail, ha="center", fontsize=10.0, color=c["text"])
    if ax is None:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
        if save_png:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            tag = structure.stem.replace(".", "_")
            png_path = OUT_DIR / f"ml_gnn_pipeline_schematic_{tag}.png"
            fig.savefig(png_path, dpi=165, bbox_inches="tight", facecolor=c["bg"])
            fig._ml_pipeline_schematic_png = str(png_path)  # type: ignore[attr-defined]
    return fig


def ml_pipeline_schematic_save_hint_html(structure: Path, lang: Lang = "ru", *, theme: Theme = "dark") -> str:
    tag = structure.stem.replace(".", "_")
    rel = f"dielectric_compare/ml_gnn_pipeline_schematic_{tag}.png"
    ink = _fig_ink(theme)
    if lang == "en":
        return (
            f'<p style="margin:6px 0 0;font-size:10.5px;color:{ink} !important;">'
            f"Pipeline schematic saved separately: <code>{rel}</code> — use «Save PNG» / «Copy PNG» above the figure.</p>"
        )
    return (
        f'<p style="margin:6px 0 0;font-size:10.5px;color:{ink} !important;">'
        f"Схема pipeline сохранена отдельно: <code>{rel}</code> — кнопки «Сохранить PNG» / «Скопировать PNG» над рисунком.</p>"
    )


def plot_gnn_combined_panel(
    structure: Path | None = None,
    *,
    theme: Theme = "light",
    lang: Lang = "ru",
    summary: dict[str, Any] | None = None,
    plane: str | None = None,
    view_mode: Literal["2d", "3d"] = "2d",
) -> plt.Figure:
    """2D neighbour graph with INPUT | graph | OUTPUT side panels."""
    from ase.io import read
    from matplotlib.gridspec import GridSpec

    structure = structure or DEFAULT_STRUCTURE
    atoms = read(str(structure))
    if summary is None:
        summary = build_pipeline_summary(structure)
    pos = atoms.get_positions()
    plane_auto, _, _, aspect = _pick_graph_plane(pos, lang)
    plane_use = normalize_gnn_plane(plane, fallback=plane_auto)
    xlab, ylab = plane_axis_labels(plane_use, lang)
    n_atoms = len(atoms)
    fw, fh = _neighbor_figsize(n_atoms, aspect, combined=False)

    c = _theme_colors(theme)
    fig = plt.figure(figsize=(fw * 1.35, fh), facecolor=c["bg"])
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.76, 3.0, 0.76], wspace=0.05)
    ax_in = fig.add_subplot(gs[0])
    ax_graph = fig.add_subplot(gs[1])
    ax_out = fig.add_subplot(gs[2])
    _prepare_gnn_io_side_axes(ax_in, ax_out, c=c)
    in_lines = _gnn_io_input_lines(structure, atoms, lang=lang)
    _draw_gnn_io_input_panel(
        ax_in,
        in_lines=in_lines,
        lang=lang,
        c=c,
        box_y=0.08,
        box_h=0.84,
        arrow_y=0.03,
    )
    _draw_gnn_io_output_panel(
        ax_out,
        summary=summary,
        lang=lang,
        c=c,
        alignn_y=0.52,
        alignn_h=0.40,
        gnnopt_y=0.08,
        gnnopt_h=0.40,
    )

    ax_graph.set_facecolor(c["bg"])
    _draw_neighbor_graph_on_ax(
        ax_graph,
        atoms,
        plane=plane_use,
        cutoff=GNNOPT_CUTOFF_ANG,
        theme=theme,
        lang=lang,
        show_labels=n_atoms <= 48,
    )
    ax_graph.set_xlabel(xlab, color=c["text"])
    ax_graph.set_ylabel(ylab, color=c["text"])
    ax_graph.tick_params(colors=c["text"])
    for spine in ax_graph.spines.values():
        spine.set_color(c["edge"])
    ax_graph.set_title(
        _l(
            lang,
            f"▼ 2D граф соседства · {n_atoms} узлов · {plane_short_label(plane_use, lang)}"
            + getattr(ax_graph, "_gnn_spread_note", ""),
            f"▼ 2D neighbour graph · {n_atoms} nodes · {plane_short_label(plane_use, lang)}"
            + getattr(ax_graph, "_gnn_spread_note", ""),
        ),
        color=c["text"],
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.suptitle(
        _l(
            lang,
            "Inference only — frozen ALIGNN / GNNOpt (не обучение на вашей ячейке)",
            "Inference only — frozen ALIGNN / GNNOpt (not trained on your cell)",
        ),
        fontsize=12.5,
        fontweight="bold",
        color=c["text"],
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)
    return fig


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("structure", nargs="?", type=Path, default=DEFAULT_STRUCTURE)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument(
        "--export-teaching",
        action="store_true",
        help="Save §2b teaching PNGs (line graph, PyG directed, training-vs-viz table) to generated_films_GPT_38/",
    )
    args = ap.parse_args()
    if args.export_teaching:
        paths = save_gnn_teaching_schematics_png()
        for p in paths:
            print(p)
    else:
        s = build_pipeline_summary(args.structure)
        print(json.dumps(s, indent=2))
        fig = plot_ml_gnn_pipeline_schematic(args.structure)
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
        else:
            plt.show()

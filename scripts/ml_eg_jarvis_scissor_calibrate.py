#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JARVIS-DFT class calibration for Eg_scissor Supplementary trend (no VASP).

Workflow (future paper — cheap ML vs expensive HSE06):
  Phase A: N≥3 JARVIS entries with paired OptB88vdW + TB-mBJ on the *same* cell.
           ⟨δ⟩ = mean(Eg_mBJ − Eg_OptB88) on oxyhalide / NbO*₂-like class.
  Phase B: Run ALIGNN mBJ on target (nboi2.xyz, nbocl2.xyz, film extxyz).
           Eg_trend = Eg_ML(mBJ) + ⟨δ⟩  — Supplementary rank, not publication HSE06.
  Phase C: Validate vs Mortazavi HSE06 bulk (1.60 / 1.72 / 1.69 eV) and Dawei Li 1.76 eV.

MANUSCRIPT_PICKUP — summary JSON must include calibration_fit_rows (all N Phase-A pairs).
validation_rows is Phase C only (NbOI2 vs Mortazavi); do not use it alone for Supplementary Table S_Eg.

Usage:
    python ml_eg_jarvis_scissor_calibrate.py
    python ml_eg_jarvis_scissor_calibrate.py --run-alignn
    python ml_eg_jarvis_scissor_calibrate.py --apply-widget
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated_films_GPT_38" / "ItoCl" / "dielectric_compare"
CSV_PATH = OUT_DIR / "jarvis_scissor_calibration.csv"
SUMMARY_PATH = OUT_DIR / "jarvis_scissor_summary.json"

OPTIMADE = "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures"

# Mortazavi 2022 Table 1 bulk HSE06 — validation only (not calibration input).
MORTAZAVI_BULK_HSE06 = {
    "NbOI2": 1.60,
    "NbOCl2": 1.72,
    "NbOBr2": 1.69,
}
DAWEI_LI_BULK_HSE06 = 1.76

# Local repo bulk cells (docx / Mortazavi C2 family).
LOCAL_STRUCTURES = [
    ROOT / "nboi2.xyz",
    ROOT / "nbocl2.xyz",
]

# Known JARVIS anchors (OPTIMADE id = dft_3d_* or dft_2d_* — not bare JVASP id).
KNOWN_JIDS = [
    "JVASP-26367",  # NbI2O C2 bulk opt+mbj
    "JVASP-29372",  # NbI2O C2 polymorph opt+mbj
    "JVASP-28028",  # NbIO2 2D opt+mbj
    "JVASP-25591",  # NbIO2 3D opt+mbj
    "JVASP-29443",  # NbIO2 3D opt+mbj
    "JVASP-25875",  # NbCl3O opt+mbj
    "JVASP-5392",   # NbCl3O polymorph opt+mbj
    "JVASP-28451",  # NbCl2O C2 bulk opt only (mBJ −99999)
    "JVASP-12017",  # NbBr2O C2 bulk opt only (mBJ −99999)
]

# chemical_formula_reduced queries (reliable on JARVIS OPTIMADE).
OXYHALIDE_FORMULAS = [
    "NbI2O", "NbIO2", "NbCl2O", "NbCl3O", "NbBr2O",
    "TaI2O", "TaIO2", "TaCl2O", "TaCl3O", "TaBr2O",
    "VI2O", "VIO2", "VCl2O", "VCl3O",
    "MoI2O", "MoIO2", "WI2O", "WIO2",
    "HfI2O", "ZrI2O", "TiI2O",
]


def _valid_gap(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f < 0 or f >= 9000:
        return None
    return f


def _fetch_jid(jid: str) -> dict[str, Any] | None:
    """JARVIS OPTIMADE: _jarvis_jid filter is broken; use full id= dft_3d_/dft_2d_ prefix."""
    for filt in (
        f'id="dft_3d_{jid}"',
        f'id="dft_2d_{jid}"',
        f'id="{jid}"',
    ):
        url = f"{OPTIMADE}?filter={urllib.parse.quote(filt)}&page_limit=1"
        r = requests.get(url, timeout=60)
        if r.ok and r.json().get("data"):
            attrs = r.json()["data"][0].get("attributes") or {}
            if str(attrs.get("_jarvis_jid", "")).upper() == jid.upper():
                return attrs
    return None


def _fetch_formula_family(formula: str) -> list[dict[str, Any]]:
    filt = urllib.parse.quote(f'chemical_formula_reduced="{formula}"')
    url = f"{OPTIMADE}?filter={filt}&page_limit=20"
    r = requests.get(url, timeout=60)
    if not r.ok:
        return []
    return [d.get("attributes") or {} for d in r.json().get("data") or []]



def _fetch_formula_reduced(formula: str) -> list[dict[str, Any]]:
    filt = urllib.parse.quote(f'chemical_formula_reduced="{formula}"')
    url = f"{OPTIMADE}?filter={filt}&page_limit=25"
    r = requests.get(url, timeout=60)
    if not r.ok:
        return []
    return [d.get("attributes") or {} for d in r.json().get("data") or []]


def _row_from_attrs(a: dict[str, Any]) -> dict[str, Any] | None:
    opt = _valid_gap(a.get("_jarvis_optb88vdw_bandgap"))
    mbj = _valid_gap(a.get("_jarvis_mbj_bandgap"))
    if opt is None or mbj is None:
        return None
    formula = str(a.get("_jarvis_formula") or "")
    return {
        "jid": a.get("_jarvis_jid"),
        "formula": formula,
        "formula_reduced": a.get("chemical_formula_reduced"),
        "spg": a.get("_jarvis_spg_symbol"),
        "nat": a.get("_jarvis_nat"),
        "typ": a.get("_jarvis_typ"),
        "Eg_opt_JARVIS": opt,
        "Eg_mbj_JARVIS": mbj,
        "delta_mbj_minus_opt": round(mbj - opt, 4),
        "slme_pct": a.get("_jarvis_slme"),
        "reference": a.get("_jarvis_reference"),
        "in_calibration": False,
    }


def _is_target_family(formula: str) -> bool:
    """Layered oxyhalide MOX₂-like — Nb-family first; widen if N<3."""
    if "O" not in formula:
        return False
    if not any(x in formula for x in ("Cl", "Br", "I")):
        return False
    return any(m in formula for m in ("Nb", "Ta", "V", "Mo", "W", "Hf", "Zr", "Ti"))


def _is_broad_oxyhalide(formula: str, nat: int | None) -> bool:
    """Broader Supplementary class: O + halogen ternary, ≤32 atoms (bulk-like)."""
    if "O" not in formula:
        return False
    if not any(x in formula for x in ("Cl", "Br", "I")):
        return False
    if nat is not None and int(nat) > 32:
        return False
    return True


def collect_jarvis_pairs(*, max_pages: int = 15, page_size: int = 100) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    for jid in KNOWN_JIDS:
        a = _fetch_jid(jid)
        if a:
            row = _row_from_attrs(a)
            if row:
                rows[str(row["jid"])] = row

    for fam in (
        "NbOI2", "NbOCl2", "NbOBr2",
        "TaOI2", "TaOCl2", "TaOBr2",
        "VOI2", "VOCl2", "VOBr2",
        "MoOI2", "WOI2", "HfOI2",
    ):
        for a in _fetch_formula_family(fam):
            row = _row_from_attrs(a)
            if row:
                rows[str(row["jid"])] = row

    for formula in OXYHALIDE_FORMULAS:
        for a in _fetch_formula_reduced(formula):
            row = _row_from_attrs(a)
            if row:
                rows[str(row["jid"])] = row

    return list(rows.values())


def mark_calibration_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nb/Ta/V oxyhalide ternaries with paired JARVIS gaps (no Mortazavi in fit)."""
    nb_family = [
        r
        for r in rows
        if _is_target_family(str(r.get("formula", "")))
        and r.get("Eg_mbj_JARVIS") is not None
    ]
    use_broad = len(nb_family) < 3
    for r in rows:
        f = str(r.get("formula", ""))
        if _is_target_family(f) and r.get("Eg_mbj_JARVIS") is not None:
            r["in_calibration"] = True
            r["calib_tier"] = "Nb/Ta/V oxyhalide (JARVIS paired DFT)"
        elif use_broad and _is_broad_oxyhalide(f, r.get("nat")) and r.get("Eg_mbj_JARVIS"):
            r["in_calibration"] = True
            r["calib_tier"] = "broad oxyhalide (Supplementary)"
        else:
            r["in_calibration"] = False
            r["calib_tier"] = ""
    return rows


def validation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-check JARVIS mBJ vs Mortazavi HSE06 (validation — geometry may differ)."""
    out: list[dict[str, Any]] = []

    def _mortazavi_key(formula: str) -> str | None:
        f = formula.replace(" ", "")
        if "Nb" not in f or "O" not in f:
            return None
        if "I" in f and f in ("NbI2O", "NbOI2", "NbIO2"):
            return "NbOI2"
        if "Cl" in f and f in ("NbCl2O", "NbOCl2"):
            return "NbOCl2"
        if "Br" in f and f in ("NbBr2O", "NbOBr2"):
            return "NbOBr2"
        return None

    for r in rows:
        f = str(r.get("formula", "")).replace(" ", "")
        key = _mortazavi_key(f)
        if not key:
            continue
        hse = MORTAZAVI_BULK_HSE06.get(key)
        mbj = r.get("Eg_mbj_JARVIS")
        if hse is None or mbj is None:
            continue
        implied_delta = round(float(hse) - float(mbj), 3)
        out.append(
            {
                "compound": key,
                "jid": r["jid"],
                "formula_jarvis": f,
                "Mortazavi_HSE06_bulk": hse,
                "JARVIS_mBJ": mbj,
                "implied_HSE06_minus_mBJ": implied_delta,
                "note": "Validation only — JARVIS cell may ≠ Mortazavi HSE06 geometry",
            }
        )
    return out


def run_alignn_local() -> list[dict[str, Any]]:
    from ml_eg_bandgap_runner import predict_bandgap, ALIGNN_MODELS, _count_atoms

    results: list[dict[str, Any]] = []
    for path in LOCAL_STRUCTURES:
        if not path.is_file():
            continue
        for key in ("optb88", "mbj"):
            meta = ALIGNN_MODELS[key]
            eg = predict_bandgap(path, model_name=meta["name"])
            out = {
                "structure": str(path),
                "n_atoms": _count_atoms(path),
                "model_key": key,
                "Eg_eV_ML": eg,
            }
            tag = path.stem.replace(".", "_")
            suffix = "" if key == "optb88" else "_mbj"
            jpath = OUT_DIR / f"alignn_eg_{tag}{suffix}.json"
            jpath.parent.mkdir(parents=True, exist_ok=True)
            jpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
            results.append(out)
    return results


def _is_nbi2o_stoich(formula: str) -> bool:
    return str(formula or "").replace(" ", "") in ("NbI2O", "NbOI2", "I2NbO")


def scheme_iii_nbi2o_anchors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """NbI₂O 1:1:2 JARVIS rows for scheme III mBJ reference (excludes NbIO₂ 1:2:1)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if not _is_nbi2o_stoich(str(r.get("formula") or "")):
            continue
        if r.get("Eg_mbj_JARVIS") is None:
            continue
        out.append(r)
    return out


def load_rows_from_csv(path: Path = CSV_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for key in ("Eg_opt_JARVIS", "Eg_mbj_JARVIS", "delta_mbj_minus_opt", "slme_pct", "nat"):
                if key in row and row[key] not in (None, ""):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            if "nat" in row and row["nat"] not in (None, ""):
                try:
                    row["nat"] = int(float(row["nat"]))
                except ValueError:
                    pass
            row["in_calibration"] = str(row.get("in_calibration", "")).lower() in ("true", "1", "yes")
            rows.append(row)
    return rows


def calibration_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase-A fit rows for manuscript / widget (all in_calibration pairs)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if not r.get("in_calibration"):
            continue
        out.append(
            {
                "jid": r.get("jid"),
                "formula": r.get("formula"),
                "spg": r.get("spg"),
                "typ": r.get("typ"),
                "Eg_opt_JARVIS": r.get("Eg_opt_JARVIS"),
                "Eg_mbj_JARVIS": r.get("Eg_mbj_JARVIS"),
                "delta_mbj_minus_opt": r.get("delta_mbj_minus_opt"),
            }
        )
    return out


def build_summary(rows: list[dict[str, Any]], alignn: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cal = [r for r in rows if r.get("in_calibration")]
    deltas = [float(r["delta_mbj_minus_opt"]) for r in cal]
    n_cal = len(cal)
    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    # Supplementary scissor proxy (JARVIS class): shift ML-mBJ by mean intra-JARVIS XC lift.
    # Validation (separate): compare Eg_trend / Eg_scissor vs Mortazavi & Dawei Li HSE06.
    eg_ml_nboi2_mbj = None
    if alignn:
        for a in alignn:
            if a.get("model_key") == "mbj" and "nboi2" in str(a.get("structure", "")).lower():
                eg_ml_nboi2_mbj = float(a["Eg_eV_ML"])
    if eg_ml_nboi2_mbj is None:
        p = OUT_DIR / "alignn_eg_nboi2_bulk_mbj.json"
        if p.is_file():
            eg_ml_nboi2_mbj = float(json.loads(p.read_text(encoding="utf-8"))["Eg_eV_ML"])

    eg_trend = (
        eg_ml_nboi2_mbj + mean_delta
        if eg_ml_nboi2_mbj is not None and n_cal >= 3 and deltas
        else float("nan")
    )

    # Validation: implied HSE06−mBJ if Mortazavi bulk matched by halogen (not used in fit).
    val_implied = validation_table(rows)
    mean_implied_hse_mbj = (
        statistics.mean(r["implied_HSE06_minus_mBJ"] for r in val_implied)
        if val_implied
        else float("nan")
    )
    eg_scissor_vs_mortazavi = (
        eg_ml_nboi2_mbj + mean_implied_hse_mbj
        if eg_ml_nboi2_mbj is not None and val_implied
        else float("nan")
    )
    # Scheme III: Dawei gives one HSE06 (1.76 eV); JARVIS ref = mean mBJ on canonical NbI₂O 1:1:2 only.
    # Do NOT mix NbIO₂ polymorphs (1:2:1, mBJ 0.92–1.31) — different stoichiometry.
    anchor_rows = scheme_iii_nbi2o_anchors(rows)
    dawei_anchor_jids = [str(r.get("jid") or "") for r in anchor_rows if r.get("jid")]
    dawei_anchor_mbj_mean: float | None = None
    dawei_anchor_jid: str | None = None
    if anchor_rows and eg_ml_nboi2_mbj is not None:
        dawei_anchor_mbj_mean = statistics.mean(float(r["Eg_mbj_JARVIS"]) for r in anchor_rows)
        eg_scissor_vs_dawei = eg_ml_nboi2_mbj + (DAWEI_LI_BULK_HSE06 - dawei_anchor_mbj_mean)
        dawei_anchor_jid = dawei_anchor_jids[0] if len(dawei_anchor_jids) == 1 else "+".join(dawei_anchor_jids)
    else:
        eg_scissor_vs_dawei = float("nan")

    return {
        "calibration_mode": "JARVIS paired OptB88vdW + TB-mBJ (same cell, no Mortazavi HSE06 in fit)",
        "n_calibration": n_cal,
        "min_required": 3,
        "mean_delta_mbj_minus_opt_eV": round(mean_delta, 4) if n_cal else None,
        "stdev_delta_eV": round(stdev, 4) if n_cal > 1 else None,
        "widget_delta_hint_eV": round(mean_delta, 3) if n_cal >= 3 else 0.0,
        "widget_n_hint": n_cal if n_cal >= 3 else 0,
        "eg_alignn_mbj_nboi2": eg_ml_nboi2_mbj,
        "eg_supplementary_trend_eV": round(eg_trend, 3) if eg_trend == eg_trend else None,
        "eg_scissor_vs_mortazavi_validation_eV": round(eg_scissor_vs_mortazavi, 3)
        if eg_scissor_vs_mortazavi == eg_scissor_vs_mortazavi
        else None,
        "eg_scissor_vs_dawei_li_validation_eV": round(eg_scissor_vs_dawei, 3)
        if eg_scissor_vs_dawei == eg_scissor_vs_dawei
        else None,
        "mean_implied_HSE06_minus_mBJ_validation_eV": round(mean_implied_hse_mbj, 3)
        if mean_implied_hse_mbj == mean_implied_hse_mbj
        else None,
        "validation_mortazavi_bulk_HSE06": MORTAZAVI_BULK_HSE06,
        "validation_dawei_li_bulk_HSE06": DAWEI_LI_BULK_HSE06,
        "dawei_scheme_III_note": (
            f"Mean JARVIS mBJ on NbI2O 1:1:2 (N={len(anchor_rows)} C2 bulk), "
            "not mean over NbIO2 1:2:1 — Dawei gives one HSE06, no JID pairs"
        ),
        "dawei_jarvis_anchor_jid": dawei_anchor_jid,
        "dawei_jarvis_anchor_jids": dawei_anchor_jids,
        "dawei_jarvis_anchor_mbj_mean_eV": round(dawei_anchor_mbj_mean, 4)
        if dawei_anchor_mbj_mean is not None
        else None,
        "calibration_fit_rows": calibration_fit_rows(rows),
        "validation_rows": validation_table(rows),
        "manuscript_export": {
            "MANUSCRIPT_PICKUP": True,
            "supplementary_table_id": "Supplementary Table S_Eg_JARVIS_fit",
            "use_key": "calibration_fit_rows",
            "not_key": "validation_rows",
            "csv_path": str(CSV_PATH),
            "note_ru": (
                "В статью — все N пар фазы A из calibration_fit_rows (или CSV). "
                "validation_rows — только фаза C (пример vs Mortazavi HSE06)."
            ),
            "note_en": (
                "Paper/Supplementary: export all N Phase-A rows from calibration_fit_rows (or CSV). "
                "validation_rows is Phase-C sample only (Mortazavi HSE06 check)."
            ),
        },
        "nb_br_in_jarvis": {
            "NbOI2_paired_opt_mbj": True,
            "NbOCl2_paired": any(
                str(r.get("formula", "")).replace(" ", "") in ("NbCl2O", "NbOCl2")
                and r.get("Eg_mbj_JARVIS")
                for r in rows
            ),
            "NbOBr2_paired": any(
                str(r.get("formula", "")).replace(" ", "") in ("NbBr2O", "NbOBr2")
                and r.get("Eg_mbj_JARVIS")
                for r in rows
            ),
            "note": "NbOCl2/NbOBr2 bulk often lack mBJ in JARVIS — expand to Ta/V oxyhalides for N≥3",
        },
        "paper_goal_ru": (
            "Цель статьи: дешёвый ML-пайплайн (ALIGNN/GNNOpt/MACEField) vs недели VASP HSE06; "
            "Eg_scissor — Supplementary-тренд после JARVIS-калибровки, не замена HSE06 в основном тексте."
        ),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "jid",
        "formula",
        "spg",
        "nat",
        "typ",
        "Eg_opt_JARVIS",
        "Eg_mbj_JARVIS",
        "delta_mbj_minus_opt",
        "slme_pct",
        "calib_tier",
        "in_calibration",
        "reference",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS class calibration for Eg_scissor Supplementary")
    ap.add_argument("--run-alignn", action="store_true", help="Run ALIGNN optb88/mbj on local xyz")
    ap.add_argument("--apply-widget", action="store_true", help="Print widget field hints only")
    ap.add_argument("--max-pages", type=int, default=5)
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="Rebuild summary from jarvis_scissor_calibration.csv (no OPTIMADE fetch)",
    )
    args = ap.parse_args()

    if args.from_csv:
        print(f"Loading rows from {CSV_PATH} …")
        rows = load_rows_from_csv()
    else:
        print("Fetching JARVIS OPTIMADE pairs …")
        rows = collect_jarvis_pairs(max_pages=args.max_pages)
        rows = mark_calibration_set(rows)
    cal = [r for r in rows if r["in_calibration"]]
    print(f"Total paired rows: {len(rows)} | Nb/Ta/V oxyhalide calibration: {len(cal)}")

    alignn: list[dict[str, Any]] | None = None
    if args.run_alignn:
        print("Running ALIGNN on local structures …")
        alignn = run_alignn_local()

    summary = build_summary(rows, alignn)
    write_csv(rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nCSV -> {CSV_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")

    if args.apply_widget or summary.get("n_calibration", 0) >= 3:
        d = summary.get("widget_delta_hint_eV")
        n = summary.get("widget_n_hint")
        print(f"\nWidget hints: <delta>_class = {d} eV , N = {n}")
        print("(Written to jarvis_scissor_summary.json — widget loads automatically; manual paste not required.)")

    return 0 if summary.get("n_calibration", 0) >= 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())

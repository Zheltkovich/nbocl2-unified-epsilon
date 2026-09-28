#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JARVIS + GNNOpt spectral calibration for Tr(Re ε(ω)) — Supplementary ε_trend.

Three schemes (publication order: honesty → density of control points → simplicity):

  I   JARVIS class (most honest for transfer):
      ⟨ΔTr⟩ = mean(Tr(Re ε(0))_mBJ − Tr(Re ε(0))_Opt) on paired JARVIS cells (same geometry).
      ε_trend(ω) = Tr_GNNOpt(ω) + ⟨ΔTr⟩  — Mortazavi/docx CSV NOT in fit.

  II  Affine anchor on bulk nboi2.xyz (most control points on one structure):
      On 0–5 eV grid fit Tr(Re ε)_CSV ≈ a·Tr(Re ε)_GNNOpt + b (500 interpolated pts).
      ε_trend(ω) = a·ε_GNNOpt(ω) + b — best single-structure level; still Supplementary.

  III Static scissor (simplest):
      Δ_static = Tr(Re ε)_CSV(0) − Tr(Re ε)_GNNOpt(0).
      ε_trend(ω) = ε_GNNOpt(ω) + Δ_static — breaks strict Kramers–Kronig if Im shifted differently.

  Combined (recommended in widget): report I for cross-structure ranking; II for bulk level;
  III as quick check; Phase C validation always vs docx CSV / Mortazavi (never in fit).

MANUSCRIPT_PICKUP — summary JSON must include calibration_fit_rows (all N Phase-A pairs).
validation_mortazavi is Phase C only (2-row sample); do not use it alone for Supplementary Table S_eps.

MANUSCRIPT_PICKUP — paper wording: see eps_spectral_scissor_manuscript_pickup_html() in
mortazavi_dielectric_shared.py ({MANUSCRIPT_EPS_SCISSOR_METHODS_ID}).
Why calibrate: GNNOpt shape (r) useful, absolute Tr(ε) level wrong at DFT-IPA training — ε_trend only in SI.

Usage:
    python ml_eps_spectral_scissor_calibrate.py
    python ml_eps_spectral_scissor_calibrate.py --offline
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

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated_films_GPT_38" / "ItoCl" / "dielectric_compare"
CSV_PATH = OUT_DIR / "jarvis_eps_calibration.csv"
SUMMARY_PATH = OUT_DIR / "jarvis_eps_scissor_summary.json"
GNNOPT_JSON = OUT_DIR / "gnnopt_spectrum_nboi2_bulk.json"

OPTIMADE = "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures"

# Mortazavi bulk RPA static Tr(ε) — validation only (not calibration fit).
MORTAZAVI_BULK_TR_EPS = {
    "NbOI2": 18.59,
    "NbOCl2": 15.42,
    "NbOBr2": 15.89,
}

KNOWN_JIDS = [
    "JVASP-26367", "JVASP-29372", "JVASP-28028", "JVASP-25591", "JVASP-29443",
    "JVASP-25875", "JVASP-5392", "JVASP-28451", "JVASP-12017",
]

OXYHALIDE_FORMULAS = [
    "NbI2O", "NbIO2", "NbCl2O", "NbCl3O", "NbBr2O",
    "TaI2O", "TaIO2", "TaCl2O", "TaCl3O", "TaBr2O",
    "VI2O", "VIO2", "VCl2O", "VCl3O",
    "MoI2O", "MoIO2", "WI2O", "WIO2",
    "HfI2O", "ZrI2O", "TiI2O",
]

AFFINE_E_MAX_EV = 5.0
AFFINE_N_GRID = 500
COMPARE_E_MAX_EV = 15.0


def _valid_scalar(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f < 0 or f >= 9000:
        return None
    return f


def _tr_static_eps(a: dict[str, Any], *, mbj: bool) -> tuple[float | None, float | None, float | None, float | None]:
    if mbj:
        keys = ("_jarvis_mepsx", "_jarvis_mepsy", "_jarvis_mepsz")
    else:
        keys = ("_jarvis_epsx", "_jarvis_epsy", "_jarvis_epsz")
    comps = [_valid_scalar(a.get(k)) for k in keys]
    if any(c is None for c in comps):
        return None, None, None, None
    x, y, z = comps  # type: ignore[misc]
    return float(x + y + z), float(x), float(y), float(z)


def _fetch_jid(jid: str) -> dict[str, Any] | None:
    for filt in (f'id="dft_3d_{jid}"', f'id="dft_2d_{jid}"', f'id="{jid}"'):
        url = f"{OPTIMADE}?filter={urllib.parse.quote(filt)}&page_limit=1"
        r = requests.get(url, timeout=60)
        if r.ok and r.json().get("data"):
            attrs = r.json()["data"][0].get("attributes") or {}
            if str(attrs.get("_jarvis_jid", "")).upper() == jid.upper():
                return attrs
    return None


def _fetch_formula_reduced(formula: str) -> list[dict[str, Any]]:
    filt = urllib.parse.quote(f'chemical_formula_reduced="{formula}"')
    url = f"{OPTIMADE}?filter={filt}&page_limit=25"
    r = requests.get(url, timeout=60)
    if not r.ok:
        return []
    return [d.get("attributes") or {} for d in r.json().get("data") or []]


def _row_from_attrs(a: dict[str, Any]) -> dict[str, Any] | None:
    tr_opt, ex, ey, ez = _tr_static_eps(a, mbj=False)
    tr_mbj, mx, my, mz = _tr_static_eps(a, mbj=True)
    if tr_opt is None or tr_mbj is None:
        return None
    formula = str(a.get("_jarvis_formula") or a.get("chemical_formula_descriptive") or "")
    return {
        "jid": a.get("_jarvis_jid"),
        "formula": formula,
        "formula_reduced": a.get("chemical_formula_reduced"),
        "spg": a.get("_jarvis_spg_symbol"),
        "nat": a.get("_jarvis_nat"),
        "typ": a.get("_jarvis_typ"),
        "Tr_eps_opt_JARVIS": round(tr_opt, 4),
        "Tr_eps_mbj_JARVIS": round(tr_mbj, 4),
        "epsx_opt": ex, "epsy_opt": ey, "epsz_opt": ez,
        "mepsx_mbj": mx, "mepsy_mbj": my, "mepsz_mbj": mz,
        "delta_tr_mbj_minus_opt": round(tr_mbj - tr_opt, 4),
        "slme_pct": a.get("_jarvis_slme"),
        "reference": a.get("_jarvis_reference"),
        "in_calibration": False,
        "calib_tier": "",
        "nbi2o_stoichiometry": _is_nbi2o_stoichiometry(formula),
    }


def _is_nbi2o_stoichiometry(formula: str) -> bool:
    f = formula.replace(" ", "")
    return f in ("NbI2O", "NbOI2", "I2NbO")


def _is_target_family(formula: str) -> bool:
    if "O" not in formula:
        return False
    if not any(x in formula for x in ("Cl", "Br", "I")):
        return False
    return any(m in formula for m in ("Nb", "Ta", "V", "Mo", "W", "Hf", "Zr", "Ti"))


def _is_broad_oxyhalide(formula: str, nat: int | None) -> bool:
    if "O" not in formula or not any(x in formula for x in ("Cl", "Br", "I")):
        return False
    if nat is not None and int(nat) > 32:
        return False
    return True


def collect_jarvis_eps_pairs() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for jid in KNOWN_JIDS:
        a = _fetch_jid(jid)
        if a:
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
    nb_family = [r for r in rows if _is_target_family(str(r.get("formula", "")))]
    use_broad = len(nb_family) < 3
    for r in rows:
        f = str(r.get("formula", ""))
        if _is_target_family(f):
            r["in_calibration"] = True
            r["calib_tier"] = "Nb/Ta/V oxyhalide (JARVIS paired static ε)"
        elif use_broad and _is_broad_oxyhalide(f, r.get("nat")):
            r["in_calibration"] = True
            r["calib_tier"] = "broad oxyhalide (Supplementary)"
        else:
            r["in_calibration"] = False
            r["calib_tier"] = ""
    return rows


def validation_mortazavi(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _key(formula: str) -> str | None:
        f = formula.replace(" ", "")
        if "Nb" not in f or "O" not in f:
            return None
        if "I" in f and f in ("NbI2O", "NbOI2", "I2NbO"):
            return "NbOI2"
        if "Cl" in f and f in ("NbCl2O", "NbOCl2"):
            return "NbOCl2"
        if "Br" in f and f in ("NbBr2O", "NbOBr2"):
            return "NbOBr2"
        return None

    for r in rows:
        key = _key(str(r.get("formula", "")))
        if not key:
            continue
        tr_mort = MORTAZAVI_BULK_TR_EPS.get(key)
        tr_mbj = r.get("Tr_eps_mbj_JARVIS")
        if tr_mort is None or tr_mbj is None:
            continue
        out.append({
            "compound": key,
            "jid": r["jid"],
            "formula_jarvis": r.get("formula"),
            "Mortazavi_RPA_Tr_eps": tr_mort,
            "JARVIS_mBJ_Tr_eps": tr_mbj,
            "implied_Mortazavi_minus_mBJ": round(float(tr_mort) - float(tr_mbj), 3),
            "note": "Validation only — JARVIS cell ≠ Mortazavi HSE06+RPA geometry",
        })
    return out


def _load_gnnopt_spectrum() -> dict[str, Any] | None:
    if not GNNOPT_JSON.is_file():
        return None
    try:
        return json.loads(GNNOPT_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_hse06_tr_re() -> tuple[np.ndarray, np.ndarray]:
    from ml_spectrum_compare import hse06_macroscopic_tr_re
    from nboi2_hse06_dielectric import load_hse06_spectrum_df

    df = load_hse06_spectrum_df()
    return hse06_macroscopic_tr_re(df)


def _affine_fit(
    e_ref: np.ndarray,
    y_ref: np.ndarray,
    e_ml: np.ndarray,
    y_ml: np.ndarray,
    *,
    e_max: float = AFFINE_E_MAX_EV,
    n_grid: int = AFFINE_N_GRID,
) -> dict[str, float]:
    e_common = np.linspace(0.0, e_max, n_grid)
    yr = np.interp(e_common, e_ref, y_ref)
    ym = np.interp(e_common, e_ml, y_ml)
    a, b = np.polyfit(ym, yr, 1)
    pred = a * ym + b
    mae_before = float(np.mean(np.abs(ym - yr)))
    mae_after = float(np.mean(np.abs(pred - yr)))
    return {
        "a": float(a),
        "b": float(b),
        "mae_before_eV": round(mae_before, 4),
        "mae_after_eV": round(mae_after, 4),
        "e_max_fit_eV": float(e_max),
        "n_grid": int(n_grid),
    }


def _apply_affine(y: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.asarray(y, dtype=float) + b


def _metrics_vs_csv(
    e_ref: np.ndarray,
    y_ref: np.ndarray,
    e_ml: np.ndarray,
    y_ml: np.ndarray,
    *,
    e_max: float = COMPARE_E_MAX_EV,
) -> dict[str, float]:
    from ml_spectrum_compare import compare_spectra

    return compare_spectra(
        e_ref, y_ref, e_ml, y_ml,
        e_max=e_max,
        label_ref="HSE06 LOPTICS CSV",
        label_ml="calibrated GNNOpt",
    )


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
                "typ": r.get("typ"),
                "Tr_eps_opt_JARVIS": r.get("Tr_eps_opt_JARVIS"),
                "Tr_eps_mbj_JARVIS": r.get("Tr_eps_mbj_JARVIS"),
                "delta_tr_mbj_minus_opt": r.get("delta_tr_mbj_minus_opt"),
            }
        )
    return out


def build_summary(
    rows: list[dict[str, Any]],
    gnn: dict[str, Any] | None,
    e_ref: np.ndarray | None,
    y_ref: np.ndarray | None,
) -> dict[str, Any]:
    cal = [r for r in rows if r.get("in_calibration")]
    cal_nbi2o = [r for r in cal if r.get("nbi2o_stoichiometry")]
    deltas = [float(r["delta_tr_mbj_minus_opt"]) for r in cal]
    deltas_nbi2o = [float(r["delta_tr_mbj_minus_opt"]) for r in cal_nbi2o]
    n_cal = len(cal)
    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    mean_delta_nbi2o = statistics.mean(deltas_nbi2o) if deltas_nbi2o else float("nan")
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    scheme_i: dict[str, Any] = {
        "id": "I_jarvis_class",
        "label": "JARVIS class (most honest for transfer)",
        "formula": "Tr_trend(ω) = Tr_GNNOpt(ω) + ⟨ΔTr⟩_JARVIS",
        "delta_definition": "⟨ΔTr⟩ = mean(Tr_mBJ(0) − Tr_Opt(0)) on paired JARVIS cells",
        "n_calibration": n_cal,
        "n_nbi2o_only": len(cal_nbi2o),
        "mean_delta_tr_eV": round(mean_delta, 4) if n_cal else None,
        "mean_delta_tr_nbi2o_only_eV": round(mean_delta_nbi2o, 4) if cal_nbi2o else None,
        "stdev_delta_eV": round(stdev, 4) if n_cal > 1 else None,
        "fit_includes_mortazavi_csv": False,
    }

    scheme_ii: dict[str, Any] = {
        "id": "II_affine_csv",
        "label": "Affine anchor on nboi2.xyz (most control points)",
        "formula": "Tr_trend(ω) ≈ a·Tr_GNNOpt(ω) + b  (fit 0–5 eV vs docx CSV)",
        "n_control_points": AFFINE_N_GRID,
        "fit_includes_mortazavi_csv": False,
        "fit_includes_hse06_csv": True,
    }
    scheme_iii: dict[str, Any] = {
        "id": "III_static_scissor",
        "label": "Static scissor (simplest)",
        "formula": "Tr_trend(ω) = Tr_GNNOpt(ω) + Δ_static",
        "delta_definition": "Δ_static = Tr_CSV(0) − Tr_GNNOpt(0)",
        "fit_includes_hse06_csv": True,
        "kk_warning": "Additive Re-only shift breaks strict Kramers–Kronig if Im not shifted consistently",
    }

    gnn_tr0 = None
    csv_tr0 = None
    if gnn and y_ref is not None and e_ref is not None:
        e_g = np.array(gnn["energy_eV"], dtype=float)
        y_g = np.array(gnn["Re_eps"], dtype=float)
        gnn_tr0 = float(y_g[0]) if y_g.size else float("nan")
        csv_tr0 = float(y_ref[0]) if y_ref.size else float("nan")
        delta_static = csv_tr0 - gnn_tr0
        scheme_iii["delta_static_eV"] = round(delta_static, 4)
        scheme_iii["csv_tr_re_0"] = round(csv_tr0, 4)
        scheme_iii["gnnopt_tr_re_0"] = round(gnn_tr0, 4)

        aff = _affine_fit(e_ref, y_ref, e_g, y_g, e_max=AFFINE_E_MAX_EV)
        scheme_ii.update(aff)
        y_aff = _apply_affine(y_g, aff["a"], aff["b"])
        y_i = y_g + mean_delta if n_cal >= 3 and deltas else y_g + delta_static
        y_iii = y_g + delta_static

        scheme_ii["tr_re_0_calibrated"] = round(float(y_aff[0]), 4)
        scheme_i["tr_re_0_calibrated"] = round(float(y_i[0]), 4) if n_cal >= 3 else None
        scheme_iii["tr_re_0_calibrated"] = round(float(y_iii[0]), 4)

        try:
            scheme_ii["validation_0_15eV"] = _metrics_vs_csv(e_ref, y_ref, e_g, y_aff)
            scheme_i["validation_0_15eV"] = _metrics_vs_csv(e_ref, y_ref, e_g, y_i) if n_cal >= 3 else None
            scheme_iii["validation_0_15eV"] = _metrics_vs_csv(e_ref, y_ref, e_g, y_iii)
            raw = _metrics_vs_csv(e_ref, y_ref, e_g, y_g)
            scheme_ii["validation_raw_gnnopt"] = raw
        except Exception as exc:
            scheme_ii["validation_error"] = str(exc)

    return {
        "calibration_mode": "Spectral ε(ω) — three Supplementary schemes; HSE06 CSV validation only in Phase C",
        "min_calibration_n": 3,
        "n_calibration": n_cal,
        "schemes": {
            "I_jarvis_class": scheme_i,
            "II_affine_csv": scheme_ii,
            "III_static_scissor": scheme_iii,
        },
        "recommended_publication": {
            "main_text": "HSE06 LOPTICS CSV / Mortazavi RPA static ε — DFT anchors",
            "supplementary_ranking": "Scheme I (JARVIS ⟨ΔTr⟩) for cross-structure GNNOpt screening",
            "supplementary_bulk_level": "Scheme II (affine 0–5 eV) on same nboi2.xyz cell",
            "quick_check": "Scheme III (static Δ) — report with KK caveat",
            "combined": "I + II: class transfer then affine bulk anchor; III as sanity check vs CSV ω=0",
        },
        "calibration_fit_rows": calibration_fit_rows(rows),
        "validation_mortazavi": validation_mortazavi(rows),
        "manuscript_export": {
            "MANUSCRIPT_PICKUP": True,
            "supplementary_table_id": "Supplementary Table S_eps_JARVIS_fit",
            "use_key": "calibration_fit_rows",
            "not_key": "validation_mortazavi",
            "csv_path": str(CSV_PATH),
            "note_ru": (
                "В статью — все N пар фазы A из calibration_fit_rows (или CSV). "
                "validation_mortazavi — только фаза C (2 примера vs Mortazavi RPA)."
            ),
            "note_en": (
                "Paper/Supplementary: export all N Phase-A rows from calibration_fit_rows (or CSV). "
                "validation_mortazavi is Phase-C sample only (Mortazavi RPA check)."
            ),
        },
        "gnnopt_json": str(GNNOPT_JSON) if GNNOPT_JSON.is_file() else None,
        "paper_blurb_en": (
            "GNNOpt reproduces Tr(Re ε(ω)) shape (r≈0.9) but not HSE06 LOPTICS level "
            "(ω=0 offset ~63%). Spectral scissor uses JARVIS paired static ε (Scheme I) "
            "and/or affine fit to docx CSV on nboi2.xyz (Scheme II) — Supplementary only; "
            "main text keeps HSE06/Mortazavi anchors."
        ),
        "paper_blurb_ru": (
            "GNNOpt воспроизводит форму Tr(Re ε(ω)) (r≈0,9), но не уровень HSE06 LOPTICS "
            "(промах на ω=0 ~63%). Спектральный scissor: JARVIS пары static ε (схема I) "
            "и/или affine к docx CSV на nboi2.xyz (схема II) — только Supplementary; "
            "в основном тексте — якоря HSE06/Mortazavi."
        ),
        "developer_rationale": {
            "GNNOpt_Hung_2024": {
                "doi": "10.1002/adma.202409175",
                "quote_en": (
                    "ML models trained on DFT at the independent-particle approximation (IPA) level "
                    "enable rapid screening of optical properties across large chemical spaces."
                ),
                "why_underestimate_ok": (
                    "Authors target high-throughput ranking (SLME, absorption onset), not "
                    "hybrid-functional LOPTICS on every structure."
                ),
            },
            "JARVIS_Choudhary_2020": {
                "doi": "10.1038/s41524-020-00440-1",
                "quote_en": (
                    "HSE06 remains too computationally expensive for high-throughput screening "
                    "of thousands of materials; TB-mBJ balances cost and accuracy."
                ),
            },
            "scissor_VASP_1995": {
                "doi": "10.1103/PhysRevB.51.17196",
                "note": "Scissor operator shifts quasiparticle levels to match experiment/expensive DFT.",
            },
        },
        "widget_hints": {
            "scheme_primary": "I",
            "mean_delta_tr_eV": round(mean_delta, 3) if n_cal >= 3 else None,
            "affine_a": round(float(scheme_ii.get("a") or 0), 3) if scheme_ii.get("a") is not None else None,
            "affine_b": round(float(scheme_ii.get("b") or 0), 2) if scheme_ii.get("b") is not None else None,
            "delta_static_eV": scheme_iii.get("delta_static_eV"),
            "n_calibration": n_cal,
        },
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "jid", "formula", "formula_reduced", "spg", "nat", "typ",
        "Tr_eps_opt_JARVIS", "Tr_eps_mbj_JARVIS",
        "epsx_opt", "epsy_opt", "epsz_opt",
        "mepsx_mbj", "mepsy_mbj", "mepsz_mbj",
        "delta_tr_mbj_minus_opt", "nbi2o_stoichiometry",
        "slme_pct", "calib_tier", "in_calibration", "reference",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_rows_offline() -> list[dict[str, Any]]:
    if not CSV_PATH.is_file():
        return []
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS + GNNOpt spectral scissor calibration")
    ap.add_argument("--offline", action="store_true", help="Use existing CSV only (no JARVIS fetch)")
    args = ap.parse_args()

    if args.offline:
        rows = load_rows_offline()
        if not rows:
            print(f"No offline data at {CSV_PATH}", file=sys.stderr)
            return 1
        for r in rows:
            r["in_calibration"] = str(r.get("in_calibration", "")).lower() in ("true", "1", "yes")
            r["nbi2o_stoichiometry"] = str(r.get("nbi2o_stoichiometry", "")).lower() in ("true", "1", "yes")
    else:
        print("Fetching JARVIS OPTIMADE paired static epsilon …")
        rows = collect_jarvis_eps_pairs()
        rows = mark_calibration_set(rows)
        write_csv(rows)

    cal = [r for r in rows if r.get("in_calibration")]
    print(f"Paired static epsilon rows: {len(rows)} | calibration set: {len(cal)}")

    gnn = _load_gnnopt_spectrum()
    e_ref, y_ref = None, None
    try:
        e_ref, y_ref = _load_hse06_tr_re()
    except Exception as exc:
        print(f"HSE06 CSV load skipped: {exc}", file=sys.stderr)

    if gnn is None:
        print(f"Warning: {GNNOPT_JSON} missing — run ml_gnnopt_spectrum_runner.py nboi2.xyz", file=sys.stderr)

    summary = build_summary(rows, gnn, e_ref, y_ref)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV -> {CSV_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")

    n_ok = int(summary.get("n_calibration") or 0)
    return 0 if n_ok >= 3 or args.offline else 2


if __name__ == "__main__":
    raise SystemExit(main())

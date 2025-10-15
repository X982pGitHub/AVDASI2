#!/usr/bin/env python3
"""
naca5_xfoil_sweep.py

Performs a full exhaustive sweep of all standard NACA 5-digit airfoils
using XFOIL’s internal geometry generation and aerodynamic analysis.

Features:
    - Enumerates all L, P, S, TT combinations.
    - Runs XFOIL in parallel via subprocess.
    - Handles timeouts and retries with relaxed parameters.
    - Logs all outputs and results for later analysis.
    - Computes a user-defined selection metric (default: max CL/CD).
    - Saves progress incrementally to CSV with resume capability.

Requirements:
    - Python 3.8+
    - XFOIL available in PATH
    - Optional: pandas for CSV handling

Author:
    Mostly ChatGPT, some by Thitut Uhthalye, 2025
"""

import os
import subprocess
import time
import numpy as np
import csv
import concurrent.futures
from itertools import product
import datetime
from typing import Any, Optional

try:
    import pandas as pd
except Exception:
    pd = None


# =============================================================================
# CONFIGURATION
# =============================================================================
XFOIL_CMD: str = "xfoil"
WORKDIR: str = "./xfoil_runs"
RESULTS_CSV: str = os.path.join(WORKDIR, "results.csv")

L_vals = list(range(0, 10))
P_vals = [1, 2, 3, 4, 5]
S_vals = [0, 1]
TT_vals = list(range(6, 25))

COARSE_SETTINGS = {
    "panels": 160,      # Increased slightly for better resolution
    "aseq": "-3 10 1",  # More conservative angle range
    "visc": 1e6,        # Always use viscous mode
    "iter": 300,        # More iterations for convergence
    "timeout": 120,     # Longer timeout
}

FINE_SETTINGS = {
    "panels": 280,          # Reduced from 600 to avoid array size issues
    "aseq": "-5 12 0.5",   # More focused angle range
    "visc": 1e6,           # Same Reynolds number
    "iter": 400,           # Increased iterations
    "timeout": 300,
}

MAX_RETRIES = 3
RELAXED_ASEQ = "-2 8 1.0"     # Even more conservative angle range
RELAXED_PANELS = 140          # Don't reduce panels too much
MAX_WORKERS = min(14, (os.cpu_count() or 1))  # Reduced parallel load


# =============================================================================
# METRIC FUNCTION
# =============================================================================
def selection_metric_from_polar_rows(polar_rows: list[dict[str, Any]]) -> float:
    """
    Computes a scalar selection metric from an XFOIL polar dataset.

    Args:
        polar_rows (list[dict[str, Any]]): Parsed polar data as a list of
            dictionaries with at least 'CL' and 'CD' keys.

    Returns:
        float: The computed scalar metric. Default behaviour is the maximum
        lift-to-drag ratio (CL/CD) across valid polar entries. Returns
        -inf if no valid data points are found.
    """
    best = -1e9
    for r in polar_rows:
        try:
            cl = float(r.get("CL", float("nan")))
            cd = float(r.get("CD", float("nan")))
        except Exception:
            continue
        if cd <= 0 or np.isnan(cl) or np.isnan(cd):
            continue
        val = cl / cd
        if val > best:
            best = val
    if best < -1e8:
        return float("-inf")
    return best


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_workdir() -> None:
    """
    Ensures that the working directory and its subdirectories exist.

    Creates the following structure:
        WORKDIR/
            polars/
            logs/
    """
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(os.path.join(WORKDIR, "polars"), exist_ok=True)
    os.makedirs(os.path.join(WORKDIR, "logs"), exist_ok=True)


def naca5_code_to_str(L: int, P: int, S: int, TT: int) -> str:
    """
    Formats a NACA 5-digit airfoil code.

    Args:
        L (int): Design lift coefficient index (0–9).
        P (int): Position of maximum camber index (1–5).
        S (int): Reflex flag (0 = normal, 1 = reflexed).
        TT (int): Maximum thickness in percent (06–24).

    Returns:
        str: Formatted code, e.g. '23012'.
    """
    return f"{int(L)}{int(P)}{int(S):d}{int(TT):02d}"


def build_xfoil_script_for_naca(
    code_str: str,
    polar_outpath: str,
    panels: int,
    aseq: str,
    visc: Optional[float],
    iter_limit: int,
) -> bytes:
    """
    Builds the batch command script to send to XFOIL.

    Args:
        code_str (str): NACA 5-digit code string.
        polar_outpath (str): Destination path for the polar file.
        panels (int): Number of surface panels for meshing.
        aseq (str): Alpha sequence command string (e.g. '-5 15 0.5').
        visc (float | None): Reynolds number for viscous mode.
        iter_limit (int): Iteration limit per angle.

    Returns:
        bytes: Encoded XFOIL command script for subprocess input.
    """
    lines = [
        "PLOP",
        "G F",
        "",
        f"NACA {code_str}",
        "PANE",
        "PPAR",
        f"N {panels}",
        "",
        "",
        "OPER"
    ]
    if visc is not None:
        lines.extend([
            "v",
            f"Re {visc}",
            "M 0.05",  # Add explicit Mach number
            "INIT",   # Initialize BL
            ""
        ])
    lines.extend([
        f"ITER {iter_limit}",
        f"PACC {polar_outpath} {polar_outpath}.dump",
        f"ASEQ {aseq}",
        "PACC",
        "",
        "QUIT"
        "EOF"
    ])

    return ("\n".join(lines) + "\n").encode("utf-8")


def run_xfoil_script(script_bytes: bytes, timeout: int) -> tuple[int, str, str]:
    """
    Executes a single XFOIL subprocess using the provided command script.

    Args:
        script_bytes (bytes): Byte-encoded command script.
        timeout (int): Timeout in seconds before termination.

    Returns:
        tuple[int, str, str]: (returncode, stdout, stderr).
    """
    process = None
    try:
        process = subprocess.Popen(
            [XFOIL_CMD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=script_bytes, timeout=timeout)
        return process.returncode, stdout.decode(errors="ignore"), stderr.decode(errors="ignore")
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        return -1, "", f"timeout after {timeout}s"
    finally:
        if process and process.poll() is None:
            process.kill()


def parse_polar_file(polar_path: str) -> Optional[list[dict[str, float]]]:
    """
    Parses an XFOIL polar file into structured numeric data.

    Args:
        polar_path (str): Path to the `.polar` file produced by XFOIL.

    Returns:
        list[dict[str, float]] | None: List of dictionaries containing
        aerodynamic quantities, or None if parsing failed.
    """
    if not os.path.exists(polar_path):
        return None
    rows = []
    with open(polar_path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    for ln in lines:
        if not ln:
            continue
        parts = ln.split()
        try:
            _ = float(parts[0])
            if len(parts) >= 4:
                nums = [float(p) for p in parts[:5]]
                row = {
                    "alpha": nums[0],
                    "CL": nums[1],
                    "CD": nums[2],
                    "CDp": nums[3] if len(nums) > 3 else float("nan"),
                    "Cm": nums[4] if len(nums) > 4 else float("nan"),
                }
                rows.append(row)
        except Exception:
            continue
    return rows if rows else None


def log_case(case_id: int, code_str: str, status: str, info: str, logfile_path: str) -> None:
    """
    Appends a single event line to a log file for a given case.

    Args:
        case_id (int): Unique case identifier.
        code_str (str): NACA code string.
        status (str): Status label (e.g. 'OK', 'FAILED').
        info (str): Additional diagnostic information.
        logfile_path (str): Path to log file for this case.
    """
    t = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(logfile_path, "a") as f:
        f.write(f"{t},{case_id},{code_str},{status},{info}\n")


# =============================================================================
# CORE EVALUATION FUNCTION
# =============================================================================
def evaluate_naca_case(
    case_id: int,
    L: int,
    P: int,
    S: int,
    TT: int,
    settings: dict[str, Any],
    retry_relaxed: bool = True,
) -> dict[str, Any]:  # type: ignore[return]
    """
    Runs XFOIL for a specific NACA 5-digit airfoil and computes its metric.

    Handles retries, timeouts, and output parsing.

    Args:
        case_id (int): Unique numeric ID for the case.
        L (int): Design lift index.
        P (int): Camber position index.
        S (int): Reflex flag.
        TT (int): Thickness percentage.
        settings (dict[str, Any]): Dictionary of XFOIL solver settings.
        retry_relaxed (bool, optional): Whether to retry failed cases with
            relaxed settings. Defaults to True.

    Returns:
        dict[str, Any]: Result record with keys:
            - case_id (int)
            - code (str)
            - L, P, S, TT (int)
            - ok (bool)
            - metric (float)
            - polar (str | None)
            - log (str)
    """
    code_str = naca5_code_to_str(L, P, S, TT)
    polar_name = f"naca_{code_str}.polar"
    polar_path = os.path.join(WORKDIR, "polars", polar_name)
    log_path = os.path.join(WORKDIR, "logs", f"case_{case_id}_{code_str}.log")

    panels = settings.get("panels", 140)
    aseq = settings.get("aseq", "-5 15 0.5")
    visc = settings.get("visc", None)
    iter_limit = settings.get("iter", 200)
    timeout = settings.get("timeout", 60)

    attempt = 0
    stdout_total = ""
    stderr_total = ""

    result = {
        "case_id": case_id,
        "code": code_str,
        "L": L,
        "P": P,
        "S": S,
        "TT": TT,
        "ok": False,
        "metric": float("-inf"),
        "polar": None,
        "log": log_path,
    }
    
    while attempt <= MAX_RETRIES:
        attempt += 1
        script = build_xfoil_script_for_naca(code_str, polar_path, panels, aseq, visc, iter_limit)
        rc, out, err = run_xfoil_script(script, timeout)
        stdout_total += out
        stderr_total += err

        polar_rows = parse_polar_file(polar_path)
        if rc == 0 and polar_rows:
            metric_val = selection_metric_from_polar_rows(polar_rows)
            log_case(case_id, code_str, "OK", f"attempt={attempt} metric={metric_val}", log_path)
            with open(log_path, "w") as lf:
                lf.write("=== STDOUT ===\n" + stdout_total + "\n=== STDERR ===\n" + stderr_total)
            result.update({
                "ok": True,
                "metric": metric_val,
                "polar": polar_path
            })
            return result

        log_case(case_id, code_str, "FAILED", f"attempt={attempt} rc={rc}", log_path)
        if attempt <= MAX_RETRIES and retry_relaxed:
            panels = max(RELAXED_PANELS, panels // 2)
            aseq = RELAXED_ASEQ
            timeout = max(timeout * 2, 120)
            iter_limit = max(50, iter_limit // 2)
            time.sleep(0.5)
            continue

        with open(log_path, "w") as lf:
            lf.write("=== STDOUT ===\n" + stdout_total + "\n=== STDERR ===\n" + stderr_total)
            result.update({
                "ok": False,
                "metric": float("-inf"),
                "polar": polar_path if os.path.exists(polar_path) else None
            })
            return result
# =============================================================================
# CSV HANDLERS
# =============================================================================
CSV_FIELDS = ["timestamp", "case_id", "code", "L", "P", "S", "TT", "ok", "metric", "polar", "log"]


def write_result_row(res: dict[str, Any]) -> None:
    """
    Appends one result row to the global results CSV.

    Args:
        res (dict[str, Any]): Case result dictionary from `evaluate_naca_case`.
    """
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **{k: res.get(k) for k in ("case_id", "code", "L", "P", "S", "TT", "ok", "metric", "polar", "log")},
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_results_csv() -> list[dict[str, Any]]:
    """
    Loads the results CSV file into memory, coercing numeric fields.

    Returns:
        list[dict[str, Any]]: Parsed results list.
    """
    rows = []
    if not os.path.exists(RESULTS_CSV):
        return rows
    with open(RESULTS_CSV, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                r["metric"] = float(r["metric"]) if r["metric"] else float("-inf")
            except Exception:
                r["metric"] = float("-inf")
            r["ok"] = str(r.get("ok")).lower() in ("true", "1")
            rows.append(r)
    return rows


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main() -> None:
    """
    Main control flow for exhaustive XFOIL evaluation.

    Steps:
        1. Prepare directories and candidate list.
        2. Skip completed cases (resume support).
        3. Run parallel coarse sweep.
        4. Identify top candidates and re-evaluate at high fidelity.
        5. Save results and print summary.
    """
    ensure_workdir()
    candidates = [(L, P, S, TT) for L, P, S, TT in product(L_vals, P_vals, S_vals, TT_vals)]
    print(f"Total candidates: {len(candidates)}")

    completed_codes = set()
    if os.path.exists(RESULTS_CSV):
        existing = load_results_csv()
        completed_codes = {r["code"] for r in existing}

    worklist = [(i, L, P, S, TT) for i, (L, P, S, TT) in enumerate(candidates, 1) if naca5_code_to_str(L, P, S, TT) not in completed_codes]
    print(f"Worklist size: {len(worklist)}")

    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_naca_case, cid, L, P, S, TT, COARSE_SETTINGS, True): (cid, L, P, S, TT) for cid, L, P, S, TT in worklist}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            write_result_row(res)
            print(f"[{len(results)}/{len(worklist)}] code={res['code']} ok={res['ok']} metric={res['metric']:.3f}")

    all_rows = load_results_csv()
    ranked = sorted(all_rows, key=lambda r: r["metric"], reverse=True)
    top = ranked[:10]
    print("Top coarse candidates:")
    for r in top:
        print(r["code"], r["metric"])

    print("Re-running top 10 at high fidelity...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(MAX_WORKERS, 10)) as ex:
        for fut in concurrent.futures.as_completed(
            {ex.submit(evaluate_naca_case, 10000 + i, int(r["code"][0]), int(r["code"][1]), int(r["code"][2]), int(r["code"][3:5]), FINE_SETTINGS, True): r["code"] for i, r in enumerate(top)}
        ):
            hf = fut.result()
            write_result_row(hf)
            print("HF:", hf["code"], "metric=", hf["metric"])

    final_rows = load_results_csv()
    best = max(final_rows, key=lambda r: r["metric"])
    print(f"Best final NACA code: {best['code']} metric={best['metric']}")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Total time: {(time.time() - start)/60:.2f} min")


if __name__ == "__main__":
    main()

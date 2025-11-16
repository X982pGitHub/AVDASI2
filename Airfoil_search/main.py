"""
Performs a full exhaustive sweep of all standard NACA 4 to 5-digit airfoils
using XFOIL’s internal geometry generation and aerodynamic analysis.

Features:
    - Enumerates NACA combinations.
    - Runs XFOIL in parallel via subprocess.
    - Handles timeouts and retries with relaxed parameters.
    - Logs all outputs and results for later analysis.
    - Computes and ranked based on selection metric.
        - 3D airfoil calculation
    - Saves to CSV.

Author:
    Was mostly ChatGPT, now decent amount by Thitut Uhthalye, 2025
"""

import os
import shutil
import subprocess
import time
import numpy as np
import csv
import concurrent.futures
from itertools import product
import datetime
from typing import Any, Optional, Hashable

import pandas as pd
from config import settings


# =============================================================================
# METRIC FUNCTION
# =============================================================================
def values_from_polar_rows(polar_rows: list[dict[str, Any]], NACA: str | float) -> dict[str, float]:
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
    C_l, C_d, AoA = [], [], []
    for r in polar_rows:
        try:
            CL = float(r.get("CL", float("nan")))
            CD = float(r.get("CD", float("nan")))
            alpha = float(r.get("alpha", float("nan")))
            if CD <= 0 or np.isnan(CL) or np.isnan(CD) or np.isnan(alpha):
                continue
            C_l.append(CL)
            C_d.append(CD)
            AoA.append(alpha)
        except Exception:
            continue
    val = compute_metric(AoA, C_l,  C_d, NACA, settings.flight_params.aspect_ratio, settings.flight_params.span_efficiency, settings.flight_params.friction_coeff, [-2.0, 6.0])
    if (val["metric"] <= -1e8):
        val["metric"] = -np.inf
    return val


def compute_metric(AoA, C_l2d, C_d2d, NACA, AR, e, mu_s, fit_range_deg):
    t_max = int(NACA[-2:]) / 100.0
    AoA = np.asarray(AoA)
    AoA_rad = np.radians(AoA)
    C_l2d = np.asarray(C_l2d)
    C_d2d = np.asarray(C_d2d)

    # Linear fit to estimate a2D and zero-lift AoA
    fit_mask = (AoA >= fit_range_deg[0]) & (AoA <= fit_range_deg[1])
    A = np.vstack([AoA_rad[fit_mask], np.ones_like(AoA_rad[fit_mask])]).T
    a2D, b = np.linalg.lstsq(A, C_l2d[fit_mask], rcond=None)[0]
    alpha_L0 = -b / a2D  # zero-lift AoA [rad]

    # Compute 3D lift and induced drag
    a3D = a2D / (1 + a2D / (np.pi * AR * e))
    C_l3d = a3D * (AoA_rad - alpha_L0)
    C_di = C_l3d**2 / (np.pi * AR * e)
    C_dtotal = C_d2d + C_di

    M = C_dtotal - mu_s * C_l3d
    LD = np.divide(C_l3d, C_dtotal, out=np.full_like(C_l3d, np.nan), where=C_dtotal > 0)

    idxmaxC_l = np.argmax(C_l3d)
    AoAmaxC_l = AoA[idxmaxC_l]

    values = np.array([np.min(M), np.max(LD), np.max(C_l3d), AoAmaxC_l, np.max(C_dtotal), t_max])

    weights = settings.metric_weights.array
    metric = values.dot(weights)
    return {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'metric': metric, 
            'Min M': np.min(M),
            'Max L/D': np.max(LD),
            'C_l': np.max(C_l3d),
            'AoA at max C_l': AoAmaxC_l,
            'Max drag': np.max(C_dtotal),
            'Max thickness': t_max,
            }

def compute_normalised_results(interm_result = settings.paths.result_csv) -> pd.DataFrame|None:
    if not os.path.exists(interm_result):
        print("No intermediate results found, normalisation skipped.")
        return
    
    df = pd.read_csv(interm_result, dtype={"code": str})

    df_ok = df[df['ok'] == True].copy()

    if len(df_ok) == 0:
        print("Intermediate results yields no successful run, normalisation skipped.")
        return
    
    metric_cols = ['Min M', 'Max L/D', 'C_l', 'AoA at max C_l', 'Max drag', 'Max thickness']

    missing_cols = [col for col in metric_cols if col not in df_ok.columns]
    if missing_cols:
        print(f"Warning: Missing columns for normalisation: {missing_cols}")
        return
    
    metric_cols = [c for c in metric_cols if c in df_ok.columns]
    max_vals = df_ok[metric_cols].abs().max().to_numpy()
    norm_vals = np.divide(df_ok[metric_cols].to_numpy(dtype=float), max_vals, where=max_vals != 0)

    weights = settings.metric_weights.array

    metric_norm = norm_vals.dot(weights)

    df_norm = pd.DataFrame(norm_vals, columns=metric_cols)
    df_norm.insert(0, "code", df_ok["code"].to_numpy())
    df_norm["metric"] = metric_norm

    df_norm.sort_values("metric", ascending=False, inplace=True, ignore_index=True)
    return df_norm


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_workdir(purge = True) -> None:
    """
    Ensures that the working directory and its subdirectories exist.

    Creates the following structure:
        WORKDIR/
            polars/
            logs/
    """
    workdir = str(settings.paths.workdir)
    if purge:
        shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "polars"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "logs"), exist_ok=True)

def loading_wheel(future: concurrent.futures.Future[Any]) -> None:
    animation = "|/-\\"
    idx = 0
    while not future.done():
        print(animation[idx % len(animation)], end="\r")
        idx += 1
        time.sleep(0.1)


def naca5_code_to_str(L: int, P: int, S: int, TT: int) -> Optional[str]:
    """
    Formats a NACA 5-digit airfoil code.

    Args:
        L (int): Design lift coefficient index (0–9).
        P (int): Position of maximum camber index (1–5).
        S (int): Reflex flag (0 = normal, 1 = reflexed).
        TT (int): Maximum thickness in percent (06–24).

    Returns:
        Optional[str]: Formatted code, e.g. '23012', or None if resulting code
        would be invalid NACA5 for XFOIL
    """
    code = f"{int(L)}{int(P)}{int(S):d}{int(TT):02d}"
    # Check valid NACA5
    if (L != 0) and not (P in range(1, 6) and (S == 0)):
        return None
    return code


def build_xfoil_script_for_naca(
    code_str: str,
    polar_outpath: str,
    panels: int,
    aseq: str,
    Re: Optional[float],
    mach: Optional[float],
    iter_limit: int,
    save_dump: bool = False
) -> bytes:
    """
    Builds the batch command script to send to XFOIL.

    Args:
        code_str (str): NACA 5-digit code string.
        polar_outpath (str): Destination path for the polar file.
        panels (int): Number of surface panels for meshing.
        aseq (str): Alpha sequence command string (e.g. '-5 15 0.5').
        Re (float | None): Reynolds number for viscous mode.
        mach (float | None): Mach number for viscous mode.
        iter_limit (int): Iteration limit per angle.
        save_dump (bool): Save polar dump file, default: `False`

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
    if Re is not None:
        lines.extend([
            f"v {Re}",
            f"M {mach}",
        ])
    lines.extend([
        f"ITER {iter_limit}",
        f"PACC",
        f"{polar_outpath}",
        f"{f"{polar_outpath}.dump" if save_dump else ""}",
        f"ASEQ {aseq}",
        "PACC",
        "",
        "QUIT"
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
            [settings.paths.xfoil_cmd],
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
    xfoil_settings: settings.XFOILParams,
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
        xfoil_settings settings.XFOILParams: XFOIL solver settings.
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
    maybe_code = naca5_code_to_str(L, P, S, TT)
    assert maybe_code is not None, "Invalid NACA code should have been filtered out"
    code_str = maybe_code
    polar_name = f"naca_{code_str}.polar"
    polar_path = os.path.join(settings.paths.workdir, "polars", polar_name)
    log_path = os.path.join(settings.paths.workdir, "logs", f"case_{case_id}_{code_str}.log")

    panels = xfoil_settings.panels or 140
    aseq = xfoil_settings.aseq or "-5 15 0.5"
    Re = xfoil_settings.Re or None
    mach = xfoil_settings.mach or None
    iter_limit = xfoil_settings.iter or 200
    timeout = xfoil_settings.timeout or 60

    attempt = 0
    stdout_total = ""
    stderr_total = ""

    result = {
        "case_id": case_id,
        "code": code_str,
        "ok": False,
        "metric": float("-inf"),
        "polar": None,
        "log": log_path,
    }
    
    while attempt <= settings.solver_params.max_retries:
        attempt += 1
        if os.path.exists(polar_path):
            os.remove(polar_path)
        script = build_xfoil_script_for_naca(code_str, polar_path, panels, aseq, Re, mach, iter_limit, settings.execution.save_pol_dump)
        rc, out, err = run_xfoil_script(script, timeout)
        stdout_total += out
        stderr_total += err

        polar_rows = parse_polar_file(polar_path)
        if rc == 0 and polar_rows:
            val = values_from_polar_rows(polar_rows, code_str)
            log_case(case_id, code_str, "OK", f"attempt={attempt} metric={val["metric"]}", log_path)
            with open(log_path, "w") as lf:
                lf.write("=== STDOUT ===\n" + stdout_total + "\n=== STDERR ===\n" + stderr_total)
            result.update(val)
            result.update({
                "ok": True,
                "polar": polar_path
            })
            return result

        log_case(case_id, code_str, "FAILED", f"attempt={attempt} rc={rc}", log_path)
        if attempt <= settings.solver_params.max_retries and retry_relaxed:
            panels = max(settings.solver_params.relaxed.panels, panels // 2)
            aseq = settings.solver_params.relaxed.aseq
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
CSV_FIELDS = ["timestamp", "case_id", "code", "ok", "metric", 'Min M', 'Max L/D', 'C_l', 'AoA at max C_l', 'Max drag', 'Max thickness', "polar", "log"]


def write_result_row(res: dict[str, Any], update_existing: bool = False) -> None:
    """Appends or updates a row in the results CSV using pandas."""
    row = {k: res.get(k) for k in CSV_FIELDS}

    row_df = pd.DataFrame([row])
    row_df["code"] = row_df["code"].astype(str)
    
    if update_existing and os.path.exists(settings.paths.result_csv):
        # Load existing data
        df = pd.read_csv(settings.paths.result_csv, dtype={"code": str})

        df["code"] = df["code"].astype(str)
        df = df[df["code"] != row["code"]]

        df = pd.concat([df, row_df], ignore_index=True)[CSV_FIELDS]
        df.to_csv(settings.paths.result_csv, index=False)
    else:
        # Append mode
        row_df.to_csv(settings.paths.result_csv, mode='a', header=not os.path.exists(settings.paths.result_csv), index=False)


def load_results_csv(csv_path) -> list[dict[str | Hashable, Any]]:
    """
    Loads the results CSV file into memory, coercing numeric fields.

    Returns:
        list[dict[str, Any]]: Parsed results list.
    """
    if not os.path.exists(csv_path):
        return []
        
    df = pd.read_csv(csv_path, dtype={"code": str})
    # Convert metric to float, replacing errors with -inf
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce").fillna(float("-inf"))
    # Convert ok to boolean
    df["ok"] = df["ok"].astype(str).str.lower().isin(["true", "1"])
    # Convert code to string
    df['code'] = df['code'].str.zfill(5)
    # Convert to list of dicts
    return df.to_dict("records")


def reserve_hf_case(code: str, case_id: int) -> None:
    """Reserve a high-fidelity case ID for a code to prevent race conditions."""
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case_id": case_id,
        "code": code,
        "ok": False,
        "metric": float("-inf"),
        "polar": None,
        "log": None
    }
    if os.path.exists(settings.paths.result_csv):
        rows = []
        with open(settings.paths.result_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["code"] != code]
        rows.append(row)
        with open(settings.paths.result_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(settings.paths.result_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not os.path.exists(settings.paths.result_csv):
                writer.writeheader()
            writer.writerow(row)


def get_next_candidates(n: int) -> list[dict]:
    """Get next n best candidates that haven't been run in high fidelity"""
    all_rows = load_results_csv(settings.paths.result_csv)
    ranked = sorted(all_rows, key=lambda r: r["metric"], reverse=True)
    # Filter out already HF-evaluated (case_id >= 10000)
    evaluated_codes = {r["code"] for r in ranked if int(r["case_id"]) >= 10000}
    candidates = [r for r in ranked if r["code"] not in evaluated_codes][:n]
    
    return candidates


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
    ensure_workdir(settings.execution.purge)
    # Generate all possible combinations
    all_candidates = list(product(settings.naca_ranges.L_vals, settings.naca_ranges.P_vals, settings.naca_ranges.S_vals, settings.naca_ranges.TT_vals))
    print(f"Total possible combinations: {len(all_candidates)}")
    
    # Filter valid NACA codes first
    candidates = []
    for L, P, S, TT in all_candidates:
        if naca5_code_to_str(L, P, S, TT) is not None:
            candidates.append((L, P, S, TT))
    print(f"Valid NACA codes (4+ digits): {len(candidates)}")

    completed_codes = set()
    if os.path.exists(settings.paths.result_csv):
        existing = load_results_csv(settings.paths.result_csv)
        completed_codes = {r["code"] for r in existing}

    worklist = [(i, L, P, S, TT) for i, (L, P, S, TT) in enumerate(candidates, 1) 
                if naca5_code_to_str(L, P, S, TT) not in completed_codes]
    print(f"Remaining work items: {len(worklist)}")

    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings.execution.max_workers) as ex:
        futures = {ex.submit(evaluate_naca_case, cid, L, P, S, TT, settings.solver_params.coarse, True): (cid, L, P, S, TT) for cid, L, P, S, TT in worklist}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            write_result_row(res)
            status = "OK" if res['ok'] else "FAILED"
            print(f"[{len(results)}/{len(worklist)}] NACA {res['code']} status={status} metric={res['metric']:.3f}")

    print("\nGenerating normalized results...")
    results_norm: pd.DataFrame = compute_normalised_results()

    results_norm.to_csv(settings.paths.normalised_result_csv, index=False)

    print(f"\nNormalized results saved to: {settings.paths.normalised_result_csv}")
    print("Top 3 profiles by normalised metric:")
    print(results_norm[['code', 'metric']].head(3).to_string(index=False))

    all_rows = load_results_csv(settings.paths.result_csv)
    print("\nStarting high-fidelity validation...")
    running_tasks = {}
    hf_case_id = 10000
    stable_optimum_found = False
    
    max_hf_evals = hf_case_id + 30  # Limit HF evaluations
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings.execution.max_workers) as ex:
        # Initial fill of the queue
        next_batch = get_next_candidates(min(settings.execution.max_workers, max_hf_evals - hf_case_id))
        print("Top coarse candidates:")
        for c in next_batch:
            # Reserve case ID immediately
            reserve_hf_case(c["code"], hf_case_id)
            print(f'NACA {c["code"]}, metric={c["metric"]}')
            
            future = ex.submit(
                evaluate_naca_case,
                hf_case_id,
                int(c["code"][0]),
                int(c["code"][1]),
                int(c["code"][2]),
                int(c["code"][3:5]),
                settings.solver_params.fine,
                True
            )
            running_tasks[future] = c["code"]
            hf_case_id += 1
            
        while running_tasks:

            done, _ = concurrent.futures.wait(
                running_tasks.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                code = running_tasks.pop(future)
                try:
                    hf = future.result()
                    write_result_row(hf, update_existing=True)
                    print(f"HF complete: NACA {code} metric={hf['metric']:.3f}")
                    
                    # Check if we found a stable optimum
                    all_rows = load_results_csv(settings.paths.result_csv)
                    current_best = max(all_rows, key=lambda r: r["metric"])
                    if int(current_best["case_id"]) >= 10000 and current_best["case_id"] == {code}:
                        print(f"\nFound stable optimum: {current_best['code']} "
                              f"with metric={current_best['metric']:.3f}")
                        stable_optimum_found = True
                except Exception as e:
                    print(f"HF failed for {code}: {str(e)}")
            
            # Add new tasks if queue not full, optimum not found, and under limit
            remaining_evals = max_hf_evals - hf_case_id
            if len(running_tasks) < settings.execution.max_workers and not stable_optimum_found and remaining_evals > 0:
                slots_available = min(settings.execution.max_workers - len(running_tasks), remaining_evals)
                next_batch = get_next_candidates(slots_available)
                if not next_batch:
                    print("Reached HF iteration limit")
                    break  # No more candidates to process
                    
                for c in next_batch:
                    # Reserve case ID immediately
                    reserve_hf_case(c["code"], hf_case_id)
                    
                    future = ex.submit(
                        evaluate_naca_case,
                        hf_case_id,
                        int(c["code"][0]),
                        int(c["code"][1]),
                        int(c["code"][2]),
                        int(c["code"][3:5]),
                        settings.solver_params.fine,
                        True
                    )
                    running_tasks[future] = c["code"]
                    hf_case_id += 1
                    loading_wheel(future)

    print(f"\nFinal Results:")
    final_rows = load_results_csv(settings.paths.result_csv)
    best = max(final_rows, key=lambda r: r["metric"])
    print(f"Best NACA code: {best['code']} metric={best['metric']:.3f}")
    print(f"Results CSV: {settings.paths.result_csv}")
    print(f"Total time: {(time.time() - start)/60:.2f} min")

if __name__ == "__main__":
    main()

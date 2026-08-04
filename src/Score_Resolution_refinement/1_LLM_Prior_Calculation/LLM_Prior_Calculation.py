# -*- coding: utf-8 -*-

import os
import json
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PAIR_JSON_FOLDER = "pair_json_outputs"

SE_SUMMARY_CSV = "semantic_entropy_outputs/semantic_entropy_summary.csv"

OUTPUT_FOLDER = "llm_prior_outputs"

# Manual sparse baseline prior pi
BASELINE_PI = 0.10

# Numerical safety to avoid log(0) and log(infinity)
EPS = 1e-8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        val = float(x)
        if np.isfinite(val):
            return val
        return default
    except Exception:
        return default


def clip_prob(p, eps=EPS):
   
    p = safe_float(p, default=np.nan)

    if pd.isna(p):
        return np.nan

    return min(max(float(p), eps), 1.0 - eps)


def get_parsed_response(resp):
    

    parsed = resp.get("parsed_response")

    if isinstance(parsed, dict):
        return parsed

    raw = resp.get("raw_json_text")

    if isinstance(raw, str) and raw.strip():
        try:
            parsed_raw = json.loads(raw)
            if isinstance(parsed_raw, dict):
                return parsed_raw
        except Exception:
            return {}

    return {}


def interpret_plausible_value(parsed_response):
    

    plausible = parsed_response.get("plausible", None)

    if plausible is True:
        return 1

    if plausible is False:
        return 0

       # This is only used when parsed_response["plausible"] is absent.
    judgment = clean_text(parsed_response.get("judgment", "")).lower()

    if "implausible" in judgment or "not plausible" in judgment:
        return 0

    if "plausible" in judgment:
        return 1

    return np.nan


# ============================================================
# PROCESS JSON FILES
# ============================================================

def process_one_pair_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        pair_data = json.load(f)

    pair_file = os.path.basename(json_path)

    pair_index = pair_data.get("pair_index", None)
    cause_variable = clean_text(pair_data.get("cause_variable", ""))
    effect_variable = clean_text(pair_data.get("effect_variable", ""))
    direction = clean_text(
        pair_data.get("direction", f"{cause_variable} -> {effect_variable}")
    )

    responses = pair_data.get("responses", [])

    response_rows = []

    for i, resp in enumerate(responses, start=1):
        parsed = get_parsed_response(resp)

        plausible_binary = interpret_plausible_value(parsed)

        response_rows.append({
            "pair_file": pair_file,
            "pair_index": pair_index,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "response_index": resp.get("response_index", i),
            "response_id": resp.get("response_id", ""),
            "plausible_raw": parsed.get("plausible", None),
            "judgment": parsed.get("judgment", ""),
            "plausible_binary": plausible_binary
        })

    return response_rows


def load_all_response_plausibility():
    json_files = sorted(glob.glob(os.path.join(PAIR_JSON_FOLDER, "*.json")))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON files found inside folder: {PAIR_JSON_FOLDER}"
        )

    all_rows = []

    for json_path in json_files:
        all_rows.extend(process_one_pair_json(json_path))

    response_df = pd.DataFrame(all_rows)

    return response_df


# ============================================================
# CALCULATE p_LLM = plausible_count / total_valid_count
# ============================================================

def calculate_pair_level_p_llm(response_df):
    group_cols = [
        "pair_file",
        "pair_index",
        "cause_variable",
        "effect_variable",
        "direction"
    ]

    pair_rows = []

    for keys, g in response_df.groupby(group_cols, dropna=False):
        pair_file, pair_index, cause_variable, effect_variable, direction = keys

        valid_g = g.dropna(subset=["plausible_binary"]).copy()

        total_responses = len(g)
        total_valid_responses = len(valid_g)

        if total_valid_responses == 0:
            plausible_count = 0
            implausible_count = 0
            p_llm = np.nan
            majority_label = "no_valid_responses"
        else:
            plausible_count = int(valid_g["plausible_binary"].sum())
            implausible_count = int(total_valid_responses - plausible_count)

            p_llm = plausible_count / total_valid_responses

            if plausible_count > implausible_count:
                majority_label = "majority_plausible"
            elif implausible_count > plausible_count:
                majority_label = "majority_implausible"
            else:
                majority_label = "tie"

        pair_rows.append({
            "pair_file": pair_file,
            "pair_index": pair_index,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "total_responses": total_responses,
            "total_valid_responses": total_valid_responses,
            "plausible_count": plausible_count,
            "implausible_count": implausible_count,
            "p_llm_proportion_plausible": p_llm,
            "majority_label": majority_label
        })

    pair_df = pd.DataFrame(pair_rows)

    return pair_df


# ============================================================
# CALCULATE LLM PRIOR USING NORMALIZED SE
# ============================================================

def calculate_prior_values(p_llm, se_norm, baseline_pi):
   
    p_llm = safe_float(p_llm, default=np.nan)
    se_norm = safe_float(se_norm, default=np.nan)

    if pd.isna(p_llm):
        return {
            "w_reliability": np.nan,
            "p_adj_llm_prior_probability": np.nan,
            "llm_prior_log_odds": np.nan,
            "baseline_log_odds": np.nan,
            "llm_prior_baseline_relative_log_odds": np.nan,
            "prior_status": "missing_p_llm"
        }

    if pd.isna(se_norm):
        return {
            "w_reliability": np.nan,
            "p_adj_llm_prior_probability": np.nan,
            "llm_prior_log_odds": np.nan,
            "baseline_log_odds": np.nan,
            "llm_prior_baseline_relative_log_odds": np.nan,
            "prior_status": "missing_semantic_entropy_normalized"
        }

    
    se_norm = min(max(float(se_norm), 0.0), 1.0)

    # Reliability weight
    w = 1.0 - se_norm

    # Adjusted prior probability
    p_adj = baseline_pi + w * (p_llm - baseline_pi)

    # Clip after adjustment to avoid log(0)
    p_adj_clipped = clip_prob(p_adj)
    baseline_pi_clipped = clip_prob(baseline_pi)

    llm_prior_log_odds = math.log(
        p_adj_clipped / (1.0 - p_adj_clipped)
    )

    baseline_log_odds = math.log(
        baseline_pi_clipped / (1.0 - baseline_pi_clipped)
    )

    baseline_relative_log_odds = (
        llm_prior_log_odds - baseline_log_odds
    )

    return {
        "w_reliability": w,
        "p_adj_llm_prior_probability": p_adj,
        "p_adj_clipped_for_log": p_adj_clipped,
        "llm_prior_log_odds": llm_prior_log_odds,
        "baseline_log_odds": baseline_log_odds,
        "llm_prior_baseline_relative_log_odds": baseline_relative_log_odds,
        "prior_status": "ok"
    }


def merge_with_se_and_calculate_prior(pair_df, se_df):
    required_cols = {
        "cause_variable",
        "effect_variable",
        "direction",
        "semantic_entropy_normalized"
    }

    missing = required_cols - set(se_df.columns)

    if missing:
        raise ValueError(
            f"SE summary CSV is missing required columns: {missing}"
        )

    se_cols = [
        "cause_variable",
        "effect_variable",
        "direction",
        "semantic_entropy_nats",
        "semantic_entropy_bits",
        "semantic_entropy_normalized",
        "n_clusters",
        "n_valid_responses_used"
    ]

    se_cols = [c for c in se_cols if c in se_df.columns]

    merged = pair_df.merge(
        se_df[se_cols],
        on=["cause_variable", "effect_variable", "direction"],
        how="left"
    )

    result_rows = []

    for _, row in merged.iterrows():
        prior_values = calculate_prior_values(
            p_llm=row["p_llm_proportion_plausible"],
            se_norm=row["semantic_entropy_normalized"],
            baseline_pi=BASELINE_PI
        )

        output_row = row.to_dict()
        output_row["baseline_pi"] = BASELINE_PI
        output_row.update(prior_values)

        result_rows.append(output_row)

    final_df = pd.DataFrame(result_rows)

    return final_df


# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(SE_SUMMARY_CSV):
        raise FileNotFoundError(
            f"Semantic entropy summary file not found: {SE_SUMMARY_CSV}"
        )

    se_df = pd.read_csv(SE_SUMMARY_CSV)

    response_df = load_all_response_plausibility()

    pair_df = calculate_pair_level_p_llm(response_df)

    final_df = merge_with_se_and_calculate_prior(pair_df, se_df)

    response_output = os.path.join(
        OUTPUT_FOLDER,
        "01_response_plausibility_counts.csv"
    )

    final_output = os.path.join(
        OUTPUT_FOLDER,
        "02_llm_prior_from_proportion_plausible.csv"
    )

    response_df.to_csv(response_output, index=False)
    final_df.to_csv(final_output, index=False)

    print("LLM prior calculation completed.")
    print(f"Responses processed: {len(response_df)}")
    print(f"Directed pairs processed: {len(final_df)}")
    print(f"Manual baseline pi: {BASELINE_PI}")

    print("\nSaved:")
    print(response_output)
    print(final_output)

    print("\nImportant final columns:")
    important_cols = [
        "cause_variable",
        "effect_variable",
        "direction",
        "plausible_count",
        "implausible_count",
        "p_llm_proportion_plausible",
        "semantic_entropy_normalized",
        "w_reliability",
        "p_adj_llm_prior_probability",
        "llm_prior_log_odds",
        "llm_prior_baseline_relative_log_odds",
        "prior_status"
    ]

    for col in important_cols:
        print(col)

    print("\nPreview:")
    preview_cols = [c for c in important_cols if c in final_df.columns]
    print(final_df[preview_cols].head(10))


if __name__ == "__main__":
    main()
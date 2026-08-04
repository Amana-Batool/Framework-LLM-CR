# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import networkx as nx

# =====================================================
# 1. USER INPUTS
# =====================================================

common_adj_path = "common_edges_adjacency.csv"
disputed_edges_path = "disputed_edges.csv"

bic_scores_path = "bic_scores_for_disputed_edges.csv"

hsic_scores_path = "fast_hsic_scores_for_disputed_edges.csv"

llm_prior_path = "02_llm_prior_from_proportion_plausible.csv"

output_dir = "combined_score_disputed_edge_results"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 2. CHOOSE LLM PRIOR TO USE
# =====================================================


llm_prior_mode = "llm_prior_log_odds"

# =====================================================
# 3. SCORE WEIGHTS
# =====================================================

w_bic = 0.40
w_hsic = 0.25
w_llm = 0.35
combined_score_threshold = 0.0

# =====================================================
# 4. LOAD FILES
# =====================================================

common_adj = pd.read_csv(common_adj_path, index_col=0)
common_adj = common_adj.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

disputed_df = pd.read_csv(disputed_edges_path)
bic_df = pd.read_csv(bic_scores_path)
hsic_df = pd.read_csv(hsic_scores_path)
llm_df = pd.read_csv(llm_prior_path)

variables = list(common_adj.index)

# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def normalize_edge_columns(df):
   

    df = df.copy()

    if "parent" in df.columns and "child" in df.columns:
        df["parent"] = df["parent"].astype(str).str.strip()
        df["child"] = df["child"].astype(str).str.strip()
        return df

    if "cause_variable" in df.columns and "effect_variable" in df.columns:
        df["parent"] = df["cause_variable"].astype(str).str.strip()
        df["child"] = df["effect_variable"].astype(str).str.strip()
        return df

    if "edge" in df.columns:
        split_edges = df["edge"].astype(str).str.split("->", expand=True)

        if split_edges.shape[1] != 2:
            raise ValueError("edge column must have format parent->child")

        df["parent"] = split_edges[0].str.strip()
        df["child"] = split_edges[1].str.strip()
        return df

    if "direction" in df.columns:
        split_edges = df["direction"].astype(str).str.split("->", expand=True)

        if split_edges.shape[1] != 2:
            raise ValueError("direction column must have format parent -> child")

        df["parent"] = split_edges[0].str.strip()
        df["child"] = split_edges[1].str.strip()
        return df

    raise ValueError(
        "DataFrame must contain parent/child, cause_variable/effect_variable, edge, or direction columns."
    )


def adjacency_to_edges(adj_df):
    edges = set()

    for parent in adj_df.index:
        for child in adj_df.columns:
            if parent == child:
                continue

            if adj_df.loc[parent, child] == 1:
                edges.add((str(parent), str(child)))

    return edges


def is_acyclic(edges, variables):
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(G)


def creates_cycle(current_edges, new_edge, variables):
    test_edges = set(current_edges)
    test_edges.add(new_edge)
    return not is_acyclic(test_edges, variables)


def robust_normalize(series):
    
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)

    median = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or abs(iqr) < 1e-12:
        std = s.std()

        if pd.isna(std) or std < 1e-12:
            return pd.Series(0.0, index=series.index)

        z = (s - median) / std
    else:
        z = (s - median) / iqr

    z = z.clip(-3, 3) / 3.0
    z = z.fillna(0.0)

    return z


def prepare_llm_prior(llm_df, mode="adjusted_probability"):
    

    llm_df = normalize_edge_columns(llm_df)

    if mode == "adjusted_probability":
        required_col = "p_adj_llm_prior_probability"

        if required_col not in llm_df.columns:
            raise ValueError(f"Missing column in LLM file: {required_col}")

        llm_df["llm_prior_raw"] = pd.to_numeric(
            llm_df[required_col],
            errors="coerce"
        )

        llm_df["llm_prior_type"] = required_col
        llm_df["llm_is_probability"] = True

    elif mode == "proportion_plausible":
        required_col = "p_llm_proportion_plausible"

        if required_col not in llm_df.columns:
            raise ValueError(f"Missing column in LLM file: {required_col}")

        llm_df["llm_prior_raw"] = pd.to_numeric(
            llm_df[required_col],
            errors="coerce"
        )

        llm_df["llm_prior_type"] = required_col
        llm_df["llm_is_probability"] = True

    elif mode == "entropy_inverse":
        required_col = "semantic_entropy_normalized"

        if required_col not in llm_df.columns:
            raise ValueError(f"Missing column in LLM file: {required_col}")

        entropy = pd.to_numeric(llm_df[required_col], errors="coerce")
        llm_df["llm_prior_raw"] = 1.0 - entropy

        llm_df["llm_prior_type"] = "1_minus_semantic_entropy_normalized"
        llm_df["llm_is_probability"] = True

    elif mode == "relative_log_odds":
        required_col = "llm_prior_baseline_relative_log_odds"

        if required_col not in llm_df.columns:
            raise ValueError(f"Missing column in LLM file: {required_col}")

        llm_df["llm_prior_raw"] = pd.to_numeric(
            llm_df[required_col],
            errors="coerce"
        )

        llm_df["llm_prior_type"] = required_col
        llm_df["llm_is_probability"] = False

    else:
        raise ValueError(
            "llm_prior_mode must be one of: "
            "'adjusted_probability', 'proportion_plausible', "
            "'entropy_inverse', 'relative_log_odds'"
        )

    keep_cols = [
        "parent",
        "child",
        "direction",
        "total_responses",
        "total_valid_responses",
        "plausible_count",
        "implausible_count",
        "p_llm_proportion_plausible",
        "majority_label",
        "semantic_entropy_normalized",
        "p_adj_llm_prior_probability",
        "llm_prior_log_odds",
        "baseline_log_odds",
        "llm_prior_baseline_relative_log_odds",
        "prior_status",
        "llm_prior_raw",
        "llm_prior_type",
        "llm_is_probability"
    ]

    keep_cols = [c for c in keep_cols if c in llm_df.columns]

    llm_keep = llm_df[keep_cols].copy()

    # If duplicate directed edges exist, aggregate them
    numeric_cols = llm_keep.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in llm_keep.columns if c not in numeric_cols]

    agg_dict = {}

    for col in numeric_cols:
        if col not in ["parent", "child"]:
            agg_dict[col] = "mean"

    for col in non_numeric_cols:
        if col not in ["parent", "child"]:
            agg_dict[col] = "first"

    llm_keep = llm_keep.groupby(["parent", "child"], as_index=False).agg(agg_dict)

    return llm_keep


# =====================================================
# 6. CLEAN EDGE FILES
# =====================================================

disputed_df = normalize_edge_columns(disputed_df)
bic_df = normalize_edge_columns(bic_df)
hsic_df = normalize_edge_columns(hsic_df)
llm_prior_df = prepare_llm_prior(llm_df, mode=llm_prior_mode)

# Base disputed edge file
base_cols = [
    "parent",
    "child",
    "edge",
    "present_in",
    "absent_in",
    "dispute_type",
    "reverse_edge"
]

base_cols = [c for c in base_cols if c in disputed_df.columns]

base_df = disputed_df[base_cols].copy()

if "edge" not in base_df.columns:
    base_df["edge"] = base_df["parent"] + "->" + base_df["child"]

# BIC
if "bic_gain" not in bic_df.columns:
    raise ValueError("BIC file must contain column: bic_gain")

bic_cols = [
    "parent",
    "child",
    "bic_without_edge",
    "bic_with_edge",
    "bic_gain",
    "decision_by_bic"
]

bic_cols = [c for c in bic_cols if c in bic_df.columns]
bic_keep = bic_df[bic_cols].copy()

# HSIC
if "hsic_gain" not in hsic_df.columns:
    raise ValueError("HSIC file must contain column: hsic_gain")

hsic_cols = [
    "parent",
    "child",
    "hsic_without_edge",
    "hsic_with_edge",
    "hsic_gain",
    "decision_by_hsic"
]

hsic_cols = [c for c in hsic_cols if c in hsic_df.columns]
hsic_keep = hsic_df[hsic_cols].copy()

# Aggregate duplicates safely
bic_keep = bic_keep.groupby(["parent", "child"], as_index=False).agg({
    col: "mean" if pd.api.types.is_numeric_dtype(bic_keep[col]) else "first"
    for col in bic_keep.columns
    if col not in ["parent", "child"]
})

hsic_keep = hsic_keep.groupby(["parent", "child"], as_index=False).agg({
    col: "mean" if pd.api.types.is_numeric_dtype(hsic_keep[col]) else "first"
    for col in hsic_keep.columns
    if col not in ["parent", "child"]
})

# =====================================================
# 7. MERGE BIC + HSIC + LLM PRIOR
# =====================================================

combined_df = base_df.merge(
    bic_keep,
    on=["parent", "child"],
    how="left"
)

combined_df = combined_df.merge(
    hsic_keep,
    on=["parent", "child"],
    how="left"
)

combined_df = combined_df.merge(
    llm_prior_df,
    on=["parent", "child"],
    how="left"
)

# =====================================================
# 8. CHECK MISSING SCORES
# =====================================================

missing_bic = combined_df["bic_gain"].isna().sum()
missing_hsic = combined_df["hsic_gain"].isna().sum()
missing_llm = combined_df["llm_prior_raw"].isna().sum()

print("\n========== MISSING SCORE CHECK ==========")
print("Missing BIC scores:", missing_bic)
print("Missing HSIC scores:", missing_hsic)
print("Missing LLM priors:", missing_llm)

# Fill missing LLM prior with neutral 0.5 if probability mode
if llm_prior_mode in ["adjusted_probability", "proportion_plausible", "entropy_inverse"]:
    combined_df["llm_prior_raw"] = pd.to_numeric(
        combined_df["llm_prior_raw"],
        errors="coerce"
    ).fillna(0.5)

    combined_df["llm_prior_raw"] = combined_df["llm_prior_raw"].clip(0.0, 1.0)

else:
    combined_df["llm_prior_raw"] = pd.to_numeric(
        combined_df["llm_prior_raw"],
        errors="coerce"
    )

# =====================================================
# 9. NORMALIZE SCORES
# =====================================================

combined_df["bic_norm"] = robust_normalize(combined_df["bic_gain"])
combined_df["hsic_norm"] = robust_normalize(combined_df["hsic_gain"])

if llm_prior_mode in ["adjusted_probability", "proportion_plausible", "entropy_inverse"]:
    # Probability [0, 1] to score [-1, 1]
    combined_df["llm_norm"] = 2.0 * combined_df["llm_prior_raw"] - 1.0
else:
    # Log-odds style value
    combined_df["llm_norm"] = robust_normalize(combined_df["llm_prior_raw"])

# =====================================================
# 10. COMBINED SCORE
# =====================================================

weight_sum = w_bic + w_hsic + w_llm

w_bic = w_bic / weight_sum
w_hsic = w_hsic / weight_sum
w_llm = w_llm / weight_sum

combined_df["combined_score"] = (
    w_bic * combined_df["bic_norm"]
    + w_hsic * combined_df["hsic_norm"]
    + w_llm * combined_df["llm_norm"]
)

combined_df["initial_decision_by_combined_score"] = np.where(
    combined_df["combined_score"] > combined_score_threshold,
    "candidate_keep",
    "reject_low_score"
)

# =====================================================
# 11. GREEDY ACYCLIC SELECTION
# =====================================================

common_edges = adjacency_to_edges(common_adj)
selected_edges = set(common_edges)

selected_rows = []
rejected_rows = []

sorted_candidates = combined_df.sort_values(
    by="combined_score",
    ascending=False
).copy()

for _, row in sorted_candidates.iterrows():

    parent = str(row["parent"])
    child = str(row["child"])
    edge = (parent, child)

    row_dict = row.to_dict()

    if parent == child:
        row_dict["final_decision"] = "reject_self_edge"
        rejected_rows.append(row_dict)
        continue

    if parent not in variables or child not in variables:
        row_dict["final_decision"] = "reject_unknown_variable"
        rejected_rows.append(row_dict)
        continue

    if row["combined_score"] <= combined_score_threshold:
        row_dict["final_decision"] = "reject_low_score"
        rejected_rows.append(row_dict)
        continue

    if creates_cycle(selected_edges, edge, variables):
        row_dict["final_decision"] = "reject_cycle"
        rejected_rows.append(row_dict)
        continue

    selected_edges.add(edge)
    row_dict["final_decision"] = "keep"
    selected_rows.append(row_dict)

selected_disputed_df = pd.DataFrame(selected_rows)
rejected_disputed_df = pd.DataFrame(rejected_rows)

# =====================================================
# 12. FINAL ADJACENCY MATRIX
# =====================================================

final_adj = pd.DataFrame(
    0,
    index=variables,
    columns=variables
)

for parent, child in selected_edges:
    final_adj.loc[parent, child] = 1

final_edge_df = pd.DataFrame(
    sorted(selected_edges),
    columns=["parent", "child"]
)

final_edge_df["edge"] = final_edge_df["parent"] + "->" + final_edge_df["child"]

# =====================================================
# 13. SAVE OUTPUTS
# =====================================================

combined_df.to_csv(
    os.path.join(output_dir, "combined_scores_for_disputed_edges.csv"),
    index=False
)

combined_df.sort_values(
    by="combined_score",
    ascending=False
).to_csv(
    os.path.join(output_dir, "combined_scores_for_disputed_edges_sorted.csv"),
    index=False
)

selected_disputed_df.to_csv(
    os.path.join(output_dir, "combined_accepted_disputed_edges.csv"),
    index=False
)

rejected_disputed_df.to_csv(
    os.path.join(output_dir, "combined_rejected_disputed_edges.csv"),
    index=False
)

final_adj.to_csv(
    os.path.join(output_dir, "common_plus_combined_selected_adjacency.csv")
)

final_edge_df.to_csv(
    os.path.join(output_dir, "common_plus_combined_selected_edges.csv"),
    index=False
)

if selected_disputed_df.empty:
    selected_only_df = pd.DataFrame(
        columns=["parent", "child", "edge", "combined_score"]
    )
else:
    selected_only_df = selected_disputed_df[
        ["parent", "child", "edge", "combined_score"]
    ].copy()

selected_only_df.to_csv(
    os.path.join(output_dir, "selected_disputed_edges_only.csv"),
    index=False
)

summary = {
    "llm_prior_mode": llm_prior_mode,
    "common_edges_count": len(common_edges),
    "disputed_edges_total": len(combined_df),
    "selected_disputed_edges": len(selected_disputed_df),
    "rejected_disputed_edges": len(rejected_disputed_df),
    "final_edges_count": len(selected_edges),
    "missing_bic_scores_before_fill": int(missing_bic),
    "missing_hsic_scores_before_fill": int(missing_hsic),
    "missing_llm_prior_before_fill": int(missing_llm),
    "w_bic": w_bic,
    "w_hsic": w_hsic,
    "w_llm": w_llm,
    "combined_score_threshold": combined_score_threshold,
    "is_final_graph_acyclic": is_acyclic(selected_edges, variables)
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(output_dir, "combined_score_summary.csv"),
    index=False
)

# =====================================================
# 14. PRINT SUMMARY
# =====================================================

print("\n========== COMBINED SCORE SUMMARY ==========")

for k, v in summary.items():
    print(f"{k}: {v}")

print("\nFiles saved in:")
print(output_dir)

print("\nSaved files:")
print("1. combined_scores_for_disputed_edges.csv")
print("2. combined_scores_for_disputed_edges_sorted.csv")
print("3. combined_accepted_disputed_edges.csv")
print("4. combined_rejected_disputed_edges.csv")
print("5. selected_disputed_edges_only.csv")
print("6. common_plus_combined_selected_adjacency.csv")
print("7. common_plus_combined_selected_edges.csv")
print("8. combined_score_summary.csv")
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import networkx as nx

# =====================================================
# 1. USER INPUTS
# =====================================================

data_path = "Metabolic_Power_data.csv"


current_dag_path = "common_plus_combined_selected_adjacency.csv"


llm_prior_path = "02_llm_prior_from_proportion_plausible.csv"

output_dir = "dag_recovery_results"
os.makedirs(output_dir, exist_ok=True)

# Choose score mode for BIC
score_mode = "continuous"

# LLM plausibility settings
llm_prior_mode = "llm_prior_log_odds"
llm_plausible_threshold = 0.7

# Combined recovery weights
w_bic = 0.40
w_hsic = 0.25
w_llm = 0.35

# Edge is recovered only if combined_recovery_score > this
combined_recovery_threshold = 0.9

# Fast HSIC settings
max_rows_for_hsic = 5000
n_rff_features = 128
random_state = 42
encode_non_numeric = True

# If True, require BIC gain to be positive as an extra hard condition.
# If False, combined score decides.
require_positive_bic_gain = False

# If True, require HSIC gain to be positive as an extra hard condition.
# If False, combined score decides.
require_positive_hsic_gain = False


# =====================================================
# 2. LOAD FILES
# =====================================================

data = pd.read_csv(data_path)

current_adj = pd.read_csv(current_dag_path, index_col=0)
current_adj = current_adj.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

llm_df = pd.read_csv(llm_prior_path)

variables = list(current_adj.index)

missing_vars = [v for v in variables if v not in data.columns]
if missing_vars:
    raise ValueError(f"These variables are missing from data.csv: {missing_vars}")

data = data[variables].copy()


# =====================================================
# 3. BASIC HELPERS
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
        df["parent"] = split_edges[0].str.strip()
        df["child"] = split_edges[1].str.strip()
        return df

    if "direction" in df.columns:
        split_edges = df["direction"].astype(str).str.split("->", expand=True)
        df["parent"] = split_edges[0].str.strip()
        df["child"] = split_edges[1].str.strip()
        return df

    raise ValueError(
        "LLM prior file must contain parent/child, cause_variable/effect_variable, edge, or direction columns."
    )


def adjacency_to_edges(adj_df):
    edges = set()

    for parent in adj_df.index:
        for child in adj_df.columns:
            if parent == child:
                continue
            if int(adj_df.loc[parent, child]) == 1:
                edges.add((str(parent), str(child)))

    return edges


def get_parents_from_edges(edges, child):
    return sorted([p for p, c in edges if c == child])


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


# =====================================================
# 4. PREPARE LLM PRIOR FILE
# =====================================================

def prepare_llm_prior(llm_df, mode="adjusted_probability"):
   

    llm_df = normalize_edge_columns(llm_df)

    if mode == "adjusted_probability":
        col = "p_adj_llm_prior_probability"
        if col not in llm_df.columns:
            raise ValueError(f"Missing required LLM column: {col}")
        llm_df["llm_prior"] = pd.to_numeric(llm_df[col], errors="coerce")

    elif mode == "proportion_plausible":
        col = "p_llm_proportion_plausible"
        if col not in llm_df.columns:
            raise ValueError(f"Missing required LLM column: {col}")
        llm_df["llm_prior"] = pd.to_numeric(llm_df[col], errors="coerce")

    elif mode == "entropy_inverse":
        col = "semantic_entropy_normalized"
        if col not in llm_df.columns:
            raise ValueError(f"Missing required LLM column: {col}")
        entropy = pd.to_numeric(llm_df[col], errors="coerce")
        llm_df["llm_prior"] = 1.0 - entropy

    else:
        raise ValueError(
            "llm_prior_mode must be one of: adjusted_probability, proportion_plausible, entropy_inverse"
        )

    llm_df["llm_prior"] = llm_df["llm_prior"].clip(0.0, 1.0)
    plausible_col = "p_llm_proportion_plausible"

    if plausible_col not in llm_df.columns:
        raise ValueError(
            f"Missing required LLM plausibility column: {plausible_col}"
        )

    llm_df[plausible_col] = pd.to_numeric(
        llm_df[plausible_col],
        errors="coerce"
    ).clip(0.0, 1.0)

    llm_df["llm_plausible"] = (
        llm_df[plausible_col] >= llm_plausible_threshold
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
        "semantic_entropy_normalized",
        "p_adj_llm_prior_probability",
        "majority_label",
        "prior_status",
        "llm_prior",
        "llm_plausible"
    ]

    keep_cols = [c for c in keep_cols if c in llm_df.columns]

    llm_keep = llm_df[keep_cols].copy()

    # Aggregate duplicates if any
    agg_dict = {}

    for col in llm_keep.columns:
        if col in ["parent", "child"]:
            continue

        if col == "llm_plausible":
            agg_dict[col] = "max"
        elif pd.api.types.is_numeric_dtype(llm_keep[col]):
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"

    llm_keep = llm_keep.groupby(["parent", "child"], as_index=False).agg(agg_dict)

    return llm_keep

llm_prior_df = prepare_llm_prior(llm_df, mode=llm_prior_mode)


# =====================================================
# 5. NUMERIC DATA FOR HSIC
# =====================================================

def make_numeric_dataframe(df, encode_non_numeric=True):
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        numeric_col = pd.to_numeric(df[col], errors="coerce")

        if numeric_col.notna().sum() == len(df[col].dropna()):
            out[col] = numeric_col
        else:
            if not encode_non_numeric:
                raise ValueError(f"Column {col} is non-numeric.")

            codes, _ = pd.factorize(df[col].astype(str))
            out[col] = codes.astype(float)

    return out


data_num = make_numeric_dataframe(data, encode_non_numeric=encode_non_numeric)


# =====================================================
# 6. BIC FUNCTIONS
# =====================================================

def local_bic_continuous(data_df, child, parents):
    cols = [child] + parents
    df = data_df[cols].dropna().copy()

    if df.empty:
        return -np.inf

    y = pd.to_numeric(df[child], errors="coerce").values.astype(float)

    if np.isnan(y).any():
        raise ValueError(f"{child} is not numeric. Use score_mode='discrete'.")

    n = len(y)

    if parents:
        X = df[parents].apply(pd.to_numeric, errors="coerce").values.astype(float)

        if np.isnan(X).any():
            raise ValueError("Parent contains non-numeric values. Use score_mode='discrete'.")

        X = np.column_stack([np.ones(n), X])
    else:
        X = np.ones((n, 1))

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    rss = np.sum(residuals ** 2)
    sigma2 = max(rss / n, 1e-12)

    log_likelihood = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)

    k = X.shape[1] + 1

    bic = log_likelihood - 0.5 * k * math.log(n)

    return bic


def local_bic_discrete(data_df, child, parents):
    cols = [child] + parents
    df = data_df[cols].dropna().copy()

    if df.empty:
        return -np.inf

    for col in cols:
        df[col] = df[col].astype(str)

    n = len(df)

    r_i = df[child].nunique()

    if r_i <= 1:
        return 0.0

    if not parents:
        counts = df[child].value_counts()

        log_likelihood = 0.0
        for count in counts:
            prob = count / n
            log_likelihood += count * math.log(prob)

        q_i = 1
        num_params = q_i * (r_i - 1)

        bic = log_likelihood - 0.5 * num_params * math.log(n)
        return bic

    q_i = 1
    for p in parents:
        q_i *= df[p].nunique()

    parent_counts = df.groupby(parents).size().reset_index(name="parent_count")
    joint_counts = df.groupby(parents + [child]).size().reset_index(name="joint_count")

    merged = joint_counts.merge(parent_counts, on=parents, how="left")

    log_likelihood = 0.0

    for _, row in merged.iterrows():
        joint_count = row["joint_count"]
        parent_count = row["parent_count"]
        prob = joint_count / parent_count
        log_likelihood += joint_count * math.log(prob)

    num_params = q_i * (r_i - 1)

    bic = log_likelihood - 0.5 * num_params * math.log(n)

    return bic


if score_mode == "continuous":
    local_bic = local_bic_continuous
elif score_mode == "discrete":
    local_bic = local_bic_discrete
else:
    raise ValueError("score_mode must be either 'continuous' or 'discrete'.")


# =====================================================
# 7. FAST HSIC FUNCTIONS
# =====================================================

def standardize_vector(x):
    x = np.asarray(x, dtype=float).reshape(-1)

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if std < 1e-12:
        return x - mean

    return (x - mean) / std


residual_cache = {}

def get_linear_residual_cached(data_df, child, parents):
    parents = tuple(sorted(parents))
    cache_key = (child, parents)

    if cache_key in residual_cache:
        return residual_cache[cache_key]

    cols = [child] + list(parents)
    df = data_df[cols].dropna().copy()

    if len(df) < 5:
        residual_cache[cache_key] = (None, None)
        return None, None

    y = df[child].values.astype(float)
    y = standardize_vector(y)

    n = len(y)

    if len(parents) == 0:
        residual = y - np.mean(y)
        result = (residual, df.index)
        residual_cache[cache_key] = result
        return result

    X = df[list(parents)].values.astype(float)

    X = np.column_stack([
        standardize_vector(X[:, j])
        for j in range(X.shape[1])
    ])

    X = np.column_stack([np.ones(n), X])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta

    result = (residual, df.index)
    residual_cache[cache_key] = result

    return result


def subsample_xy(x, y, max_rows=500, random_state=42):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    n = len(x)

    if n != len(y):
        raise ValueError("x and y must have same length.")

    if max_rows is None or n <= max_rows:
        return x, y

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)

    return x[idx], y[idx]


def rff_features_1d(x, n_features=128, random_state=42):
    x = standardize_vector(x).reshape(-1, 1)

    rng = np.random.default_rng(random_state)

    sigma = 1.0

    W = rng.normal(
        loc=0.0,
        scale=1.0 / sigma,
        size=(1, n_features)
    )

    b = rng.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=n_features
    )

    Z = np.sqrt(2.0 / n_features) * np.cos(x @ W + b)

    return Z


def fast_hsic_rff(x, y, n_features=128, random_state=42):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) < 5:
        return np.nan

    Zx = rff_features_1d(
        x,
        n_features=n_features,
        random_state=random_state
    )

    Zy = rff_features_1d(
        y,
        n_features=n_features,
        random_state=random_state + 999
    )

    Zx = Zx - Zx.mean(axis=0, keepdims=True)
    Zy = Zy - Zy.mean(axis=0, keepdims=True)

    n = len(x)

    Cxy = (Zx.T @ Zy) / n

    hsic_value = np.sum(Cxy ** 2)

    return float(hsic_value)


def compute_hsic_gain(data_num, parent, child, old_parents, new_parents):
    residual_without, idx_without = get_linear_residual_cached(
        data_num,
        child,
        old_parents
    )

    residual_with, idx_with = get_linear_residual_cached(
        data_num,
        child,
        new_parents
    )

    if residual_without is None or residual_with is None:
        return np.nan, np.nan, np.nan

    x_without = data_num.loc[idx_without, parent].values.astype(float)
    x_with = data_num.loc[idx_with, parent].values.astype(float)

    x_without, residual_without = subsample_xy(
        x_without,
        residual_without,
        max_rows=max_rows_for_hsic,
        random_state=random_state
    )

    x_with, residual_with = subsample_xy(
        x_with,
        residual_with,
        max_rows=max_rows_for_hsic,
        random_state=random_state
    )

    hsic_without = fast_hsic_rff(
        x_without,
        residual_without,
        n_features=n_rff_features,
        random_state=random_state
    )

    hsic_with = fast_hsic_rff(
        x_with,
        residual_with,
        n_features=n_rff_features,
        random_state=random_state
    )

    hsic_gain = hsic_without - hsic_with

    return hsic_without, hsic_with, hsic_gain


# =====================================================
# 8. FIND MISSING LLM-PLAUSIBLE EDGES
# =====================================================

current_edges = adjacency_to_edges(current_adj)

llm_prior_df = llm_prior_df[
    (llm_prior_df["parent"].isin(variables)) &
    (llm_prior_df["child"].isin(variables)) &
    (llm_prior_df["parent"] != llm_prior_df["child"])
].copy()

llm_prior_df["edge"] = llm_prior_df["parent"] + "->" + llm_prior_df["child"]

llm_prior_df["already_in_current_dag"] = llm_prior_df.apply(
    lambda row: (row["parent"], row["child"]) in current_edges,
    axis=1
)

candidate_df = llm_prior_df[
    (llm_prior_df["llm_plausible"] == True) &
    (llm_prior_df["already_in_current_dag"] == False)
].copy()

candidate_df = candidate_df.reset_index(drop=True)

print("\n========== RECOVERY CANDIDATES ==========")
print("Current DAG edges:", len(current_edges))
print("LLM plausible missing candidate edges:", len(candidate_df))


# =====================================================
# 9. SCORE EACH RECOVERY CANDIDATE
# =====================================================

candidate_results = []

for _, row in candidate_df.iterrows():

    parent = str(row["parent"])
    child = str(row["child"])
    candidate_edge = (parent, child)

    old_parents = get_parents_from_edges(current_edges, child)
    new_parents = sorted(set(old_parents + [parent]))

   
    acyclic_if_added = not creates_cycle(current_edges, candidate_edge, variables)

    bic_without = local_bic(data, child, old_parents)
    bic_with = local_bic(data, child, new_parents)
    bic_gain = bic_with - bic_without

    hsic_without, hsic_with, hsic_gain = compute_hsic_gain(
        data_num,
        parent,
        child,
        old_parents,
        new_parents
    )

    candidate_results.append({
        "parent": parent,
        "child": child,
        "edge": f"{parent}->{child}",
        "llm_prior": row["llm_prior"],
        "llm_plausible": row["llm_plausible"],
        "majority_label": row.get("majority_label", ""),
        "p_llm_proportion_plausible": row.get("p_llm_proportion_plausible", np.nan),
        "semantic_entropy_normalized": row.get("semantic_entropy_normalized", np.nan),
        "p_adj_llm_prior_probability": row.get("p_adj_llm_prior_probability", np.nan),
        "old_parents_of_child": ", ".join(old_parents),
        "new_parents_of_child": ", ".join(new_parents),
        "acyclic_if_added_to_original_current_dag": acyclic_if_added,
        "bic_without_edge": bic_without,
        "bic_with_edge": bic_with,
        "bic_gain": bic_gain,
        "hsic_without_edge": hsic_without,
        "hsic_with_edge": hsic_with,
        "hsic_gain": hsic_gain,
        "preliminary_status": "scored"
    })


scores_df = pd.DataFrame(candidate_results)

if scores_df.empty:
    scores_df = pd.DataFrame(columns=[
        "parent", "child", "edge", "llm_prior", "llm_plausible",
        "old_parents_of_child", "new_parents_of_child",
        "acyclic_if_added_to_original_current_dag",
        "bic_without_edge", "bic_with_edge", "bic_gain",
        "hsic_without_edge", "hsic_with_edge", "hsic_gain",
        "combined_recovery_score", "final_recovery_decision"
    ])


# =====================================================
# 10. NORMALIZE AND COMBINE SCORES
# =====================================================

if not scores_df.empty:
    scores_df["bic_norm"] = robust_normalize(scores_df["bic_gain"])
    scores_df["hsic_norm"] = robust_normalize(scores_df["hsic_gain"])

    # LLM prior probability [0, 1] -> [-1, 1]
    scores_df["llm_norm"] = 2.0 * pd.to_numeric(
        scores_df["llm_prior"],
        errors="coerce"
    ).fillna(0.5) - 1.0

    weight_sum = w_bic + w_hsic + w_llm
    w_bic = w_bic / weight_sum
    w_hsic = w_hsic / weight_sum
    w_llm = w_llm / weight_sum

    scores_df["combined_recovery_score"] = (
        w_bic * scores_df["bic_norm"]
        + w_hsic * scores_df["hsic_norm"]
        + w_llm * scores_df["llm_norm"]
    )
else:
    scores_df["combined_recovery_score"] = []


# =====================================================
# 11. ADD EDGES FIRST, THEN REMOVE LOWEST-SCORE EDGES FROM CYCLES
# =====================================================



recovered_edges = set(current_edges)

initial_accepted_rows = []
rejected_rows = []
cycle_removed_rows = []

scores_sorted = scores_df.sort_values(
    by="combined_recovery_score",
    ascending=False
).copy()

for _, row in scores_sorted.iterrows():

    parent = str(row["parent"])
    child = str(row["child"])
    edge = (parent, child)

    row_dict = row.to_dict()

    if row["combined_recovery_score"] <= combined_recovery_threshold:
        row_dict["final_recovery_decision"] = "reject_no_combined_score_improvement"
        rejected_rows.append(row_dict)
        continue

    if require_positive_bic_gain and row["bic_gain"] <= 0:
        row_dict["final_recovery_decision"] = "reject_bic_not_improved"
        rejected_rows.append(row_dict)
        continue

    if require_positive_hsic_gain and row["hsic_gain"] <= 0:
        row_dict["final_recovery_decision"] = "reject_hsic_not_improved"
        rejected_rows.append(row_dict)
        continue

    
    recovered_edges.add(edge)
    row_dict["final_recovery_decision"] = "recover_keep_before_cycle_repair"
    initial_accepted_rows.append(row_dict)

initial_accepted_df = pd.DataFrame(initial_accepted_rows)
initial_recovered_edges = recovered_edges - current_edges

edge_to_row = {}
edge_to_score = {}

for row_dict in initial_accepted_rows:
    edge = (str(row_dict["parent"]), str(row_dict["child"]))
    edge_to_row[edge] = row_dict
    edge_to_score[edge] = float(row_dict.get("combined_recovery_score", -np.inf))


def build_digraph(edges, variables):
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(edges)
    return G


def cycle_nodes_to_edges(cycle_nodes):
   
    cycle_edges = []
    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % len(cycle_nodes)]
        cycle_edges.append((u, v))
    return cycle_edges


# Remove lowest-score newly recovered edge from each detected cycle.
# Repeat until the recovered graph becomes acyclic.
cycle_repair_iteration = 0
unresolved_current_graph_cycle = False

while True:
    G = build_digraph(recovered_edges, variables)

    if nx.is_directed_acyclic_graph(G):
        break

    try:
        cycle_nodes = next(nx.simple_cycles(G))
    except StopIteration:
        break

    cycle_repair_iteration += 1
    cycle_edges = cycle_nodes_to_edges(cycle_nodes)

    # Only remove newly recovered edges. Original current DAG edges are fixed.
    removable_edges = [
        edge for edge in cycle_edges
        if edge in initial_recovered_edges and edge in recovered_edges
    ]

    if not removable_edges:
       
        unresolved_current_graph_cycle = True
        cycle_removed_rows.append({
            "cycle_repair_iteration": cycle_repair_iteration,
            "removed_parent": "",
            "removed_child": "",
            "removed_edge": "",
            "removed_edge_score": np.nan,
            "cycle_nodes": " -> ".join(map(str, cycle_nodes)),
            "cycle_edges": ", ".join([f"{u}->{v}" for u, v in cycle_edges]),
            "reason": "cycle_contains_no_removable_recovered_edge"
        })
        break

    # Remove the newly recovered edge with the lowest combined recovery score.
    edge_to_remove = min(
        removable_edges,
        key=lambda e: edge_to_score.get(e, -np.inf)
    )

    recovered_edges.remove(edge_to_remove)

    removed_row = edge_to_row.get(edge_to_remove, {}).copy()
    removed_row["cycle_repair_iteration"] = cycle_repair_iteration
    removed_row["removed_parent"] = edge_to_remove[0]
    removed_row["removed_child"] = edge_to_remove[1]
    removed_row["removed_edge"] = f"{edge_to_remove[0]}->{edge_to_remove[1]}"
    removed_row["removed_edge_score"] = edge_to_score.get(edge_to_remove, np.nan)
    removed_row["cycle_nodes"] = " -> ".join(map(str, cycle_nodes))
    removed_row["cycle_edges"] = ", ".join([f"{u}->{v}" for u, v in cycle_edges])
    removed_row["reason"] = "removed_lowest_score_recovered_edge_in_cycle"
    removed_row["final_recovery_decision"] = "reject_removed_lowest_score_in_cycle"

    cycle_removed_rows.append(removed_row)


# Final accepted/rejected tables after cycle repair.
removed_edges_after_repair = {
    (str(row.get("removed_parent", "")), str(row.get("removed_child", "")))
    for row in cycle_removed_rows
    if row.get("removed_parent", "") != "" and row.get("removed_child", "") != ""
}

accepted_rows = []

for row_dict in initial_accepted_rows:
    edge = (str(row_dict["parent"]), str(row_dict["child"]))

    if edge in removed_edges_after_repair:
        continue

    final_row = row_dict.copy()
    final_row["final_recovery_decision"] = "recover_keep_after_cycle_repair"
    accepted_rows.append(final_row)

# Add cycle-removed rows to rejected rows for one complete rejected table.
rejected_rows.extend(cycle_removed_rows)

accepted_df = pd.DataFrame(accepted_rows)
rejected_df = pd.DataFrame(rejected_rows)
cycle_removed_df = pd.DataFrame(cycle_removed_rows)

# Attach final decisions back to scores_df for easier inspection.
final_decision_map = {}

for row_dict in accepted_rows:
    final_decision_map[(str(row_dict["parent"]), str(row_dict["child"]))] = "recover_keep_after_cycle_repair"

for row_dict in rejected_rows:
    if "parent" in row_dict and "child" in row_dict:
        final_decision_map[(str(row_dict["parent"]), str(row_dict["child"]))] = row_dict.get(
            "final_recovery_decision",
            "rejected"
        )

if not scores_df.empty:
    scores_df["final_recovery_decision"] = scores_df.apply(
        lambda row: final_decision_map.get(
            (str(row["parent"]), str(row["child"])),
            "not_selected"
        ),
        axis=1
    )


# =====================================================
# 12. BUILD RECOVERED FINAL DAG
# =====================================================

recovered_adj = pd.DataFrame(
    0,
    index=variables,
    columns=variables
)

for parent, child in recovered_edges:
    recovered_adj.loc[parent, child] = 1

recovered_edges_df = pd.DataFrame(
    sorted(recovered_edges),
    columns=["parent", "child"]
)

recovered_edges_df["edge"] = recovered_edges_df["parent"] + "->" + recovered_edges_df["child"]

newly_recovered_edges_df = pd.DataFrame(
    sorted(recovered_edges - current_edges),
    columns=["parent", "child"]
)

if newly_recovered_edges_df.empty:
    newly_recovered_edges_df = pd.DataFrame(columns=["parent", "child", "edge"])
else:
    newly_recovered_edges_df["edge"] = (
        newly_recovered_edges_df["parent"] + "->" + newly_recovered_edges_df["child"]
    )


# =====================================================
# 13. SAVE OUTPUTS
# =====================================================

scores_df.to_csv(
    os.path.join(output_dir, "recovery_candidate_scores.csv"),
    index=False
)

scores_sorted.to_csv(
    os.path.join(output_dir, "recovery_candidate_scores_sorted.csv"),
    index=False
)

initial_accepted_df.to_csv(
    os.path.join(output_dir, "recovery_initial_accepted_before_cycle_repair.csv"),
    index=False
)

accepted_df.to_csv(
    os.path.join(output_dir, "recovery_accepted_edges.csv"),
    index=False
)

rejected_df.to_csv(
    os.path.join(output_dir, "recovery_rejected_edges.csv"),
    index=False
)

cycle_removed_df.to_csv(
    os.path.join(output_dir, "recovery_cycle_removed_edges.csv"),
    index=False
)

newly_recovered_edges_df.to_csv(
    os.path.join(output_dir, "newly_recovered_edges_only.csv"),
    index=False
)

recovered_edges_df.to_csv(
    os.path.join(output_dir, "recovered_final_edges.csv"),
    index=False
)

recovered_adj.to_csv(
    os.path.join(output_dir, "recovered_final_adjacency.csv")
)

summary = {
    "score_mode": score_mode,
    "llm_prior_mode": llm_prior_mode,
    "current_dag_edges_before_recovery": len(current_edges),
    "llm_plausible_missing_candidates": len(candidate_df),
    "initial_recovered_edges_before_cycle_repair": len(initial_recovered_edges),
    "cycle_removed_edges": len(removed_edges_after_repair),
    "recovered_edges_added": len(recovered_edges - current_edges),
    "rejected_recovery_candidates": len(rejected_df),
    "final_edges_after_recovery": len(recovered_edges),
    "is_recovered_graph_acyclic": is_acyclic(recovered_edges, variables),
    "unresolved_current_graph_cycle": unresolved_current_graph_cycle,
    "w_bic": w_bic,
    "w_hsic": w_hsic,
    "w_llm": w_llm,
    "combined_recovery_threshold": combined_recovery_threshold,
    "llm_plausible_threshold": llm_plausible_threshold,
    "require_positive_bic_gain": require_positive_bic_gain,
    "require_positive_hsic_gain": require_positive_hsic_gain
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(output_dir, "recovery_summary.csv"),
    index=False
)


# =====================================================
# 14. PRINT SUMMARY
# =====================================================

print("\n========== DAG RECOVERY SUMMARY ==========")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\nFiles saved in:")
print(output_dir)

print("\nSaved files:")
print("1. recovery_candidate_scores.csv")
print("2. recovery_candidate_scores_sorted.csv")
print("3. recovery_initial_accepted_before_cycle_repair.csv")
print("4. recovery_accepted_edges.csv")
print("5. recovery_rejected_edges.csv")
print("6. recovery_cycle_removed_edges.csv")
print("7. newly_recovered_edges_only.csv")
print("8. recovered_final_edges.csv")
print("9. recovered_final_adjacency.csv")
print("10. recovery_summary.csv")

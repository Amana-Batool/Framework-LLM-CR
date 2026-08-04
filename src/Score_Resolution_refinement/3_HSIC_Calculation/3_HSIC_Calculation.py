import os
import numpy as np
import pandas as pd
import networkx as nx

# =====================================================
# 1. USER INPUTS
# =====================================================

data_path = "Metabolic_Power_data.csv"

# Files generated from common/disputed edge separation
common_adj_path = "common_edges_adjacency.csv"
disputed_edges_path = "disputed_edges.csv"


output_dir = "fast_hsic_disputed_edge_results"
os.makedirs(output_dir, exist_ok=True)

# Fast settings
max_rows_for_hsic = 5000          
n_rff_features = 128             
random_state = 42

hsic_gain_threshold = 0.001
encode_non_numeric = True

# =====================================================
# 2. LOAD DATA
# =====================================================

data = pd.read_csv(data_path)

common_adj = pd.read_csv(common_adj_path, index_col=0)
common_adj = common_adj.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

disputed_df = pd.read_csv(disputed_edges_path)

variables = list(common_adj.index)

missing_vars = [v for v in variables if v not in data.columns]
if missing_vars:
    raise ValueError(f"These graph variables are missing from data.csv: {missing_vars}")

data = data[variables].copy()

# =====================================================
# 3. MAKE DATA NUMERIC
# =====================================================

def make_numeric_dataframe(df, encode_non_numeric=True):
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        numeric_col = pd.to_numeric(df[col], errors="coerce")

        if numeric_col.notna().sum() == len(df[col].dropna()):
            out[col] = numeric_col
        else:
            if not encode_non_numeric:
                raise ValueError(
                    f"Column {col} is non-numeric. "
                    f"Set encode_non_numeric=True."
                )

            codes, _ = pd.factorize(df[col].astype(str))
            out[col] = codes.astype(float)

    return out


data_num = make_numeric_dataframe(data, encode_non_numeric=encode_non_numeric)

# =====================================================
# 4. GRAPH HELPERS
# =====================================================

def adjacency_to_edges(adj_df):
    edges = set()

    for parent in adj_df.index:
        for child in adj_df.columns:
            if parent == child:
                continue
            if adj_df.loc[parent, child] == 1:
                edges.add((parent, child))

    return edges


def get_parents_from_edges(edges, child):
    return sorted([p for p, c in edges if c == child])


def is_acyclic_after_adding(edges, candidate_edge, variables):
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(edges)
    G.add_edge(candidate_edge[0], candidate_edge[1])

    return nx.is_directed_acyclic_graph(G)


# =====================================================
# 5. RESIDUAL FUNCTIONS WITH CACHING
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


# =====================================================
# 6. FAST APPROXIMATE HSIC USING RANDOM FOURIER FEATURES
# =====================================================

def subsample_xy(x, y, max_rows=500, random_state=42):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    n = len(x)

    if n != len(y):
        raise ValueError("x and y must have the same length.")

    if max_rows is None or n <= max_rows:
        return x, y

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)

    return x[idx], y[idx]


def rff_features_1d(x, n_features=128, random_state=42):
    
    x = standardize_vector(x).reshape(-1, 1)

    rng = np.random.default_rng(random_state)

    # Since x is standardized, sigma = 1 is a practical fast choice.
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

    # Center features
    Zx = Zx - Zx.mean(axis=0, keepdims=True)
    Zy = Zy - Zy.mean(axis=0, keepdims=True)

    n = len(x)

    # Cross-covariance in feature space
    Cxy = (Zx.T @ Zy) / n

    hsic_value = np.sum(Cxy ** 2)

    return float(hsic_value)


# =====================================================
# 7. SCORE DISPUTED EDGES
# =====================================================

common_edges = adjacency_to_edges(common_adj)

results = []
cycle_edges = []

for _, row in disputed_df.iterrows():

    parent = row["parent"]
    child = row["child"]

    candidate_edge = (parent, child)

    if parent == child:
        continue

    if parent not in variables or child not in variables:
        continue

    if candidate_edge in common_edges:
        continue

    acyclic = is_acyclic_after_adding(
        common_edges,
        candidate_edge,
        variables
    )

    if not acyclic:
        cycle_edges.append({
            "parent": parent,
            "child": child,
            "edge": f"{parent}->{child}",
            "reason": "Adding this edge creates a cycle"
        })

        results.append({
            "parent": parent,
            "child": child,
            "edge": f"{parent}->{child}",
            "present_in": row.get("present_in", ""),
            "absent_in": row.get("absent_in", ""),
            "dispute_type": row.get("dispute_type", ""),
            "reverse_edge": row.get("reverse_edge", ""),
            "acyclic": False,
            "old_parents_of_child": "",
            "new_parents_of_child": "",
            "hsic_without_edge": np.nan,
            "hsic_with_edge": np.nan,
            "hsic_gain": -np.inf,
            "decision_by_hsic": "reject_cycle"
        })

        continue

    old_parents = get_parents_from_edges(common_edges, child)
    new_parents = sorted(set(old_parents + [parent]))

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
        results.append({
            "parent": parent,
            "child": child,
            "edge": f"{parent}->{child}",
            "present_in": row.get("present_in", ""),
            "absent_in": row.get("absent_in", ""),
            "dispute_type": row.get("dispute_type", ""),
            "reverse_edge": row.get("reverse_edge", ""),
            "acyclic": True,
            "old_parents_of_child": ", ".join(old_parents),
            "new_parents_of_child": ", ".join(new_parents),
            "hsic_without_edge": np.nan,
            "hsic_with_edge": np.nan,
            "hsic_gain": np.nan,
            "decision_by_hsic": "reject_too_few_samples"
        })

        continue

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

    decision = "keep" if hsic_gain > hsic_gain_threshold else "reject"

    results.append({
        "parent": parent,
        "child": child,
        "edge": f"{parent}->{child}",
        "present_in": row.get("present_in", ""),
        "absent_in": row.get("absent_in", ""),
        "dispute_type": row.get("dispute_type", ""),
        "reverse_edge": row.get("reverse_edge", ""),
        "acyclic": True,
        "old_parents_of_child": ", ".join(old_parents),
        "new_parents_of_child": ", ".join(new_parents),
        "hsic_without_edge": hsic_without,
        "hsic_with_edge": hsic_with,
        "hsic_gain": hsic_gain,
        "decision_by_hsic": decision
    })


# =====================================================
# 8. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(output_dir, "fast_hsic_scores_for_disputed_edges.csv"),
    index=False
)

results_sorted_df = results_df.sort_values(
    by="hsic_gain",
    ascending=False
)

results_sorted_df.to_csv(
    os.path.join(output_dir, "fast_hsic_scores_for_disputed_edges_sorted.csv"),
    index=False
)

accepted_df = results_df[results_df["decision_by_hsic"] == "keep"].copy()
rejected_df = results_df[results_df["decision_by_hsic"] != "keep"].copy()

accepted_df.to_csv(
    os.path.join(output_dir, "fast_hsic_accepted_disputed_edges.csv"),
    index=False
)

rejected_df.to_csv(
    os.path.join(output_dir, "fast_hsic_rejected_disputed_edges.csv"),
    index=False
)

cycle_df = pd.DataFrame(cycle_edges)

if cycle_df.empty:
    cycle_df = pd.DataFrame(columns=["parent", "child", "edge", "reason"])

cycle_df.to_csv(
    os.path.join(output_dir, "fast_hsic_rejected_due_to_cycle.csv"),
    index=False
)

# =====================================================
# 9. CREATE FINAL ADJACENCY
# =====================================================

hsic_selected_edges = set(common_edges)

for _, row in accepted_df.iterrows():
    hsic_selected_edges.add((row["parent"], row["child"]))

hsic_adj = pd.DataFrame(
    0,
    index=variables,
    columns=variables
)

for parent, child in hsic_selected_edges:
    hsic_adj.loc[parent, child] = 1

hsic_adj.to_csv(
    os.path.join(output_dir, "common_plus_fast_hsic_selected_adjacency.csv")
)

selected_edges_df = pd.DataFrame(
    sorted(hsic_selected_edges),
    columns=["parent", "child"]
)

selected_edges_df["edge"] = selected_edges_df["parent"] + "->" + selected_edges_df["child"]

selected_edges_df.to_csv(
    os.path.join(output_dir, "common_plus_fast_hsic_selected_edges.csv"),
    index=False
)

# =====================================================
# 10. SAVE SUMMARY
# =====================================================

summary = {
    "common_edges_count": len(common_edges),
    "disputed_edges_tested": len(results_df),
    "fast_hsic_accepted_disputed_edges": len(accepted_df),
    "fast_hsic_rejected_disputed_edges": len(rejected_df),
    "cycle_rejected_edges": len(cycle_df),
    "final_edges_count_common_plus_fast_hsic": len(hsic_selected_edges),
    "max_rows_for_hsic": max_rows_for_hsic,
    "n_rff_features": n_rff_features,
    "hsic_gain_threshold": hsic_gain_threshold
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(output_dir, "fast_hsic_disputed_edge_summary.csv"),
    index=False
)

# =====================================================
# 11. PRINT SUMMARY
# =====================================================

print("\n========== FAST HSIC DISPUTED EDGE SUMMARY ==========")

for k, v in summary.items():
    print(f"{k}: {v}")

print("\nFiles saved in:")
print(output_dir)

print("\nSaved files:")
print("1. fast_hsic_scores_for_disputed_edges.csv")
print("2. fast_hsic_scores_for_disputed_edges_sorted.csv")
print("3. fast_hsic_accepted_disputed_edges.csv")
print("4. fast_hsic_rejected_disputed_edges.csv")
print("5. fast_hsic_rejected_due_to_cycle.csv")
print("6. common_plus_fast_hsic_selected_adjacency.csv")
print("7. common_plus_fast_hsic_selected_edges.csv")
print("8. fast_hsic_disputed_edge_summary.csv")
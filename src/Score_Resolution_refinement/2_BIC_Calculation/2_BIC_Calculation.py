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


common_adj_path = "common_edges_adjacency.csv"
disputed_edges_path = "disputed_edges.csv"


# "continuous" = linear Gaussian BIC
# "discrete"   = multinomial/discrete BIC
score_mode = "continuous"

output_dir = "bic_disputed_edge_results"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 2. LOAD FILES
# =====================================================

data = pd.read_csv(data_path)

common_adj = pd.read_csv(common_adj_path, index_col=0)
common_adj = common_adj.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

disputed_df = pd.read_csv(disputed_edges_path)

variables = list(common_adj.index)

# Keep only variables available in data
missing_vars = [v for v in variables if v not in data.columns]
if missing_vars:
    raise ValueError(f"These graph variables are missing from data.csv: {missing_vars}")

data = data[variables].copy()

# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def adjacency_to_edges(adj_df):
    """
    Convert adjacency matrix to directed edge set.
    """
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
# 4. CONTINUOUS LINEAR GAUSSIAN LOCAL BIC
# =====================================================

def local_bic_continuous(data, child, parents):
   
    cols = [child] + parents
    df = data[cols].dropna().copy()

    if df.empty:
        return -np.inf

    y = pd.to_numeric(df[child], errors="coerce").values.astype(float)

    if np.isnan(y).any():
        raise ValueError(f"Child variable {child} is not numeric. Use score_mode='discrete'.")

    n = len(y)

    if parents:
        X = df[parents].apply(pd.to_numeric, errors="coerce").values.astype(float)

        if np.isnan(X).any():
            raise ValueError(
                f"Some parent values for {child} are not numeric. "
                f"Use score_mode='discrete' if your data are categorical."
            )

        # Add intercept
        X = np.column_stack([np.ones(n), X])
    else:
        X = np.ones((n, 1))

    # Least squares fit
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    rss = np.sum(residuals ** 2)

    # Avoid log(0)
    sigma2 = max(rss / n, 1e-12)

    log_likelihood = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)

    # coefficients + variance
    k = X.shape[1] + 1

    bic = log_likelihood - 0.5 * k * math.log(n)

    return bic


# =====================================================
# 5. DISCRETE LOCAL BIC
# =====================================================

def local_bic_discrete(data, child, parents):
   
    cols = [child] + parents
    df = data[cols].dropna().copy()

    if df.empty:
        return -np.inf

    # Convert to string categories for safety
    for col in cols:
        df[col] = df[col].astype(str)

    n = len(df)

    child_states = sorted(df[child].unique())
    r_i = len(child_states)

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

    # Parent cardinality product
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


# =====================================================
# 6. SELECT BIC FUNCTION
# =====================================================

if score_mode == "continuous":
    local_bic = local_bic_continuous
elif score_mode == "discrete":
    local_bic = local_bic_discrete
else:
    raise ValueError("score_mode must be either 'continuous' or 'discrete'.")


# =====================================================
# 7. SCORE EACH DISPUTED EDGE
# =====================================================

common_edges = adjacency_to_edges(common_adj)

results = []
cycle_edges = []

for _, row in disputed_df.iterrows():

    parent = row["parent"]
    child = row["child"]

    candidate_edge = (parent, child)

    # Skip malformed/self edges
    if parent == child:
        continue

    if parent not in variables or child not in variables:
        continue

    # If edge already exists in common graph, no need to test
    if candidate_edge in common_edges:
        continue

    # Check cycle
    acyclic = is_acyclic_after_adding(common_edges, candidate_edge, variables)

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
            "dispute_type": row.get("dispute_type", ""),
            "acyclic": False,
            "bic_without_edge": np.nan,
            "bic_with_edge": np.nan,
            "bic_gain": -np.inf,
            "decision_by_bic": "reject_cycle"
        })

        continue

    # Existing parents of child in common graph
    old_parents = get_parents_from_edges(common_edges, child)

    # Parents after adding candidate edge
    new_parents = sorted(set(old_parents + [parent]))

    # Local BIC before and after adding edge
    bic_without = local_bic(data, child, old_parents)
    bic_with = local_bic(data, child, new_parents)

    bic_gain = bic_with - bic_without

    decision = "keep" if bic_gain > 0 else "reject"

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
        "bic_without_edge": bic_without,
        "bic_with_edge": bic_with,
        "bic_gain": bic_gain,
        "decision_by_bic": decision
    })

# =====================================================
# 8. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    results_df = pd.DataFrame(columns=[
        "parent",
        "child",
        "edge",
        "present_in",
        "absent_in",
        "dispute_type",
        "reverse_edge",
        "acyclic",
        "old_parents_of_child",
        "new_parents_of_child",
        "bic_without_edge",
        "bic_with_edge",
        "bic_gain",
        "decision_by_bic"
    ])

results_df.to_csv(
    os.path.join(output_dir, "bic_scores_for_disputed_edges.csv"),
    index=False
)

# Sorted by best BIC gain
results_sorted_df = results_df.sort_values(
    by="bic_gain",
    ascending=False
)

results_sorted_df.to_csv(
    os.path.join(output_dir, "bic_scores_for_disputed_edges_sorted.csv"),
    index=False
)

# Save accepted and rejected separately
accepted_df = results_df[results_df["decision_by_bic"] == "keep"].copy()
rejected_df = results_df[results_df["decision_by_bic"] != "keep"].copy()

accepted_df.to_csv(
    os.path.join(output_dir, "bic_accepted_disputed_edges.csv"),
    index=False
)

rejected_df.to_csv(
    os.path.join(output_dir, "bic_rejected_disputed_edges.csv"),
    index=False
)

# Save cycle edges
cycle_df = pd.DataFrame(cycle_edges)

if cycle_df.empty:
    cycle_df = pd.DataFrame(columns=["parent", "child", "edge", "reason"])

cycle_df.to_csv(
    os.path.join(output_dir, "bic_rejected_due_to_cycle.csv"),
    index=False
)

# =====================================================
# 9. CREATE ADJACENCY FOR BIC-ACCEPTED EDGES
# =====================================================

bic_selected_edges = set(common_edges)

for _, row in accepted_df.iterrows():
    bic_selected_edges.add((row["parent"], row["child"]))

bic_adj = pd.DataFrame(
    0,
    index=variables,
    columns=variables
)

for parent, child in bic_selected_edges:
    bic_adj.loc[parent, child] = 1

bic_adj.to_csv(
    os.path.join(output_dir, "common_plus_bic_selected_disputed_edges_adjacency.csv")
)

# Also save selected edge list
selected_edges_df = pd.DataFrame(
    sorted(bic_selected_edges),
    columns=["parent", "child"]
)

selected_edges_df["edge"] = selected_edges_df["parent"] + "->" + selected_edges_df["child"]

selected_edges_df.to_csv(
    os.path.join(output_dir, "common_plus_bic_selected_edges.csv"),
    index=False
)

# =====================================================
# 10. SAVE SUMMARY
# =====================================================

summary = {
    "score_mode": score_mode,
    "common_edges_count": len(common_edges),
    "disputed_edges_tested": len(results_df),
    "bic_accepted_disputed_edges": len(accepted_df),
    "bic_rejected_disputed_edges": len(rejected_df),
    "cycle_rejected_edges": len(cycle_df),
    "final_edges_count_common_plus_bic": len(bic_selected_edges)
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(output_dir, "bic_disputed_edge_summary.csv"),
    index=False
)

# =====================================================
# 11. PRINT SUMMARY
# =====================================================

print("\n========== BIC DISPUTED EDGE SCORING SUMMARY ==========")

for k, v in summary.items():
    print(f"{k}: {v}")

print("\nFiles saved in:")
print(output_dir)

print("\nSaved files:")
print("1. bic_scores_for_disputed_edges.csv")
print("2. bic_scores_for_disputed_edges_sorted.csv")
print("3. bic_accepted_disputed_edges.csv")
print("4. bic_rejected_disputed_edges.csv")
print("5. bic_rejected_due_to_cycle.csv")
print("6. common_plus_bic_selected_disputed_edges_adjacency.csv")
print("7. common_plus_bic_selected_edges.csv")
print("8. bic_disputed_edge_summary.csv")
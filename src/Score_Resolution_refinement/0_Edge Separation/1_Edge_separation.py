import os
import pandas as pd

# =====================================================
# 1. USER INPUT FILES
# =====================================================

dag1_path = "Data_DAG.csv"
dag2_path = "LLM_DAG.csv"

dag1_name = "Data"
dag2_name = "LLM"

output_dir = "1_Edge_separation_results"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 2. LOAD ADJACENCY MATRIX
# =====================================================

def load_adj(path):
  
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return df


A = load_adj(dag1_path)
B = load_adj(dag2_path)

# =====================================================
# 3. CHECK VARIABLE CONSISTENCY
# =====================================================

if list(A.index) != list(A.columns):
    raise ValueError(f"{dag1_name} adjacency rows and columns are not identical.")

if list(B.index) != list(B.columns):
    raise ValueError(f"{dag2_name} adjacency rows and columns are not identical.")

if set(A.index) != set(B.index):
    missing_in_B = set(A.index) - set(B.index)
    missing_in_A = set(B.index) - set(A.index)

    raise ValueError(
        f"The two DAGs do not contain the same variables.\n"
        f"Missing in {dag2_name}: {missing_in_B}\n"
        f"Missing in {dag1_name}: {missing_in_A}"
    )


B = B.loc[A.index, A.columns]

variables = list(A.index)

# =====================================================
# 4. CONVERT ADJACENCY TO EDGE SET
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


edges_A = adjacency_to_edges(A)
edges_B = adjacency_to_edges(B)

# =====================================================
# 5. COMMON AND DISPUTED EDGES
# =====================================================

common_edges = edges_A & edges_B

dag1_only_edges = edges_A - edges_B
dag2_only_edges = edges_B - edges_A

disputed_edges = dag1_only_edges | dag2_only_edges

# =====================================================
# 6. REVERSED DIRECTION CONFLICTS
# =====================================================

reversed_conflicts = []

for parent, child in sorted(dag1_only_edges):
    if (child, parent) in dag2_only_edges:
        reversed_conflicts.append({
            "variable_1": parent,
            "variable_2": child,
            f"{dag1_name}_edge": f"{parent}->{child}",
            f"{dag2_name}_edge": f"{child}->{parent}",
            "conflict_type": "reversed_direction"
        })

reversed_conflicts_df = pd.DataFrame(reversed_conflicts)

# =====================================================
# 7. SAVE COMMON EDGES
# =====================================================

common_df = pd.DataFrame(
    sorted(common_edges),
    columns=["parent", "child"]
)

if not common_df.empty:
    common_df["edge"] = common_df["parent"] + "->" + common_df["child"]
else:
    common_df = pd.DataFrame(columns=["parent", "child", "edge"])

common_df["edge_type"] = "common"

common_df.to_csv(
    os.path.join(output_dir, "common_edges.csv"),
    index=False
)

# =====================================================
# 8. SAVE DISPUTED EDGES
# =====================================================

disputed_rows = []

for parent, child in sorted(dag1_only_edges):
    reverse_exists = (child, parent) in dag2_only_edges

    disputed_rows.append({
        "parent": parent,
        "child": child,
        "edge": f"{parent}->{child}",
        "present_in": dag1_name,
        "absent_in": dag2_name,
        "reverse_edge_present_in_other_DAG": reverse_exists,
        "reverse_edge": f"{child}->{parent}" if reverse_exists else "",
        "dispute_type": "direction_conflict" if reverse_exists else f"{dag1_name}_only"
    })

for parent, child in sorted(dag2_only_edges):
    reverse_exists = (child, parent) in dag1_only_edges

    disputed_rows.append({
        "parent": parent,
        "child": child,
        "edge": f"{parent}->{child}",
        "present_in": dag2_name,
        "absent_in": dag1_name,
        "reverse_edge_present_in_other_DAG": reverse_exists,
        "reverse_edge": f"{child}->{parent}" if reverse_exists else "",
        "dispute_type": "direction_conflict" if reverse_exists else f"{dag2_name}_only"
    })

disputed_df = pd.DataFrame(disputed_rows)

if disputed_df.empty:
    disputed_df = pd.DataFrame(columns=[
        "parent",
        "child",
        "edge",
        "present_in",
        "absent_in",
        "reverse_edge_present_in_other_DAG",
        "reverse_edge",
        "dispute_type"
    ])

disputed_df.to_csv(
    os.path.join(output_dir, "disputed_edges.csv"),
    index=False
)

# =====================================================
# 9. SAVE METHOD-SPECIFIC ONLY EDGES
# =====================================================

dag1_only_df = pd.DataFrame(
    sorted(dag1_only_edges),
    columns=["parent", "child"]
)

if not dag1_only_df.empty:
    dag1_only_df["edge"] = dag1_only_df["parent"] + "->" + dag1_only_df["child"]
else:
    dag1_only_df = pd.DataFrame(columns=["parent", "child", "edge"])

dag1_only_df.to_csv(
    os.path.join(output_dir, f"{dag1_name}_only_edges.csv"),
    index=False
)


dag2_only_df = pd.DataFrame(
    sorted(dag2_only_edges),
    columns=["parent", "child"]
)

if not dag2_only_df.empty:
    dag2_only_df["edge"] = dag2_only_df["parent"] + "->" + dag2_only_df["child"]
else:
    dag2_only_df = pd.DataFrame(columns=["parent", "child", "edge"])

dag2_only_df.to_csv(
    os.path.join(output_dir, f"{dag2_name}_only_edges.csv"),
    index=False
)

# =====================================================
# 10. SAVE REVERSED CONFLICTS
# =====================================================

if reversed_conflicts_df.empty:
    reversed_conflicts_df = pd.DataFrame(columns=[
        "variable_1",
        "variable_2",
        f"{dag1_name}_edge",
        f"{dag2_name}_edge",
        "conflict_type"
    ])

reversed_conflicts_df.to_csv(
    os.path.join(output_dir, "reversed_direction_conflicts.csv"),
    index=False
)

# =====================================================
# 11. SAVE ADJACENCY MATRICES
# =====================================================

def edges_to_adjacency(edges, variables):
    adj = pd.DataFrame(
        0,
        index=variables,
        columns=variables
    )

    for parent, child in edges:
        adj.loc[parent, child] = 1

    return adj


common_adj = edges_to_adjacency(common_edges, variables)
disputed_adj = edges_to_adjacency(disputed_edges, variables)
dag1_only_adj = edges_to_adjacency(dag1_only_edges, variables)
dag2_only_adj = edges_to_adjacency(dag2_only_edges, variables)

common_adj.to_csv(
    os.path.join(output_dir, "common_edges_adjacency.csv")
)

disputed_adj.to_csv(
    os.path.join(output_dir, "disputed_edges_adjacency.csv")
)

dag1_only_adj.to_csv(
    os.path.join(output_dir, f"{dag1_name}_only_edges_adjacency.csv")
)

dag2_only_adj.to_csv(
    os.path.join(output_dir, f"{dag2_name}_only_edges_adjacency.csv")
)

# =====================================================
# 12. SAVE SUMMARY
# =====================================================

summary = {
    f"{dag1_name}_total_edges": len(edges_A),
    f"{dag2_name}_total_edges": len(edges_B),
    "common_edges": len(common_edges),
    "disputed_edges_total": len(disputed_edges),
    f"{dag1_name}_only_edges": len(dag1_only_edges),
    f"{dag2_name}_only_edges": len(dag2_only_edges),
    "reversed_direction_conflicts": len(reversed_conflicts_df)
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(output_dir, "common_disputed_summary.csv"),
    index=False
)

# =====================================================
# 13. PRINT RESULTS
# =====================================================

print("\n========== EDGE SUMMARY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\n========== FILES SAVED ==========")
print(f"All files saved inside directory: {output_dir}")

print("\nSaved files:")
print("1. common_edges.csv")
print("2. disputed_edges.csv")
print(f"3. {dag1_name}_only_edges.csv")
print(f"4. {dag2_name}_only_edges.csv")
print("5. reversed_direction_conflicts.csv")
print("6. common_edges_adjacency.csv")
print("7. disputed_edges_adjacency.csv")
print(f"8. {dag1_name}_only_edges_adjacency.csv")
print(f"9. {dag2_name}_only_edges_adjacency.csv")
print("10. common_disputed_summary.csv")
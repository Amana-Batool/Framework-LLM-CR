# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.GraphUtils import GraphUtils


# =========================
# Edit these paths/settings
# =========================
CSV_PATH = r"Sample_data.csv"   
OUTPUT_DIR = r"ges_bic_output"         
MAX_PARENTS = None                     


def load_csv_as_numeric_matrix(csv_path):
   
    df = pd.read_csv(csv_path)

    # Save variable names
    variable_names = df.columns.tolist()

    # Force all columns to numeric
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except Exception as e:
        raise ValueError(
            "Your CSV contains at least one non-numeric column.\n"
            f"Original error: {e}"
        )

    # Check missing values
    if df.isnull().values.any():
        raise ValueError(
            "Your CSV contains missing values. "
            "Please clean the file before running this script."
        )

    # Convert dataframe to numpy array
    X = df.to_numpy(dtype=float)

    return df, X, variable_names


def graph_to_raw_matrix_df(G, labels):

    raw = np.array(G.graph, dtype=int)
    raw_df = pd.DataFrame(raw, index=labels, columns=labels)
    return raw_df


def cpdag_to_adjacency_df(G, labels):
    
    raw = np.array(G.graph, dtype=int)
    p = raw.shape[0]
    adj = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue

            # Directed edge i -> j
            if raw[i, j] == -1 and raw[j, i] == 1:
                adj[i, j] = 1

            # Undirected edge i -- j
            elif raw[i, j] == -1 and raw[j, i] == -1:
                adj[i, j] = 1

    adj_df = pd.DataFrame(adj, index=labels, columns=labels)
    return adj_df


def dag_to_adjacency_df(G, labels):
   
    raw = np.array(G.graph, dtype=int)
    p = raw.shape[0]
    adj = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue

            if raw[i, j] == -1 and raw[j, i] == 1:
                adj[i, j] = 1

    adj_df = pd.DataFrame(adj, index=labels, columns=labels)
    return adj_df


def save_graph_png(G, labels, png_path):
  
    pyd = GraphUtils.to_pydot(G, labels=labels)
    pyd.write_png(png_path)


def main():
  
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # Step 1: Read CSV
    # -------------------------
    df, X, variable_names = load_csv_as_numeric_matrix(CSV_PATH)

    print("=" * 70)
    print("Data shape:", X.shape)
    print("Variable names:", variable_names)
    print("=" * 70)

    # -------------------------
    # Step 2: Run GES with BIC
    # -------------------------
    print("\nRunning GES with BIC score ...")
    record = ges(
        X,
        score_func="local_score_BIC",
        maxP=MAX_PARENTS
    )

    # Learned equivalence-class graph
    cpdag = record["G"]

    print("GES finished.")
    print("Final score:", record["score"])

    # -------------------------
    # Step 3: Save CPDAG
    # -------------------------
    cpdag_raw_df = graph_to_raw_matrix_df(cpdag, variable_names)
    cpdag_adj_df = cpdag_to_adjacency_df(cpdag, variable_names)

    cpdag_raw_csv = os.path.join(OUTPUT_DIR, "ges_bic_cpdag_raw_matrix.csv")
    cpdag_adj_csv = os.path.join(OUTPUT_DIR, "ges_bic_cpdag_adjacency.csv")
    cpdag_png = os.path.join(OUTPUT_DIR, "ges_bic_cpdag.png")

    cpdag_raw_df.to_csv(cpdag_raw_csv)
    cpdag_adj_df.to_csv(cpdag_adj_csv)
    save_graph_png(cpdag, variable_names, cpdag_png)

    print("\nSaved CPDAG raw matrix to:", cpdag_raw_csv)
    print("Saved CPDAG adjacency matrix to:", cpdag_adj_csv)
    print("Saved CPDAG image to:", cpdag_png)

    # -------------------------
    # Step 4: Convert CPDAG -> DAG
    # -------------------------
    try:
        dag = pdag2dag(cpdag)

        dag_raw_df = graph_to_raw_matrix_df(dag, variable_names)
        dag_adj_df = dag_to_adjacency_df(dag, variable_names)

        dag_raw_csv = os.path.join(OUTPUT_DIR, "ges_bic_dag_raw_matrix.csv")
        dag_adj_csv = os.path.join(OUTPUT_DIR, "ges_bic_dag_adjacency.csv")
        dag_png = os.path.join(OUTPUT_DIR, "ges_bic_dag.png")

        dag_raw_df.to_csv(dag_raw_csv)
        dag_adj_df.to_csv(dag_adj_csv)
        save_graph_png(dag, variable_names, dag_png)

        print("\nSaved DAG raw matrix to:", dag_raw_csv)
        print("Saved DAG adjacency matrix to:", dag_adj_csv)
        print("Saved DAG image to:", dag_png)

    except Exception as e:
        print("\nCould not convert the learned CPDAG to a DAG.")
        print("Reason:", str(e))

    print("\nAll results are saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
    
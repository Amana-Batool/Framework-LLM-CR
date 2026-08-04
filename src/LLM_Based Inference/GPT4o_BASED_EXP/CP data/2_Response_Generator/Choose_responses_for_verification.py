""

import os
import json
import glob
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = r"pairwise_pair_json_outputs"
OUTPUT_CSV_PATH = r"pairwise_majority_summary.csv"

MAJORITY_THRESHOLD = 7


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MAP RESPONSE TO CATEGORY
# ============================================================

def categorize_response(parsed_response):
   
    judgment = parsed_response.get("judgment", "")
    plausible = parsed_response.get("plausible", None)

    if judgment == "plausible_directed_cause" or plausible is True:
        return "plausible"

    if judgment in [
        "not_plausible",
        "reverse_direction_more_plausible",
        "correlation_only",
        "invalid_identifier_or_excluded"
    ] or plausible is False:
        return "not_plausible"

    if judgment == "uncertain":
        return "uncertain"

    return "uncertain"


# ============================================================
# PROCESS ONE PAIR FILE
# ============================================================

def process_pair_file(json_path, majority_threshold):
    data = load_json(json_path)

    pair_index = data.get("pair_index")
    cause = data.get("cause_variable")
    effect = data.get("effect_variable")
    direction = data.get("direction")

    responses = data.get("responses", [])

    plausible_count = 0
    not_plausible_count = 0
    uncertain_count = 0

    response_categories = []

    for response in responses:
        parsed = response.get("parsed_response", {})
        category = categorize_response(parsed)

        response_categories.append(category)

        if category == "plausible":
            plausible_count += 1
        elif category == "not_plausible":
            not_plausible_count += 1
        else:
            uncertain_count += 1

    total_responses = len(responses)

    if plausible_count >= majority_threshold:
        final_decision = "plausible"
    elif not_plausible_count >= majority_threshold:
        final_decision = "not_plausible"
    elif uncertain_count >= majority_threshold:
        final_decision = "uncertain"
    else:
        final_decision = "no_majority"

    return {
        "pair_index": pair_index,
        "cause_variable": cause,
        "effect_variable": effect,
        "direction": direction,
        "total_responses": total_responses,
        "plausible_count": plausible_count,
        "not_plausible_count": not_plausible_count,
        "uncertain_count": uncertain_count,
        "final_decision": final_decision,
        "response_categories": ";".join(response_categories),
        "source_file": os.path.basename(json_path)
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def categorize_all_pair_files(input_dir, output_csv_path, majority_threshold):
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    # Exclude index.json
    json_files = [
        path for path in json_files
        if os.path.basename(path).lower() != "index.json"
    ]

    json_files = sorted(json_files)

    if len(json_files) == 0:
        raise FileNotFoundError(f"No pair JSON files found in: {input_dir}")

    print(f"Found {len(json_files)} pair JSON files.")

    rows = []

    for i, json_path in enumerate(json_files, start=1):
        print(f"Processing {i}/{len(json_files)}: {os.path.basename(json_path)}")

        row = process_pair_file(
            json_path=json_path,
            majority_threshold=majority_threshold
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(by=["pair_index"])

    df.to_csv(output_csv_path, index=False)

    print("\nDone.")
    print("Saved CSV to:", output_csv_path)

    return df


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    summary_df = categorize_all_pair_files(
        input_dir=INPUT_DIR,
        output_csv_path=OUTPUT_CSV_PATH,
        majority_threshold=MAJORITY_THRESHOLD
    )

    print("\nDecision counts:")
    print(summary_df["final_decision"].value_counts())
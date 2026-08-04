
import os
import json
import time
import pandas as pd
from openai import OpenAI


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV_PATH = r"pairwise_majority_summary.csv"
SCHEMA_JSON_PATH = r"generated_schema.json"
METADATA_PATH = r"metadata_clean.json"  
OUTPUT_CSV_PATH = r"schema_metadata_domain_verified_pairwise_summary.csv"
OUTPUT_JSON_PATH = r"schema_metadata_domain_verification_audit.json"
OUTPUT_PLAUSIBLE_EDGES_CSV = r"verified_plausible_edges.csv"

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    return OpenAI(api_key=api_key)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_decision(value):
    if pd.isna(value):
        return "uncertain"

    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    mapping = {
        "plausible": "plausible",
        "yes": "plausible",
        "true": "plausible",

        "not_plausible": "not_plausible",
        "implausible": "not_plausible",
        "notplausible": "not_plausible",
        "no": "not_plausible",
        "false": "not_plausible",

        "uncertain": "uncertain",
        "unknown": "uncertain",
        "unclear": "uncertain",

        "no_majority": "no_majority"
    }

    return mapping.get(value, "uncertain")


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def infer_edge_action(previous_decision, corrected_outcome):
    previous_decision = normalize_decision(previous_decision)
    corrected_outcome = normalize_decision(corrected_outcome)

    previous_has_edge = previous_decision == "plausible"
    corrected_has_edge = corrected_outcome == "plausible"

    if previous_has_edge and corrected_has_edge:
        return "keep_edge"

    if previous_has_edge and not corrected_has_edge:
        return "remove_edge"

    if not previous_has_edge and corrected_has_edge:
        return "add_edge"

    return "no_edge"


# ============================================================
# LOAD FILES
# ============================================================

def load_majority_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    required_columns = [
        "pair_index",
        "cause_variable",
        "effect_variable",
        "direction",
        "total_responses",
        "plausible_count",
        "not_plausible_count",
        "uncertain_count",
        "final_decision"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError("Missing required CSV columns: " + str(missing))

    df["cause_variable"] = df["cause_variable"].astype(str).str.strip()
    df["effect_variable"] = df["effect_variable"].astype(str).str.strip()
    df["final_decision"] = df["final_decision"].apply(normalize_decision)

    return df


def load_schema(schema_json_path):
    with open(schema_json_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if not isinstance(schema, dict):
        raise ValueError("Schema JSON must be a dictionary.")

    return schema


def load_metadata(metadata_path):
    

    if metadata_path is None or str(metadata_path).strip() == "":
        return {}

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    ext = os.path.splitext(metadata_path)[1].lower()

    if ext == ".json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "metadata_type": "json",
            "content": metadata
        }

    if ext == ".csv":
        df = pd.read_csv(metadata_path)
        df.columns = df.columns.astype(str).str.strip()
        return {
            "metadata_type": "csv",
            "columns": list(df.columns),
            "records": df.to_dict(orient="records")
        }

    if ext in [".txt", ".md"]:
        with open(metadata_path, "r", encoding="utf-8") as f:
            text = f.read()
        return {
            "metadata_type": "text",
            "content": text
        }

    raise ValueError("Unsupported metadata format. Use .json, .csv, .txt, or .md")


# ============================================================
# SCHEMA HELPERS
# ============================================================

def get_variable_schema_info(schema, variable_name):
    

    variables = schema.get("variables", [])

    if isinstance(variables, list):
        for item in variables:
            if not isinstance(item, dict):
                continue

            possible_names = [
                item.get("name"),
                item.get("variable"),
                item.get("variable_name"),
                item.get("Variable"),
                item.get("Variable Name")
            ]

            possible_names = [
                str(x).strip() for x in possible_names if x is not None
            ]

            if variable_name in possible_names:
                return item

    if isinstance(variables, dict):
        return variables.get(variable_name, {})

    return {}


def compress_schema_for_prompt(schema):
    

    useful_keys = [
        "semantic_categories",
        "variables",
        "category_level_direction_rules",
        "allowed_edge_rules",
        "forbidden_edge_rules",
        "discouraged_edge_rules",
        "parent_only_variables",
        "intervened_variables",
        "excluded_variables",
        "identifier_variables",
        "temporal_rules",
        "general_schema_notes",
        "causal_assumptions"
    ]

    compressed = {}

    for key in useful_keys:
        if key in schema:
            compressed[key] = schema[key]

    return compressed


# ============================================================
# METADATA HELPERS
# ============================================================

def get_variable_metadata_info(metadata, variable_name):
    
    if not metadata:
        return {}

    metadata_type = metadata.get("metadata_type")

    if metadata_type == "text":
        return {
            "note": "Metadata is unstructured text. Use full metadata context."
        }

    if metadata_type == "csv":
        records = metadata.get("records", [])

        for row in records:
            if not isinstance(row, dict):
                continue

            possible_names = [
                row.get("name"),
                row.get("variable"),
                row.get("variable_name"),
                row.get("Variable"),
                row.get("Variable Name")
            ]

            possible_names = [
                str(x).strip() for x in possible_names if pd.notna(x)
            ]

            if variable_name in possible_names:
                return row

        return {}

    if metadata_type == "json":
        content = metadata.get("content", {})

        if isinstance(content, dict) and "variables" in content:
            variables = content["variables"]

            if isinstance(variables, list):
                for item in variables:
                    if not isinstance(item, dict):
                        continue

                    possible_names = [
                        item.get("name"),
                        item.get("variable"),
                        item.get("variable_name"),
                        item.get("Variable"),
                        item.get("Variable Name")
                    ]

                    possible_names = [
                        str(x).strip() for x in possible_names if x is not None
                    ]

                    if variable_name in possible_names:
                        return item

            if isinstance(variables, dict):
                return variables.get(variable_name, {})

        if isinstance(content, dict):
            if variable_name in content:
                return content[variable_name]

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                possible_names = [
                    item.get("name"),
                    item.get("variable"),
                    item.get("variable_name"),
                    item.get("Variable"),
                    item.get("Variable Name")
                ]

                possible_names = [
                    str(x).strip() for x in possible_names if x is not None
                ]

                if variable_name in possible_names:
                    return item

    return {}


def compress_metadata_for_prompt(metadata, max_text_chars=12000, max_records=300):
    

    if not metadata:
        return {}

    metadata_type = metadata.get("metadata_type")

    if metadata_type == "text":
        text = metadata.get("content", "")
        return {
            "metadata_type": "text",
            "content": text[:max_text_chars]
        }

    if metadata_type == "csv":
        records = metadata.get("records", [])
        return {
            "metadata_type": "csv",
            "columns": metadata.get("columns", []),
            "records": records[:max_records]
        }

    if metadata_type == "json":
        return metadata

    return metadata


def extract_previous_responses_if_available(row):
    
    possible_cols = [
        "responses",
        "previous_responses",
        "rationales",
        "selected_responses",
        "all_responses",
        "llm_responses",
        "reasoning"
    ]

    found = {}

    for col in possible_cols:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                found[col] = str(value)

    return found


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_verifier_system_prompt():
    return """
You are an expert causal-relationship verifier.

Your task is to verify one directed candidate causal pair X -> Y.

You will receive:
1. generated schema,
2. metadata,
3. variable-specific schema information,
4. variable-specific metadata information,
5. previous majority decision,
6. previous LLM response counts,
7. previous rationales if available.

You must decide whether the previous majority decision should be kept or corrected.

Core rules:

1. Use generated schema as the main rulebook.
   - The schema may contain variable categories.
   - Category-level rules only tell whether relationships between types of variables are generally plausible or implausible.
   - Category compatibility is supporting context only.
   - Category compatibility alone is NOT enough to keep or add an edge.

2. Use metadata to understand the actual meaning of each variable.
   - Metadata may contain variable descriptions, measurement type, interventions, temporal role, biological role, clinical role, or experimental role.
   - If metadata says a variable is intervened, manipulated, randomized, fixed, controlled, externally assigned, or parent-only, then it should not receive incoming causal edges.
   - Intervened or parent-only variables may still have outgoing edges if X -> Y is specifically supported.

3. Do NOT connect all variables from one category to all variables of another category.
   - Even if schema says category A can affect category B, you must verify the specific variable pair X -> Y.
   - You may mark X -> Y plausible only if there is a variable-specific causal reason.

4. Keep or add an edge only when:
   - X -> Y does not violate schema,
   - X -> Y does not violate metadata,
   - X -> Y has a pair-specific causal mechanism or strong domain-knowledge rationale,
   - the direction X -> Y makes sense.

5. Remove or reject an edge when:
   - Y is intervened, randomized, manipulated, fixed, controlled, or parent-only,
   - X or Y is excluded from causal reasoning,
   - X -> Y violates a forbidden edge rule,
   - X -> Y contradicts metadata,
   - X -> Y is temporally impossible,
   - X -> Y is biologically, medically, logically, or mechanistically implausible,
   - the only support is category-level compatibility.

6. If schema/metadata do not clearly support or forbid the edge:
   - use domain knowledge cautiously.
   - If evidence is weak or ambiguous, choose uncertain.
   - Do not force plausible.

7. Preserve directionality.
   - Verify exactly X -> Y.
   - Do not reverse the edge.
   - If Y -> X is plausible but X -> Y is not, mark X -> Y as not_plausible or uncertain.

8. Previous majority decision:
   - You may keep it if schema, metadata, and domain knowledge support it.
   - You must correct it if it contradicts schema, metadata, or strong domain knowledge.
   - You may change uncertain/not_plausible to plausible only if there is a strong pair-specific causal reason.

9. Output must be strict JSON only.
   - No markdown.
   - No explanation outside JSON.

Decision labels:
- plausible
- not_plausible
- uncertain

Edge actions:
- keep_edge: previous decision was plausible and corrected outcome is plausible.
- remove_edge: previous decision was plausible but corrected outcome is not_plausible or uncertain.
- add_edge: previous decision was not_plausible, uncertain, or no_majority, but corrected outcome is plausible.
- no_edge: previous decision was not_plausible, uncertain, or no_majority, and corrected outcome remains not_plausible or uncertain.

Very important:
Do not mark an edge plausible by saying only:
"the schema says this category can affect that category."

A valid plausible decision must include a variable-specific reason, such as:
- biological mechanism,
- clinical pathway,
- temporal order,
- experimental design,
- known functional relationship,
- measurement/definition-based dependency.
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_verifier_user_prompt(schema, metadata, row):
    cause = row["cause_variable"]
    effect = row["effect_variable"]

    cause_schema_info = get_variable_schema_info(schema, cause)
    effect_schema_info = get_variable_schema_info(schema, effect)

    cause_metadata_info = get_variable_metadata_info(metadata, cause)
    effect_metadata_info = get_variable_metadata_info(metadata, effect)

    previous_responses = extract_previous_responses_if_available(row)

    payload = {
        "task": "Verify directed causal pair X -> Y",

        "candidate_pair": {
            "pair_index": safe_int(row["pair_index"]),
            "cause_variable_X": cause,
            "effect_variable_Y": effect,
            "direction": row["direction"]
        },

        "previous_majority_decision": {
            "final_decision": normalize_decision(row["final_decision"]),
            "total_responses": safe_int(row["total_responses"]),
            "plausible_count": safe_int(row["plausible_count"]),
            "not_plausible_count": safe_int(row["not_plausible_count"]),
            "uncertain_count": safe_int(row["uncertain_count"])
        },

        "previous_responses_if_available": previous_responses,

        "generated_schema": compress_schema_for_prompt(schema),

        "metadata": compress_metadata_for_prompt(metadata),

        "cause_variable_schema_info": cause_schema_info,
        "effect_variable_schema_info": effect_schema_info,

        "cause_variable_metadata_info": cause_metadata_info,
        "effect_variable_metadata_info": effect_metadata_info,

        "strict_instruction": (
            "Do not add or keep X -> Y only because category-level schema says "
            "the source category can affect the target category. The final decision "
            "must be based on the specific variable pair and must include a "
            "knowledge-based causal reason."
        )
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


# ============================================================
# RESPONSE JSON SCHEMA
# ============================================================

def get_verifier_response_schema():
    return {
        "name": "schema_metadata_domain_verification_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pair_index": {"type": "integer"},
                "cause_variable": {"type": "string"},
                "effect_variable": {"type": "string"},
                "direction": {"type": "string"},

                "previous_decision": {
                    "type": "string",
                    "enum": [
                        "plausible",
                        "not_plausible",
                        "uncertain",
                        "no_majority"
                    ]
                },

                "schema_consistency": {
                    "type": "string",
                    "enum": [
                        "consistent",
                        "inconsistent",
                        "partially_consistent",
                        "unclear"
                    ]
                },

                "metadata_consistency": {
                    "type": "string",
                    "enum": [
                        "consistent",
                        "inconsistent",
                        "partially_consistent",
                        "unclear"
                    ]
                },

                "domain_knowledge_support": {
                    "type": "string",
                    "enum": [
                        "supports",
                        "contradicts",
                        "weak_or_unclear",
                        "not_applicable"
                    ]
                },

                "category_level_support_only": {
                    "type": "boolean"
                },

                "pair_specific_reason_available": {
                    "type": "boolean"
                },

                "contradiction_to_schema_or_metadata": {
                    "type": "boolean"
                },

                "contradiction_to_domain_knowledge": {
                    "type": "boolean"
                },

                "contradiction_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "effect_is_intervened_or_parent_only",
                        "forbidden_edge_rule",
                        "excluded_or_identifier_variable",
                        "unsupported_by_category_rules",
                        "category_only_without_pair_specific_reason",
                        "violates_variable_role",
                        "violates_temporal_order",
                        "contradicts_metadata_description",
                        "domain_knowledge_conflict",
                        "other"
                    ]
                },

                "corrected_outcome": {
                    "type": "string",
                    "enum": [
                        "plausible",
                        "not_plausible",
                        "uncertain"
                    ]
                },

                "edge_action": {
                    "type": "string",
                    "enum": [
                        "keep_edge",
                        "remove_edge",
                        "add_edge",
                        "no_edge"
                    ]
                },

                "schema_rule_used": {
                    "type": "string"
                },

                "metadata_evidence_used": {
                    "type": "string"
                },

                "domain_knowledge_reason": {
                    "type": "string"
                },

                "one_line_reason": {
                    "type": "string"
                },

                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": [
                "pair_index",
                "cause_variable",
                "effect_variable",
                "direction",
                "previous_decision",
                "schema_consistency",
                "metadata_consistency",
                "domain_knowledge_support",
                "category_level_support_only",
                "pair_specific_reason_available",
                "contradiction_to_schema_or_metadata",
                "contradiction_to_domain_knowledge",
                "contradiction_type",
                "corrected_outcome",
                "edge_action",
                "schema_rule_used",
                "metadata_evidence_used",
                "domain_knowledge_reason",
                "one_line_reason",
                "confidence"
            ]
        }
    }


# ============================================================
# CALL LLM VERIFIER
# ============================================================

def call_llm_verifier(
    client,
    system_prompt,
    user_prompt,
    model_name,
    temperature,
    max_retries
):
    response_schema = get_verifier_response_schema()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": response_schema
                },
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            return parsed

        except Exception as e:
            last_error = e
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    raise RuntimeError("LLM verifier failed: " + str(last_error))


# ============================================================
# MAIN PIPELINE
# ============================================================

def verify_majority_decisions_against_schema_metadata_domain(
    input_csv_path,
    schema_json_path,
    metadata_path,
    output_csv_path,
    output_json_path,
    output_plausible_edges_csv,
    model_name,
    temperature,
    max_retries
):
    print("Loading majority CSV...")
    df = load_majority_csv(input_csv_path)

    print("Loading generated schema JSON...")
    schema = load_schema(schema_json_path)

    print("Loading metadata file...")
    metadata = load_metadata(metadata_path)

    client = get_openai_client()
    system_prompt = build_verifier_system_prompt()

    audit_results = {
        "input_csv": input_csv_path,
        "schema_json": schema_json_path,
        "metadata_path": metadata_path,
        "model_name": model_name,
        "verifications": []
    }

    output_rows = []

    for _, row in df.iterrows():
        pair_index = safe_int(row["pair_index"])
        cause = row["cause_variable"]
        effect = row["effect_variable"]
        previous_decision = normalize_decision(row["final_decision"])

        print("=" * 80)
        print(f"Verifying pair {pair_index}: {cause} -> {effect}")
        print(f"Previous majority decision: {previous_decision}")

        user_prompt = build_verifier_user_prompt(
            schema=schema,
            metadata=metadata,
            row=row
        )

        verification = call_llm_verifier(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries
        )

        corrected_outcome = normalize_decision(
            verification["corrected_outcome"]
        )

        computed_action = infer_edge_action(
            previous_decision=previous_decision,
            corrected_outcome=corrected_outcome
        )

        # Enforce edge_action locally so it cannot be inconsistent.
        verification["edge_action"] = computed_action

        audit_results["verifications"].append(verification)

        output_row = {
            "pair_index": pair_index,
            "cause_variable": cause,
            "effect_variable": effect,
            "direction": row["direction"],

            "previous_decision": previous_decision,
            "plausible_count": safe_int(row["plausible_count"]),
            "not_plausible_count": safe_int(row["not_plausible_count"]),
            "uncertain_count": safe_int(row["uncertain_count"]),
            "total_responses": safe_int(row["total_responses"]),

            "schema_consistency": verification["schema_consistency"],
            "metadata_consistency": verification["metadata_consistency"],
            "domain_knowledge_support": verification["domain_knowledge_support"],

            "category_level_support_only": verification[
                "category_level_support_only"
            ],
            "pair_specific_reason_available": verification[
                "pair_specific_reason_available"
            ],

            "contradiction_to_schema_or_metadata": verification[
                "contradiction_to_schema_or_metadata"
            ],
            "contradiction_to_domain_knowledge": verification[
                "contradiction_to_domain_knowledge"
            ],
            "contradiction_type": verification["contradiction_type"],

            "corrected_outcome": corrected_outcome,
            "edge_action": computed_action,

            "schema_rule_used": verification["schema_rule_used"],
            "metadata_evidence_used": verification["metadata_evidence_used"],
            "domain_knowledge_reason": verification["domain_knowledge_reason"],
            "verification_reason": verification["one_line_reason"],
            "verifier_confidence": verification["confidence"]
        }

        output_rows.append(output_row)

        # Incremental save
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_csv_path, index=False)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_BETWEEN_CALLS)

    final_df = pd.DataFrame(output_rows)
    final_df.to_csv(output_csv_path, index=False)

    plausible_edges_df = final_df[
        final_df["corrected_outcome"] == "plausible"
    ].copy()

    plausible_edges_df = plausible_edges_df[
        [
            "pair_index",
            "cause_variable",
            "effect_variable",
            "direction",
            "previous_decision",
            "corrected_outcome",
            "edge_action",
            "verification_reason",
            "verifier_confidence"
        ]
    ]

    plausible_edges_df.to_csv(output_plausible_edges_csv, index=False)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Saved verified CSV to:", output_csv_path)
    print("Saved audit JSON to:", output_json_path)
    print("Saved plausible edges to:", output_plausible_edges_csv)

    print("\nCorrected outcome counts:")
    print(final_df["corrected_outcome"].value_counts())

    print("\nEdge action counts:")
    print(final_df["edge_action"].value_counts())

    print("\nCategory-level-only support counts:")
    print(final_df["category_level_support_only"].value_counts())

    return final_df, audit_results, plausible_edges_df


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    verified_df, audit, plausible_edges_df = verify_majority_decisions_against_schema_metadata_domain(
        input_csv_path=INPUT_CSV_PATH,
        schema_json_path=SCHEMA_JSON_PATH,
        metadata_path=METADATA_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        output_plausible_edges_csv=OUTPUT_PLAUSIBLE_EDGES_CSV,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES
    )
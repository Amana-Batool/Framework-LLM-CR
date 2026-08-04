
import os
import json
import time
import pandas as pd
from openai import OpenAI


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = r"Metabolic_Power_data.csv"
METADATA_JSON_PATH = r"metadata_clean.json"
OUTPUT_JSON_PATH = r"generated_schema.json"

DATASET_NAME = "Gait Analysis Dataset"

DOMAIN_HINT = (
    """Dataset from a gait lab including neuromuscular control variables, 
    demographic variables, anthropometric variables, and gait-related outcomes 
    from children with Cerebral Palsy."""
)

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_RETRIES = 3


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key is None:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment before running this script."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# LOAD CSV
# ============================================================

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Dataset CSV is empty.")

    return df


# ============================================================
# LOAD METADATA JSON
# ============================================================

def load_metadata(metadata_json_path):
    if not os.path.exists(metadata_json_path):
        raise FileNotFoundError(f"Metadata JSON file not found: {metadata_json_path}")

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must be a dictionary/object.")

    return metadata


# ============================================================
# NORMALIZE VARIABLE NAMES
# ============================================================

def normalize_name(name):
    text = str(name).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


# ============================================================
# INFER COLUMN DATA TYPE
# ============================================================

def infer_column_data_type(series):
    s = series.dropna()

    if len(s) == 0:
        return "unknown"

    if pd.api.types.is_bool_dtype(s):
        return "binary"

    if pd.api.types.is_numeric_dtype(s):
        unique_count = s.nunique()

        if unique_count == 2:
            return "binary"

        if pd.api.types.is_integer_dtype(s) and unique_count <= 10:
            return "ordinal"

        return "continuous"

    unique_count = s.nunique()

    if unique_count == 2:
        return "binary"

    if unique_count <= 20:
        return "categorical"

    return "text"


def infer_dataset_types(df):
    inferred_types = {}

    for col in df.columns:
        inferred_types[col] = infer_column_data_type(df[col])

    return inferred_types


# ============================================================
# MATCH METADATA TO CSV COLUMNS
# ============================================================

def match_metadata_to_columns(columns, metadata):
    matched_metadata = {}

    meta_lookup = {}
    for key, value in metadata.items():
        meta_lookup[normalize_name(key)] = value

    for col in columns:
        normalized_col = normalize_name(col)
        matched_metadata[col] = meta_lookup.get(normalized_col, {})

    return matched_metadata


# ============================================================
# BUILD VARIABLE PAYLOAD
# ============================================================

def build_variable_payload(df, matched_metadata, inferred_types):
    variable_payload = []

    for col in df.columns:
        meta = matched_metadata.get(col, {})

        if not isinstance(meta, dict):
            meta = {"description": str(meta)}

        variable_record = {
            "name": col,
            "alias": meta.get("alias", ""),
            "description": (
                meta.get("meaning", "")
                or meta.get("description", "")
                or meta.get("definition", "")
            ),
            "measured_as": meta.get("measured_as", ""),
            "unit": meta.get("unit", ""),
            "inferred_data_type": inferred_types.get(col, "unknown"),
            "non_null_count": int(df[col].notna().sum()),
            "unique_count": int(df[col].nunique(dropna=True))
        }

        variable_payload.append(variable_record)

    return variable_payload


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():
    return """
You are an expert schema/ontology generator for scientific causal discovery datasets. Your task is to generate a SIMPLE causal schema for dataset variables according to the domain and metadata provided.
You can use the prior knowledge to infer the schema. This is a closed system and do not use other variables except from the dataset. The main goal of this schema is to help LLMs 
decribe the causal relationships among variables in the next step. That's why the schema should be scientifically correct, categorizes variables according to their causal nature
and also define what role each variable plays in the dataset. How much its presence matters in the causal diagram.

Moreover, following are some critical rules that you should follow while generating the schema.
CRITICAL RULES:
1. Return STRICT JSON only.
2. Do not include markdown.
3. Do not include explanations outside JSON.
4. Use categories which are shared by multiple variables whenever possible and these categories should be causal not according to the variable datatype.
5. Do NOT create overly specific categories.
6. If a dataset has very few variables and one category must contain only one variable, explain why in schema_quality_warnings.
7. Categories must support causal reasoning, not just statistical typing.
8. Prefer broad, reusable categories
9. Do not create fragmented categories 
10. Identifier variables should be marked as identifier and excluded from causal reasoning.
11. Outcome variables can be children/effects but should rarely be parents unless scientifically justified.
12. Be conservative. If uncertain, use unknown and explain briefly in notes.

The schema must help later LLM components:
- find pairwise causal relationships
- verify whether proposed relationships obey schema rules
- detect forbidden or implausible causal directions
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(dataset_name, domain_hint, variable_payload):
    return f"""
Dataset name:
{dataset_name}

Domain hint:
{domain_hint}

Dataset variables and metadata:
{json.dumps(variable_payload, indent=2)}

Generate the dataset schema using the required JSON structure.
"""


# ============================================================
# EXPECTED JSON SCHEMA FOR STRUCTURED OUTPUT
# ============================================================

def get_response_json_schema():
    return {
        "name": "dataset_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dataset_name": {"type": "string"},
                "domain": {"type": "string"},

                "semantic_categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "category_name": {"type": "string"},
                            "description": {"type": "string"},
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "category_name",
                            "description",
                            "variables"
                        ]
                    }
                },

                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "alias": {"type": "string"},
                            "description": {"type": "string"},
                            "measured_as": {"type": "string"},
                            "data_type": {"type": "string"},
                            "semantic_category": {"type": "string"},
                            "causal_role": {
                                "type": "string",
                                "enum": [
                                    "root_like",
                                    "intermediate",
                                    "outcome_like",
                                    "identifier",
                                    "unknown"
                                ]
                            },
                            "can_be_parent": {"type": "boolean"},
                            "can_be_child": {"type": "boolean"},
                            "exclude_from_causal_reasoning": {"type": "boolean"},
                            "notes": {"type": "string"}
                        },
                        "required": [
                            "name",
                            "alias",
                            "description",
                            "measured_as",
                            "data_type",
                            "semantic_category",
                            "causal_role",
                            "can_be_parent",
                            "can_be_child",
                            "exclude_from_causal_reasoning",
                            "notes"
                        ]
                    }
                },

                "category_level_direction_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rule_id": {"type": "string"},
                            "from_category": {"type": "string"},
                            "to_category": {"type": "string"},
                            "relationship": {
                                "type": "string",
                                "enum": [
                                    "plausible",
                                    "implausible",
                                    "uncertain"
                                ]
                            },
                            "reason": {"type": "string"}
                        },
                        "required": [
                            "rule_id",
                            "from_category",
                            "to_category",
                            "relationship",
                            "reason"
                        ]
                    }
                },

                "parent_only_variables": {
                    "type": "array",
                    "items": {"type": "string"}
                },

                "forbidden_edge_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rule_id": {"type": "string"},
                            "from_variable_or_category": {"type": "string"},
                            "to_variable_or_category": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": [
                            "rule_id",
                            "from_variable_or_category",
                            "to_variable_or_category",
                            "reason"
                        ]
                    }
                },

                "schema_quality_warnings": {
                    "type": "array",
                    "items": {"type": "string"}
                },

                "general_schema_notes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "dataset_name",
                "domain",
                "semantic_categories",
                "variables",
                "category_level_direction_rules",
                "parent_only_variables",
                "forbidden_edge_rules",
                "schema_quality_warnings",
                "general_schema_notes"
            ]
        }
    }


# ============================================================
# CALL LLM
# ============================================================

def call_llm_for_schema(system_prompt,
                        user_prompt,
                        model_name,
                        temperature,
                        max_retries):
    client = get_openai_client()
    response_schema = get_response_json_schema()

    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"Calling LLM attempt {attempt + 1}/{max_retries}...")

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
            schema = json.loads(content)

            return schema

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    raise RuntimeError("LLM call failed after all retries: " + str(last_error))


# ============================================================
# VALIDATE GENERATED SCHEMA
# ============================================================

def validate_schema(schema, expected_columns):
    required_top_keys = [
        "dataset_name",
        "domain",
        "semantic_categories",
        "variables",
        "category_level_direction_rules",
        "parent_only_variables",
        "forbidden_edge_rules",
        "schema_quality_warnings",
        "general_schema_notes"
    ]

    for key in required_top_keys:
        if key not in schema:
            raise ValueError(f"Missing key in schema: {key}")

    if not isinstance(schema["semantic_categories"], list):
        raise ValueError("'semantic_categories' must be a list.")

    if not isinstance(schema["variables"], list):
        raise ValueError("'variables' must be a list.")

    if len(schema["semantic_categories"]) < 3:
        raise ValueError("Schema has fewer than 3 semantic categories.")

    if len(schema["semantic_categories"]) > 6:
        raise ValueError("Schema has more than 6 semantic categories.")

    schema_variable_names = [v.get("name", "") for v in schema["variables"]]

    missing = []
    for col in expected_columns:
        if col not in schema_variable_names:
            missing.append(col)

    if len(missing) > 0:
        raise ValueError("Schema missing these dataset variables: " + str(missing))

    category_names = [
        c.get("category_name", "")
        for c in schema["semantic_categories"]
    ]

    for var in schema["variables"]:
        if var["semantic_category"] not in category_names:
            raise ValueError(
                f"Variable {var['name']} uses unknown category: "
                f"{var['semantic_category']}"
            )

    small_categories = []

    for category in schema["semantic_categories"]:
        category_name = category["category_name"]
        category_vars = category["variables"]

        if len(category_vars) < 2:
            small_categories.append(category_name)

    if len(small_categories) > 0:
        print("\nWARNING: Some categories contain fewer than 2 variables:")
        print(small_categories)
        print("This may be acceptable only for very small datasets or identifiers.")

    print("Schema validation passed.")


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, output_json_path):
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_dataset_schema(csv_path,
                            metadata_json_path,
                            output_json_path,
                            dataset_name,
                            domain_hint,
                            model_name,
                            temperature,
                            max_retries):

    print("Loading dataset...")
    df = load_dataset(csv_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_json_path)

    print("Inferring column data types...")
    inferred_types = infer_dataset_types(df)

    print("Matching metadata to dataset columns...")
    matched_metadata = match_metadata_to_columns(
        columns=list(df.columns),
        metadata=metadata
    )

    print("Building variable payload...")
    variable_payload = build_variable_payload(
        df=df,
        matched_metadata=matched_metadata,
        inferred_types=inferred_types
    )

    print("Building system prompt...")
    system_prompt = build_system_prompt()

    print("Building user prompt...")
    user_prompt = build_user_prompt(
        dataset_name=dataset_name,
        domain_hint=domain_hint,
        variable_payload=variable_payload
    )

    print("Calling LLM schema generator...")
    schema = call_llm_for_schema(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries
    )

    print("Validating generated schema...")
    validate_schema(schema, list(df.columns))

    print("Saving schema...")
    save_json(schema, output_json_path)

    return schema


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERAL LLM SCHEMA GENERATOR")
    print("=" * 70)

    generated_schema = generate_dataset_schema(
        csv_path=CSV_PATH,
        metadata_json_path=METADATA_JSON_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        dataset_name=DATASET_NAME,
        domain_hint=DOMAIN_HINT,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES
    )

    print("\nSchema successfully generated.")
    print("Saved to:", OUTPUT_JSON_PATH)

    print("\nGenerated schema preview:")
    print(json.dumps(generated_schema, indent=2, ensure_ascii=False))
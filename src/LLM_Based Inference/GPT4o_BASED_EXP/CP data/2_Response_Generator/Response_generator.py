
import os
import json
import time
import itertools
import pandas as pd
from openai import OpenAI


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = r"Metabolic_Power_data.csv"
METADATA_JSON_PATH = r"metadata_clean.json"
SCHEMA_JSON_PATH = r"generated_schema.json"

OUTPUT_DIR = r"pairwise_pair_json_outputs"

DATASET_NAME = "Gait Analysis Dataset"

DOMAIN_HINT = (
    "Dataset from a gait lab including neuromuscular control variables, "
    "demographic variables, anthropometric variables, and gait-related outcomes "
    "from children with Cerebral Palsy."
)

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.1
N_RESPONSES_PER_PAIR = 10
MAX_RETRIES = 3

EXCLUDE_NON_CAUSAL_VARIABLES = True


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key is None:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    return OpenAI(api_key=api_key)


# ============================================================
# LOAD FILES
# ============================================================

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Dataset CSV is empty.")

    return df


def load_json(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object.")

    return data




# ============================================================
# METADATA HELPERS
# ============================================================

def normalize_name(name):
    text = str(name).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def match_metadata_to_columns(columns, metadata):
    meta_lookup = {}

    for key, value in metadata.items():
        meta_lookup[normalize_name(key)] = value

    matched = {}

    for col in columns:
        matched[col] = meta_lookup.get(normalize_name(col), {})

    return matched


def get_variable_metadata(variable_name, matched_metadata):
    meta = matched_metadata.get(variable_name, {})

    if not isinstance(meta, dict):
        meta = {"description": str(meta)}

    return {
        "name": variable_name,
        "alias": meta.get("alias", ""),
        "description": (
            meta.get("meaning", "")
            or meta.get("description", "")
            or meta.get("definition", "")
        ),
        "measured_as": meta.get("measured_as", ""),
        "unit": meta.get("unit", "")
    }


# ============================================================
# SCHEMA HELPERS
# ============================================================

def get_schema_variable_info(schema, variable_name):
    for item in schema.get("variables", []):
        if item.get("name") == variable_name:
            return item

    return {}


def get_causal_variables_from_schema(df_columns, schema, exclude_non_causal=True):
    if not exclude_non_causal:
        return list(df_columns)

    causal_vars = []

    for col in df_columns:
        info = get_schema_variable_info(schema, col)

        exclude = info.get("exclude_from_causal_reasoning", False)
        role = info.get("causal_role", "")

        if exclude is True:
            continue

        if role == "identifier":
            continue

        causal_vars.append(col)

    return causal_vars


def build_ordered_pairs(variables):
    pairs = []

    for cause, effect in itertools.permutations(variables, 2):
        pairs.append({
            "cause": cause,
            "effect": effect
        })

    return pairs


# ============================================================
# OUTPUT FILE HELPERS
# ============================================================

def safe_filename(text):
    text = str(text)
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(":", "_")
    text = text.replace("*", "_")
    text = text.replace("?", "_")
    text = text.replace('"', "_")
    text = text.replace("<", "_")
    text = text.replace(">", "_")
    text = text.replace("|", "_")
    return text


def make_pair_output_path(output_dir, pair_index, cause, effect):
    cause_safe = safe_filename(cause)
    effect_safe = safe_filename(effect)

    filename = f"pair_{pair_index:03d}_{cause_safe}_to_{effect_safe}.json"

    return os.path.join(output_dir, filename)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_pairwise_system_prompt():
    return """
You are an expert causal reasoning assistant for scientific datasets. Your task is to judge whether a proposed ordered causal relationship X -> Y is plausible.For each pair generate a response
and one-line reason to choose that response.
Use metadata, schema and your own knowledge to generate responses.

CRITICAL RULES:
1. Return STRICT JSON only.
2. Do not include markdown.
3. Do not include explanations outside JSON.
4. Judge only the proposed direction: cause_variable -> effect_variable.
5. Do not assume association means causation.
6. Use the dataset metadata, variable meanings, and generated schema.
7. Use schema rules as constraints.
8. If the schema says a variable is parent-only, it should not be treated as an effect.
9. If a variable is an identifier, it should not participate in causal reasoning.
10. Be conservative.
11. If the direction is scientifically weak or unclear, mark it as uncertain or not_plausible.
12. Always provide one_line_reason as exactly one concise sentence.
13. The one_line_reason must explain why the edge is plausible, not plausible, uncertain, or reversed.
14. Do not invent unsupported domain facts.

You must produce JSON matching the required schema.
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_pairwise_user_prompt(dataset_name,
                               domain_hint,
                               schema,
                               cause_variable,
                               effect_variable,
                               cause_metadata,
                               effect_metadata,
                               cause_schema_info,
                               effect_schema_info):

    pair_payload = {
        "dataset_name": dataset_name,
        "domain_hint": domain_hint,
        "candidate_edge": {
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": f"{cause_variable} -> {effect_variable}"
        },
        "cause_variable_metadata": cause_metadata,
        "effect_variable_metadata": effect_metadata,
        "cause_variable_schema_info": cause_schema_info,
        "effect_variable_schema_info": effect_schema_info,
        "generated_schema": {
            "semantic_categories": schema.get("semantic_categories", []),
            "category_level_direction_rules": schema.get(
                "category_level_direction_rules", []
            ),
            "parent_only_variables": schema.get("parent_only_variables", []),
            "forbidden_edge_rules": schema.get("forbidden_edge_rules", []),
            "general_schema_notes": schema.get("general_schema_notes", [])
        }
    }

    return json.dumps(pair_payload, indent=2)


# ============================================================
# RESPONSE JSON SCHEMA
# ============================================================

def get_pairwise_response_json_schema():
    return {
        "name": "pairwise_causal_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cause_variable": {"type": "string"},
                "effect_variable": {"type": "string"},
                "plausible": {"type": "boolean"},

                "judgment": {
                    "type": "string",
                    "enum": [
                        "plausible_directed_cause",
                        "not_plausible",
                        "reverse_direction_more_plausible",
                        "correlation_only",
                        "uncertain",
                        "invalid_identifier_or_excluded"
                    ]
                },

                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },

                "one_line_reason": {"type": "string"},

                "schema_consistency": {
                    "type": "string",
                    "enum": [
                        "consistent",
                        "violates_schema",
                        "partially_consistent",
                        "not_enough_information"
                    ]
                },

                "schema_rule_used": {"type": "string"},
                "possible_reverse_edge": {"type": "boolean"},
                "reverse_edge_reason": {"type": "string"}
            },
            "required": [
                "cause_variable",
                "effect_variable",
                "plausible",
                "judgment",
                "confidence",
                "one_line_reason",
                "schema_consistency",
                "schema_rule_used",
                "possible_reverse_edge",
                "reverse_edge_reason"
            ]
        }
    }


# ============================================================
# LOGPROB EXTRACTION
# ============================================================

def extract_logprobs_from_choice(choice, parsed_json):
    result = {
        "available": False,

        "full_response_token_logprobs": [],
        "full_response_total_logprob": None,
        "full_response_mean_logprob": None,

        "reason_sentence": parsed_json.get("one_line_reason", ""),
        "reason_token_logprobs": [],
        "reason_total_logprob": None,
        "reason_mean_logprob": None
    }

    try:
        logprobs_obj = choice.logprobs

        if logprobs_obj is None or logprobs_obj.content is None:
            return result

        tokens = []
        full_logprob_values = []

        for token_item in logprobs_obj.content:
            token = getattr(token_item, "token", "")
            logprob = getattr(token_item, "logprob", None)

            tokens.append(token)

            result["full_response_token_logprobs"].append({
                "token": token,
                "logprob": logprob
            })

            if logprob is not None:
                full_logprob_values.append(float(logprob))

        if len(full_logprob_values) > 0:
            result["available"] = True
            result["full_response_total_logprob"] = float(
                sum(full_logprob_values)
            )
            result["full_response_mean_logprob"] = float(
                sum(full_logprob_values) / len(full_logprob_values)
            )

        reason_sentence = parsed_json.get("one_line_reason", "")

        if reason_sentence == "":
            return result

        full_generated_text = "".join(tokens)
        reason_start = full_generated_text.find(reason_sentence)

        if reason_start == -1:
            return result

        reason_end = reason_start + len(reason_sentence)

        current_char_pos = 0
        reason_logprob_values = []

        for token_item in logprobs_obj.content:
            token = getattr(token_item, "token", "")
            logprob = getattr(token_item, "logprob", None)

            token_start = current_char_pos
            token_end = current_char_pos + len(token)

            overlaps_reason = (
                token_start < reason_end and token_end > reason_start
            )

            if overlaps_reason:
                result["reason_token_logprobs"].append({
                    "token": token,
                    "logprob": logprob
                })

                if logprob is not None:
                    reason_logprob_values.append(float(logprob))

            current_char_pos = token_end

        if len(reason_logprob_values) > 0:
            result["reason_total_logprob"] = float(
                sum(reason_logprob_values)
            )
            result["reason_mean_logprob"] = float(
                sum(reason_logprob_values) / len(reason_logprob_values)
            )

        return result

    except Exception:
        return result


# ============================================================
# CALL LLM FOR ONE PAIR RESPONSE
# ============================================================

def call_llm_for_one_pair_response(client,
                                   system_prompt,
                                   user_prompt,
                                   model_name,
                                   temperature,
                                   max_retries):

    response_schema = get_pairwise_response_json_schema()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                logprobs=True,
                top_logprobs=5,
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

            choice = response.choices[0]
            content = choice.message.content
            parsed_json = json.loads(content)

            logprob_info = extract_logprobs_from_choice(
                choice=choice,
                parsed_json=parsed_json
            )

            usage = None

            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return {
                "raw_json_text": content,
                "parsed_response": parsed_json,
                "logprobs": logprob_info,
                "finish_reason": choice.finish_reason,
                "model": response.model,
                "usage": usage
            }

        except Exception as e:
            last_error = e
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    raise RuntimeError(
        "LLM pairwise call failed after all retries: " + str(last_error)
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_pairwise_llm_responses(csv_path,
                                    metadata_json_path,
                                    schema_json_path,
                                    output_dir,
                                    dataset_name,
                                    domain_hint,
                                    model_name,
                                    temperature,
                                    n_responses_per_pair,
                                    max_retries,
                                    exclude_non_causal_variables=True):

    print("Loading dataset...")
    df = load_dataset(csv_path)

    print("Loading metadata...")
    metadata = load_json(metadata_json_path, "Metadata JSON")

    print("Loading generated schema...")
    schema = load_json(schema_json_path, "Generated schema JSON")


    print("Matching metadata to columns...")
    matched_metadata = match_metadata_to_columns(df.columns, metadata)

    print("Selecting causal variables...")
    variables = get_causal_variables_from_schema(
        df_columns=list(df.columns),
        schema=schema,
        exclude_non_causal=exclude_non_causal_variables
    )

    print("Variables used:")
    print(variables)

    ordered_pairs = build_ordered_pairs(variables)

    print(f"Total ordered pairs: {len(ordered_pairs)}")
    print(f"Responses per pair: {n_responses_per_pair}")
    print(f"Total LLM calls: {len(ordered_pairs) * n_responses_per_pair}")

    os.makedirs(output_dir, exist_ok=True)

    client = get_openai_client()
    system_prompt = build_pairwise_system_prompt()

    summary_index = {
        "dataset_name": dataset_name,
        "domain_hint": domain_hint,
        "model_name": model_name,
        "temperature": temperature,
        "n_responses_per_pair": n_responses_per_pair,
        "total_ordered_pairs": len(ordered_pairs),
        "variables_used": variables,
        "output_directory": output_dir,
        "pair_files": []
    }

    index_path = os.path.join(output_dir, "index.json")

    for pair_index, pair in enumerate(ordered_pairs, start=1):
        cause = pair["cause"]
        effect = pair["effect"]

        print("=" * 70)
        print(f"Pair {pair_index}/{len(ordered_pairs)}: {cause} -> {effect}")

        pair_output_path = make_pair_output_path(
            output_dir=output_dir,
            pair_index=pair_index,
            cause=cause,
            effect=effect
        )

        cause_metadata = get_variable_metadata(cause, matched_metadata)
        effect_metadata = get_variable_metadata(effect, matched_metadata)

        cause_schema_info = get_schema_variable_info(schema, cause)
        effect_schema_info = get_schema_variable_info(schema, effect)

        user_prompt = build_pairwise_user_prompt(
            dataset_name=dataset_name,
            domain_hint=domain_hint,
            schema=schema,
            cause_variable=cause,
            effect_variable=effect,
            cause_metadata=cause_metadata,
            effect_metadata=effect_metadata,
            cause_schema_info=cause_schema_info,
            effect_schema_info=effect_schema_info
        )

        pair_record = {
            "dataset_name": dataset_name,
            "domain_hint": domain_hint,
            "model_name": model_name,
            "temperature": temperature,
            "n_responses_per_pair": n_responses_per_pair,
            "pair_index": pair_index,
            "cause_variable": cause,
            "effect_variable": effect,
            "direction": f"{cause} -> {effect}",
            "pair_output_file": pair_output_path,
            "responses": []
        }

        for response_index in range(1, n_responses_per_pair + 1):
            print(f"  Response {response_index}/{n_responses_per_pair}")

            one_response = call_llm_for_one_pair_response(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries
            )

            one_response["response_index"] = response_index
            one_response["response_id"] = (
                f"pair_{pair_index:03d}_response_{response_index:02d}"
            )

            pair_record["responses"].append(one_response)

            # Save this pair after every response
            with open(pair_output_path, "w", encoding="utf-8") as f:
                json.dump(pair_record, f, indent=2, ensure_ascii=False)

        summary_index["pair_files"].append({
            "pair_index": pair_index,
            "cause_variable": cause,
            "effect_variable": effect,
            "direction": f"{cause} -> {effect}",
            "file_path": pair_output_path
        })

        # Save/update index
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(summary_index, f, indent=2, ensure_ascii=False)

    print("All pair JSON files saved.")
    print("Output directory:", output_dir)

    return summary_index


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2: LLM PAIRWISE CAUSAL RELATIONSHIP FINDER")
    print("=" * 70)

    results = generate_pairwise_llm_responses(
        csv_path=CSV_PATH,
        metadata_json_path=METADATA_JSON_PATH,
        schema_json_path=SCHEMA_JSON_PATH,
        output_dir=OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        domain_hint=DOMAIN_HINT,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        n_responses_per_pair=N_RESPONSES_PER_PAIR,
        max_retries=MAX_RETRIES,
        exclude_non_causal_variables=EXCLUDE_NON_CAUSAL_VARIABLES
    )

    print("\nDone.")
    print("Saved pairwise JSON files in:", OUTPUT_DIR)
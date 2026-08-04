
import os
import json
import math
import glob
import re
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# 1) CONFIG
# =========================================================

INPUT_DIR = "pair_json_outputs"
OUTPUT_DIR = "semantic_entropy_outputs"

OUTPUT_SUMMARY_CSV = "semantic_entropy_summary.csv"
OUTPUT_RESPONSE_LEVEL_CSV = "semantic_entropy_response_level.csv"
OUTPUT_CLUSTER_LEVEL_CSV = "semantic_entropy_cluster_level.csv"
OUTPUT_DETAILS_JSON = "semantic_entropy_details.json"

ENTAILMENT_MODEL_NAME = "microsoft/deberta-large-mnli"
MAX_LENGTH = 512

REASON_LOGPROB_FIELD = "reason_mean_logprob"
# REASON_LOGPROB_FIELD = "reason_total_logprob"

# Semantic text construction.
TEXT_MODE = "judgment_reason_sentence"


STRICT_REASON_SENTENCE = True


STRICT_REASON_LOGPROB = True

# Numerical clipping for exp(logprob)
MIN_LOGPROB_FOR_EXP = -700.0


# =========================================================
# 2) VALIDATE CONFIG
# =========================================================

def validate_config():
    allowed_logprob_fields = {"reason_mean_logprob", "reason_total_logprob"}

    if REASON_LOGPROB_FIELD is None:
        raise ValueError(
            "\nYou must choose REASON_LOGPROB_FIELD manually.\n"
            "Set one of these:\n"
            '    REASON_LOGPROB_FIELD = "reason_mean_logprob"\n'
            '    REASON_LOGPROB_FIELD = "reason_total_logprob"\n'
        )

    if REASON_LOGPROB_FIELD not in allowed_logprob_fields:
        raise ValueError(
            f"Invalid REASON_LOGPROB_FIELD: {REASON_LOGPROB_FIELD}\n"
            f"Allowed values: {allowed_logprob_fields}"
        )

    if TEXT_MODE != "judgment_reason_sentence":
        raise ValueError(
            "This modified script is written for TEXT_MODE='judgment_reason_sentence'."
        )


# =========================================================
# 3) DEVICE
# =========================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# =========================================================
# 4) BASIC HELPERS
# =========================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(x):
    try:
        if x is None:
            return None
        val = float(x)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return None


def safe_exp(log_value: float) -> float:
    
    if log_value is None or not np.isfinite(log_value):
        return None

    log_value = max(float(log_value), MIN_LOGPROB_FOR_EXP)
    return float(math.exp(log_value))


def normalize_probs(masses: List[float]) -> List[float]:
    masses = [float(m) for m in masses if m is not None and m >= 0]
    total = float(sum(masses))

    if total <= 0:
        return []

    return [m / total for m in masses]


def entropy_nats(probs: List[float]) -> float:
    probs = [float(p) for p in probs if p is not None and p > 0]

    if len(probs) == 0:
        return None

    return float(-sum(p * math.log(p) for p in probs))


def compute_mean_logprob_from_token_list(token_logprobs: List[Dict[str, Any]]):
    vals = []

    for item in token_logprobs or []:
        lp = safe_float(item.get("logprob"))
        if lp is not None:
            vals.append(lp)

    if len(vals) == 0:
        return None

    return float(np.mean(vals))


def compute_total_logprob_from_token_list(token_logprobs: List[Dict[str, Any]]):
    vals = []

    for item in token_logprobs or []:
        lp = safe_float(item.get("logprob"))
        if lp is not None:
            vals.append(lp)

    if len(vals) == 0:
        return None

    return float(np.sum(vals))


# =========================================================
# 5) RECOVER FIELDS FROM YOUR JSON FORMAT
# =========================================================

def get_parsed_response(resp: Dict[str, Any]) -> Dict[str, Any]:

    parsed = resp.get("parsed_response")

    if isinstance(parsed, dict):
        return parsed

    raw = resp.get("raw_json_text")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}

    return {}


def recover_judgment_and_reason_sentence(resp: Dict[str, Any]) -> Tuple[str, str]:
   

    parsed = get_parsed_response(resp)
    logprobs = resp.get("logprobs", {})

   
   
    judgment = parsed.get("judgment", parsed.get("judgement", ""))

    reason_sentence = ""
    if isinstance(logprobs, dict):
        reason_sentence = logprobs.get("reason_sentence", "")

    judgment = normalize_text(judgment)
    reason_sentence = normalize_text(reason_sentence)

    return judgment, reason_sentence


def recover_reason_logprob(resp: Dict[str, Any], field_name: str):
   
    logprobs = resp.get("logprobs", {})

    if not isinstance(logprobs, dict):
        return None

    if not logprobs.get("available", False):
        return None

    val = safe_float(logprobs.get(field_name))

    if val is not None:
        return val

    reason_token_logprobs = logprobs.get("reason_token_logprobs", [])

    if field_name == "reason_mean_logprob":
        return compute_mean_logprob_from_token_list(reason_token_logprobs)

    if field_name == "reason_total_logprob":
        return compute_total_logprob_from_token_list(reason_token_logprobs)

    raise ValueError(f"Unsupported reason logprob field: {field_name}")


def build_semantic_text(cause_variable: str,
                        effect_variable: str,
                        judgment: str,
                        reason_sentence: str) -> str:
    
    cause_variable = normalize_text(cause_variable)
    effect_variable = normalize_text(effect_variable)
    judgment = normalize_text(judgment)
    reason_sentence = normalize_text(reason_sentence)

    return (
        f"For the directed pair {cause_variable} -> {effect_variable}, "
        f"the judgment is {judgment}. "
        f"Reason: {reason_sentence}"
    )


# =========================================================
# 6) ENTAILMENT MODEL
# =========================================================

@torch.no_grad()
def predict_nli_label(text_a: str,
                      text_b: str,
                      tokenizer,
                      nli_model,
                      id2label,
                      device) -> str:
    encoded = tokenizer(
        text_a,
        text_b,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = nli_model(**encoded).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())

    return id2label[pred_id]


def is_entailment(label: str) -> bool:
    return "ENTAIL" in str(label).upper()


def bidirectional_entailment(text_a: str,
                             text_b: str,
                             cache: Dict[Tuple[str, str], str],
                             tokenizer,
                             nli_model,
                             id2label,
                             device) -> bool:
    key_ab = (text_a, text_b)
    key_ba = (text_b, text_a)

    if key_ab not in cache:
        cache[key_ab] = predict_nli_label(
            text_a, text_b, tokenizer, nli_model, id2label, device
        )

    if key_ba not in cache:
        cache[key_ba] = predict_nli_label(
            text_b, text_a, tokenizer, nli_model, id2label, device
        )

    return is_entailment(cache[key_ab]) and is_entailment(cache[key_ba])


# =========================================================
# 7) SEMANTIC CLUSTERING
# =========================================================

def cluster_by_semantics(items: List[Dict[str, Any]],
                         tokenizer,
                         nli_model,
                         id2label,
                         device) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
   
    clusters = []
    nli_cache = {}

    for item in items:
        placed = False

        for cluster in clusters:
            representative = cluster["members"][0]

            same_meaning = bidirectional_entailment(
                representative["semantic_text"],
                item["semantic_text"],
                nli_cache,
                tokenizer,
                nli_model,
                id2label,
                device
            )

            if same_meaning:
                cluster["members"].append(item)
                placed = True
                break

        if not placed:
            clusters.append({
                "cluster_id": len(clusters) + 1,
                "members": [item]
            })

    return clusters, nli_cache


# =========================================================
# 8) PREPARE RESPONSES FROM ONE PAIR FILE
# =========================================================

def prepare_items(pair_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cause_variable = normalize_text(pair_data.get("cause_variable", ""))
    effect_variable = normalize_text(pair_data.get("effect_variable", ""))
    direction = normalize_text(pair_data.get("direction", f"{cause_variable} -> {effect_variable}"))

    responses = pair_data.get("responses", [])

    items = []
    skipped_rows = []

    for idx, resp in enumerate(responses, start=1):
        response_index = resp.get("response_index", idx)
        response_id = resp.get("response_id", f"response_{idx}")

        judgment, reason_sentence = recover_judgment_and_reason_sentence(resp)
        reason_logprob = recover_reason_logprob(resp, REASON_LOGPROB_FIELD)

        skip_reason = None

        if judgment == "":
            skip_reason = "missing_judgment"

        elif reason_sentence == "" and STRICT_REASON_SENTENCE:
            skip_reason = "missing_reason_sentence"

        elif reason_logprob is None and STRICT_REASON_LOGPROB:
            skip_reason = f"missing_{REASON_LOGPROB_FIELD}"

        if skip_reason is not None:
            skipped_rows.append({
                "response_index": response_index,
                "response_id": response_id,
                "skip_reason": skip_reason
            })
            continue

        semantic_text = build_semantic_text(
            cause_variable=cause_variable,
            effect_variable=effect_variable,
            judgment=judgment,
            reason_sentence=reason_sentence
        )

        mass = safe_exp(reason_logprob)

        items.append({
            "response_index": response_index,
            "response_id": response_id,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "judgment": judgment,
            "reason_sentence": reason_sentence,
            "semantic_text": semantic_text,
            "reason_logprob_field_used": REASON_LOGPROB_FIELD,
            "reason_logprob_value": float(reason_logprob),
            "mass_unnormalized": mass
        })

    return items, skipped_rows


# =========================================================
# 9) COMPUTE SEMANTIC ENTROPY FROM CLUSTERS
# =========================================================

def compute_logprob_semantic_entropy(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(clusters) == 0:
        return {
            "n_clusters": 0,
            "cluster_probs": [],
            "semantic_entropy_nats": None,
            "semantic_entropy_bits": None,
            "semantic_entropy_normalized": None
        }

    cluster_masses = []

    for cluster in clusters:
        masses = [
            member["mass_unnormalized"]
            for member in cluster["members"]
            if member["mass_unnormalized"] is not None
        ]

        cluster_mass = float(sum(masses)) if len(masses) > 0 else 0.0

        cluster["cluster_mass_unnormalized"] = cluster_mass
        cluster["cluster_size"] = len(cluster["members"])

        cluster_masses.append(cluster_mass)

    cluster_probs = normalize_probs(cluster_masses)

    for cluster, p in zip(clusters, cluster_probs):
        cluster["cluster_probability"] = float(p)

    se_nats = entropy_nats(cluster_probs)
    se_bits = None if se_nats is None else float(se_nats / math.log(2))

    if se_nats is None:
        se_normalized = None
    elif len(cluster_probs) <= 1:
        se_normalized = 0.0
    else:
        se_normalized = float(se_nats / math.log(len(cluster_probs)))

    return {
        "n_clusters": len(clusters),
        "cluster_probs": cluster_probs,
        "semantic_entropy_nats": se_nats,
        "semantic_entropy_bits": se_bits,
        "semantic_entropy_normalized": se_normalized
    }


# =========================================================
# 10) PROCESS ONE JSON FILE
# =========================================================

def process_one_pair_file(filepath: str,
                          tokenizer,
                          nli_model,
                          id2label,
                          device) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        pair_data = json.load(f)

    pair_file = os.path.basename(filepath)

    cause_variable = normalize_text(pair_data.get("cause_variable", ""))
    effect_variable = normalize_text(pair_data.get("effect_variable", ""))
    direction = normalize_text(pair_data.get("direction", f"{cause_variable} -> {effect_variable}"))
    pair_index = pair_data.get("pair_index", None)

    responses = pair_data.get("responses", [])
    total_responses_in_file = len(responses)

    items, skipped_rows = prepare_items(pair_data)

    response_level_rows = []

    for item in items:
        row = {
            "pair_file": pair_file,
            "pair_index": pair_index,
            **item
        }
        response_level_rows.append(row)

    if len(items) == 0:
        summary = {
            "pair_file": pair_file,
            "pair_index": pair_index,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "n_total_responses_in_file": total_responses_in_file,
            "n_valid_responses_used": 0,
            "n_skipped_responses": len(skipped_rows),
            "reason_logprob_field_used": REASON_LOGPROB_FIELD,
            "n_clusters": 0,
            "semantic_entropy_nats": None,
            "semantic_entropy_bits": None,
            "semantic_entropy_normalized": None,
            "status": "no_valid_responses"
        }

        detail = {
            "pair_file": pair_file,
            "pair_index": pair_index,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "skipped_responses": skipped_rows,
            "clusters": [],
            "status": "no_valid_responses"
        }

        return summary, response_level_rows, [], detail

    clusters, nli_cache = cluster_by_semantics(
        items,
        tokenizer,
        nli_model,
        id2label,
        device
    )

    se_stats = compute_logprob_semantic_entropy(clusters)

    summary = {
        "pair_file": pair_file,
        "pair_index": pair_index,
        "cause_variable": cause_variable,
        "effect_variable": effect_variable,
        "direction": direction,
        "n_total_responses_in_file": total_responses_in_file,
        "n_valid_responses_used": len(items),
        "n_skipped_responses": len(skipped_rows),
        "reason_logprob_field_used": REASON_LOGPROB_FIELD,
        "n_clusters": se_stats["n_clusters"],
        "semantic_entropy_nats": se_stats["semantic_entropy_nats"],
        "semantic_entropy_bits": se_stats["semantic_entropy_bits"],
        "semantic_entropy_normalized": se_stats["semantic_entropy_normalized"],
        "n_nli_comparisons_cached": len(nli_cache),
        "status": "ok"
    }

    cluster_level_rows = []
    detail_clusters = []

    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        cluster_size = int(cluster["cluster_size"])
        cluster_mass = float(cluster["cluster_mass_unnormalized"])
        cluster_probability = float(cluster["cluster_probability"])

        member_ids = [str(m["response_id"]) for m in cluster["members"]]

        representative_text = cluster["members"][0]["semantic_text"]

        cluster_level_rows.append({
            "pair_file": pair_file,
            "pair_index": pair_index,
            "cause_variable": cause_variable,
            "effect_variable": effect_variable,
            "direction": direction,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "cluster_mass_unnormalized": cluster_mass,
            "cluster_probability": cluster_probability,
            "member_response_ids": "; ".join(member_ids),
            "representative_semantic_text": representative_text
        })

        detail_clusters.append({
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "cluster_mass_unnormalized": cluster_mass,
            "cluster_probability": cluster_probability,
            "members": [
                {
                    "response_index": member["response_index"],
                    "response_id": member["response_id"],
                    "judgment": member["judgment"],
                    "reason_sentence": member["reason_sentence"],
                    "semantic_text": member["semantic_text"],
                    "reason_logprob_field_used": member["reason_logprob_field_used"],
                    "reason_logprob_value": float(member["reason_logprob_value"]),
                    "mass_unnormalized": float(member["mass_unnormalized"])
                }
                for member in cluster["members"]
            ]
        })

    detail = {
        "pair_file": pair_file,
        "pair_index": pair_index,
        "cause_variable": cause_variable,
        "effect_variable": effect_variable,
        "direction": direction,
        "n_total_responses_in_file": total_responses_in_file,
        "n_valid_responses_used": len(items),
        "n_skipped_responses": len(skipped_rows),
        "reason_logprob_field_used": REASON_LOGPROB_FIELD,
        "semantic_entropy_nats": se_stats["semantic_entropy_nats"],
        "semantic_entropy_bits": se_stats["semantic_entropy_bits"],
        "semantic_entropy_normalized": se_stats["semantic_entropy_normalized"],
        "n_nli_comparisons_cached": len(nli_cache),
        "skipped_responses": skipped_rows,
        "clusters": detail_clusters,
        "status": "ok"
    }

    return summary, response_level_rows, cluster_level_rows, detail


# =========================================================
# 11) MAIN
# =========================================================

def main():
    validate_config()

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found inside: {INPUT_DIR}")

    device = get_device()

    print("\nLoading entailment model:", ENTAILMENT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL_NAME)
    nli_model.to(device)
    nli_model.eval()

    id2label = {
        int(k): str(v).upper()
        for k, v in nli_model.config.id2label.items()
    }

    print("NLI labels:", id2label)
    print(f"\nFound {len(json_files)} JSON pair files.")
    print(f"Using reason logprob field: {REASON_LOGPROB_FIELD}\n")

    summary_rows = []
    response_rows_all = []
    cluster_rows_all = []
    detail_rows = []

    for i, filepath in enumerate(json_files, start=1):
        print(f"[{i}/{len(json_files)}] Processing: {os.path.basename(filepath)}")

        try:
            summary, response_rows, cluster_rows, detail = process_one_pair_file(
                filepath,
                tokenizer,
                nli_model,
                id2label,
                device
            )

        except Exception as e:
            summary = {
                "pair_file": os.path.basename(filepath),
                "pair_index": None,
                "cause_variable": None,
                "effect_variable": None,
                "direction": None,
                "n_total_responses_in_file": None,
                "n_valid_responses_used": None,
                "n_skipped_responses": None,
                "reason_logprob_field_used": REASON_LOGPROB_FIELD,
                "n_clusters": None,
                "semantic_entropy_nats": None,
                "semantic_entropy_bits": None,
                "semantic_entropy_normalized": None,
                "status": f"error: {str(e)}"
            }

            response_rows = []
            cluster_rows = []

            detail = {
                "pair_file": os.path.basename(filepath),
                "reason_logprob_field_used": REASON_LOGPROB_FIELD,
                "status": f"error: {str(e)}"
            }

        summary_rows.append(summary)
        response_rows_all.extend(response_rows)
        cluster_rows_all.extend(cluster_rows)
        detail_rows.append(detail)

    df_summary = pd.DataFrame(summary_rows)
    df_response = pd.DataFrame(response_rows_all)
    df_cluster = pd.DataFrame(cluster_rows_all)

    if "semantic_entropy_nats" in df_summary.columns:
        df_summary = df_summary.sort_values(
            by="semantic_entropy_nats",
            ascending=False,
            na_position="last"
        )

    summary_path = os.path.join(OUTPUT_DIR, OUTPUT_SUMMARY_CSV)
    response_path = os.path.join(OUTPUT_DIR, OUTPUT_RESPONSE_LEVEL_CSV)
    cluster_path = os.path.join(OUTPUT_DIR, OUTPUT_CLUSTER_LEVEL_CSV)
    details_path = os.path.join(OUTPUT_DIR, OUTPUT_DETAILS_JSON)

    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    df_response.to_csv(response_path, index=False, encoding="utf-8")
    df_cluster.to_csv(cluster_path, index=False, encoding="utf-8")

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(detail_rows, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Saved:", summary_path)
    print("Saved:", response_path)
    print("Saved:", cluster_path)
    print("Saved:", details_path)

    print("\nTop rows:")
    print(df_summary.head(10))


if __name__ == "__main__":
    main()
import re
import json
import string
import csv
import argparse
from collections import Counter

# ---------------------------------
# 1. TEXT NORMALIZATION
# ---------------------------------

def normalize(text: str) -> str:
    """
    Standard VQA text normalization applied to BOTH the generated answer
    and each ground-truth answer before comparison.

    Steps (in order):
        1. Lowercase
        2. Strip leading/trailing whitespace
        3. Remove all punctuation (via str.translate)
        4. Remove English articles: 'a', 'an', 'the'
        5. Collapse multiple spaces into one and strip again

    This is consistent with the normalization used in the original
    VQA v1/v2 evaluation scripts (Antol et al., 2015).
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. MATCHING FUNCTION
# ---------------------------------------------------------------------------

def answer_in_generated(gt_answer: str, generated: str) -> bool:
    """
    Returns True if `gt_answer` appears as a complete word or phrase inside
    `generated` (both already normalized).

    Uses a word-boundary regex (\b) so that, e.g., the GT answer "right"
    does not match the substring "rights" or "upright".

    Why containment match instead of exact match?
    - Qwen2-VL outputs full descriptive sentences, not short answers.
    - Exact match would yield ~0% accuracy even for correct responses.
    - Containment match checks whether the model's response *includes*
      the correct answer somewhere in its output, which is semantically
      valid for a sentence-generating VLM.

    Edge case: empty gt_answer strings return False immediately.
    """
    if not gt_answer:
        return False
    pattern = r'\b' + re.escape(gt_answer) + r'\b'
    return bool(re.search(pattern, generated))


# ---------------------------------------------------------------------------
# 3. PARSING generated_outputs.txt
# ---------------------------------------------------------------------------

def parse_generated_outputs(filepath: str) -> list[dict]:
    """
    Parses the flat text file produced by the Qwen2-VL inference script.

    File format (repeating block):
    Question: [...]
    Answer: [...]
    Image: [...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"Question:\s*(.*?)\s*"
        r"Answer:\s*(.*?)\s*"
        r"Image:\s*([^\n]+)",
    )

    parsed = []

    for match in pattern.finditer(content):
        image_path = match.group(3).strip()

        parsed.append({
            "image": image_path.split("/")[-1].split("\\")[-1],
            "generated_answer": match.group(2).strip()
        })

    print(f"[Parser] Parsed {len(parsed)} records")
    return parsed

    """
    blocks = content.strip().split("system\n")
    blocks = [b for b in blocks if b.strip()]

    parsed = []
    parse_failures = 0

    for block in blocks:
        # Extract image filename
        img_match = re.search(r'VizWiz_val_\d+\.jpg', block)
        if not img_match:
            parse_failures += 1
            continue

        image_file = img_match.group(0)

        # Extract generated answer text
        # Pattern: everything between "assistant\n" and the trailing ", val\...<image>"
        ans_match = re.search(
            r'Answer:(.*),\s*Image:[^\n]*VizWiz_val_\d+\.jpg',
            block,
            re.DOTALL
        )
        if not ans_match:
            parse_failures += 1
            continue

        parsed.append({
            "image": image_file,
            "generated_answer": ans_match.group(1).strip()
        })

    print(f"[Parser] Total blocks: {len(blocks)} | "
          f"Parsed: {len(parsed)} | Failures: {parse_failures}")
    return parsed

    """


# ---------------------------------------------------------------------------
# 4. LOADING GROUND TRUTH
# ---------------------------------------------------------------------------

def load_ground_truth(filepath: str) -> dict[str, list[str]]:
    """
    Loads the VizWiz annotation JSON (train.json / val.json).

    Expected JSON structure:
        [
          {
            "image": "VizWiz_val_XXXXXXXX.jpg",
            "question": "...",
            "answers": [{"answer": "...", "answer_confidence": "..."}, ...],
            ...
          }, ...
        ]

    Returns a dict mapping image filename -> {
        "answers":     list of raw answer strings,
        "answer_type": str ("other", "yes/no", or "unanswerable")
    }
    Each sample has exactly 10 crowd-sourced answers.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["image"]: {
            "answers": [a["answer"].lower().strip() for a in item["answers"]],
            "answer_type": item.get("answer_type", "unknown")
        }
        for item in data
    }


# ---------------------------------------------------------------------------
# 5. SCORING
# ---------------------------------------------------------------------------

def score_sample(generated_answer: str, gt_answers: list[str]) -> dict:
    """
    Scores a single sample.

    Args:
        generated_answer: Raw string from the model output (NOT normalized).
        gt_answers:       List of 10 raw GT answer strings.

    Returns a dict with:
        - match_count: number of GT answers whose normalized form was found
                       in the normalized generated output (can exceed 3)
        - accuracy:    min(match_count / 3, 1.0)

    Note: All 10 GT answers are checked; match_count can range 0–10 before
    being capped at 3 by the formula.
    """
    if "image quality issues" in generated_answer:
        generated_answer = "unanswerable"
    gen_norm = normalize(generated_answer)
    gt_norms = [normalize(a) for a in gt_answers]

    match_count = sum(
        1 for a in gt_norms if answer_in_generated(a, gen_norm)
    )
    accuracy = min(match_count / 3, 1.0)

    return {
        "match_count": match_count,
        "accuracy": round(accuracy, 4)
    }


# ---------------------------------------------------------------------------
# 6. HELPER: per-type summary block
# ---------------------------------------------------------------------------

def _print_type_summary(label: str, subset: list[dict]):
    """Prints the standard accuracy breakdown for a filtered subset of results."""
    n = len(subset)
    if n == 0:
        print(f"\n  [{label}]  No samples found.")
        return

    avg_acc     = sum(r["accuracy"] for r in subset) / n
    full_credit = sum(1 for r in subset if r["match_count"] >= 3)
    two_thirds  = sum(1 for r in subset if r["match_count"] == 2)
    one_third   = sum(1 for r in subset if r["match_count"] == 1)
    no_credit   = sum(1 for r in subset if r["match_count"] == 0)

    print(f"\n  Answer type : {label}  (n={n})")
    print(f"  Avg Accuracy: {avg_acc:.4f}  ({avg_acc * 100:.2f}%)")
    print(f"    acc = 1.00  (match_count >= 3) : {full_credit:5d}  ({full_credit/n*100:.1f}%)")
    print(f"    acc = 0.67  (match_count == 2) : {two_thirds:5d}  ({two_thirds/n*100:.1f}%)")
    print(f"    acc = 0.33  (match_count == 1) : {one_third:5d}  ({one_third/n*100:.1f}%)")
    print(f"    acc = 0.00  (match_count == 0) : {no_credit:5d}  ({no_credit/n*100:.1f}%)")


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------

def evaluate(outputs_path: str, gt_path: str, out_csv: str):
    """
    Full evaluation pipeline:
        1. Parse generated outputs
        2. Load ground truth
        3. Score each sample
        4. Print summary statistics
        5. Save per-sample results to CSV
    """
    predictions = parse_generated_outputs(outputs_path)
    gt_lookup   = load_ground_truth(gt_path)

    results = []
    skipped = 0

    for pred in predictions:
        image = pred["image"]
        if image not in gt_lookup:
            skipped += 1
            continue
        
        gt_entry = gt_lookup[image]

        score = score_sample(pred["generated_answer"], gt_entry["answers"])
        results.append({
            "image":            image,
            "generated_answer": pred["generated_answer"],
            "gt_answers":       " | ".join(gt_entry["answers"]),
            "answer_type":      gt_entry["answer_type"],
            "match_count":      score["match_count"],
            "accuracy":         score["accuracy"]
        })

    if skipped:
        print(f"[Warning] {skipped} predictions had no matching GT entry and were skipped.")

    # --- Summary Statistics ---
    n = len(results)
    avg_acc     = sum(r["accuracy"] for r in results) / n
    full_credit = sum(1 for r in results if r["match_count"] >= 3)
    two_thirds  = sum(1 for r in results if r["match_count"] == 2)
    one_third   = sum(1 for r in results if r["match_count"] == 1)
    no_credit   = sum(1 for r in results if r["match_count"] == 0)

    print()
    print("=" * 58)
    print("  Qwen2-VL-2B  —  VizWiz VQA Accuracy Results")
    print("=" * 58)
    print(f"  Samples scored        : {n}")
    print(f"  Average VQA Accuracy  : {avg_acc:.4f}  ({avg_acc * 100:.2f}%)")
    print()
    print(f"  Score Breakdown:")
    print(f"    acc = 1.00  (match_count >= 3) : {full_credit:5d}  ({full_credit/n*100:.1f}%)")
    print(f"    acc = 0.67  (match_count == 2) : {two_thirds:5d}  ({two_thirds/n*100:.1f}%)")
    print(f"    acc = 0.33  (match_count == 1) : {one_third:5d}  ({one_third/n*100:.1f}%)")
    print(f"    acc = 0.00  (match_count == 0) : {no_credit:5d}  ({no_credit/n*100:.1f}%)")
    print("=" * 58)

    # --- Per-Answer-Type Breakdown ---
    print()
    print("=" * 58)
    print("  Accuracy Breakdown by Answer Type")
    print("=" * 58)
    for answer_type in ["other", "yes/no", "unanswerable"]:
        subset = [r for r in results if r["answer_type"] == answer_type]
        _print_type_summary(answer_type, subset)

    # Surface any unexpected answer_type values present in the data
    known_types = {"other", "yes/no", "unanswerable"}
    found_types = {r["answer_type"] for r in results}
    for unknown_type in sorted(found_types - known_types):
        subset = [r for r in results if r["answer_type"] == unknown_type]
        _print_type_summary(f"UNKNOWN: {unknown_type}", subset)

    print()
    print("=" * 58)

    # --- Save CSV ---
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "generated_answer", "gt_answers", "answer_type", "match_count", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Output] Per-sample results saved to: {out_csv}")
    return results


# ---------------------------------------------------------------------------
# 8. CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute VQA accuracy for VLM outputs on VizWiz dataset."
    )
    parser.add_argument(
        "--outputs",
        default="generated_outputs/generated_outputs[text_recognition].txt",
        help="Path to the model's generated outputs text file."
    )
    parser.add_argument(
        "--gt",
        default="Annotations/val.json",
        help="Path to the VizWiz ground truth JSON (val.json or train.json)."
    )
    parser.add_argument(
        "--out_csv",
        default="vqa_accuracy_results.csv",
        help="Path to save the per-sample CSV results."
    )
    args = parser.parse_args()
    evaluate(args.outputs, args.gt, args.out_csv)
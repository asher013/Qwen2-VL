# FLOW OF SCRIPT
# 1. ITERATE THROUGH EACH ANSWER
# 2. IF ANSWER IS INCORRECTLY FORMATTED, CORRECT IT
# 3. STANDARDIZE ANSWER FIELD (LOWERCASE, STRIP WHITESPACE)
# 4. INGEST GROUND TRUTH ANSWERS FROM VALIDATION SET
# 5. CALCULATE ACCURACY USING VIZWIZ FORMULA (min(gt_answers==answer/3,1))
# 6. AGGREGATE ALL ANSWER ACCURACIES INTO ONE FINAL SCORE, ALSO SEPARATING ACCURACIES BETWEEN ANSWER TYPES

import json
import re # for regex searching
from collections import defaultdict # using dictionaries for faster lookup time

OUTPUT_FILE = "generated_outputs\\generated_outputs[text_recognition].txt"
GT_FILE = "Annotations\\val_text.json"

# ── 1. LOAD GROUND TRUTH ──────────────────────────────────────────────────────
 
with open(GT_FILE, "r", encoding="utf-8") as f:
    raw_gt = json.load(f)
 
# Build lookup: image filename → list of ground truth answer strings
ground_truth = {}
for item in raw_gt:
    img = item["image"].split("/")[-1].split("\\")[-1]
    answers = item.get("answers", [])
    if answers and isinstance(answers[0], dict):
        answers = [a["answer"] for a in answers]
    ground_truth[img] = answers
 
print(f"Loaded {len(ground_truth)} ground truth entries.")

# ── 2. PARSE GENERATED OUTPUT FILE ───────────────────────────────────────────
 
def parse_line(line):
    """Extract (image_filename, predicted_answer) from a generated output line."""
    line = line.strip()
    if not line:
        return None
 
    # Find image filename
    img_match = re.search(r'VizWiz_\w+\.jpg', line)
    if not img_match:
        return None
    image = img_match.group()
 
    # Strip markdown code blocks, extract JSON
    raw = re.sub(r'```json|```', '', line)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return None
    
    # Strategy 1: find all {...} blocks and merge them
    # Handles both normal JSON and split JSON across multiple blocks
    blocks = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
    merged = {}
    for block in blocks:
        try:
            merged.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    
    if "answer" in merged:
        return image, merged["answer"]

    # Strategy 2: regex extraction of answer field
    # Handles broken JSON where fields leaked outside the closing brace
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if answer_match:
        return image, answer_match.group(1)
 
    return None
 
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()
 
parsed = []
parse_errors_list = []
parse_errors = 0
for line in lines:
    result = parse_line(line)
    if result:
        parsed.append(result)
    else:
        parse_errors += 1
        parse_errors_list.append(line)
 
print(f"Parsed {len(parsed)} records. Parse errors: {parse_errors_list}")

# ── 3. STANDARDIZE ANSWERS ────────────────────────────────────────────────────
 
def standardize(answer):
    """Lowercase, strip whitespace and trailing punctuation."""
    if not isinstance(answer, str):
        return ""
    answer = answer.lower().strip()
    answer = re.sub(r'[.!?,;:]+$', '', answer).strip() # strip of any special characters
    return answer

# ── 4 & 5. CALCULATE VIZWIZ ACCURACY ─────────────────────────────────────────
# Formula: min(number of GT answers matching prediction / 3, 1.0)
 
scores = []
not_found = 0
unanswerable_scores = []
answerable_scores = []
 
for image, predicted in parsed:
    if image not in ground_truth:
        not_found += 1
        continue
 
    gt_answers = ground_truth[image]
    pred_std = standardize(predicted)
    gt_std = [standardize(a) for a in gt_answers]
 
    match_count = sum(1 for gt in gt_std if gt == pred_std)
    accuracy = min(match_count / 3, 1.0)
    scores.append(accuracy)
 
    if pred_std == "unanswerable":
        unanswerable_scores.append(accuracy)
    else:
        answerable_scores.append(accuracy)

# ── 6. AGGREGATE AND PRINT RESULTS ───────────────────────────────────────────
 
overall = sum(scores) / len(scores) if scores else 0.0
ans_acc = sum(answerable_scores) / len(answerable_scores) if answerable_scores else 0.0
unans_acc = sum(unanswerable_scores) / len(unanswerable_scores) if unanswerable_scores else 0.0
 
print("\n" + "="*50)
print("VIZWIZ ACCURACY RESULTS")
print("="*50)
print(f"Total evaluated:         {len(scores)}")
print(f"Not found in GT:         {not_found}")
print(f"Parse errors:            {parse_errors}")
print(f"\nOverall accuracy:        {overall:.4f} ({overall*100:.2f}%)")
print(f"Answerable accuracy:     {ans_acc:.4f}  (n={len(answerable_scores)})")
print(f"Unanswerable accuracy:   {unans_acc:.4f}  (n={len(unanswerable_scores)})")
print("="*50)

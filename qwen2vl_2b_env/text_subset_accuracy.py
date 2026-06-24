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
 
def parse_text_file(file_path):
    """Parse records with question answer image pattern."""

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"Question:\s*(.*?)\s*"
        r"Answer:\s*(.*?)\s*"
        r"Image:\s*([^\n]+)",
        re.DOTALL
    )

    parsed = []

    for match in pattern.finditer(text):
        answer = match.group(2).strip()
        image_path = match.group(3).strip()

        image = image_path.split("/")[-1].split("\\")[-1]

        parsed.append((image, answer))

    return parsed

parsed = parse_text_file(OUTPUT_FILE)
print(f"Parsed {len(parsed)} records.")



# ── 3. STANDARDIZE ANSWERS ────────────────────────────────────────────────────
 
def standardize(answer):
    """Lowercase, strip whitespace and trailing punctuation."""
    if not isinstance(answer, str):
        return ""
    answer = answer.lower().strip()
    # remove whitespace
    answer = re.sub(r"\s+", " ", answer)
    # remove punctuation
    answer = re.sub(r"[^\w\s]+$", "", answer)
    return answer

# ── 4. NORMALIZE ANSWERS ────────────────────────────────────────────────────

ARTICLES = {"a", "an", "the"}

def vizwiz_normalize(text):
    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    tokens = [
        tok for tok in text.split()
        if tok not in ARTICLES
    ]

    return " ".join(tokens)

# ── 5. STRIP LEADING PATTERNS ────────────────────────────────────────────────────

LEADING_PATTERNS = [
    r"^the image shows\s+",
    r"^image shows\s+",
    r"^this is\s+",
    r"^the book in the image is\s+",
    r"^the dvd in the image is\s+",
    r"^the can in the image is\s+",
    r"^the bottle in the picture is\s+",
    r"^the item in the image is\s+",
    r"^the text on the image reads\s+",
]

def strip_boilerplate(text):
    text = vizwiz_normalize(text)

    for pattern in LEADING_PATTERNS:
        text = re.sub(pattern, "", text)

    return text.strip()


# ── 6. TOKEN OVERLAP FOR ANSWER MATCHING ────────────────────────────────────────────────────

def token_f1(pred, gt):
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())

    overlap = len(pred_tokens & gt_tokens)

    if overlap == 0:
        return 0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)

# ── 7 & 8. CALCULATE VIZWIZ ACCURACY ─────────────────────────────────────────
# Formula: min(number of GT answers matching prediction / 3, 1.0)
 
scores = []
 
for image, predicted in parsed:
    if image not in ground_truth:
        continue
 
    gt_answers = ground_truth[image]
    pred_std = standardize(predicted)
    pred_std = vizwiz_normalize(pred_std)
    pred_std = strip_boilerplate(pred_std)
    gt_std = [vizwiz_normalize(standardize(a)) for a in gt_answers]
 
    match_count = sum(1 for gt in gt_std if token_f1(pred_std, gt) >= 0.8)
    print(pred_std, gt_std)
    accuracy = min(match_count / 3, 1.0)
    scores.append(accuracy)

# ── 8. AGGREGATE AND PRINT RESULTS ───────────────────────────────────────────
 
overall = sum(scores) / len(scores) if scores else 0.0
 
print("\n" + "="*50)
print("VIZWIZ ACCURACY RESULTS")
print("="*50)
print(f"Total evaluated:         {len(scores)}")
print(f"\nOverall accuracy:        {overall:.4f} ({overall*100:.2f}%)")
print("="*50)

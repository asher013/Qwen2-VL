import os
os.environ["PYTHONUTF8"] = "1"
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, Dataset
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(torch.version.cuda)
print(f"Using device: {device}")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_CHECKPOINTING = True  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False
OPTIM = "paged_adamw_8bit" # saves memory compared to adamw_torch or paged_adamw_32bit
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
EVAL_STEPS = 50
SAVE_STEPS = 50
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_STEPS = 0
DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs
REMOVE_UNUSED_COLUMNS = False # VLM thing
MAX_SEQ_LEN=256 
NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS
print(f"NUM_STEPS: {NUM_STEPS}")

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in OCR for blind and low-vision users. 
Your task is to process and extract text from images taken by blind and low-vision users, leveraging multimodal understanding to provide accurate and contextually relevant information. 
Extract all text exactly as it appears.
Do not provide summaries or interpretations unless explicitly requested by the user.
Only respond in short answers (2-3 words) and do not elaborate or use captions for the images.
"""

def aggregate_answers(answers):
    confidence_weights = {"yes": 1.0, "maybe": 0.5, "no": 0.25}
    answer_scores = {}
    for a in answers:
        ans = a["answer"].strip().lower()
        weight = confidence_weights[a["answer_confidence"]]
        answer_scores[ans] = answer_scores.get(ans, 0) + weight
    return max(answer_scores, key=answer_scores.get)

def format_data(sample):
    target_answer = aggregate_answers(sample["answers"])
    return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "Images\\val\\val\\"+sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_answer}],
            },
    ]

def text_generator(sample_data):
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )

    image_inputs = sample_data[1]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images = image_inputs,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

    new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]

    output_text = processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    del inputs
    # actual_answer = sample_data[2]["content"][0]["text"]
    return sample_data[1]["content"][1]["text"], output_text[0], image_inputs

# ========== MODEL CREATION =================
if device == "cuda":
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config
        )

else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.bfloat16,
        device_map="auto"
        )

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# ======== DATASET IMPORT AND CONSTRUCTION ============
dataset = load_dataset("json",data_files="Annotations\\val_text.json")
train_dataset = dataset["train"]

train_test = train_dataset.train_test_split(test_size=0.2, seed=42)
train_val = train_test["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_val["train"]
eval_dataset = train_val["test"]
test_dataset = train_test["test"]

print(len(train_dataset), len(eval_dataset), len(test_dataset))

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]


with open("generated_outputs/generated_outputs[text_recognition].txt", "w", encoding='utf-8') as f:
    f.write("="*25)
    f.write("SYSTEM MESSAGE") 
    f.write("="*25)
    f.write("\n")
    f.write(f"{system_message}\n")
    f.write("="*25) 
    f.write("END")
    f.write("="*25) 
    f.write("\n")


# ========= PRE-TRAINING EVALUATION ===========
"""
for i in range(100):
    user_question, generated_text, img = text_generator(train_dataset[i])
    print(f"Writing to file...")
    with open("generated_outputs/generated_outputs[text_recognition].txt", "a", encoding='utf-8') as f:
        f.write(f"Question: {user_question}, Answer: {generated_text}, Image: {img}\n")
    print(f"Image {i} processed.")
    torch.cuda.empty_cache()
"""

# ======== PARAMETER-EFFICIENT FINE-TUNING | Low-Rank Adaptation Configuration ===========
peft_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout=0.05,
    r=16,
    bias='none',
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    task_type='CAUSAL_LM',
)

print(f"Before adapter parameters: {model.num_parameters()}")
"""peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
"""

# ============ SUPERVISED FINE-TUNING CONFIGURATION ====================
training_args = SFTConfig(
    output_dir = "./output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    dataset_kwargs=DATASET_KWARGS,
    max_length=MAX_SEQ_LEN,
    max_steps=5,
    remove_unused_columns=REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
)

collate_sample = [train_dataset[0], train_dataset[1]]

# ============= COLLATING AND STANDARDIZING DATA FOR LLM TRAINING =================
def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [Image.open(example[1]["content"][0]["image"]).convert("RGB") for example in examples]
    answers = [example[2]["content"][0]["text"] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = torch.full_like(batch['input_ids'], -100)
    
    for i, answer in enumerate(answers):
        answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = batch["input_ids"][i].tolist()
        
        for start in range(len(input_ids) - len(answer_ids) + 1):
            if input_ids[start:start + len(answer_ids)] == answer_ids:
                labels[i, start:start + len(answer_ids)] = batch["input_ids"][i, start:start + len(answer_ids)]
                break
    
    batch["labels"] = labels
    return batch

collated_data = collate_fn(collate_sample)
print(collated_data.keys())

# ======== SUPERVISED FINE-TUNING TRAINER ==============
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor,
)

sample_batch = collate_fn([train_dataset[0], train_dataset[1]])
print("Labels shape:", sample_batch["labels"].shape)
print("Non-ignored tokens:", (sample_batch["labels"] != -100).sum().item())
print("Labels:", sample_batch["labels"][0][:50])
print("Input IDs:", sample_batch["input_ids"][0][:50])

non_masked = sample_batch["labels"][0][sample_batch["labels"][0] != -100]
print(processor.tokenizer.decode(non_masked))

# =========== TRAINER EVALUATION ====================
"""
print("-"*30)
print("Initial Evaluation")
metric = trainer.evaluate()
print(metric)
print("-"*30)
"""
torch.cuda.empty_cache()
print("Training")
trainer.train()
print("-"*30)

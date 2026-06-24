import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore")

import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(torch.version.cuda)
print(f"Using device: {device}")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING = True,  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False,
OPTIM = "paged_adamw_32bit"
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
MAX_SEQ_LEN=128
NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS
print(f"NUM_STEPS: {NUM_STEPS}")

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in OCR for blind and low-vision users. 
Your task is to process and extract text from images taken by blind and low-vision users, leveraging multimodal understanding to provide accurate and contextually relevant information. 
Extract all text exactly as it appears.
Do not provide summaries or interpretations unless explicitly requested by the user.
Only respond in short answers (2-3 words) and do not elaborate or use captions for the images.
"""


def format_data(sample):
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
                    "image": "val\\val\\"+sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["question"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answers"]}],
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


if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=bnb_config
        )

else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.bfloat16,
        device_map="auto"
        )

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files="Annotations\\val_text.json")
train_dataset = dataset["train"]
train_dataset = [format_data(sample) for sample in train_dataset]

with open("generated_outputs/generated_outputs[text_recognition].txt", "a", encoding='utf-8') as f:
    f.write("="*25)
    f.write("SYSTEM MESSAGE") 
    f.write("="*25)
    f.write("\n")
    f.write(f"{system_message}\n")
    f.write("="*25) 
    f.write("END")
    f.write("="*25) 
    f.write("\n")


# Serialized version of text output
for i in range(100):
    user_question, generated_text, img = text_generator(train_dataset[i])
    print(f"Writing to file...")
    with open("generated_outputs/generated_outputs[text_recognition].txt", "a", encoding='utf-8') as f:
        f.write(f"Question: {user_question}, Answer: {generated_text}, Image: {img}\n")
    print(f"Image {i} processed.")
    torch.cuda.empty_cache()
import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
import torch
torch.zeros(1).cuda()
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI
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

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data for blind and low-vision users. 
Your task is to process and extract meaningful insights from images taken by blind and low-vision users, leveraging multimodal understanding to provide accurate and contextually relevant information. 
Your responses should have two parts to them: First, if there are image quality issues, say "image quality issues:" and list them out, otherwise say "good image". Second, if the question is unanswerable, label it as unanswerable, otherwise tell the answer."""
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

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    del inputs
    # actual_answer = sample_data[2]["content"][0]["text"]
    return output_text[0], image_inputs


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

# MPI implementation for node parallelization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    dataset = load_dataset("json", data_files="Annotations\\val.json")
    train_dataset = dataset["train"]
    train_dataset = [format_data(sample) for sample in train_dataset]
    indexed_dataset = list(enumerate(train_dataset)) # Global indexing for ordering
    chunk_size = len(indexed_dataset) // size # size for each rank
    chunks = [indexed_dataset[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
    for i, item in enumerate(indexed_dataset[chunk_size*size:]): # remainder-handler, so if it divides unevenly across ranks the rest will be assigned to the first N chunks
        chunks[i].append(item)
else:
    chunks = None

# Scatter chunks to ranks
local_chunk = comm.scatter(chunks, root=0)

# Process chunk for each rank
local_results = []
for global_idx, sample in local_chunk:
    text, img = text_generator(sample)
    local_results.append((global_idx, text, img))

# Gather all results to rank 0
gathered = comm.gather(local_results, root=0)

# Rank 0 sorts by original index and writes output
if rank == 0:
    all_results = [] # flattened 1-D array version of gathered
    for sublist in gathered:       # iterate over each rank's results
        for item in sublist:       # iterate over each (idx, text, img) tuple
            all_results.append(item)
    all_results.sort(key=lambda x: x[0])
    with open("generated_outputs[image_quality_issues_fixed].txt", "w", encoding="utf-8") as f:
        for _, text, img in all_results:
            f.write(f"{text}, {img}\n")
"""
# Serialized version of code
for i in range(len(train_dataset)):
    generated_text, img = text_generator(train_dataset[i])
    print(f"Writing to file...")
    with open("generated_outputs[image_quality_issues_fixed].txt", "a", encoding='utf-8') as f:
        f.write(f"{generated_text}, {img}\n")
    print(f"Image {i} processed.")
"""
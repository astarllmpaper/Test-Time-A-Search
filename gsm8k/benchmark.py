import json
import time
import os
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

os.makedirs("gsm8k_results", exist_ok=True)

start_time = time.time()

dataset = load_dataset("gsm8k", "main", split="test")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def gpt(prompt):
    response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens = 256
    )
    return response.choices[0].message.content.strip()

for local_idx, problem in enumerate(tqdm(dataset, desc="Running GSM8K")):
    question = problem["question"] + "Let's think step by step"
    ground_truth = problem["answer"]

    try:
        llm_answer = gpt(question)
    except Exception as e:
        llm_answer = f"[ERROR] {e}"

    file_path = f"gsm8k_results/Problem_{local_idx}.txt"
    with open(file_path, 'w') as f:
        f.write(f"Problem {local_idx}\n")
        f.write(f"Question:\n{question}\n")
        f.write("-" * 50 + "\n")
        f.write(f"LLM Answer:\n{llm_answer}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Ground Truth:\n{ground_truth}\n")
        f.write("*" * 50 + "\n")
elapsed = time.time() - start_time
print(f"✅ Completed in {elapsed:.2f} seconds. Results saved in 'gsm8k_results/' directory.")

import json
from openai import OpenAI
import time
import os

if not os.path.exists('401_results'):
    os.makedirs('401_results')

start_time = time.time()

data = []
with open('math401-llm/math401.json', 'r') as file:
    for line in file:
        data.append(json.loads(line))

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def gpt(prompt):
    response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}]
    )
    x = response.choices[0].message.content
    return response.choices[0].message.content

count = 0

for problem in data:
    question = problem["query"]
    ground_truth = problem["response"]
    llm_answer = gpt(question) + "Let's think step by step"

    count += 1

    file_name1 = "401_results/Problem" + str(count) + ".txt"

    with open(file_name1, 'w') as file:
    
        file.write(f"Problem {count}\n")
        file.write(f"Question:\n{question}\n")
        file.write("-" * 50 + "\n")
        file.write(f"LLM Answer:\n{llm_answer}\n")
        file.write("-" * 50 + "\n")
        file.write(f"Ground Truth:\n{ground_truth}\n")
        file.write("*" * 50 + "\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

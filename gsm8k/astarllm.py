import json
import time
import os
import re
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def num_tokens(text):
    return len(tokenizer.encode(text))

def gpt(prompt, model="meta-llama/Meta-Llama-3-8B-Instruct", max_context=4096, max_output_tokens=1024):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_context - max_output_tokens:
        print("⚠️ Prompt too long, truncating.")
        tokens = tokens[:max_context - max_output_tokens]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=1500,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ LLM call failed:", e)
        return "LLM timed out or failed."


class Node:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.critique = self.generateCritic()
        self.h = self.selfConsistScore(num_times=2)

    def getScore(self):
        return self.h

    def getCritic(self):
        return self.critique

    def getAnswer(self):
        return self.answer

    def generateCritic(self):
        prompt = (
            f"Question:\n{self.question}\n\n"
            f"Answer:\n{self.answer}\n\n"
            f"Please provide detailed constructive criticism, yet highlight what is already correct. "
            f"Point the student in the right direction. Do not solve the problem. Grade harshly."
            f"Provide a grade (out of 100) in the format 'Grade: xx'."
        )
        return gpt(prompt)

    def parseScore(self):
        match = re.search(r'Grade:\s*(\d{1,3})', self.critique)
        return max(0, min(int(match.group(1)), 100)) if match else 50

    def selfConsistScore(self, num_times):
        total = float(self.parseScore())
        for _ in range(num_times):
            score = self.parseScore()
            total += score
        return total / (num_times + 1)

# A* LLM search
class AStarLLM:
    def __init__(self, question, max_iter):
        self.question = question
        self.max_iter = max_iter
        self.all_answers = []
        self.answer_scores = []
        self.best_answer = None
        self.best_score = 0
        self.nodesToVisit = []

    def search(self):
        root_answer = gpt(f"Let's think step-by-step: \n{self.question}")
        root = Node(self.question, root_answer)
        current_node = root
      
        self.best_answer = root
        self.best_score = root.getScore()

        for _ in range(self.max_iter):
            if current_node.getScore() < 95:
                child_prompt = (
                    f"Question:\n{self.question}\n\n"
                    f"Previous Answer:\n{current_node.getAnswer()}\n\n"
                    f"Critique:\n{current_node.getCritic()}\n\n"
                    f"Given the feedback above, please try to solve the original problem again, step by step."
                )
                child1 = Node(self.question, gpt(child_prompt))
                child2 = Node(self.question, gpt(child_prompt))
                self.nodesToVisit.extend([child1, child2])

                current_node = max(self.nodesToVisit, key=lambda n: n.getScore())
                self.nodesToVisit.remove(current_node)
                self.all_answers.append(current_node)
                self.answer_scores.append(current_node.getScore())

                if current_node.getScore() > self.best_score:
                    self.best_score = current_node.getScore()
                    self.best_answer = current_node

        return self.best_answer, self.best_score

if __name__ == '__main__':
    os.makedirs("gsm8k_8_iter_results", exist_ok=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    start_time = time.time()

    for index in range(0, 1000):  
        print(f"\n--- Problem {index + 1} ---\n")
        sample = dataset[index]
        print(sample["question"])
        question = sample["question"]
        ground_truth = sample["answer"]

        astar = AStarLLM(question, max_iter=8)
        final_answer, score = astar.search()

        print("LLM Answer:\n", final_answer.getAnswer())
        print("Critique Score:", score)
        print("*" * 50)

        
        problem_filename = f"gsm8k_8_iter_results/Problem{index + 1}.txt"
        with open(problem_filename, 'w') as f:
            f.write(f"Problem {index + 1}\n")
            f.write(f"Question:\n{question}\n")
            f.write("-" * 50 + "\n")
            f.write(f"LLM Answer:\n{final_answer.getAnswer()}\n")
            f.write("-" * 50 + "\n")
            f.write("-" * 50 + "\n")
            f.write(f"Score:\n{score}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Ground Truth:\n{ground_truth}\n")
            f.write("*" * 50 + "\n")

    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.2f} seconds. Results saved to 'gsm8k_8_iter_results/'.")

import json
import re
import asyncio
from tqdm import tqdm
import ollama
from datetime import datetime
import os
import aiohttp

class AsyncOllamaCyberMetricEvaluator:
    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = model_name
        self.client = ollama.Client()
        self.semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent executions

    def read_json_file(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except UnicodeDecodeError:
            with open(self.file_path, 'r', encoding='cp1252') as file:
                return json.load(file)

    @staticmethod
    def extract_answer(response):
        if response.strip():
            match = re.search(r"<answer>\s*ANSWER:\s*([A-D])\s*</answer>", response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).upper()
            
            match = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            match = re.search(r"\b([A-D])\)", response)
            if match:
                return match.group(1).upper()
        
        return None

    async def ask_llm(self, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = f"""
You are tasked with answering multiple choice questions. Here's how to approach this task:

First, you will be presented with a question and a set of options. 

<question>
{question}
</question>

<options>
{options}
</options>

To answer the question correctly, follow these steps:

1. Carefully read the question and all the provided options.
2. Analyze each option and consider how well it answers the question.
3. Eliminate any options that are clearly incorrect or irrelevant.
4. Compare the remaining options and determine which one best answers the question.
5. Select the letter (A, B, C, or D) corresponding to the best answer.

When you have determined the correct answer, present it in the following format:

<answer>
ANSWER: X
</answer>

Replace 'X' with the letter of the correct option (A, B, C, or D).

Important notes:
- Only provide the letter of the correct answer. Do not include any explanation or justification.
- Always use the exact format specified above, including the "ANSWER: " prefix and the letter.
- Do not include any additional text or information in your response.

Now, please analyze the question and options provided, and give your answer using the specified format."""

        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.chat,
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a security expert who answers questions."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    if response['message']['content']:
                        result = self.extract_answer(response['message']['content'])
                        if result:
                            return result
                        else:
                            print("Incorrect answer format detected. Attempting the question again.")
                except Exception as e:
                    print(f"Error: {e}. Attempting the question again.")
            return None

    def save_json_report(self, results, accuracy):
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'total_questions': len(results),
            'correct_answers': sum(1 for r in results if r['is_correct']),
            'accuracy': accuracy,
            'results': results
        }
        
        filename = f"ollama_evaluation_report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved as {filename}")

    async def process_question(self, item, progress_bar, correct_count, incorrect_answers):
        question = item['question']
        answers = item['answers']
        correct_answer = item['solution']
        llm_answer = await self.ask_llm(question, answers)
        
        result = {
            'question': question,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'is_correct': llm_answer == correct_answer
        }
        
        if result['is_correct']:
            correct_count.append(1)
        else:
            incorrect_answers.append(result)
        
        accuracy_rate = len(correct_count) / (progress_bar.n + 1) * 100
        progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
        progress_bar.update(1)
        
        return result

    async def run_evaluation(self):
        json_data = self.read_json_file()
        questions_data = json_data['questions']
        correct_count = []
        incorrect_answers = []
        all_results = []
        
        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            tasks = [self.process_question(item, progress_bar, correct_count, incorrect_answers) for item in questions_data]
            all_results = await asyncio.gather(*tasks)
        
        accuracy = len(correct_count) / len(questions_data) * 100
        print(f"Final Accuracy: {accuracy:.2f}%")
        
        self.save_json_report(all_results, accuracy)
        
        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

async def main():
    file_paths = ['questions/CyberMetric-80-v1.json']
    model_names = [
    # "gemma2:9b",
    # "phi3:medium",
    # "mistral:7b",
    # "qwen2:7b",
    # "llama3:8b",
    "llama3.1:8b",
    # "gemma2:9b",
    # "gemma2:27b",
    # "mixtral:8x7b",
    # "qwen2:72b",
    # "llama3:70b",
    "llama3.1:70b"
]

    for i in range(1, 10):
        for file_path in file_paths:
            for model_name in model_names:
                evaluator = AsyncOllamaCyberMetricEvaluator(file_path=file_path, model_name=model_name)
                await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
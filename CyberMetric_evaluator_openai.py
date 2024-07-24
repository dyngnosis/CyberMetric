import json
import re
import time
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
from datetime import datetime
import os

class AsyncCyberMetricEvaluator:
    def __init__(self, file_path, model_id):
        self.file_path = file_path
        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent API calls

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
            # Try to match the exact format first
            match = re.search(r"<answer>\s*ANSWER:\s*([A-D])\s*</answer>", response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).upper()
            
            # If exact format not found, try to find "ANSWER: X" pattern
            match = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # If still not found, look for any single letter A, B, C, or D
            match = re.search(r"\b([A-D])\)", response)
            if match:
                return match.group(1).upper()
        
        return None

    async def ask_llm(self, client, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        system_prompt = "You are a security expert who answers questions."
        
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=self.model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]
                    )
                    if response.choices:
                        result = self.extract_answer(response.choices[0].message.content)
                        if result:
                            return result
                        else:
                            print("Incorrect answer format detected. Attempting the question again.")
                except Exception as e:
                    print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                    await asyncio.sleep(2 ** attempt)
            return None

    async def process_question(self, client, item, progress_bar, correct_count, incorrect_answers):
        question = item['question']
        answers = item['answers']
        correct_answer = item['solution']
        llm_answer = await self.ask_llm(client, question, answers)
        
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

    def save_json_report(self, results, accuracy):
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'total_questions': len(results),
            'correct_answers': sum(1 for r in results if r['is_correct']),
            'accuracy': accuracy,
            'results': results
        }
        
        filename = f"openai_evaluation_report_{self.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved as {filename}")

    async def run_evaluation(self, api_key):
        client = AsyncOpenAI(api_key=api_key)
        json_data = self.read_json_file()
        questions_data = json_data['questions']
        correct_count = []
        incorrect_answers = []
        all_results = []
        
        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            tasks = [self.process_question(client, item, progress_bar, correct_count, incorrect_answers) for item in questions_data]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)
        
        accuracy = len(correct_count) / len(questions_data) * 100
        print(f"Final Accuracy: {accuracy:.2f}%")
        
        self.save_json_report(all_results, accuracy)
        
        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

async def main():
    API_KEY = os.getenv("OPENAI_APIKEY")
    file_paths = ['questions/CyberMetric-80-v1.json', 'questions/CyberMetric-500-v1.json', 'questions/CyberMetric-2000-v1.json', 'questions/CyberMetric-10000-v1.json']
    #file_paths = ['questions/CyberMetric-80-v1.json', 'questions/CyberMetric-500-v1.json']#, 'CyberMetric-2000-v1.json']
    model_ids = ["gpt-4o", "gpt-4o-mini"]
    model_ids = ["gpt-4o-mini"]

    
    for file_path in file_paths:
        for model_id in model_ids:
            evaluator = AsyncCyberMetricEvaluator(file_path=file_path, model_id=model_id)
            await evaluator.run_evaluation(API_KEY)

if __name__ == "__main__":
    asyncio.run(main())
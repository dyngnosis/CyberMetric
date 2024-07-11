import json
import re
import asyncio
from tqdm import tqdm
from anthropic import AsyncAnthropic
from datetime import datetime
import os

class AsyncCyberMetricEvaluator:
    def __init__(self, file_path, model_id):
        self.file_path = file_path
        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls

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
            # Check for SKIP response
            # if 'SKIP' in response.upper():
            #     print("SKIP detected. Skipping this question.")
            #     return {'answer': 'SKIP', 'final_assessment': 'SKIP', 'detailed_evaluation': {}}
            
            # Extract the final assessment
            print(response)
            final_assessment_match = re.search(r"<final_assessment>\s*(GOOD|NEEDS_IMPROVEMENT|POOR)\s*</final_assessment>", response, re.IGNORECASE)
            if final_assessment_match:
                final_assessment = final_assessment_match.group(1).upper()
            else:
                final_assessment = "UNKNOWN"
            
            # Try to match the exact format first for answer
            answer_match = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).upper()
            else:
                answer = None
            
            # Extract detailed evaluation
            detailed_evaluation = {
                "clarity": re.search(r"Clarity:\s*(.*)", response, re.IGNORECASE),
                "grammar": re.search(r"Grammar:\s*(.*)", response, re.IGNORECASE),
                "sense": re.search(r"Sense:\s*(.*)", response, re.IGNORECASE),
                "answer_quality": re.search(r"Answer quality:\s*(.*)", response, re.IGNORECASE),
                "correctness": re.search(r"Correctness:\s*(.*)", response, re.IGNORECASE),
            }
            for key, match in detailed_evaluation.items():
                if match:
                    detailed_evaluation[key] = match.group(1).strip()
                else:
                    detailed_evaluation[key] = "Not Evaluated"
            
            return {
                "answer": answer,
                "final_assessment": final_assessment,
                "detailed_evaluation": detailed_evaluation
            }
        
        return None



    async def ask_llm(self, client, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = fprompt = f"""
You are tasked with evaluating the quality of synthetically generated multiple-choice questions. Some of these questions may be ambiguous, have poor grammar, unclear answers, or may not make sense at all. Your job is to carefully analyze the question and its options, then flag any issues you find.

Here is the question to evaluate:

<question>
{question}
</question>

And here are the provided options:

<options>
{options}
</options>

Please evaluate the question and options based on the following criteria:
1. Clarity: Is the question clear and unambiguous?
2. Grammar: Is the grammar correct in both the question and options?
3. Sense: Does the question make logical sense?
4. Answer quality: Are the options clear and do they relate logically to the question?
5. Correctness: Is there a clear, unambiguous correct answer?

In your evaluation, provide your reasoning for each criterion. Then, flag any issues you've identified using the following tags: <flag_grammar></flag_grammar> <flag_answer_quality></flag_answer_quality>.
After your detailed evaluation, provide a final assessment of the question's overall quality using one of the following ratings inside <final_assessment> tags: GOOD, NEEDS_IMPROVEMENT, POOR.
Finally, provide your answer choice (A, B, C, or D) in the format "ANSWER: X", where X is the letter corresponding to the best answer.
Here are two examples of how your response should be structured:
Example 1 (Good Question):
Clarity: The question is clear and unambiguous. It asks about a specific cybersecurity concept.
Grammar: The grammar is correct in both the question and options.
Sense: The question makes logical sense within the context of cybersecurity.
Answer quality: The options are clear and directly related to the question.
Correctness: There is a clear, unambiguous correct answer (option B in this case).
<final_assessment>GOOD</final_assessment>
ANSWER: B
Example 2 (Poor Question):
Clarity: The question is ambiguous and poorly worded. <flag_answer_quality>It's unclear what specific aspect of network security is being asked about.</flag_answer_quality>
Grammar: <flag_grammar>There are grammatical errors in the question, such as missing articles and incorrect verb tense.</flag_grammar>
Sense: The question doesn't make complete logical sense due to its vague nature.
Answer quality: <flag_answer_quality>The options are not clearly related to the question and some are too similar, making it difficult to choose a definitive answer.</flag_answer_quality>
Correctness: Due to the issues mentioned, there is no clear, unambiguous correct answer.
<final_assessment>POOR</final_assessment>
Begin your evaluation now.
"""

        system_prompt = "You are a security expert who answers questions."
        
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await client.messages.create(
                        model=self.model_id,
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": prompt},
                        ]
                    )
                    if response.content:
                        result = self.extract_answer(response.content[0].text)
                        if result:
                            print(result)
                            return result
                        else:
                            print("Incorrect answer format detected. Attempting the question again.")
                except Exception as e:
                    print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                    await asyncio.sleep(2 ** attempt)
            return None


    def save_json_report(self, results, accuracy):
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'total_questions': len(results),
            'correct_answers': sum(1 for r in results if r['is_correct']),
            'accuracy': accuracy,
            'results': results
        }
        
        filename = f"BADQ_anthropic_evaluation_report_{self.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved as {filename}")

    async def process_question(self, client, item, progress_bar, correct_count, incorrect_answers):
        question = item['question']
        answers = item['answers']
        correct_answer = item['solution']
        llm_response = await self.ask_llm(client, question, answers)
        
        if llm_response:
            llm_answer = llm_response['answer']
            final_assessment = llm_response['final_assessment']
            detailed_evaluation = llm_response['detailed_evaluation']
        else:
            llm_answer = None
            final_assessment = "UNKNOWN"
            detailed_evaluation = {}

        result = {
            'question': question,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'is_correct': llm_answer == correct_answer,
            'final_assessment': final_assessment,
            'detailed_evaluation': detailed_evaluation
        }
        
        if result['is_correct']:
            correct_count.append(1)
        else:
            incorrect_answers.append(result)
        
        accuracy_rate = len(correct_count) / (progress_bar.n + 1) * 100
        progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
        progress_bar.update(1)
        
        return result


    async def run_evaluation(self, api_key):
        client = AsyncAnthropic(api_key=api_key)
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
    API_KEY = os.getenv("ANTHROPIC_APIKEY")
    #file_paths = ['CyberMetric-80-v1.json', 'CyberMetric-500-v1.json', 'CyberMetric-2000-v1.json', 'CyberMetric-10000-v1.json']
    file_paths = ['CyberMetric-80-v1.json', 'CyberMetric-500-v1.json']
    model_ids = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]
    
    for file_path in file_paths:
        for model_id in model_ids:
            evaluator = AsyncCyberMetricEvaluator(file_path=file_path, model_id=model_id)
            await evaluator.run_evaluation(API_KEY)

if __name__ == "__main__":
    asyncio.run(main())
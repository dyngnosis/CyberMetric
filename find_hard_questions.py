import json
import os
from collections import defaultdict
import argparse
import sys

def read_json_file(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}", file=sys.stderr)
            return None
    print(f"Unable to read file {file_path} with any of the attempted encodings.", file=sys.stderr)
    return None

def analyze_reports(directory):
    incorrect_answers = defaultdict(lambda: defaultdict(int))
    model_counts = defaultdict(int)

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            data = read_json_file(file_path)
            if data is None:
                continue

            model_id = data.get('model_id')
            if not model_id:
                print(f"Warning: 'model_id' not found in file {filename}", file=sys.stderr)
                continue

            model_counts[model_id] += 1

            results = data.get('results', [])
            for result in results:
                if not result.get('is_correct', True):
                    question = result.get('question')
                    if question:
                        incorrect_answers[question][model_id] += 1
                    else:
                        print(f"Warning: 'question' not found in a result in file {filename}", file=sys.stderr)

    return incorrect_answers, model_counts

def calculate_percentages(incorrect_answers, model_counts):
    percentages = {}
    for question, models in incorrect_answers.items():
        percentages[question] = {
            model: (count / model_counts[model]) * 100
            for model, count in models.items()
        }
    return percentages

def sort_questions(percentages):
    return sorted(
        percentages.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

def print_results(sorted_questions, model_counts):
    safe_print("Questions most frequently answered incorrectly across all models:")
    safe_print("=" * 80)
    for question, model_percentages in sorted_questions:
        safe_print(f"\nQuestion: {question}")
        safe_print("Model percentages:")
        for model, percentage in model_percentages.items():
            safe_print(f"  {model}: {percentage:.2f}%")
    
    safe_print("\n" + "=" * 80)
    safe_print("Total reports analyzed per model:")
    for model, count in model_counts.items():
        safe_print(f"  {model}: {count}")

def main(directory):
    if not os.path.isdir(directory):
        print(f"Error: The specified path '{directory}' is not a valid directory.", file=sys.stderr)
        return

    incorrect_answers, model_counts = analyze_reports(directory)
    if not incorrect_answers:
        print(f"No valid JSON reports found in the directory: {directory}", file=sys.stderr)
        return

    percentages = calculate_percentages(incorrect_answers, model_counts)
    sorted_questions = sort_questions(percentages)
    print_results(sorted_questions, model_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze incorrect answers from JSON reports.")
    parser.add_argument("directory", help="Path to the directory containing JSON reports")
    args = parser.parse_args()

    main(args.directory)
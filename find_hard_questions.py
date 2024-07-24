import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
import re
import html  # Import the html module for escaping

def load_questions_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
    return questions_data

def process_evaluation_reports(input_directory, questions_file):
    questions_data = load_questions_json(questions_file)
    
    # Create a dictionary to store question details
    question_details = {q['question']: q for q in questions_data['questions']}
    
    question_stats = defaultdict(lambda: {
        "total_attempts": defaultdict(int),
        "incorrect_attempts": defaultdict(int),
        "null_answers": defaultdict(int),
        "correct_answer": "",
        "selected_answers": defaultdict(lambda: defaultdict(int)),
        "answers_text": {}  # Add a field to store answer option text
    })
    model_stats = defaultdict(lambda: {"total_attempts": 0, "incorrect_attempts": 0})
    all_models = set()

    for filename in os.listdir(input_directory):
        if re.match(r'(ollama|openai|mistral|anthropic)_evaluation_report_.*\.json', filename):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r') as file:
                report = json.load(file)
                model_name = report.get("model_id") or report.get("model_name")
                all_models.add(model_name)
                
                for result in report["results"]:
                    question = result["question"]
                    is_correct = result["is_correct"]
                    llm_answer = result["llm_answer"]
                    correct_answer = result["correct_answer"]

                    # Store answer option text
                    question_details_obj = question_details.get(question)
                    if question_details_obj:
                        question_stats[question]["answers_text"] = question_details_obj["answers"]

                    question_stats[question]["total_attempts"][model_name] += 1
                    model_stats[model_name]["total_attempts"] += 1
                    question_stats[question]["correct_answer"] = correct_answer
                    
                    if not is_correct:
                        question_stats[question]["incorrect_attempts"][model_name] += 1
                        model_stats[model_name]["incorrect_attempts"] += 1
                    
                    if llm_answer is None:
                        question_stats[question]["null_answers"][model_name] += 1
                    else:
                        question_stats[question]["selected_answers"][model_name][llm_answer] += 1

    return question_stats, model_stats, list(all_models)

def rank_hard_questions(question_stats, all_models):
    ranked_questions = []
    for question, stats in question_stats.items():
        total_incorrect = sum(stats["incorrect_attempts"].values())
        total_attempts = sum(stats["total_attempts"].values())
        overall_failure_rate = (total_incorrect / total_attempts) * 100 if total_attempts > 0 else 0
        models_failed_count = sum(1 for model in all_models if stats["incorrect_attempts"].get(model, 0) > 0)
        
        ranked_questions.append({
            "question": question,
            "overall_failure_rate": overall_failure_rate,
            "total_attempts": stats["total_attempts"],
            "incorrect_attempts": stats["incorrect_attempts"],
            "null_answers": stats["null_answers"],
            "models_failed_count": models_failed_count,
            "correct_answer": stats["correct_answer"],
            "selected_answers": stats["selected_answers"],
            "answers_text": stats["answers_text"]  # Include answers text in the output
        })

    ranked_questions.sort(key=lambda x: (x["models_failed_count"], x["overall_failure_rate"]), reverse=True)
    return ranked_questions

def sort_models_by_accuracy(model_stats):
    model_accuracy = {}
    for model, stats in model_stats.items():
        total = stats["total_attempts"]
        incorrect = stats["incorrect_attempts"]
        accuracy = (1 - incorrect / total) * 100 if total > 0 else 0
        model_accuracy[model] = accuracy
        print(f"Debug - Model: {model}, Accuracy: {accuracy:.2f}%")  # Debug print

    sorted_models = sorted(model_accuracy.items(), key=lambda x: x[1], reverse=True)
    print("Debug - Sorted models:")  # Debug print
    for model, accuracy in sorted_models:
        print(f"  {model}: {accuracy:.2f}%")  # Debug print

    return sorted_models

def create_interactive_heatmap(top_hard_questions, all_models, output_file):
    sorted_models = [model for model, _ in sort_models_by_accuracy(model_stats)]
    
    data = []
    hover_texts = []
    for q in reversed(top_hard_questions):
        row = []
        row_hover_texts = []
        for model in sorted_models:
            total = q["total_attempts"].get(model, 0)
            incorrect = q["incorrect_attempts"].get(model, 0)
            percentage = (incorrect / total) * 100 if total > 0 else 0
            row.append(percentage)
            
            selected_answers = q["selected_answers"].get(model, {})
            selected_answers_text = "<br>".join(
                [f"{html.escape(ans)} ({html.escape(q['answers_text'].get(ans, ''))}): {count}" for ans, count in selected_answers.items()]
            )  # Escape selected answers and include answer text
            
            correct_answer_text = f"{html.escape(q['correct_answer'])} ({html.escape(q['answers_text'].get(q['correct_answer'], ''))})"
            
            hover_text = (
                f"<b>Question:</b> {html.escape(q['question'])}<br>"  # Escape question
                f"<b>Correct Answer:</b> {correct_answer_text}<br>"  # Include and escape correct answer text
                f"<b>Selected Answers:</b><br>{selected_answers_text}"
            )
            row_hover_texts.append(hover_text)
        
        data.append(row)
        hover_texts.append(row_hover_texts)

    df = pd.DataFrame(data, columns=sorted_models, index=[f"Q{i+1}" for i in range(len(top_hard_questions))])

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        hoverongaps=False,
        hovertemplate='<b>Model:</b> %{x}<br>' +
                      '<b>Incorrect Answers:</b> %{z:.1f}%<br>' +
                      '%{customdata}<extra></extra>',
        customdata=hover_texts,
        colorscale='YlOrRd',
        zmin=0,
        zmax=100
    ))

    fig.update_layout(
        title={
            'text': 'Hardest Questions Across Models (Percentage of Incorrect Answers)',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=36)  # Keep the increased title font size
        },
        yaxis_title='Questions',
        height=max(1000, len(top_hard_questions) * 20),
        width=max(1200, len(sorted_models) * 50),
    )

    fig.update_layout(
        xaxis=dict(
            side="top",
            tickangle=45,
            tickfont=dict(size=10),  # Reduce font size of top labels
        )
    )

    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="bottom",
            matches="x",
            ticktext=sorted_models,
            tickvals=list(range(len(sorted_models))),
            tickmode="array",
            ticks="",
            showticklabels=True,
            tickangle=45,
            tickfont=dict(size=10),  # Reduce font size of bottom labels
            anchor="y",
            showgrid=False
        )
    )

    fig.update_layout(
        margin=dict(t=220, b=150, l=100, r=50),  # Increase top margin further
        autosize=False
    )

    fig.update_traces(xgap=1, ygap=1)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='black', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=1)
    )

    fig.write_html(output_file, full_html=True, include_plotlyjs='cdn')
    print(f"Interactive heatmap visualization saved as '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Find hard questions from LLM evaluation reports.")
    parser.add_argument("input_directory", help="Path to the directory containing evaluation reports")
    parser.add_argument("-o", "--output", default="hard_questions.json", help="Output JSON file name (default: hard_questions.json)")
    parser.add_argument("-n", "--num_questions", type=int, default=100, help="Number of top hard questions to output (default: 500)")
    parser.add_argument("-v", "--visual", default="hard_questions_heatmap.html", help="Output visual file name (default: hard_questions_heatmap.html)")
    parser.add_argument("-q", "--questions_file", default="questions/CyberMetric-10000-v1.json", help="Path to questions JSON file")
    args = parser.parse_args()

    global question_stats, model_stats, all_models
    question_stats, model_stats, all_models = process_evaluation_reports(args.input_directory, args.questions_file)
    ranked_questions = rank_hard_questions(question_stats, all_models)

    top_hard_questions = ranked_questions[:args.num_questions]

    # Save top hard questions to JSON file
    with open(args.output, "w") as outfile:
        json.dump(top_hard_questions, outfile, indent=2)

    print(f"Top {args.num_questions} hard questions have been identified and saved as '{args.output}'")

    # Create and save interactive heatmap visualization
    create_interactive_heatmap(top_hard_questions, all_models, args.visual)

    # Print summary to console (limiting to top 10 for brevity)
    print("\nTop 10 Hardest Questions:")
    for i, q in enumerate(top_hard_questions[:10], 1):
        print(f"{i}. Overall Failure Rate: {q['overall_failure_rate']:.2f}%, Models Failed: {q['models_failed_count']}")
        print(f"   Question: {q['question']}")
        print(f"   Correct Answer: {q['correct_answer']}")
        print("   Failure Rates by Model:")
        for model in all_models:
            total = q["total_attempts"].get(model, 0)
            incorrect = q["incorrect_attempts"].get(model, 0)
            percentage = (incorrect / total) * 100 if total > 0 else 0
            print(f"     - {model}: {percentage:.2f}% ({incorrect}/{total})")
            selected_answers = q["selected_answers"].get(model, {})
            if selected_answers:
                print("       Selected Answers:")
                for ans, count in selected_answers.items():
                    print(f"         {ans}: {count}")
        print()

if __name__ == "__main__":
    main()

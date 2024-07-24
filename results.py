import os
import json
import argparse
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Define the companies and their respective model IDs
companies_models = {
    "Mistral AI": ["open-mixtral-8x7b", "open-mixtral-8x22b", "mistral-large-2402", "mistral:7b", "mixtral:8x7b"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],
    "Alibaba": ["qwen2:7b", "qwen2:72b"],
    "Meta": ["llama3:8b", "llama3:70b", "llama3.1:8b", "llama3.1:70b"],
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "Microsoft": ["phi3:medium"],
    "Google": ["gemma2:27b", "gemma2:9b"]
}

hard_coded_sizes = {
    "open-mixtral-8x7b": "47B", # 47B total with 13B active: https://mistral.ai/news/mixtral-8x22b/
    "gpt-40": "> 175B", # https://aimlapi.com/comparisons/claude-sonnet-3-5-vs-chatgpt-4o
    "phi3:medium": "14B", # https://ollama.com/library/phi3:14b
    "mixtral:8x7b": "47B",
    "claude-3-5-sonnet-20240620": "N/A",
    "open-mixtral-8x22b": "141B", # 141B total with 39B active https://mistral.ai/news/mixtral-8x22b/
    "mixtral:8x22b": "141B", # 141B total with 39B active https://mistral.ai/news/mixtral-8x22b/
}

# Initialize an empty dictionary to dynamically build the leaderboard
leaderboard = {}

# Function to get company name from model ID
def get_company_from_model_id(model_id):
    for company, models in companies_models.items():
        if model_id in models:
            return company
    return "Unknown"

# Function to extract size from model ID
def get_size_from_model_id(model_id):
    # Check the hard coded sizes dictionary first
    if model_id in hard_coded_sizes:
        return hard_coded_sizes[model_id]
    
    # If not found, try to extract size from model ID
    match = re.search(r':(\d+[bB])$', model_id)
    if match:
        return match.group(1).upper()
    
    return "N/A"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process JSON files and generate leaderboard.')
parser.add_argument('input_dir', type=str, help='Directory containing JSON files')
args = parser.parse_args()

# Define the pattern to match JSON files
file_pattern = os.path.join(args.input_dir, "*.json")

# Iterate over each file in the directory
for file_path in glob(file_pattern):
    with open(file_path, 'r') as file:
        data = json.load(file)
        model_id = data.get("model_id") or data.get("model_name")  # Support both model_id and model name
        accuracy = round(data["accuracy"], 2)  # Limit accuracy to two decimal places
        total_questions = data["total_questions"]
        
        if model_id not in leaderboard:
            # Add model entry to leaderboard if not present
            leaderboard[model_id] = {
                "Company": get_company_from_model_id(model_id),
                "Size": get_size_from_model_id(model_id),
                "License": "Unknown",  # Replace with actual data if available
                "Acc 80 Q": [0, 0],    # [total accuracy, number of files]
                "Acc 500 Q": [0, 0],
                "Acc 2k Q": [0, 0],
                "Acc 10k Q": [0, 0]
            }
        
        # Update accuracy and file count
        if total_questions == 80:
            leaderboard[model_id]["Acc 80 Q"][0] += accuracy
            leaderboard[model_id]["Acc 80 Q"][1] += 1
        elif total_questions == 500:
            leaderboard[model_id]["Acc 500 Q"][0] += accuracy
            leaderboard[model_id]["Acc 500 Q"][1] += 1
        elif total_questions == 2000:
            leaderboard[model_id]["Acc 2k Q"][0] += accuracy
            leaderboard[model_id]["Acc 2k Q"][1] += 1
        elif total_questions == 10211:
            leaderboard[model_id]["Acc 10k Q"][0] += accuracy
            leaderboard[model_id]["Acc 10k Q"][1] += 1

# Calculate average accuracy and format the results
for model_id, stats in leaderboard.items():
    for key in ["Acc 80 Q", "Acc 500 Q", "Acc 2k Q", "Acc 10k Q"]:
        total_accuracy, file_count = stats[key]
        if file_count > 0:
            average_accuracy = round(total_accuracy / file_count, 2)
            leaderboard[model_id][key] = f"{average_accuracy} ({file_count})"
        else:
            leaderboard[model_id][key] = None

# Define the output file path
output_file = "leaderboard.json"

# Save the dynamically built leaderboard to the output file
with open(output_file, 'w') as outfile:
    json.dump(leaderboard, outfile, indent=4)

# Convert leaderboard dictionary to a pandas DataFrame for better visualization
leaderboard_df = pd.DataFrame(leaderboard).transpose()

# Order the table by highest average accuracy in the Q500 column
leaderboard_df.sort_values(by="Acc 2k Q", ascending=False, inplace=True)

# Define the color map for the background based on value ranges
def color_map(val):
    try:
        score = float(val.split()[0])
        if score >= 90:
            return '#90EE90'  # light green
        elif score >= 80:
            return '#FFFFE0'  # light yellow
        elif score >= 70:
            return '#FFD700'  # yellow
        elif score >= 60:
            return '#FF8C00'  # dark orange
        elif score >= 50:
            return '#FF6347'  # tomato
        else:
            return '#FF4500'  # orange red
    except:
        return 'w'

# Save the styled table as an image
def render_mpl_table(data, col_width=1.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([col_width*0.8, row_height])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    # Rename the index column to "Model Name"
    data.index.name = "Model Name"

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell_text = cell.get_text().get_text()
            cell.set_facecolor(color_map(cell_text))
        
        # Make the first column bold
        if k[1] == -1:  # This is the row label (index) column
            cell.set_text_props(weight='bold')
        
        # Adjust the column widths
        if k[1] == -1:  # This is the row label (index) column
            cell.set_width(2 * col_width / (data.shape[1] + 1))
        else:
            cell.set_width(0.5 * col_width / (data.shape[1] + 1))

    plt.savefig("styled_table_new_format.png", bbox_inches='tight')
    plt.show()

# Render and save the adjusted table
render_mpl_table(leaderboard_df)
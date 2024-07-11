import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new data format
new_data = {
    "Mistral large(api)": {
        "Company": "Mistral AI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 96.25,
        "Acc 500 Q": 91.0,
        "Acc 2k Q": 91.25,
        "Acc 10k Q": 85.18
    },
    "Mixtral 8x22b (api)": {
        "Company": "Mistral AI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 96.25,
        "Acc 500 Q": 88.2,
        "Acc 2k Q": 91.25,
        "Acc 10k Q": 85.18
    },    
    "Mixtral 8x7b (api)": {
        "Company": "Mistral AI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 91.25,
        "Acc 500 Q": 87.0,
        "Acc 2k Q": 84.6,
        "Acc 10k Q": 85.18
    },
    "GPT-4o": {
        "Company": "OpenAI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 96.25,
        "Acc 500 Q": 93.4,
        "Acc 2k Q": 91.25,
        "Acc 10k Q": 88.89
    },
    "Claude 3.5 Sonnet": {
        "Company": "Anthropic",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 93.75,
        "Acc 500 Q": 94.39,
        "Acc 2k Q": 91.14,
        "Acc 10k Q": 88.02
    },
    "GPT-4o": {
        "Company": "OpenAI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 96.25,
        "Acc 500 Q": 93.4,
        "Acc 2k Q": 91.25,
        "Acc 10k Q": 88.89
    },    
    "Mixtral-8x7B-Instruct": {
        "Company": "Mistral AI",
        "Size": "45B",
        "License": "Apache 2.0",
        "Acc 80 Q": 92.5,
        "Acc 500 Q": 91.8,
        "Acc 2k Q": 91.1,
        "Acc 10k Q": 87.0
    },
    "GPT-4-turbo": {
        "Company": "OpenAI",
        "Size": "N/A",
        "License": "Proprietary",
        "Acc 80 Q": 96.25,
        "Acc 500 Q": 93.3,
        "Acc 2k Q": 91.0,
        "Acc 10k Q": 88.5
    },
    "Falcon-180B-Chat": {
        "Company": "TII",
        "Size": "180B",
        "License": "Apache 2.0",
        "Acc 80 Q": 90.0,
        "Acc 500 Q": 87.8,
        "Acc 2k Q": 87.1,
        "Acc 10k Q": 87.0
    },
    "GPT-3.5-turbo": {
        "Company": "OpenAI",
        "Size": "175B",
        "License": "Proprietary",
        "Acc 80 Q": 90.0,
        "Acc 500 Q": 87.3,
        "Acc 2k Q": 88.1,
        "Acc 10k Q": 80.3
    },
    "GEMINI-pro 1.0": {
        "Company": "Google",
        "Size": "137B",
        "License": "Proprietary",
        "Acc 80 Q": 90.0,
        "Acc 500 Q": 85.05,
        "Acc 2k Q": 84.0,
        "Acc 10k Q": 87.5
    },
    "Mistral-7B-Instruct-v0.2": {
        "Company": "Mistral AI",
        "Size": "7B",
        "License": "Apache 2.0",
        "Acc 80 Q": 78.75,
        "Acc 500 Q": 78.4,
        "Acc 2k Q": 76.4,
        "Acc 10k Q": 74.82
    },
    "Gemma-1.1-7b-it": {
        "Company": "Google",
        "Size": "7B",
        "License": "Open",
        "Acc 80 Q": 82.5,
        "Acc 500 Q": 75.4,
        "Acc 2k Q": 75.75,
        "Acc 10k Q": 73.32
    },
    "Meta-Llama-3-8B-Instruct": {
        "Company": "Meta",
        "Size": "8B",
        "License": "Open",
        "Acc 80 Q": 81.25,
        "Acc 500 Q": 76.2,
        "Acc 2k Q": 73.05,
        "Acc 10k Q": 71.25
    },
    "Flan-T5-XXL": {
        "Company": "Google",
        "Size": "11B",
        "License": "Apache 2.0",
        "Acc 80 Q": 81.94,
        "Acc 500 Q": 71.1,
        "Acc 2k Q": 69.0,
        "Acc 10k Q": 67.5
    },
    "Llama 2-70B": {
        "Company": "Meta",
        "Size": "70B",
        "License": "Apache 2.0",
        "Acc 80 Q": 75.0,
        "Acc 500 Q": 73.4,
        "Acc 2k Q": 71.6,
        "Acc 10k Q": 66.1
    },
    "Zephyr-7B-beta": {
        "Company": "HuggingFace",
        "Size": "7B",
        "License": "MIT",
        "Acc 80 Q": 80.94,
        "Acc 500 Q": 76.4,
        "Acc 2k Q": 72.5,
        "Acc 10k Q": 65.0
    },
    "Qwen1.5-MoE-A2.7B": {
        "Company": "Qwen",
        "Size": "2.7B",
        "License": "Open",
        "Acc 80 Q": 62.5,
        "Acc 500 Q": 64.6,
        "Acc 2k Q": 61.65,
        "Acc 10k Q": 60.73
    },
    "Qwen1.5-7B": {
        "Company": "Qwen",
        "Size": "7B",
        "License": "Open",
        "Acc 80 Q": 73.75,
        "Acc 500 Q": 60.6,
        "Acc 2k Q": 61.35,
        "Acc 10k Q": 59.79
    },
    "Qwen-7B": {
        "Company": "Qwen",
        "Size": "7B",
        "License": "Open",
        "Acc 80 Q": 43.75,
        "Acc 500 Q": 58.0,
        "Acc 2k Q": 55.75,
        "Acc 10k Q": 54.09
    },
    "Phi-2": {
        "Company": "Microsoft",
        "Size": "2.7B",
        "License": "MIT",
        "Acc 80 Q": 53.75,
        "Acc 500 Q": 48.0,
        "Acc 2k Q": 52.9,
        "Acc 10k Q": 52.13
    },
    "Llama3-ChatQA-1.5-8B": {
        "Company": "Nvidia",
        "Size": "8B",
        "License": "Open",
        "Acc 80 Q": 53.75,
        "Acc 500 Q": 52.8,
        "Acc 2k Q": 49.45,
        "Acc 10k Q": 49.64
    },
    "DeciLM-7B": {
        "Company": "Deci",
        "Size": "7B",
        "License": "Apache 2.0",
        "Acc 80 Q": 52.5,
        "Acc 500 Q": 47.2,
        "Acc 2k Q": 50.44,
        "Acc 10k Q": 50.75
    },
    "Qwen1.5-4B": {
        "Company": "Qwen",
        "Size": "4B",
        "License": "Open",
        "Acc 80 Q": 36.25,
        "Acc 500 Q": 41.2,
        "Acc 2k Q": 40.5,
        "Acc 10k Q": 40.29
    },
    "Genstruct-7B": {
        "Company": "NousResearch",
        "Size": "7B",
        "License": "Apache 2.0",
        "Acc 80 Q": 38.75,
        "Acc 500 Q": 40.6,
        "Acc 2k Q": 37.55,
        "Acc 10k Q": 36.93
    },
    "Meta-Llama-3-8B": {
        "Company": "Meta",
        "Size": "8B",
        "License": "Open",
        "Acc 80 Q": 38.75,
        "Acc 500 Q": 35.8,
        "Acc 2k Q": 37.0,
        "Acc 10k Q": 36.0
    },
    "Gemma-7b": {
        "Company": "Google",
        "Size": "7B",
        "License": "Open",
        "Acc 80 Q": 42.5,
        "Acc 500 Q": 37.2,
        "Acc 2k Q": 36.0,
        "Acc 10k Q": 34.28
    },
    "Dolly V2 12b BF16": {
        "Company": "Databricks",
        "Size": "12B",
        "License": "MIT",
        "Acc 80 Q": 33.75,
        "Acc 500 Q": 30.0,
        "Acc 2k Q": 28.75,
        "Acc 10k Q": 27.0
    },
    "Gemma-2b": {
        "Company": "Google",
        "Size": "2B",
        "License": "Open",
        "Acc 80 Q": 25.0,
        "Acc 500 Q": 23.2,
        "Acc 2k Q": 18.2,
        "Acc 10k Q": 19.18
    },
    "Phi-3-mini-4k-instruct": {
        "Company": "Microsoft",
        "Size": "3.8B",
        "License": "MIT",
        "Acc 80 Q": 5.0,
        "Acc 500 Q": 5.0,
        "Acc 2k Q": 4.41,
        "Acc 10k Q": 4.8
    }
}

# Convert the new data format into a DataFramedf_new = pd.DataFrame.from_dict(new_data, orient='index')
# Convert the new data format into a DataFrame
df_new = pd.DataFrame.from_dict(new_data, orient='index')

# Define the color map for the background based on value ranges
def color_map(val):
    if val >= 90:
        return '#90EE90'  # light green
    elif val >= 80:
        return '#FFFFE0'  # light yellow
    elif val >= 70:
        return '#FFD700'  # yellow
    elif val >= 60:
        return '#FF8C00'  # dark orange
    elif val >= 50:
        return '#FF6347'  # tomato
    else:
        return '#FF4500'  # orange red

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
            try:
                cell_value = float(cell_text)
                cell_color = color_map(cell_value)
                cell.set_facecolor(cell_color)
            except ValueError:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        
        # Make the first column bold
        if k[1] == -1:  # This is the row label (index) column
            cell.set_text_props(weight='bold')
        
        # Adjust the column widths
        if k[1] == -1:  # This is the row label (index) column
            cell.set_width(2 * col_width / (data.shape[1] + 1))
        else:
            cell.set_width(col_width / (data.shape[1] + 1))



    plt.savefig("styled_table_new_format.png", bbox_inches='tight')
    plt.show()

# Render and save the adjusted table
render_mpl_table(df_new)
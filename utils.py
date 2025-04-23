import random
from datasets import Dataset, load_dataset
import numpy as np


## helper functions ##
def _get_scenarios(example):
    data = example['sentences']
    # gold_label is an array with 0,1,2 in which 0 correspond to anti_stereotype and 1 correspond to stereotype
    anti_stereotype_idx = data['gold_label'].index(0)
    stereotype_idx = data['gold_label'].index(1)
    scenario = ""
    if 'BLANK' not in example['context']:
        scenario += example['context'] + " "
    return (
        scenario + data['sentence'][anti_stereotype_idx], 
        scenario + data['sentence'][stereotype_idx], 
    )


def bias_dataset(ntrain=150, user_tag="", assistant_tag="", seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the bias of the following scenario.\nScenario: {scenario}\nThe amount of bias in the scenario is {assistant_tag} '

    dataset = load_dataset('stereoset', 'intrasentence')['validation'].shuffle(seed=1)
    test_dataset = load_dataset('stereoset', 'intersentence')['validation'].shuffle(seed=1)


    dataset = dataset.filter(lambda x: x['bias_type'] == "race")
    test_dataset = test_dataset.filter(lambda x: x['bias_type'] == "race")


    paired_train_data = []
    m = min([ntrain, len(dataset)])
    for d in dataset.select(range(m)):
        anti_steotype, stereotype = _get_scenarios(d)

        ## We get anti_steotype as a direction to control fairness/bias
        paired_train_data.append([anti_steotype, stereotype])

    train_labels = []
    for d in paired_train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    # test_dataset = dataset.select(range(ntrain, len(dataset)))
    test_dataset = dataset.select(range(len(dataset)))
    test_data = []
    for d in test_dataset:
        anti_steotype, stereotype = _get_scenarios(d)
        current_group = [anti_steotype, stereotype]
        test_data.extend(current_group)    

    train_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in np.concatenate(paired_train_data)]
    test_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in test_data]

    return {
            'train': {'data': train_data, 'labels': train_labels},
            'test': {'data': test_data, 'labels': [[1,0]* len(test_data)]}
        }

    
import json
import random
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

def literary_openings_dataset(data_dir, ntrain=16, seed=0):
    random.seed(seed)

    with open(os.path.join(data_dir, "literary_openings/real.json")) as file:
        seen_docs = json.load(file)

    with open(os.path.join(data_dir, "literary_openings/fake.json")) as file:
        unseen_docs = json.load(file)

    data = [[s.replace("...", ""),u.replace("...", "")] for s,u in zip(seen_docs, unseen_docs)]
    train_data =  data[:ntrain]
    test_data = data

    docs_train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        docs_train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    docs_train_labels = docs_train_labels

    template_str = "{s} "

    docs_train_data = [template_str.format(s=s) for s in train_data]
    docs_test_data = [template_str.format(s=s) for s in test_data]
    return docs_train_data, docs_train_labels, docs_test_data

def quotes_dataset(data_dir, ntrain=16, seed=0):
    random.seed(0)

    with open(os.path.join(data_dir, "quotes/popular_quotes.json")) as file:
        seen_quotes = json.load(file)

    with open(os.path.join(data_dir, "quotes/unseen_quotes.json")) as file:
        unseen_quotes = json.load(file)

    data = [[s,u] for s,u in zip(seen_quotes, unseen_quotes)]
    train_data =  data[:ntrain]
    test_data = data

    quote_train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        quote_train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    quote_train_labels = quote_train_labels

    template_str = "{s} "

    quote_train_data = [template_str.format(s=s) for s in train_data]
    quote_test_data = [template_str.format(s=s) for s in test_data]
    return quote_train_data, quote_train_labels, quote_test_data





import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag}{template_str.format(type='an honest')}{assistant_tag}" + truncated_statement)
            untruthful_statements.append(f"{user_tag}{template_str.format(type='an untruthful')}{assistant_tag}" + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }




def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":", wb_char='▁'):

    cmap=LinearSegmentedColormap.from_list('rg',["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace(wb_char, ' ') for token in input_ids]
    # print("words:", words)
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6.4, 2.5), dpi=200)

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8.5

    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.7
    word_width = 0

    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[0:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False
        honest_token_counts = 0    
        for word, score in zip(words[0:], rep_scores[0:]):
            if start_answer_token in word and not started:
                started = True
                continue
            if not started:
                continue
            
            color = colormap(norm(score))
            if norm(score)==1.0:
                honest_token_counts+=1
            # print("Word", word, "Color", color)
            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 1.6

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=8)
            word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
            word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=8)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1
    return fig, ax, round(honest_token_counts*100/len(words[0:]), 2)


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice,  wb_char='▁'):
    input_ids = [token.replace(wb_char, ' ') for token in input_ids]
    sns.set(font_scale=.4) 
    for rep, scores in rep_reader_scores_dict.items():
        start_tok = 0 #input_ids.index('▁A') # easier to just keep the generated string in the outputs
        # print(start_tok, np.array(scores).shape)
        standardized_scores = np.array(scores)[start_tok:start_tok+40,layer_slice]

        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(6.4, 2), dpi=200)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        # ax.set_xlabel("Token Position", fontsize=5)
        ax.set_xlabel("Tokens", fontsize=5)
        ax.set_ylabel("Layer", fontsize=5)

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 1)[1:])
        # ax.set_xticklabels(np.arange(0, len(standardized_scores), 1)[1:], fontsize=5)
        # print(np.arange(0, len(standardized_scores), 1))
        # print(input_ids)
        ax.set_xticklabels(input_ids[1:], fontsize=5)
        ax.tick_params(axis='x', rotation=90)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:], fontsize=5)
        ax.set_title("LAT Neural Activity", fontsize=5)
        plt.tight_layout()
    # plt.show()
    return fig, ax

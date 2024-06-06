import re
from copy import deepcopy
import json
import requests
from workflows.record.helpers import Task
from pydrive2.drive import GoogleDrive
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
import pandas as pd
import plotly.graph_objects as go

action_char2action: dict[str, str] = dict((action_char, action_name) for action_name, action_char in Task.action_char_map.items())
encoding = tiktoken.get_encoding("cl100k_base") ## GPT-4 tokenizer

def remove_outliers_arr(data: list[float|int], lower_percentile: float = 5., upper_percentile: float = 95.) -> list[float|int]:
    '''Remove outliers from a list of numbers'''
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return [x for x in data if (x >= lower_bound) and (x <= upper_bound)]

def get_num_steps_in_sop(drive: GoogleDrive, task: Task) -> int:
    '''Count the number of steps in the SOP of a task. Return None if no matching string is found.'''
    try:
        sop: str = task.get_sop(drive).strip()
    except:
        return None
    
    lines: list[str] = sop.split('\n')
    pattern: re.Pattern = re.compile(r'^(\d+)[.)]')
    for line in reversed(lines):
        match: re.Match = pattern.search(line)
        if match:
            # Return the digits part if a match is found
            num_steps: int = int(match.group(1))
            return num_steps
    
    # Return None if no matching string is found
    return None

def get_num_tokens_sop(drive: GoogleDrive, task: Task) -> int:
    '''Count the number of tokens in the SOP of a task with the GPT-4 tokenizer.'''
    try:
        sop: str = task.get_sop(drive).strip()
    except:
        return None
    num_tokens: int = len(encoding.encode(sop))
    return num_tokens

def get_task_action_counts(drive: GoogleDrive, task: Task) -> dict[str, int]:
    '''Count the number of occurrences of each action in the action trace of a task.'''
    
    try:
        trace_action_str: str = task.get_trace_action_str(drive)
    except:
        return dict((action, None) for action in Task.action_char_map)
    action_counts: dict[str, int] = dict((action,0) for action in Task.action_char_map)
    
    for action_char in trace_action_str:
        action_counts[action_char2action[action_char]] += 1
    
    return action_counts

def get_action_acounts(drive: GoogleDrive, tasks: list[Task]) -> dict[str, list[int]]:
    '''Get the number of occurrences of each action in the action traces of a list'''

    action_counts: dict[str, list[int]] = dict((action, []) for action in Task.action_char_map)
    for task in tasks:
        task_action_counts: dict[str, int] = get_task_action_counts(drive, task)
        if None in task_action_counts.values(): ## skip tasks with invalid trace files
            continue
        for action, count in task_action_counts.items():
            action_counts[action].append(count)
    
    return action_counts

def plot_difficulty_recording_len_corr(drive: GoogleDrive, tasks: list[Task], path_to_cache_dir: str, remove_outliers: bool = False) -> None:
    '''Plot a violin plot of the difficulty of the tasks vs. the recording length.'''

    recording_lengths: dict = {'Easy': [], 'Medium': [], 'Hard': []}
    
    for task in tasks:
        try:
            recording_len: float = get_task_recording_length(drive, task, path_to_cache_dir)
        except Exception as e:
            continue
        if recording_len < 0: continue
        difficulty: str = task.difficulty
        if difficulty in recording_lengths:
            recording_lengths[difficulty].append(recording_len)
    
    if remove_outliers:
        for difficulty in recording_lengths:
            recording_lengths[difficulty] = remove_outliers_arr(recording_lengths[difficulty])
        
    # Prepare data for violin plot
    data_to_plot = [recording_lengths['Easy'], recording_lengths['Medium'], recording_lengths['Hard']]
    
    plt.figure(figsize=(6, 4))
    plt.violinplot(data_to_plot, vert=False)
    
    plt.title('Task difficulty vs. recording length' + (' (outliers removed)' if remove_outliers else ''))
    plt.ylabel('Task difficulty')
    plt.xlabel('Recording length (in s)')
    plt.yticks(ticks=[1, 2, 3], labels=['Easy', 'Medium', 'Hard'])
    plt.grid()
    plt.show()


def get_task_recording_length(drive: GoogleDrive, task: Task, cache_dir: str = 'cache') -> float:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    csv_path = os.path.join(cache_dir, f'recording_lengths.csv')
    
    # Load existing data if available
    try:
        existing_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['task_uid', 'recording_length'])
    
    task_uid = f'{task.task_id}_{task.person}'
    if task_uid in existing_data['task_uid'].values:
        recording_length = existing_data[existing_data['task_uid'] == task_uid]['recording_length'].values[0]
    else:
        try:
            recording_length = task.get_recording_length(drive)
        except Exception as e:
            print(f'Error while getting recording length for task {task_uid}:', e)
            recording_length = -1
        new_data = pd.DataFrame({'task_uid': [task_uid], 'recording_length': [recording_length]})
        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
        existing_data.to_csv(csv_path, index=False)
    
    return float(recording_length)

def get_demos_per_task_counts(drive: GoogleDrive, tasks: list[Task]) -> list[int]:
    demos_per_task: dict[str, int] = {}
    for task in tasks:
        if task.task_id not in demos_per_task:
            demos_per_task[task.task_id] = 0
        demos_per_task[task.task_id] += 1
    return list(demos_per_task.values())

def plot_demos_per_task_hist(split_name: str, values: list[int]) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=range(1, 7), edgecolor='black')
    plt.title(f'Number of demos per task ({split_name})')
    plt.xlabel('Number of demos')
    plt.ylabel('Number of tasks')
    plt.xticks(np.arange(1, 7)+0.5, np.arange(1, 7))
    plt.show()

def plot_sop_steps_hist(split_name, values) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=range(0, (max(values)//10+1)*10, 2), edgecolor='black')
    plt.title(f'Number of steps in SOP ({split_name})')
    plt.xlabel('Number of steps')
    plt.ylabel('Number of SOPs')
    plt.xticks(range(0, (max(values)//10+1)*10, 5))
    plt.show()

def plot_sop_tokens_hist(split_name, values) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=range(0, (max(values)//10+1)*10, 50), edgecolor='black')
    plt.title(f'Number of tokens in SOP ({split_name})')
    plt.xlabel('Number of tokens')
    plt.ylabel('Number of SOPs')
    plt.show()

def plot_action_counts_hist(split_name, action, values) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=range(0, (max(values)//10+1)*10, 2), edgecolor='black')
    plt.title(f'Number of {action} actions in action traces ({split_name})')
    plt.xlabel(f'Number of {action} actions')
    plt.ylabel('Number of action traces')
    plt.show()

def print_stats(split_name, values) -> None:
    print(f"Number of samples in {split_name}: {len(values)}")
    print(f"Min: {min(values)}, Max: {max(values)}")
    print(f"Mean: {sum(values)/len(values):.2f}, Median: {sorted(values)[len(values)//2]}")

def get_bert_sop_embedding_data(drive: GoogleDrive, tasks: list[Task], cache_path: str = 'cache'):
    '''Computes BERT embeddings for the SOPs of the given tasks and returns the embeddings along with the difficulty and website of each task.
    This function discards tasks with invalid SOPs and tasks with multiple websites.
    The BERT model used is `bert-base-uncased`.'''
    from transformers import BertModel, BertTokenizer
    import torch

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    csv_path: str = os.path.join(cache_path, 'bert_sop_embeddings.csv')

    # Load existing data if available
    try:
        existing_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['task_uid', 'embedding', 'difficulty', 'website'])

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    batch_size = 32

    # Initialize lists to store data
    new_embeddings: list[list[float]] = []
    new_difficulty: list[str] = []
    new_website: list[str] = []
    new_sops: list[str] = []
    new_task_uids: list[str] = []

    # Process tasks and get embeddings
    for task in tasks:
        task_uid = f'{int(task.task_id)}_{task.person}'
        if task_uid in existing_data['task_uid'].values:
            continue  # Skip if embedding already exists
        try:  # only consider valid sops
            sop = task.get_sop(drive)
        except:
            continue
        if sop is not None and type(task.difficulty) == str and ',' not in task.site:
            new_sops.append(sop)
            new_difficulty.append(task.difficulty)
            new_website.append(task.site)
            new_embeddings.append(None)  # Placeholders, actual embeddings will be computed in batches
            new_task_uids.append(task_uid)

    # Compute embeddings in batches
    for i in tqdm(range(0, len(new_sops), batch_size)):
        batch: list[str] = new_sops[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings: np.ndarray = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Update the embeddings in new_embeddings list
        for j, emb in enumerate(batch_embeddings):
            new_embeddings[i+j] = emb.tolist()  # Convert numpy array to list to store in CSV

    # Append new data to existing_data DataFrame
    new_sops_df: list[dict] = []
    for i, new_task_uid in enumerate(new_task_uids):
        new_row = {'task_uid': new_task_uid, 'embedding': new_embeddings[i], 'difficulty': new_difficulty[i], 'website': new_website[i]}
        new_sops_df.append(new_row)
    new_sops_df = pd.DataFrame(new_sops_df)
    existing_data: pd.DataFrame = pd.concat([existing_data, new_sops_df], ignore_index=True)

    # Save updated DataFrame to CSV
    existing_data.to_csv(csv_path, index=False)
    embeddings: list[str] = existing_data['embedding'].to_list()
    embeddings: np.ndarray = np.array([json.loads(emb) if type(emb)==str else emb for emb in embeddings])
    difficulty: list[str] = existing_data['difficulty'].to_list()
    website: list[str] = existing_data['website'].to_list()

    return embeddings, difficulty, website

def generate_together_embeddings(text: str, model_api_string: str) -> list[float]:
    url = "https://api.together.xyz/api/v1/embeddings"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"
    }
    session = requests.Session()
    response = session.post(
        url,
        headers=headers,
        json={
            "input": text,
            "model": model_api_string
        }
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    embedding: list[float] = response.json()['data'][0]['embedding']
    return embedding

def get_m2_bert_trace_embedding_data(drive: GoogleDrive, tasks: list[Task], include_json_state: bool = False, cache_path: str = 'cache'):
    '''Gets M2-BERT embeddings (using Together API) for trace jsons of given tasks and returns the embeddings along with the difficulty and website of each task.
    This function discards tasks with invalid trace jsons, tasks with invalid difficulty and tasks with multiple websites.'''

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if include_json_state:
        csv_path = os.path.join(cache_path, 'm2_bert_trace_embeddings_with_json_state.csv')
    else:
        csv_path = os.path.join(cache_path, 'm2_bert_trace_embeddings_no_json_state.csv')

    # Load existing data if available
    try:
        existing_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['task_uid', 'embedding', 'difficulty', 'website'])
    
    ## get trace action jsons
    new_trace_json_strings: list[str] = []
    new_difficulty: list[str] = []
    new_website: list[str] = []
    new_task_uids: list[str] = []
    new_embeddings: list[list[float]] = []

    for task in tasks:

        task_uid = f'{int(task.task_id)}_{task.person}'
        if task_uid in existing_data['task_uid'].values:
            continue # Skip if embedding already exists

        try: ## only consider valid trace files
            trace_json = deepcopy(task.get_trace(drive))
        except:
            continue

        ## remove json_state and html from state objects
        for obj in trace_json['trace']:
            if obj['type'] == 'state':
                if not include_json_state:
                    obj['data'].pop('json_state')
                obj['data'].pop('html')
                
        trace_json_str = json.dumps(trace_json)
        if trace_json_str is not None and type(task.difficulty) == str and ',' not in task.site:
            new_trace_json_strings.append(trace_json_str)
            new_difficulty.append(task.difficulty)
            new_website.append(task.site)
            new_task_uids.append(task_uid)
        
    for trace_json_str in tqdm(new_trace_json_strings):
        new_embeddings.append(generate_together_embeddings(trace_json_str, 'togethercomputer/m2-bert-80M-32k-retrieval'))

    ## append new data to existing data
    new_data = pd.DataFrame({
        'task_uid': new_task_uids,
        'embedding': new_embeddings,
        'difficulty': new_difficulty,
        'website': new_website
    })
    existing_data = pd.concat([existing_data, new_data], ignore_index=True)
    existing_data.to_csv(csv_path, index=False)

    embeddings = np.array([json.loads(o) if type(o)==str else o for o in existing_data['embedding'].tolist()])
    difficulty = existing_data['difficulty'].tolist()
    website = existing_data['website'].tolist()
    
    return embeddings, difficulty, website

def plot_tsne(embeddings, difficulty, website, title: str) -> None:
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    ## group by difficulty
    color_map = {'Easy': 'blue', 'Medium': 'green', 'Hard': 'red'}
    plt.figure(figsize=(10, 6))
    for color in color_map:
        indices: list[int] = [i for i, d in enumerate(difficulty) if d == color]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[color], label=color)
    plt.title(f't-SNE plot of {title} grouped by Difficulty')
    plt.legend()
    plt.show()

    ## group by website
    color_map = {'gitlab': 'blue', 'reddit': 'green', 'shopping': 'red', 'shopping_admin': 'cyan'}
    plt.figure(figsize=(10, 6))
    for color in color_map:
        indices: list[int] = [i for i, d in enumerate(website) if d == color]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[color], label=color)
    plt.title(f't-SNE plot of {title} grouped by Website')
    plt.legend()
    plt.show()

def plot_recording_len_hist(split_name, values) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=range(0, (int(max(values))//10+1)*10, 2), edgecolor='black')
    plt.title(f'Recording length (in s) of demos ({split_name})')
    plt.xlabel('Recording length (in s)')
    plt.ylabel('Number of demos')
    plt.xticks(range(0, (int(max(values))//10+1)*10, 5))
    plt.show()

def get_split_stats_df(task_splits: dict[str, list[Task]], split_type: str, metric_fn: callable) -> pd.DataFrame:
    '''Given a dictionary of task splits, calculate the stats (min, median, max) for a metric_fn.
    The metric_fn should take a Task and return an int.'''
    metric_df = {
        split_type: [],
        'Min': [],
        'Median': [],
        'Max': [],
    }
    for split, split_tasks in task_splits.items():
        metric_values: list[int] = [metric_fn(task) for task in split_tasks]
        metric_values = [o for o in metric_values if o is not None and o != -1] ## Remove None values

        ## calculate stats
        metric_df[split_type].append(split)
        metric_df['Min'].append(int(np.min(metric_values)))
        metric_df['Median'].append(int(np.median(metric_values)))
        metric_df['Max'].append(int(np.max(metric_values)))

    metric_df = pd.DataFrame(metric_df)
    return metric_df

def get_num_demo_split_stats_df(drive: GoogleDrive, task_splits: dict[str, list[Task]], split_type: str) -> pd.DataFrame:
    '''Given a dictionary of task splits, calculate the stats (min, median, max) for a num_demos per task.'''

    num_demos_df = {
        split_type: [],
        'Min': [],
        'Median': [],
        'Max': [],
    }

    for split, split_tasks in task_splits.items():
        counts = get_demos_per_task_counts(drive, split_tasks)
        num_demos_df[split_type].append(split)
        num_demos_df['Min'].append(min(counts))
        num_demos_df['Median'].append(int(np.median(counts)))
        num_demos_df['Max'].append(max(counts))
    num_demos_df = pd.DataFrame(num_demos_df)

    return num_demos_df

def sankey_plot_action_sequences(drive: GoogleDrive, tasks: list[Task]) -> None:
    '''Given a list of tasks, plot a Sankey diagram of the action sequences and save to an html file.'''

    # Get all action sequences
    action_sequences: list[list[str]] = []
    for task in tasks:
        try:
            action_str: str = task.get_trace_action_str(drive)
        except Exception as e:
            continue
        action_seq: list[str] = [action_char2action[o] for o in action_str]
        action_sequences.append(action_seq)

    # Count transitions
    transitions: dict[tuple[str, str], int] = {}
    for action_seq in action_sequences:
        for i in range(len(action_seq) - 1):
            pair: tuple[str, str] = (action_seq[i], action_seq[i + 1])
            if pair not in transitions:
                transitions[pair] = 0
            transitions[pair] += 1
    
    states: list[str] = list(action_char2action.values())
    state_indices: dict[str, int] = {state: idx for idx, state in enumerate(states)}

    # Create source, target, and value lists
    sources: list[int] = [state_indices[pair[0]] for pair in transitions]
    targets: list[int] = [state_indices[pair[1]] for pair in transitions]
    values: list[int] = [transitions[pair] for pair in transitions]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=states,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        ))])

    fig.update_layout(title_text="Action Transitions Sankey Diagram", font_size=15, width=1000, height=500)
    fig.show()

    print('Number of action sequences:', len(action_sequences))

def plot_difficulty_by_website(tasks: list[Task]):

    websites: list[str] = ['gitlab', 'reddit', 'shopping', 'shopping_admin']
    difficulties: list[str] = ['Easy', 'Medium', 'Hard']

    difficulty_by_website: np.ndarray = np.zeros((len(websites), len(difficulties)))

    for task in tasks:
        if task.site not in websites or task.difficulty not in difficulties:
            continue
        website_idx: int = websites.index(task.site)
        difficulty_idx: int = difficulties.index(task.difficulty)
        difficulty_by_website[website_idx, difficulty_idx] += 1
    difficulty_by_website /= np.sum(difficulty_by_website)

    plt.figure(figsize=(6.5, 4))
    bubble = plt.scatter(websites*len(difficulties), difficulties*len(websites), s=difficulty_by_website.flatten()*25000, c=difficulty_by_website.flatten(), cmap='YlOrRd', alpha=0.6)

    for i in range(len(websites)):
        for j in range(len(difficulties)):
            plt.annotate(f"{difficulty_by_website[i,j]*100:.1f}%", (websites[i], difficulties[j]), ha='center', va='center')
    
    plt.xticks(range(len(websites)), websites)
    plt.yticks(range(len(difficulties)), difficulties)
    plt.title('Task difficulty split by website')
    plt.xlabel('Website')
    plt.ylabel('Difficulty')
    plt.xlim(-0.5, len(websites)-0.5)
    plt.ylim(-0.75, len(difficulties)-0.5)

    cbar = plt.colorbar(bubble)
    cbar.set_label('Bubble Size Indicator')
    plt.show()

def plot_difficulty_num_steps_corr(drive: GoogleDrive, tasks: list[Task], remove_outliers: bool = False) -> None:
    '''Plot a violin plot of the difficulty of the tasks vs. the number of steps in the action traces.'''

    num_steps_action_trace: dict = {'Easy': [], 'Medium': [], 'Hard': []}
    
    for task in tasks:
        try:
            num_steps: int = len(task.get_trace_action_str(drive))  # Assume the function is defined elsewhere
        except Exception as e:
            continue
        difficulty: str = task.difficulty
        if difficulty in num_steps_action_trace:
            num_steps_action_trace[difficulty].append(num_steps)
    
    if remove_outliers:
        for difficulty in num_steps_action_trace:
            num_steps_action_trace[difficulty] = remove_outliers_arr(num_steps_action_trace[difficulty])
    
    # Prepare data for violin plot
    data_to_plot = [num_steps_action_trace['Easy'], num_steps_action_trace['Medium'], num_steps_action_trace['Hard']]
    
    plt.figure(figsize=(6, 4))
    plt.violinplot(data_to_plot, vert=False)
    
    plt.title('Task difficulty vs. number of steps in action trace' + (' (outliers removed)' if remove_outliers else ''))
    plt.ylabel('Task difficulty')
    plt.xlabel('Number of steps in action trace')
    plt.yticks(ticks=[1, 2, 3], labels=['Easy', 'Medium', 'Hard'])
    plt.grid()
    plt.show()


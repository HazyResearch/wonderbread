import datetime
from typing import List, Optional, Set
import json
import string
import re
from typing import Any, Dict, List, Callable, Tuple
from wonderbread.helpers import (
    QACheck, Task, 
    get_files_in_folder, 
    get_folders_in_folder, 
    get_rel_path, group_tasks_by_id
)
import Levenshtein
from tqdm import tqdm
from moviepy.editor import VideoFileClip

RECORDING_LENGTH_DIFF_THRESHOLD = 10 # seconds
TRACE_ACTION_LENGTH_DIFF_THRESHOLD = 5 # number of actions
MEAN_NORMALIZED_LV_DIST_THRESHOLD = 0.5 # normalized Levenshtein edit distance
SOP_MAX_N_STEP_DIFF: int = 3 # number of steps in SOP

def sop_not_empty(tasks: List[Task], drive) -> List[QACheck]:
    """Check if SOP is empty.
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc="sop_not_empty()"):
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= (task.get_sop(drive) is not None and task.get_sop(drive) != ''),
        ))
    return qa_results


def gdrive_link_not_empty(tasks: List[Task], drive = None) -> List[QACheck]:
    """Check if GDrive link is not empty.
    """
    qa_results: List[QACheck] = []
    for task in tqdm(tasks, desc="gdrive_link_not_empty()"):
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid=(task.possible == 'No' or (task.gdrive_link is not None and task.gdrive_link != '')),
        ))
    return qa_results

def trace_not_empty(tasks: List[Task], drive = None) -> List[QACheck]:
    """Check if trace is empty.
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc="trace_not_empty()"):
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= (task.get_trace(drive) is not None and task.get_trace(drive) != ''),
        ))
    return qa_results

def screenshots_not_empty(tasks: List[Task], drive) -> List[QACheck]:
    """Check if screenshots/ subfolder exists
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc="screenshots_not_empty()"):
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= task.is_screenshots_exists(drive),
        ))
    return qa_results

def video_not_empty(tasks: List[Task], drive) -> List[QACheck]:
    """Check if .mp4 exists
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc="video_not_empty()"):
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= task.is_video_exists(drive),
        ))
    return qa_results

def truncate_trace_if_non_chrome_app(tasks: List[Task], drive) -> List[QACheck]:
    """Checks if non-Chrome app is at very start or end of trace.json
    If so, then truncate the trace.json and .mp4 to remove t he non-Chrome app
    """
    trace_errors: List[QACheck] = state_app_name_correct(tasks, drive)
    trace_errors = [ x for x in trace_errors if not x.is_valid ]
    qa_results: List[QACheck] = []
    for check in tqdm(trace_errors, desc="truncate_trace_if_non_chrome_app()"):
        # Try to fix
        seq = check.other['seq']
        if ' ' in seq.replace('0', ' ').strip():
            # If there is a non-Chrome app in the middle of the trace, then can't fix
            qa_results.append(check)
        else:
            # If non-Chrome app is at very start or end of trace, then truncate the trace.json and .mp4 video
            task = [ task for task in tasks if task.task_id == check.task_id and task.person == check.person ][0]
            # Find first instance of Chrome app
            start_idx: int = seq.find('1')
            end_idx: int = seq.rfind('1')
            if start_idx == -1 or end_idx == -1:
                qa_results.append(check)
            elif start_idx > 1 or end_idx < len(seq) - 2:
                # Only truncate one frame for now
                qa_results.append(check)
            else:
                # Calculate amt to truncate .mp4
                original_trace_start_timestamp: float = datetime.datetime.fromisoformat(task.get_trace(drive)['trace'][0]['data']['timestamp'])
                original_trace_end_timestamp: float = datetime.datetime.fromisoformat(task.get_trace(drive)['trace'][-1]['data']['timestamp'])
                new_trace_start_timestamp: float = datetime.datetime.fromisoformat(task.get_trace(drive)['trace'][start_idx]['data']['timestamp'])
                new_trace_end_timestamp: float = datetime.datetime.fromisoformat(task.get_trace(drive)['trace'][end_idx]['data']['timestamp'])
                truncate_secs_start: float = (new_trace_start_timestamp - original_trace_start_timestamp).total_seconds()
                truncate_secs_end: float = (original_trace_end_timestamp - new_trace_end_timestamp).total_seconds()
                # Truncate .mp4
                path_to_video: str = task.get_path_to_video(drive)
                with VideoFileClip(path_to_video) as video:
                    truncated_video = video.subclip(truncate_secs_start, video.duration - truncate_secs_end)
                    truncated_video.write_videofile(path_to_video)
                # Truncate trace.json
                fixed_trace = task.get_trace(drive)['trace'][start_idx:end_idx+1]
                for event_idx, event in task.trace:
                    # Update timestamps
                    fixed_trace[event_idx]['data']['timestamp'] -= truncate_secs_start
                
                qa_results.append(QACheck(
                    person=task.person,
                    task_id=task.task_id,
                    folder_url=task.gdrive_folder.get('url'),
                    is_valid=False,
                    fixed_trace_json=fixed_trace,
                    fixed_video_path=path_to_video,
                ))
    return qa_results


def sop_numbering_correct(tasks: List[Task], drive = None) -> List[QACheck]:
    """Check if SOP numbering is consecutive ascending, starting with "1.".
    Tries to repair SOP numbering if it is not correct.
    """
    def remove_numbered_prefix(s):
        pattern = re.compile(r'^\d+\.\s')
        res = pattern.sub('',s)
        return res
    
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc="sop_numbering_correct()"):
        try:
            lines = task.get_sop(drive).splitlines()[1:]
            lines = [j for j in lines if j.strip() != '' ]
            nums = [int(s.split(".")[0]) for s in lines]
        except Exception as e:
            print(str(e))
            qa_results.append(QACheck(
                person=task.person,
                task_id=task.task_id,
                folder_url=task.gdrive_folder.get('url'),
                is_valid=False,
                note=str(e),
            ))
            continue
            

        expected_nums = list(range(1, len(nums)+1))
        new_sop = None
        if nums != expected_nums:
            new_sop = task.get_sop(drive).splitlines()[0]
            for i in range(len(lines)):
                new_line = str(i+1) + ". " + remove_numbered_prefix(lines[i])
                new_sop += '\n' + new_line
        
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= (nums == expected_nums),
            fixed_sop=new_sop
        ))
        
    return qa_results

def sop_correct_task_description(tasks: List[Task], drive = None) -> List[QACheck]:
    """Check if SOP task description is first line of document AND
    CORRECT (match against task.jsons)
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc='sop_correct_task_description()'):
        try:
            s = task.get_sop(drive).splitlines()[0].strip()
            sop_task_description = s.translate(str.maketrans('', '', string.punctuation)).lower()
            lines = task.get_sop(drive).splitlines()[1:]
            lines = [j for j in lines if j!='']
        except Exception as e:
            print(str(e))
            qa_results.append(QACheck(
                person=task.person,
                task_id=task.task_id,
                folder_url=task.gdrive_folder.get('url'),
                is_valid=False,
                note=str(e),
            ))
            continue
        file_path = get_rel_path(__file__, f"../tasks/{task.task_id}.json")
        with open(file_path, 'r') as file:
            e = json.load(file)['intent']
            expected_description = e.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        
        new_sop = None
        if sop_task_description != expected_description:
            new_sop = e + '\n' + '\n'.join(lines)

        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= (sop_task_description == expected_description),
            fixed_sop=new_sop
        ))

    return qa_results

def sop_cross_ref_num_steps(tasks: List[Task], drive = None) -> List[QACheck]:
    """Cross-reference SOPs across ppl for each task. If there
    is a deviation of `SOP_MAX_N_STEP_DIFF` steps from the mean # of steps across ppl, flag
    """
    def get_step_count(task: Task) -> int:
        """ Given a task, outputs the number of steps in the SOP
        """
        lines = task.get_sop(drive).splitlines()[1:]
        lines = [j for j in lines if j!='']
        return len(lines)

    qa_results: List[QACheck] = []

    # group tasks performed by different people into task_id
    grouped_tasks = {}
    for task in tqdm(tasks, desc='sop_cross_ref_num_steps()'):
        if task.task_id not in grouped_tasks:
            grouped_tasks[task.task_id] = []
        grouped_tasks[task.task_id].append((task, get_step_count(task))) # append tuples (task, num_steps)
    
    # perform number of steps check and add to qa_results
    for _, task_group in grouped_tasks.items():
        avg_steps = 0
        for pair in task_group:
            avg_steps += pair[1]
        avg_steps /= len(task_group)

        for pair in task_group:
            task = pair[0]
            qa_results.append(QACheck(
                person=task.person,
                task_id=task.task_id,
                folder_url=task.gdrive_folder.get('url'),
                is_valid= (abs(pair[1]-avg_steps) <= SOP_MAX_N_STEP_DIFF),
            ))

    return qa_results

def sop_mentions_keywords(tasks: List[Task], drive = None) -> List[QACheck]:
    """ If certain strings/keywords are explicitly mentioned in the task description,
    then check whether or not they appear in the SOP.
    """
    def get_words_within_quotes(s):
        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, s)
        return matches

    qa_results: List[QACheck] = []
    for task in tqdm(tasks, desc='sop_mentions_keywords()'):
        file_path = "tasks/" + f"{task.task_id}.json"
        with open(file_path, 'r') as file:
            task_description = json.load(file)['intent'].strip().lower()

        keywords = get_words_within_quotes(task_description)
        sop_body = task.get_sop(drive).splitlines()[1:]
        sop_body = [j for j in sop_body if j!='']
        sop_body_text = " ".join(sop_body)
        
        passesCheck = all(keyword in sop_body_text for keyword in keywords)

        
        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid= passesCheck,
        ))

    return qa_results

def cross_ref_helper(tasks: List[Task], comparator: Callable) -> List[QACheck]:
    """Cross ref tasks
    """
    qa_results: List[QACheck] = []
    grouped_tasks = group_tasks_by_id(tasks)
    for _, task_group in tqdm(grouped_tasks.items(), 'cross_ref_helper()'):
        for task in task_group:
            is_valid, note = comparator(task, task_group)
            qa_results.append(QACheck(
                person=task.person,
                task_id=task.task_id,
                folder_url=task.gdrive_folder.get('url') if task.gdrive_folder is not None else None,
                is_valid=is_valid,
                note=note,
            ))
    return qa_results

def cross_ref_infeasible(tasks: List[Task], drive = None) -> List[QACheck]:
    """Cross ref whether someone said a task was infeasible or not.
    """
    def comparator(task: Task, task_group: List[Task]) -> Tuple[bool, Dict[str, Any]]:
        # TRUE if at least one person said task was feasible
        is_feasible: bool = any([ task.possible != 'No' for task in task_group ]) 
        return not is_feasible or task.possible != 'No', {}
    return cross_ref_helper(tasks, comparator)

def cross_ref_recording_length(tasks: List[Task], drive = None) -> List[QACheck]:
    """Cross ref whether recording length has more than threshold diff from mean length.
    """
    def comparator(task: Task, task_group: List[Task]) -> Tuple[bool, Dict[str, Any]]:
        # TRUE if at least one person said it was feasible
        mean_length = sum([task.get_recording_length() for task in task_group]) / len(task_group)
        return abs(mean_length - task.recording_length) < RECORDING_LENGTH_DIFF_THRESHOLD, {
            'recording_length' : task.recording_length,
        }
    return cross_ref_helper(tasks, comparator)

def cross_ref_trace_actions_len(tasks: List[Task], drive = None) -> List[QACheck]:
    """Cross ref whether trace action length is more than threshold diff from mean length.
    """
    def comparator(task: Task, task_group: List[Task]) -> Tuple[bool, Dict[str, Any]]:
        # TRUE if at least one person said it was feasible
        mean_length: float = sum([len(o.get_trace_action_str()) for o in task_group]) / len(task_group)
        return abs(mean_length - len(task.get_trace_action_str())) < TRACE_ACTION_LENGTH_DIFF_THRESHOLD, {
            'trace_action_length' : len(task.get_trace_action_str()),
        }
    return cross_ref_helper(tasks, comparator)

def cross_ref_trace_actions_str(tasks: List[Task], drive = None) -> List[QACheck]:
    """Cross ref to find mean Levenshtein edit distance between trace action strings.
    """
    qa_results: List[QACheck] = []
    grouped_tasks = group_tasks_by_id(tasks)
    ## calculate mean levenshtein edit distance and add to qa_results
    for _, task_group in tqdm(grouped_tasks.items()):
        for task in task_group:

            trace_action_str1: str = task.get_trace_action_str()

            ## calculate mean normalized levenshtein edit distance
            mean_normalized_lv_dist: float = 0
            for task2 in task_group:
                trace_action_str2: str = task2.get_trace_action_str()
                lv_dist: float = Levenshtein.distance(trace_action_str1, trace_action_str2)
                normalized_lv_dist: float = lv_dist / max(len(trace_action_str1),
                                                    len(trace_action_str2))
                mean_normalized_lv_dist += normalized_lv_dist
            
            mean_normalized_lv_dist /= (len(task_group) - 1)

            qa_check = QACheck(
                person=task.person,
                task_id=task.task_id,
                folder_url=task.gdrive_folder.get('url'),
                is_valid=mean_normalized_lv_dist < MEAN_NORMALIZED_LV_DIST_THRESHOLD,
                note={
                    'mean_normalized_lv_dist' : mean_normalized_lv_dist,
                    'trace' : task.trace_actions,
                },
                fixed_sop=None
            )
            qa_results.append(qa_check)

    return qa_results


def state_app_name_correct(tasks: List[Task], drive = None) -> List[QACheck]:
    """Check if Google Chrome is the active application for each state in trace.
    """
    qa_results: List[QACheck] = []
    tasks = [ x for x in tasks if x.gdrive_folder is not None ] # filter out tasks that don't have a GDrive folder
    for task in tqdm(tasks, desc='state_app_name_correct()'):
        # FALSE if at least one state has active_application_name != 'Google Chrome'
        is_valid: bool = True
        note: Optional[str] = None
        seq: str = '' # 1 = Chrome, 0 = non-Chrome
        try:
            for trace_item_idx, trace_item in enumerate(task.get_trace(drive)['trace']):
                if (
                    trace_item['type'] == 'state'
                    and trace_item['data']['active_application_name'] != 'Google Chrome'
                ):
                    is_valid = False
                    seq += '0'
                else:
                    seq += '1'
        except Exception as e:
            print(str(e))
            is_valid = False
            note = str(e)

        qa_results.append(QACheck(
            person=task.person,
            task_id=task.task_id,
            folder_url=task.gdrive_folder.get('url'),
            is_valid=is_valid,
            note=note,
            other={ 
                'seq' : seq,
            },
        ))

    return qa_results

def remove_ds_store_files(drive, folder_id: str) -> List[str]:
    """Remove all .DS_STORE files from a Google Drive folder."""
    folder_urls: List[str] = []
    task_folders = get_folders_in_folder(drive, folder_id)
    for subfolder in tqdm(task_folders, desc="remove_ds_store_files()"):
        subfolder_id = subfolder['id']
        file_list = get_files_in_folder(drive, subfolder_id)
        for file in file_list:
            if file['title'] == '.DS_STORE':
                print(f".DS_STORE in {subfolder['alternateLink']}")
                file.Delete()
                folder_urls.append(subfolder['alternateLink'])
    return folder_urls

def get_demo_folders_with_demo_subfolders(drive, folder_id: str) -> List[str]: 
    """ Returns a list of the Google Drive Folders containing extraneous directories
    """
    folder_urls: List[str] = []
    task_folders = get_folders_in_folder(drive, folder_id)
    for subfolder in tqdm(task_folders, desc="get_demo_folders_with_demo_subfolders()"):
        subfolder_id = subfolder['id']
        subfolder_list = get_folders_in_folder(drive, subfolder_id)
        if len(subfolder_list) > 1:
            print(f"Nested subdirectory detected in in {subfolder['alternateLink']}")
            folder_urls.append(subfolder['alternateLink'])
    return folder_urls

def move_folders_based_on_linkage_status(drive, source_folder_id: str, destination_folder_id: str, valid_folder_urls: Set[str], is_move_if_linked: bool = False) -> List[str]: 
    """
        Looks at all folders in `source_folder` and moves any folders that are (not) linked to a task to the `destination_folder`
        
        Move unlinked folders from root -> archive:
            move_folders_based_on_linkage_status(drive, ROOT_FOLDER_ID, ARCHIVE_FOLDER_ID, valid_folder_urls, False)
        kMove linked folders from archive -> root:
            move_folders_based_on_linkage_status(drive, ARCHIVE_FOLDER_ID, ROOT_FOLDER_ID, valid_folder_urls, True)
    """
    results: List[str] = []
    task_folders = get_folders_in_folder(drive, source_folder_id)
    valid_folder_url_ids: Set[str] = set([ url.split('/')[-1] for url in valid_folder_urls ])
    for subfolder in tqdm(task_folders, desc="move_folders_based_on_linkage_status()"):
        if subfolder['id'] == destination_folder_id:
            # ignore destination folder itself
            continue
        subfolder_url: str = subfolder['alternateLink'] # 'https://drive.google.com/drive/folders/1Z2X3Y4Z5Y6Z7Y8Z9'
        subfolder_url_id: str = subfolder_url.split('/')[-1] # '1Z2X3Y4Z5Y6Z7Y8Z9'
        
        if (not is_move_if_linked and subfolder_url_id not in valid_folder_url_ids):
            # Root -> Archive
            print(f"Found unlinked folder: {subfolder['alternateLink']}")
            subfolder['parents'] = [{"id": destination_folder_id}]
            subfolder.Upload()
            results.append(subfolder['alternateLink'])
        elif (is_move_if_linked and subfolder_url_id in valid_folder_url_ids):
            # Archive -> Root
            print(f"Found linked folder: {subfolder['alternateLink']}")
            subfolder['parents'] = [{"id": destination_folder_id}]
            subfolder.Upload()
            results.append(subfolder['alternateLink'])
    return results
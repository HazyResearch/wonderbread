"""
Usage:

python3 main.py --path_to_drive_client_secrets_json "./client_secrets.json"
"""
import argparse
import os
import pickle
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from wonderbread.helpers import ROOT_FOLDER_ID, ARCHIVE_FOLDER_ID, QACheck, get_files_in_folder, init_drive, load_tasks, Task, load_tasks_excel
from wonderbread.collect.qa.utils import (
    get_demo_folders_with_demo_subfolders,
    move_folders_based_on_linkage_status,
    remove_ds_store_files,
    sop_not_empty,
    trace_not_empty,
    video_not_empty,
    screenshots_not_empty,
    sop_numbering_correct, 
    sop_correct_task_description, 
    state_app_name_correct,
    cross_ref_infeasible,
)

PATH_TO_CACHE_DIR: str = './cache/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_drive_client_secrets_json', type=str, required=True, help='Path to Google Drive client secrets JSON file.')
    parser.add_argument('--path_to_output_dir', type=str, default=PATH_TO_CACHE_DIR, help='Path to output dir')
    return parser.parse_known_args()

def save_invalid_qas(qa_checks: List[QACheck], path_to_output_file: str) -> pd.DataFrame:
    """Saves invalid QA checks to CSV file."""
    df = pd.DataFrame({
        'person': [qa_check.person for qa_check in qa_checks],
        'task_id': [qa_check.task_id for qa_check in qa_checks],
        'folder_url': [qa_check.folder_url for qa_check in qa_checks],
        'is_valid': [qa_check.is_valid for qa_check in qa_checks],
        'note': [qa_check.note for qa_check in qa_checks],
        'fixed_sop': [qa_check.fixed_sop for qa_check in qa_checks],
    })
    df = df[~df['is_valid']]
    df = df[df['fixed_sop'].isna()]
    df = df[['task_id', 'person', 'folder_url', 'is_valid', 'note', 'fixed_sop']]
    df.to_csv(path_to_output_file, index=False, sep='\t')
    return df

def main(args):
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    path_to_drive_client_secrets_json: str = args.path_to_drive_client_secrets_json

    # # Folder structure checks
    drive = init_drive(path_to_drive_client_secrets_json)
    # Demos stored within demos
    invalid_subfolders: List[str] = get_demo_folders_with_demo_subfolders(drive, ROOT_FOLDER_ID)
    with open(os.path.join(args.path_to_output_dir, 'invalid_subfolders.txt'), 'w') as f:
        f.write('\n'.join(invalid_subfolders))
    .DS_Store files
    remove_ds_store_files(drive, ROOT_FOLDER_ID)
    # Move unlinked folders to "Archive"
    tasks_excel_df = load_tasks_excel('../data/Process Mining Task Demonstrations.xlsx')
    valid_folder_urls: List[str] = tasks_excel_df['Gdrive link'].tolist()
    valid_folder_urls = set([ 
                             url.split('?id=')[1].split('&')[0] if 'https://drive.google.com/open?id=' in url else url.split('?')[0]
                             for url in valid_folder_urls 
                             if isinstance(url, str) and url.startswith('https://')  
    ]) # remove "?usp=drive_link"
    print(f"Found {len(valid_folder_urls)} valid folder URLs.")
    # move_folders_based_on_linkage_status(drive, ROOT_FOLDER_ID, ARCHIVE_FOLDER_ID, valid_folder_urls, False)
    # Move linked folders to "Root"
    move_folders_based_on_linkage_status(drive, ARCHIVE_FOLDER_ID, ROOT_FOLDER_ID, valid_folder_urls, True)

    # Lazy load tasks
    tasks: List[Task] = load_tasks(path_to_drive_client_secrets_json, args.path_to_output_dir)

    # Run validators
    for validator in [
        sop_not_empty,
        trace_not_empty,
        video_not_empty,
        screenshots_not_empty,
        sop_numbering_correct, 
        sop_correct_task_description, 
        state_app_name_correct,
        cross_ref_infeasible,
    ]:
        qa_checks: List[QACheck] = validator(tasks, drive)
        save_invalid_qas(qa_checks, os.path.join(args.path_to_output_dir, f'{validator.__name__}.csv'))

        # Fix SOPs (if applicable)
        queries = []
        for check in qa_checks:
            if check.fixed_sop is not None and not check.is_valid:
                queries.append(check)
        for check in tqdm(queries, desc=f'Fixing SOPs for {validator.__name__}...'):
            task = [ task for task in tasks if task.task_id == check.task_id and task.person == check.person ][0]
            task.sop = check.fixed_sop
            files_in_folder: List[str] = get_files_in_folder(drive, task.gdrive_folder['id'])
            for file in files_in_folder:
                if file['title'].startswith('SOP'):
                    if len([ x for x in files_in_folder if x['title'] == f"[raw] {file['title']}" ]) == 0:
                        # Haven't created a copy of the original SOP yet, so do that now
                        drive.auth.service.files().copy(fileId=file['id'], body={'title': f"[raw] {file['title']}" }).execute()
                    file.SetContentString(task.get_sop(drive))
                    file.Upload()
                    break

        pickle.dump(tasks, open(os.path.join(args.path_to_output_dir, "tasks.pkl"), "wb"))



if __name__ == "__main__":
    args, __ = parse_args()
    main(args)
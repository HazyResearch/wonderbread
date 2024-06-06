from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from tqdm import tqdm
from wonderbread.helpers import init_drive
import pandas as pd

# Path to your client_secret.json
BASE_DIR: str = os.path.join(__file__, '../../')
PATH_TO_CLIENT_SECRET_FILE = os.path.join(BASE_DIR, "client_secrets.json")
PATH_TO_XLSX = os.path.join(BASE_DIR, "data/Process Mining Task Demonstrations.xlsx")
PATH_TO_OUTPUT_DIR: str = os.path.join(BASE_DIR, "data/demos")
os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

def download_file(file, path):
    """Download a single file to the specified path."""
    if (
        not os.path.exists(os.path.join(path, file["title"]))
        or os.path.getsize(os.path.join(path, file["title"])) == 0
    ):
        file.GetContentFile(os.path.join(path, file["title"]))


def download_folder_contents(drive, folder_id, path):
    """Recursively download the contents of a folder, including subfolders."""
    file_list = drive.ListFile(
        {"q": f"'{folder_id}' in parents and trashed=false"}
    ).GetList()
    for file in file_list:
        # If the file is a folder, recurse
        if file["mimeType"] == "application/vnd.google-apps.folder":
            # print(f'Entering folder: {file["title"]}')
            # Create the folder locally if it doesn't exist
            folder_path = os.path.join(path, file["title"])
            os.makedirs(folder_path, exist_ok=True)
            # Recurse into the folder
            download_folder_contents(drive, file["id"], folder_path)
        else:
            download_file(file, path)


def get_folder_name(drive, folder_id):
    """Get the name of the folder from its ID."""
    folder = drive.CreateFile({"id": folder_id})
    folder.FetchMetadata(fields="title")
    return folder["title"]


def extract_folder_id(x: str) -> str:
    # Extract Google Drive folder ID from URL
    # id = x.split('?id=')[1].split('&')[0] if 'https://drive.google.com/open?id=' in x else x.split('?')[0]
    if "https://drive.google.com/open?id=" in x:
        return x.split("?id=")[1].split("&")[0]
    return x[: x.index("?") if "?" in x else len(x)].split("/")[-1]


def process_row(row, drive, path_to_output_dir):
    try:
        if not str(row["Gdrive link"]).startswith("http"):
            # Ignore if no link
            return None
        folder_id = extract_folder_id(str(row["Gdrive link"]))
        folder_name = get_folder_name(drive, folder_id)
        path_to_output = os.path.join(path_to_output_dir, folder_name)
        os.makedirs(path_to_output, exist_ok=True)
        download_folder_contents(drive, folder_id, path_to_output)
        return {
            "folder_id": folder_id,
            "folder_name": folder_name,
            "url": row["Gdrive link"],
            "path_to_output": path_to_output,
        }
    except Exception as e:
        print(f"Exception: {e} for row: ", row["Gdrive link"], row["Person"], row.index)
        raise e


def main():
    # Read folder URLs
    dfs = pd.read_excel(PATH_TO_XLSX, sheet_name=None)
    df = []
    for key in dfs.keys():
        if key.startswith("Sheet "):
            df.append(dfs[key])
    df = pd.concat(df)
    print("Size of dataframe:", df.shape)

    # Setup gdrive
    drive = init_drive(PATH_TO_CLIENT_SECRET_FILE)

    # Setup threading
    n_threads: int = 20
    results = []
    with ThreadPoolExecutor(
        max_workers=n_threads
    ) as executor:  # Adjust max_workers as needed
        # Prepare futures
        futures = [
            executor.submit(process_row, row, drive, PATH_TO_OUTPUT_DIR)
            for index, row in df.iterrows()
        ]
        # Progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")
    json.dump(
        {"results": results},
        open(
            os.path.abspath(os.path.join(PATH_TO_OUTPUT_DIR, "../", "metadata.json")),
            "w",
        ),
    )


if __name__ == "__main__":
    main()

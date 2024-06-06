from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive, GoogleDriveFile
import pandas as pd
from wonderbread.helpers import get_rel_path
from typing import Dict, List
import re
import json

gauth: GoogleAuth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

df: Dict[str, pd.DataFrame] = pd.read_excel(get_rel_path(__file__, "../../data/Process Mining Task Demonstrations.xlsx"), sheet_name=None)
gdrive_link_2_folder_name: Dict[str, str] = {}

def get_folder_id_from_gdrive_link(gdrive_link: str) -> str:
    """Get folder id from Google Drive link.
    
    Example 1: https://drive.google.com/drive/folders/1t9Uw2sGzqYJzJY0Qg8CZxw9gC4vY5zZw
    Folder id: 1t9Uw2sGzqYJzJY0Qg8CZxw9gC4vY5zZw

    Example 2: https://drive.google.com/drive/folders/1VHCRPFQadBPN42Ujjd_I4DipBprZfu3O?usp=drive_link
    Folder id: 1VHCRPFQadBPN42Ujjd_I4DipBprZfu3O
    
    Example 3: https://drive.google.com/drive/folders/1SiNwknfh3Gc4K0J2FW5REVUwHBZHMo-S?usp=sharing
    Folder id: 1SiNwknfh3Gc4K0J2FW5REVUwHBZHMo-S
    
    Example 4: https://drive.google.com/open?id=14B3bojpkU0kGNSZNdf3QKH5cNZJF2GC4&usp=drive_copy
    Folder id: 14B3bojpkU0kGNSZNdf3QKH5cNZJF2GC4"""

    if 'usp=' in gdrive_link:
        gdrive_link = gdrive_link[:gdrive_link.index('usp=')-1] # Remove part after `&usp=`, `?usp=`
    
    splits: List[str] = re.split('[/=]+', gdrive_link)
    return splits[-1]

# Iterate over all sheets
for i in range(1, 6):
    links: List[str] = df[f"Sheet {i}"]["Gdrive link"].dropna().tolist()
    
    # Get folder name for each link
    for link in links:

        if 'drive.google.com' not in link:
            print(f"Skipping link {link} as it does not contain 'drive.google.com'")
            continue

        folder_id: str = get_folder_id_from_gdrive_link(link)
        folder: GoogleDriveFile = drive.CreateFile({'id': folder_id})
        try:
            folder.FetchMetadata(fields='title')
        except Exception as e:
            print(f"Error fetching metadata for folder with URL {link}: {e}")
            continue

        gdrive_link_2_folder_name[link] = folder['title']

json.dump(gdrive_link_2_folder_name, open(get_rel_path(__file__, "./data/gdrive_link_2_folder_name.json"), "w"), indent=4)

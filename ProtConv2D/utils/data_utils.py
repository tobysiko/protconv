import urllib.request
import shutil
import os

def download_file(url, destination):
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    return os.path.exists(destination)
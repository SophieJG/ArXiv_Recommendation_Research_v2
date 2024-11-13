import json
import os
from urllib.request import urlretrieve
import requests

release = "2024-11-05"
path = "/share/garg/semantic_scholar_Nov2024/"

# https://api.semanticscholar.org/datasets/v1/release/latest see dataset list


def show_progress(block_num, block_size, total_size):
    print(f"Downloading {round(block_num * block_size / total_size *100,2)}%", end="\r")


def get_files(field: str):
    api_key = os.environ.get('API_KEY')
    if api_key is None:
        raise RuntimeError("Semantic Scholar API key is None")

    response = requests.get(
            f"https://api.semanticscholar.org/datasets/v1/release/2024-11-05/dataset/{field}",
            headers={'x-api-key': api_key}
        )
    
    if "error" in response.text:
        print(json.dumps(json.loads(response.text), indent=4))
    
    response = json.loads(response.text)
    files = response["files"]
    output_dir = os.path.join(path, release, field)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("/share/garg/tmp", exist_ok=True)
    for idx, f in enumerate(files):
        output_fname = f.split("?")[0].split("/")[-1]
        output_path = os.path.join(output_dir, output_fname)
        tmp_path = os.path.join("/share/garg/tmp", output_fname)
        print(f"Downloading {field} file {idx + 1} / {len(files)} - {output_fname}")
        if os.path.exists(output_path):
            continue
        urlretrieve(f, tmp_path, show_progress)
        os.rename(tmp_path, output_path)
    

if __name__ == '__main__':
    raise RuntimeError("Are you sure you want to run this? Data is already downloaded. See semantic_scholar_path in data configs")
    for key in ["papers", "authors", "citations", "abstracts"]:
        get_files(key)

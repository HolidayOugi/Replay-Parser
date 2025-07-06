import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_DIR = "./raw"
OUTPUT_DIR = "../PS/input/new_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, (dict, list)):
                return pd.json_normalize(data)
    except Exception as e:
        print(f"❌ Errore nel file {file}: {e}")
    return None

def process_subfolder(subfolder_path):
    folder_name = os.path.basename(subfolder_path)
    json_files = [
        os.path.join(subfolder_path, f)
        for f in os.listdir(subfolder_path)
        if f.endswith(".json") and os.path.isfile(os.path.join(subfolder_path, f))
    ]

    dfs = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Elaborando {folder_name}", leave=False):
            result = future.result()
            if result is not None:
                dfs.append(result)

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, f"{folder_name}.parquet")
        full_df.to_parquet(output_path, index=False)
        return folder_name
    return None

def main():
    subfolders = [
        os.path.join(INPUT_DIR, d)
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]

    for folder in tqdm(subfolders, desc="Elaborazione cartelle"):
        result = process_subfolder(folder)
        if result:
            tqdm.write(f"✔️ Salvato {result}.parquet")

if __name__ == "__main__":
    main()

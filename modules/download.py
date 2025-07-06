import os
import requests
import json
from urllib.parse import quote
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import math
import streamlit as st

def download_files(readable_format, player, output_dir, max_replays):
    OUTPUT_BASE_DIR = output_dir
    MAX_WORKERS = 10

    tier_map_reverse = {
        "OU": "ou",
        "UBERS": "ubers",
        "UU": "uu",
        "NU": "nu",
        "RU": "ru",
        "ZU": "zu",
        "PU": "pu",
        "ANYTHINGGOES": "anythinggoes",
    }

    def fetch(url):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                return response.text
        except:
            return None
        return None

    def download_replay(replay_id, format_dir):
        filepath = os.path.join(format_dir, f"{replay_id}.json")
        try:
            with open(filepath, "x", encoding="utf-8") as f:
                url = f"https://replay.pokemonshowdown.com/{replay_id}.json"
                content = fetch(url)
                if content:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
        except FileExistsError:
            pass
        except:
            return

    def process_player_format(player, raw_format, readable_format, max_replays):
        format_dir = os.path.join(OUTPUT_BASE_DIR, readable_format)
        os.makedirs(format_dir, exist_ok=True)
        placeholder = st.empty()

        max_page = math.ceil(max_replays / 50)

        num_replays = 0

        max_valid_page = 0


        for page in range(1, max_page+1):
            encoded_player = quote(player)
            url = f"https://replay.pokemonshowdown.com/search.json?user={encoded_player}&format={raw_format}&page={page}"
            text = fetch(url)
            if not text or text == "[]":
                break

            else:
                max_valid_page += 1


        progress_bar = placeholder.progress(0)
        for page in range(1, max_valid_page+1):
            encoded_player = quote(player)
            url = f"https://replay.pokemonshowdown.com/search.json?user={encoded_player}&format={raw_format}&page={page}"
            text = fetch(url)
            if not text or text == "[]":
                break

            try:
                data = json.loads(text)
            except:
                return

            if isinstance(data, dict) and "replays" in data:
                replays = data["replays"]
            elif isinstance(data, list):
                replays = data
            else:
                replays = []

            if num_replays + len(replays) > max_replays:
                replays = replays[:max_replays - num_replays]

            num_replays += len(replays)

            new_found = False
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for replay in replays:
                    replay_id = replay.get("id")
                    if replay_id:
                        filepath = os.path.join(format_dir, f"{replay_id}.json")
                        if not os.path.exists(filepath):
                            new_found = True
                            futures.append(executor.submit(download_replay, replay_id, format_dir))

                for _ in tqdm(futures, desc=f"Scaricando replays {readable_format}", leave=False):
                    _.result()

            if not new_found:
                return
            progress_bar.progress(int(page / max_valid_page * 100))
        placeholder.empty()

    def get_raw_format(readable_format):
        match = re.match(r"\[Gen (\d+)\]\s+([A-Z]+)", readable_format)
        if not match:
            return None
        gen = match.group(1)
        tier = match.group(2).upper()
        if tier not in tier_map_reverse:
            return None
        return f"gen{gen}{tier_map_reverse[tier]}"

    raw_format = get_raw_format(readable_format)
    if raw_format:
        process_player_format(player, raw_format, readable_format, max_replays)
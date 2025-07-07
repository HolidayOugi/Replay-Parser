import pandas as pd
import re
import os
from tqdm import tqdm
import numpy as np
import json
import difflib

def load_battle(input_folder, output_folder, selected_player, selected_formats):

    def parse_log(log_text):
        lines = log_text.splitlines()

        winner = None
        forfeit = None
        p1 = None
        p2 = None
        team1 = set()
        team2 = set()
        faint1 = 0
        faint2 = 0
        tsize1 = 0
        tsize2 = 0
        switch1 = -1 #always starts with switch
        switch2 = -1
        max_turn = 0

        for line in lines:
            if line.startswith('|tie'):
                winner = 'Tie'

            elif line.startswith('|player|p1|'):
                p1 = line.split('|')[3]

            elif line.startswith('|player|p2|'):
                p2 = line.split('|')[3]

            elif line.startswith('|teamsize|p1|'):
                tsize1 = int(line.split('|')[3])

            elif line.startswith('|teamsize|p2|'):
                tsize2 = int(line.split('|')[3])

            elif line.startswith('|win|'):
                winner = line.split('|')[2]

            elif line.startswith('|poke|p1|'):
                species = line.split('|')[3].split(',')[0].strip()
                species = species.replace('’', "'")
                team1.add(species.title())

            elif line.startswith('|poke|p2|'):
                species = line.split('|')[3].split(',')[0].strip()
                species = species.replace('’', "'")
                team2.add(species.title())

            elif line.startswith('|switch|p1a:'):
                switch1 += 1
                parts = line.split('|')
                if len(parts) > 3:
                    species = parts[3].split(',')[0].strip()
                    species = species.replace('’', "'")
                    team1.add(species.title())

            elif line.startswith('|switch|p2a:'):
                switch2 += 1
                parts = line.split('|')
                if len(parts) > 3:
                    species = parts[3].split(',')[0].strip()
                    species = species.replace('’', "'")
                    team2.add(species.title())

            elif line.startswith('|turn|'):
                max_turn += 1

            elif line.startswith('|faint|p1a'):
                faint1 += 1

            elif line.startswith('|faint|p2a'):
                faint2 += 1

        if winner == p1:
            if faint2 < len(team2) or (tsize2 > 0 and faint2 < tsize2):
                forfeit = True
            else:
                forfeit = False

        elif winner == p2:
            if faint1 < len(team1) or (tsize1 > 0 and faint1 < tsize1):
                forfeit = True
            else:
                forfeit = False

        else:
            forfeit = False

        return winner, forfeit, list(team1), list(team2), max_turn, switch1, switch2

    tqdm.pandas()

    def normalize_players(value):
        if isinstance(value, (list, np.ndarray)):
            return [str(p).strip().strip('\'"').strip() for p in value]
        if isinstance(value, str):
            value = value.strip().strip('[]').strip()
            if not value:
                return []
            parts = re.split(r'\s*,\s*', value)
            return [p.strip('\'" ').strip() for p in parts]
        return []

    player_name = None

    if not os.path.exists(input_folder):
        return player_name

    for format_name in os.listdir(input_folder):
        if format_name in selected_formats:
            format_path = os.path.join(input_folder, format_name)
            if not os.path.isdir(format_path):
                continue

            json_files = [f for f in os.listdir(format_path) if f.endswith(".json")]
            if not json_files:
                continue

            all_data = []
            for filename in tqdm(json_files, desc=f"Parsing {format_name}"):
                filepath = os.path.join(format_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "log" in data and "players" in data and "uploadtime" in data:
                            all_data.append({
                                "id": os.path.splitext(filename)[0],
                                "log": data["log"],
                                "players": data["players"],
                                "rating": data["rating"],
                                "uploadtime": data["uploadtime"]
                            })
                except Exception as e:
                    print(f"Errore in {filename}: {e}")
                    continue

            if not all_data:
                continue

            df = pd.DataFrame(all_data)

            print(f"Elaborating {format_name}")

            df['uploadtime'] = pd.to_datetime(df['uploadtime'], unit='s')
            df['players'] = df['players'].apply(normalize_players)
            df['player1'] = df['players'].apply(lambda x: x[0] if len(x) > 0 else None)
            df['player2'] = df['players'].apply(lambda x: x[1] if len(x) > 1 else None)

            sets_of_players = df.apply(lambda row: {row['player1'].lower(), row['player2'].lower()}, axis=1)
            common_players = set.intersection(*sets_of_players)
            if common_players:
                if len(common_players) == 1:
                    player_name = common_players.pop()
                else:
                    matches = difflib.get_close_matches(selected_player, common_players, n=1, cutoff=0)
                    player_name = matches[0] if matches else next(iter(common_players))
            else:
                player1_counts = df['player1'].value_counts()
                player2_counts = df['player2'].value_counts()

                total_counts = player1_counts.add(player2_counts, fill_value=0)

                if not total_counts.empty:
                    player_name = total_counts.idxmax()
                else:
                    player_name = None


            print(f"Selected player: {player_name}")

            parsed = df['log'].progress_map(parse_log)
            df_new = pd.DataFrame(parsed.tolist(),
                                  columns=['Winner', 'Forfeit', 'Team 1', 'Team 2', 'Turns', '# Switches 1', '# Switches 2'])

            df = df.drop(columns=['log', 'players'], errors='ignore')
            df['format'] = format_name
            df = pd.concat([df, df_new], axis=1)

            os.makedirs(output_folder, exist_ok=True)
            output_filename = f"{format_name}.parquet"
            output_path = os.path.join(output_folder, output_filename)

            df.to_parquet(output_path, index=False)

    return player_name
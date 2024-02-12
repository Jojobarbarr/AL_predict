import matplotlib.pyplot as plt
from pathlib import Path
import json

def save_stats(save_dir: Path, name: str, d_stats: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{name}.json", "w", encoding="utf8") as json_file:
        json.dump(d_stats, json_file)
        
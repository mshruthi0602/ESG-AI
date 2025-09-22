# src/runlog.py
import os, json, datetime
from typing import Any

def begin_run(tag: str = "web") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", f"run_{ts}_{tag}")
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj: Any, path: str, name: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

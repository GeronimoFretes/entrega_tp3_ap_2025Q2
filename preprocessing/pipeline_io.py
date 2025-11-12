from __future__ import annotations
import os, json, datetime
from typing import Tuple, Dict, Any, List
from joblib import dump, load

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def save_pipeline(pipe, out_dir: str = "artifacts/prep_v1") -> str:
    """
    Saves:
      - pipeline.joblib  (the fitted pipeline object)
      - schema.json      (frozen columns, catboost_cats, metadata)
    NOTE: On load, the class definitions must be importable at the
    same module path (put your classes in a .py module, not a notebook).
    """
    os.makedirs(out_dir, exist_ok=True)
    pipe_path = os.path.join(out_dir, "pipeline.joblib")
    dump(pipe, pipe_path, compress=("xz", 3))

    meta = {
        "created_utc": _now_iso(),
        "module": pipe.__class__.__module__,
        "class": pipe.__class__.__name__,
        "n_columns": len(getattr(pipe, "columns_", [])),
        "columns": getattr(pipe, "columns_", []),
        "n_catboost_cats": len(getattr(pipe, "catboost_cats_", [])),
        "catboost_cats": getattr(pipe, "catboost_cats_", []),
    }
    with open(os.path.join(out_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_pipeline(out_dir: str = "artifacts/prep_v1") -> Tuple[object, Dict[str, Any]]:
    pipe_path = os.path.join(out_dir, "pipeline.joblib")
    pipe = load(pipe_path)
    try:
        with open(os.path.join(out_dir, "schema.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    # Optional sanity check
    if hasattr(pipe, "columns_") and meta.get("columns") and pipe.columns_ != meta["columns"]:
        print("[WARN] pipeline.columns_ differs from schema.json")
    return pipe, meta

def cat_feature_indices(pipe) -> List[int]:
    """Indices of categorical features for CatBoost, based on the frozen column order."""
    col_to_idx = {c: i for i, c in enumerate(getattr(pipe, "columns_", []))}
    return [col_to_idx[c] for c in getattr(pipe, "catboost_cats_", []) if c in col_to_idx]

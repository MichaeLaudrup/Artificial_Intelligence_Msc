import json
import random
import re
import subprocess
import time
import hashlib
import sys
import argparse
import threading
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/TFM_education_ai_analytics")
OUT = PROJECT / "reports" / "hparam_search"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.jsonl"
HISTORY_JSON = OUT / "experiments_history_random_search.json"
TRIAL_CONFIGS_DIR = OUT / "trial_configs"
TRIAL_METRICS_DIR = OUT / "trial_metrics"
TRIAL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_SPACE = {
    "batch_size": [64, 128, 256],
    "latent_d": [64, 128, 256, 512],
    "num_heads": [2, 4, 8],
    "ff_dim": [128, 256, 512, 1024],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "num_layers": [1, 2, 3, 4],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "focal_gamma": [1.5, 2.0, 2.5, 3.0],
    "static_hidden_dim": [64, 128, 256],
    "head_hidden_dim": [64, 128, 256],
    "reduce_lr_patience": [3, 5, 7],
    "early_stopping_patience": [10, 15, 20],
}

TARGET_METRIC = "val_balanced_acc"
MAX_TRIALS = 5000


def sample_cfg():
    latent_d = random.choice(SEARCH_SPACE["latent_d"])
    num_heads = random.choice(SEARCH_SPACE["num_heads"])
    while latent_d % num_heads != 0 or (latent_d == 512 and num_heads == 2):
        num_heads = random.choice(SEARCH_SPACE["num_heads"])

    return {
        "batch_size": random.choice(SEARCH_SPACE["batch_size"]),
        "latent_d": latent_d,
        "num_heads": num_heads,
        "ff_dim": random.choice(SEARCH_SPACE["ff_dim"]),
        "dropout": random.choice(SEARCH_SPACE["dropout"]),
        "num_layers": random.choice(SEARCH_SPACE["num_layers"]),
        "learning_rate": random.choice(SEARCH_SPACE["learning_rate"]),
        "focal_gamma": random.choice(SEARCH_SPACE["focal_gamma"]),
        "static_hidden_dim": random.choice(SEARCH_SPACE["static_hidden_dim"]),
        "head_hidden_dim": random.choice(SEARCH_SPACE["head_hidden_dim"]),
        "reduce_lr_patience": random.choice(SEARCH_SPACE["reduce_lr_patience"]),
        "early_stopping_patience": random.choice(SEARCH_SPACE["early_stopping_patience"]),
    }


def cfg_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.md5(s).hexdigest()[:10]


# ── helpers para leer / purgar resultados ──────────────────────────────
def load_results():
    """Carga el fichero JSONL y devuelve lista de records válidos."""
    if not RESULTS.exists():
        return []
    records = []
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def purge_failed_results():
    """Elimina del JSONL las entradas con returncode != 0 o sin métricas.
    Los resultados exitosos previos se conservan para la deduplicación."""
    records = load_results()
    good = [r for r in records if r.get("returncode") == 0 and r.get("metrics") is not None]
    removed = len(records) - len(good)
    # Reescribir el fichero solo con los buenos
    with RESULTS.open("w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
    return len(good), removed


def seen_ids():
    if not RESULTS.exists():
        return set()
    ids = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["cfg_id"])
        except Exception:
            pass
    return ids


# ── helpers ────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    """Elimina códigos de escape ANSI de una cadena."""
    return _ANSI_RE.sub("", s)


def _is_noise(line: str) -> bool:
    """Devuelve True si la línea es ruido (protobuf warnings, etc)."""
    low = line.lower()
    return (
        "protobuf" in low
        or "warnings.warn" in low
        or "absl::initializelog" in low
        or line.strip() == "warnings.warn("
        or line.strip() == ""
    )


def export_history():
    """Convierte results.jsonl al formato experiments_history_2clases.json."""
    records = load_results()
    good = [r for r in records if r.get("returncode") == 0 and r.get("metrics") is not None]
    
    out_list = []
    tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rec in good:
        cfg = rec["cfg"]
        m = rec["metrics"]
        
        # Parse timestamp from stdout if possible, otherwise use current time
        rec_tstamp = tstamp
        for ln in rec.get("stdout_tail", []):
            clean = strip_ansi(ln)
            # loguru logs usually start with timestamp like "2026-02-21 23:44:52.606 | "
            if "|" in clean and len(clean) > 20 and clean[0:4] == "2026":
                rec_tstamp = clean.split("|")[0].strip()[:19] # up to seconds
                break

        out_list.append({
            "timestamp": rec_tstamp,
            "hyperparameters": {
                "upto_week": 5,
                "num_classes": 2,
                "paper_baseline": True,
                "with_static": True,
                "batch_size": cfg.get("batch_size"),
                "latent_d": cfg.get("latent_d"),
                "num_heads": cfg.get("num_heads"),
                "ff_dim": cfg.get("ff_dim"),
                "dropout": cfg.get("dropout"),
                "num_layers": cfg.get("num_layers"),
                "learning_rate": cfg.get("learning_rate"),
                "focal_gamma": cfg.get("focal_gamma"),
                "static_hidden_dim": cfg.get("static_hidden_dim"),
                "head_hidden_dim": cfg.get("head_hidden_dim"),
                "reduce_lr_patience": cfg.get("reduce_lr_patience"),
                "early_stopping_patience": cfg.get("early_stopping_patience"),
            },
            "validation_metrics": {
                "loss": m.get("val_loss", 0),
                "accuracy": m.get("val_accuracy", 0),
                "balanced_accuracy": m.get("val_balanced_acc", 0),
                "auc_ovr": m.get("val_auc", 0),
                "precision": m.get("val_precision", 0),
                "recall": m.get("val_recall", 0),
                "f1_score": m.get("val_f1", 0)
            }
        })
        
    with HISTORY_JSON.open("w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=4)


# ── ejecución de un trial (streaming) ──────────────────────────────────
def run_trial(cfg: dict, upto_week=5):
    cid = cfg_id(cfg)
    config_json = TRIAL_CONFIGS_DIR / f"trial_{cid}.json"
    metrics_json = TRIAL_METRICS_DIR / f"trial_{cid}_metrics.json"

    run_cfg = {
        "upto_week": int(upto_week),
        "num_classes": 2,
        "paper_baseline": True,
        "eval_test": False,
        "with_static": True,
        "batch_size": cfg["batch_size"],
        "latent_d": cfg["latent_d"],
        "num_heads": cfg["num_heads"],
        "ff_dim": cfg["ff_dim"],
        "dropout": cfg["dropout"],
        "num_layers": cfg["num_layers"],
        "learning_rate": cfg["learning_rate"],
        "focal_gamma": cfg["focal_gamma"],
        "static_hidden_dim": cfg["static_hidden_dim"],
        "head_hidden_dim": cfg["head_hidden_dim"],
        "reduce_lr_patience": cfg["reduce_lr_patience"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "fast_search": True,
        "run_compare": False,
    }
    config_json.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    if metrics_json.exists():
        metrics_json.unlink()

    cmd = [
        sys.executable, "-u",  # unbuffered para streaming
        "educational_ai_analytics/2_modeling/transformers/train_transformer.py",
        "--config-json", str(config_json),
        "--metrics-out", str(metrics_json),
    ]

    start = time.time()

    # Usamos Popen para streaming de output
    proc = subprocess.Popen(
        cmd, cwd=PROJECT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,  # line-buffered
    )

    all_lines = []
    metrics = None

    for raw_line in proc.stdout:
        raw_line = raw_line.rstrip("\n")
        all_lines.append(raw_line)
        clean = strip_ansi(raw_line)

        # Parsear VAL_METRICS
        if "VAL_METRICS:" in clean:
            try:
                metrics = json.loads(clean.split("VAL_METRICS:", 1)[1].strip())
            except Exception:
                pass

        # Imprimir líneas interesantes (no ruido)
        if not _is_noise(clean):
            print(f"   │ {clean}", flush=True)

    proc.wait()
    dur = time.time() - start

    if metrics_json.exists():
        try:
            metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
            metrics = metrics_payload.get("val_metrics", metrics)
        except Exception:
            pass

    record = {
        "cfg_id": cid,
        "cfg": cfg,
        "returncode": proc.returncode,
        "seconds": dur,
        "metrics": metrics,
        "stdout_tail": all_lines[-20:],
        "stderr_tail": [],
        "config_json": str(config_json),
        "metrics_json": str(metrics_json),
    }

    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


# ── main ───────────────────────────────────────────────────────────────
def main(max_trials: int = MAX_TRIALS):
    random.seed(42)

    # 1) Purge de resultados fallidos previos
    print("=" * 70)
    print("🧹 Purgando resultados fallidos del historial...")
    n_good, n_removed = purge_failed_results()
    print(f"   Conservados: {n_good} exitosos | Eliminados: {n_removed} fallidos")
    print("=" * 70)

    seen = seen_ids()
    print(f"📋 IDs ya registrados (se saltan): {len(seen)}")
    print(f"🎯 Métrica objetivo: {TARGET_METRIC}")
    print(f"🔢 Máximo de trials: {max_trials}")
    print("=" * 70, flush=True)

    best = None
    best_score = -1e9
    n_success = 0
    n_fail = 0
    n_skip = 0
    trial_times = []

    for t in range(1, max_trials + 1):
        cfg = sample_cfg()
        cid = cfg_id(cfg)
        if cid in seen:
            n_skip += 1
            continue
        seen.add(cid)

        # Header del trial
        cfg_short = (
            f"d={cfg['latent_d']} h={cfg['num_heads']} ff={cfg['ff_dim']} "
            f"L={cfg['num_layers']} dr={cfg['dropout']} lr={cfg['learning_rate']} "
            f"γ={cfg['focal_gamma']} bs={cfg['batch_size']}"
        )
        print(f"\n{'─'*70}")
        total_tried = n_success + n_fail
        avg_time = sum(trial_times) / len(trial_times) if trial_times else 0
        print(
            f"📊 TRIAL {t} | {cid} | "
            f"OK:{n_success} FAIL:{n_fail} SKIP:{n_skip} | "
            f"Avg: {avg_time:.1f}s"
        )
        print(f"   {cfg_short}", flush=True)

        rec = run_trial(cfg, upto_week=5)
        dur_mins, dur_secs = divmod(int(rec["seconds"]), 60)

        if rec["returncode"] != 0 or rec["metrics"] is None:
            # Extraer la última línea real de error
            err_msg = ""
            for errline in reversed(rec.get("stdout_tail", [])):
                clean = strip_ansi(errline)
                if not _is_noise(clean) and clean.strip():
                    err_msg = clean.strip()[:120]
                    break
            print(f"   └─ ❌ FALLO [{dur_mins:02d}:{dur_secs:02d}] (rc={rec['returncode']}) {err_msg}", flush=True)
            n_fail += 1
            trial_times.append(rec["seconds"])
            continue

        score = rec["metrics"].get(TARGET_METRIC, None)
        trial_times.append(rec["seconds"])
        
        if score is None:
            print(f"   └─ ⚠️ no target metric [{dur_mins:02d}:{dur_secs:02d}]", flush=True)
            n_fail += 1
            continue

        n_success += 1
        m = rec["metrics"]
        print(
            f"   └─ ✅ [{dur_mins:02d}:{dur_secs:02d}] "
            f"bAcc={score:.4f} F1={m.get('val_f1', 0):.4f} "
            f"AUC={m.get('val_auc', 0):.4f} Recall={m.get('val_recall', 0):.4f}",
            flush=True
        )

        if score > best_score:
            best_score = score
            best = rec
            print(f"  🏆 ¡NUEVO BEST! {TARGET_METRIC}={best_score:.4f}")
            print(f"     Config: {best['cfg']}", flush=True)

    print("\n" + "=" * 70)
    print(f"🏁 BÚSQUEDA COMPLETADA")
    print(f"   Total ejecutados: {n_success + n_fail}")
    print(f"   Exitosos: {n_success} | Fallidos: {n_fail} | Saltados (dedup): {n_skip}")
    if best:
        print(f"\n🏆 MEJOR CONFIGURACIÓN (por {TARGET_METRIC}):")
        print(f"   Score: {best_score:.4f}")
        print(f"   Config: {json.dumps(best['cfg'], indent=2)}")
        print(f"   Todas las métricas: {json.dumps(best['metrics'], indent=2)}")
    print("=" * 70)
    
    # Exportar resultados en json normal
    export_history()
    print(f"💾 Resultados exportados (formato history) a: {HISTORY_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random search para Transformer")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS, help="Número máximo de trials")
    args = parser.parse_args()
    try:
        main(max_trials=args.max_trials)
    except KeyboardInterrupt:
        export_history()
        print("\n\n⛔ Random search detenido por el usuario.")
        print(f"   Resultados parciales guardados en: {RESULTS}")
        print(f"   JSON exportado a: {HISTORY_JSON}")

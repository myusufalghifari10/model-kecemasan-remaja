"""Verifikasi ambang kategori 4 tingkat pada prediksi out-of-fold.

Skema yang dipakai laporan adalah skema dokumen analisis sebelumnya:

    Rendah        <= 1,75
    Sedang        1,75 < skor <= 2,50
    Tinggi        2,50 < skor <= 3,25
    Sangat Tinggi > 3,25

Ambang ini tidak dicari dari data model. Ketiganya hanya bisa dikunci kalau tabel
kategori dokumen sebelumnya (115 / 153 / 35 / 3 siswa) benar-benar terpenuhi, dan
skrip ini memeriksa ulang kecocokan itu pada skor aktual 306 siswa.

Grid pencarian ambang tidak dipakai lagi: dengan kelas Sangat Tinggi berisi 3 siswa,
syarat "setiap kelas minimal 20 siswa" tidak mungkin dipenuhi, dan memilih ambang
demi QWK pada data yang sama justru menimbulkan optimisme pemilihan. Untuk
menghindari itu, ambang dibiarkan tetap dan skrip ini hanya mengukur akibatnya.

Keluaran: outputs/model/uji_ambang.json (deterministik, tanpa pelatihan ulang).
Butuh data privat `outputs/model/oof.csv` dan `data/features.parquet`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TUJUAN = Path(__file__).resolve().parent.parent / "outputs" / "model" / "uji_ambang.json"
AMBANG = (1.75, 2.50, 3.25)
LABEL = ("Rendah", "Sedang", "Tinggi", "Sangat Tinggi")
AKTUAL_DOKUMEN = [115, 153, 35, 3]      # tabel kategori dokumen analisis sebelumnya
BINER = 2.50                            # batas "berisiko tinggi" untuk sensitivitas


def kategori(v: np.ndarray) -> np.ndarray:
    """Nilai tepat di batas masuk kategori bawah."""
    return np.searchsorted(AMBANG, v, side="left")


def main() -> int:
    if not C.DATA_FEATURES.exists():
        print(f"[uji_ambang] tidak ada {C.DATA_FEATURES} — jalankan --mode cv dulu",
              file=sys.stderr)
        return 1
    oof_path = TUJUAN.parent / "oof.csv"
    if not oof_path.exists():
        print(f"[uji_ambang] tidak ada {oof_path} — jalankan --mode cv dulu",
              file=sys.stderr)
        return 1
    oof = pd.read_csv(oof_path).sort_values("row_id").reset_index(drop=True)
    y = oof["y"].to_numpy(float)
    p = oof["pred"].to_numpy(float)

    ka, kp = kategori(y), kategori(p)
    k = len(LABEL)
    cm = confusion_matrix(ka, kp, labels=range(k))
    n_aktual = cm.sum(axis=1).tolist()
    n_prediksi = cm.sum(axis=0).tolist()
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(k)]
    bias = [float((kp == i).mean() - (ka == i).mean()) for i in range(k)]
    qwk = float(cohen_kappa_score(ka, kp, weights="quadratic"))
    akurasi = float((ka == kp).mean())
    selisih = ka - kp
    meleset1 = int((np.abs(selisih) == 1).sum())
    meleset2 = int((np.abs(selisih) >= 2).sum())

    # sensitivitas/spesifisitas biner pada batas 2,50
    positif = y > BINER
    tandai = p > BINER
    tp = int((positif & tandai).sum())
    fp = int((~positif & tandai).sum())
    fn = int((positif & ~tandai).sum())
    tn = int((~positif & ~tandai).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None

    hasil = {
        "skema": "4 tingkat, ambang tetap dari dokumen analisis sebelumnya",
        "ambang": list(AMBANG),
        "label": list(LABEL),
        "aturan_batas": "nilai tepat di batas masuk kategori bawah",
        "n": int(len(y)),
        "n_aktual": n_aktual,
        "n_prediksi": n_prediksi,
        "cocok_dokumen_sebelumnya": n_aktual == AKTUAL_DOKUMEN,
        "akurasi": round(akurasi, 4),
        "qwk": round(qwk, 4),
        "tepat": int((ka == kp).sum()),
        "meleset_1_tingkat": meleset1,
        "meleset_2_tingkat_atau_lebih": meleset2,
        "tabel_silang": {LABEL[i]: {LABEL[j]: int(cm[i, j]) for j in range(k)}
                         for i in range(k)},
        "per_kategori": [
            {"kategori": LABEL[i], "n": int(n_aktual[i]), "recall": recall[i],
             "bias": round(bias[i], 4),
             "mae": round(float(np.abs(y[ka == i] - p[ka == i]).mean()), 4)
             if (ka == i).any() else None}
            for i in range(k)
        ],
        "n_tepat_di_batas": {f"{b:.2f}": int((y == b).sum()) for b in AMBANG},
        "biner_2_50": {
            "batas": BINER, "n_positif": int(positif.sum()),
            "sensitivitas": round(sens, 4) if sens else None,
            "spesifisitas": round(spec, 4) if spec else None,
            "ppv": round(ppv, 4) if ppv else None,
            "npv": round(npv, 4) if npv else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "catatan": "Ambang tidak dicari dari data ini; kecocokan dengan tabel dokumen "
                   "sebelumnya diperiksa pada skor aktual, bukan pada prediksi.",
    }
    TUJUAN.parent.mkdir(parents=True, exist_ok=True)
    TUJUAN.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n")

    print(f"[uji_ambang] skema 4 tingkat, ambang {AMBANG}")
    print(f"  n aktual      : {n_aktual} (dokumen: {AKTUAL_DOKUMEN}, "
          f"cocok: {hasil['cocok_dokumen_sebelumnya']})")
    print(f"  n prediksi    : {n_prediksi}")
    print(f"  akurasi       : {akurasi:.4f} | QWK {qwk:.4f} | tepat {hasil['tepat']} | "
          f"meleset 1: {meleset1} | >=2: {meleset2}")
    print(f"  recall        : {[round(r, 3) if r else None for r in recall]}")
    print(f"  biner >{BINER}: sens {sens:.3f} | spec {spec:.3f} | PPV {ppv:.3f} | "
          f"NPV {npv:.3f} | n positif {int(positif.sum())}")
    print(f"[uji_ambang] ditulis ke {TUJUAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

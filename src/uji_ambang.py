"""Uji ambang kategori tiga tingkat: sapuan grid + uji bersarang.

Skrip ini yang menghasilkan dua angka di laporan dan README:

  * pada prediksi out-of-fold, ambang (1,70 / 2,20) memberi QWK 0,734 dan akurasi 0,693;
  * bila ambang dipilih ulang di lipatan latih lalu dinilai di lipatan uji (5x6 = 30
    lipatan), hasilnya QWK 0,689 dan akurasi 0,660. Angka kedua inilah perkiraan yang
    tidak memakai data yang sama untuk memilih dan menilai.

Alasannya memilih dengan QWK dan bukan akurasi juga diukur di sini: sapuan yang diurutkan
menurut akurasi berakhir pada skema yang menumpuk hampir semua siswa di satu kelas.

Masukan (privat, tidak ikut repo): `outputs/model/oof.csv` dan `data/features.parquet`.
Keluaran: `outputs/model/uji_ambang.json`, deterministik (tanpa cap waktu).

Pakai:
    python src/uji_ambang.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold

import config as C

OUT_JSON = C.ROOT / "outputs" / "model" / "uji_ambang.json"
K = 3                      # jumlah kategori yang diuji


def _fail(pesan: str) -> None:
    raise SystemExit(f"[uji_ambang] {pesan}")


def muat() -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (y sebenarnya, prediksi out-of-fold) berurutan menurut row_id."""
    oof = C.ROOT / "outputs" / "model" / "oof.csv"
    if not oof.is_file():
        _fail(f"tidak ada {oof} — jalankan --mode cv dulu (berkas ini privat)")
    if not C.DATA_FEATURES.is_file():
        _fail(f"tidak ada {C.DATA_FEATURES} — matriks fitur lokal belum ada")
    d = (pd.read_csv(oof)[["row_id", "pred"]]
         .merge(pd.read_parquet(C.DATA_FEATURES)[["row_id", C.TARGET_CONT]], on="row_id")
         .sort_values("row_id"))
    if len(d) != 306:
        _fail(f"harus 306 baris, ada {len(d)}")
    return d[C.TARGET_CONT].to_numpy(float), d["pred"].to_numpy(float)


def kategori(nilai: np.ndarray, b1: float, b2: float) -> np.ndarray:
    """Batas milik kelas BAWAH: tepat 1,70 masuk Rendah, tepat 2,20 masuk Sedang."""
    return np.searchsorted([b1, b2], nilai, side="left")


def ukur(y: np.ndarray, p: np.ndarray, b1: float, b2: float) -> dict:
    ka, kp = kategori(y, b1, b2), kategori(p, b1, b2)
    beda = np.abs(ka - kp)
    n = np.bincount(ka, minlength=K)
    return {
        "b1": b1, "b2": b2,
        "akurasi": float((beda == 0).mean()),
        "qwk": float(cohen_kappa_score(ka, kp, weights="quadratic")),
        "meleset_2_tingkat": int((beda >= 2).sum()),
        "n_aktual": [int(v) for v in n],
        "n_prediksi": [int(v) for v in np.bincount(kp, minlength=K)],
        "recall": [float(((ka == i) & (kp == i)).sum() / n[i]) if n[i] else None
                   for i in range(K)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langkah", type=float, default=0.05,
                    help="langkah grid ambang (default 0,05 = resolusi data)")
    ap.add_argument("--min-kelas", type=int, default=20,
                    help="ukuran minimum tiap kategori agar pasangan ambang ikut diranking")
    a = ap.parse_args()

    y, p = muat()

    # --- grid: hanya pasangan yang tidak membuat kategori kosong atau terlalu kecil
    grid = []
    for b1 in np.round(np.arange(1.05, 3.45 + 1e-9, a.langkah), 2):
        for b2 in np.round(np.arange(float(b1) + a.langkah, 3.50 + 1e-9, a.langkah), 2):
            if np.bincount(kategori(y, float(b1), float(b2)), minlength=K).min() < a.min_kelas:
                continue
            grid.append(ukur(y, p, float(b1), float(b2)))
    if not grid:
        _fail("tidak ada pasangan ambang yang memenuhi syarat ukuran kelas")

    terbaik_qwk = max(grid, key=lambda r: (r["qwk"], r["akurasi"]))
    terbaik_acc = max(grid, key=lambda r: (r["akurasi"], r["qwk"]))
    print(f"[uji_ambang] grid {len(grid)} pasangan (langkah {a.langkah:g}, "
          f"tiap kategori >= {a.min_kelas} siswa)")
    print(f"  QWK terbaik      {terbaik_qwk['b1']:.2f}/{terbaik_qwk['b2']:.2f}: "
          f"QWK {terbaik_qwk['qwk']:.4f} | akurasi {terbaik_qwk['akurasi']:.4f} "
          f"| n {terbaik_qwk['n_aktual']}")
    print(f"  akurasi terbaik  {terbaik_acc['b1']:.2f}/{terbaik_acc['b2']:.2f}: "
          f"akurasi {terbaik_acc['akurasi']:.4f} | QWK {terbaik_acc['qwk']:.4f} "
          f"| n {terbaik_acc['n_aktual']}  <- skema kolaps, QWK jatuh")

    # --- kandidat tetap yang dibahas di laporan
    kandidat = []
    for b1, b2, nama in [(1.70, 2.20, "dipakai"), (1.75, 2.20, "pembanding"),
                         (1.65, 2.20, "pembanding"), (1.75, 2.50, "gabung skema lama"),
                         (2.00, 3.00, "pita sama lebar")]:
        r = ukur(y, p, b1, b2)
        r["nama"] = nama
        kandidat.append(r)
        print(f"  {nama:18s} {b1:.2f}/{b2:.2f}: akurasi {r['akurasi']:.4f} | "
              f"QWK {r['qwk']:.4f} | n {r['n_aktual']} | "
              f"recall {[round(v, 3) for v in r['recall']]}")

    # --- uji bersarang: ambang dipilih di lipatan latih, dinilai di lipatan uji
    bins = np.clip(np.digitize(y, np.arange(1.0, 3.5001, 0.25)), 1, 10)
    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=C.RANDOM_SEED)
    acc_pilih, qwk_pilih, acc_tetap, qwk_tetap, pilihan = [], [], [], [], []
    for tr, va in kfold.split(np.zeros(len(y)), bins):
        skor = [(g, ukur(y[tr], p[tr], g["b1"], g["b2"])) for g in grid]
        amb = max(skor, key=lambda s: (s[1]["qwk"], s[1]["akurasi"]))[0]
        r = ukur(y[va], p[va], amb["b1"], amb["b2"])
        acc_pilih.append(r["akurasi"])
        qwk_pilih.append(r["qwk"])
        pilihan.append([amb["b1"], amb["b2"]])
        rt = ukur(y[va], p[va], 1.70, 2.20)
        acc_tetap.append(rt["akurasi"])
        qwk_tetap.append(rt["qwk"])

    tersering = Counter(tuple(v) for v in pilihan)
    bersarang = {
        "n_lipatan": len(acc_pilih),
        "pilih_dengan_qwk": {
            "akurasi": float(np.mean(acc_pilih)),
            "qwk": float(np.mean(qwk_pilih)),
            "ambang_tersering": [[list(k), n] for k, n in tersering.most_common(3)],
        },
        "ambang_tetap_1_70_2_20": {
            "akurasi": float(np.mean(acc_tetap)),
            "qwk": float(np.mean(qwk_tetap)),
        },
    }
    print(f"\n[uji_ambang] uji bersarang ({len(acc_pilih)} lipatan):")
    print(f"  ambang dipilih   : akurasi {np.mean(acc_pilih):.4f} | "
          f"QWK {np.mean(qwk_pilih):.4f} | "
          f"tersering {bersarang['pilih_dengan_qwk']['ambang_tersering']}")
    print(f"  ambang tetap     : akurasi {np.mean(acc_tetap):.4f} | "
          f"QWK {np.mean(qwk_tetap):.4f}")

    # --- pembanding: skema empat tingkat lama
    t = [1.75, 2.50, 3.25]
    ka, kp = np.searchsorted(t, y, side="left"), np.searchsorted(t, p, side="left")
    empat = {"akurasi": float((ka == kp).mean()),
             "qwk": float(cohen_kappa_score(ka, kp, weights="quadratic")),
             "n_aktual": [int((ka == i).sum()) for i in range(4)]}
    print(f"  pembanding 4 tingkat 1,75/2,50/3,25: akurasi {empat['akurasi']:.4f} | "
          f"QWK {empat['qwk']:.4f} | n {empat['n_aktual']}")

    hasil = {
        "versi": "uji_ambang_v1",
        "masukan": {"prediksi": "outputs/model/oof.csv",
                    "target": "data/features.parquet (y_kontinu)",
                    "n_baris": int(len(y))},
        "grid": {"n_pasangan": len(grid), "langkah": a.langkah, "min_kelas": a.min_kelas,
                 "terbaik_qwk": terbaik_qwk, "terbaik_akurasi": terbaik_acc},
        "kandidat": kandidat,
        "bersarang": bersarang,
        "pembanding_4_kategori": empat,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(hasil, indent=2) + "\n", encoding="utf-8")
    print(f"\n[uji_ambang] ditulis ke {OUT_JSON.relative_to(C.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

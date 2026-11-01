"""Uji ambang kategori tiga tingkat: sapuan grid + uji bersarang penuh.

Skrip ini menghasilkan angka-angka ambang kategori di laporan dan README.

1. Acuan populasi: ambang tetap (1,70 / 2,20) pada prediksi out-of-fold 306 siswa memberi
   QWK 0,7342 dan akurasi 0,6928. Angka ini yang dipakai sebagai hasil utama.
2. Uji bersarang: di setiap lipatan luar (5x6 = 30 lipatan), ambang dipilih lewat validasi
   silang lima lipatan DI DALAM lipatan latih, dan model dilatih ulang di dalamnya. Ambang
   terpilih lalu dinilai pada lipatan uji yang tidak dipakai memilih, sehingga pemilihan
   ambang tidak pernah melihat siswa yang dipakai menilainya. Dua angka dilaporkan: rata-rata
   per lipatan dan QWK gabungan 306 siswa memakai ambang lipatannya masing-masing.
3. Mengapa QWK, bukan akurasi: sapuan yang diurutkan menurut akurasi berakhir pada skema yang
   menumpuk hampir semua siswa di satu kelas (akurasi 88,6%, QWK 0,54).

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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import config as C
import train as M

OUT_JSON = C.ROOT / "outputs" / "model" / "uji_ambang.json"
K = 3                      # jumlah kategori yang diuji


def _fail(pesan: str) -> None:
    raise SystemExit(f"[uji_ambang] {pesan}")


def muat() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Kembalikan (matriks fitur, y, prediksi out-of-fold), urut menurut row_id."""
    oof = C.ROOT / "outputs" / "model" / "oof.csv"
    if not oof.is_file():
        _fail(f"tidak ada {oof} — jalankan --mode cv dulu (berkas ini privat)")
    if not C.DATA_FEATURES.is_file():
        _fail(f"tidak ada {C.DATA_FEATURES} — matriks fitur lokal belum ada")
    df = pd.read_parquet(C.DATA_FEATURES).sort_values("row_id").reset_index(drop=True)
    d = (pd.read_csv(oof)[["row_id", "pred"]]
         .merge(df[["row_id", C.TARGET_CONT]], on="row_id")
         .sort_values("row_id"))
    if len(d) != len(df) or len(d) != 306:
        _fail(f"jumlah baris tidak cocok: fitur {len(df)}, oof {len(d)} (harus 306)")
    y = d[C.TARGET_CONT].to_numpy(float)
    return df, y, d["pred"].to_numpy(float)


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


def sapu(y: np.ndarray, p: np.ndarray, langkah: float, min_kelas: int) -> list[dict]:
    """Grid pasangan ambang; kelas yang terlalu kecil (menurut `y`) dilewati."""
    hasil = []
    for b1 in np.round(np.arange(1.05, 3.45 + 1e-9, langkah), 2):
        for b2 in np.round(np.arange(float(b1) + langkah, 3.50 + 1e-9, langkah), 2):
            if np.bincount(kategori(y, float(b1), float(b2)), minlength=K).min() < min_kelas:
                continue
            hasil.append(ukur(y, p, float(b1), float(b2)))
    return hasil


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langkah", type=float, default=0.05,
                    help="langkah grid ambang (default 0,05 = resolusi data)")
    ap.add_argument("--min-kelas", type=int, default=20,
                    help="ukuran minimum tiap kategori agar pasangan ambang ikut diranking")
    ap.add_argument("--inner-folds", type=int, default=5,
                    help="jumlah lipatan validasi silang di dalam lipatan latih")
    a = ap.parse_args()

    df, y, p_oof = muat()
    grid = sapu(y, p_oof, a.langkah, a.min_kelas)
    if not grid:
        _fail("tidak ada pasangan ambang yang memenuhi syarat ukuran kelas")

    terbaik_qwk = max(grid, key=lambda r: (r["qwk"], r["akurasi"]))
    terbaik_acc = max(grid, key=lambda r: (r["akurasi"], r["qwk"]))
    acuan = ukur(y, p_oof, 1.70, 2.20)
    print(f"[uji_ambang] grid {len(grid)} pasangan (langkah {a.langkah:g}, "
          f"tiap kategori >= {a.min_kelas} siswa)")
    print(f"  QWK terbaik      {terbaik_qwk['b1']:.2f}/{terbaik_qwk['b2']:.2f}: "
          f"QWK {terbaik_qwk['qwk']:.4f} | akurasi {terbaik_qwk['akurasi']:.4f} "
          f"| n aktual {terbaik_qwk['n_aktual']} prediksi {terbaik_qwk['n_prediksi']}")
    print(f"  akurasi terbaik  {terbaik_acc['b1']:.2f}/{terbaik_acc['b2']:.2f}: "
          f"akurasi {terbaik_acc['akurasi']:.4f} | QWK {terbaik_acc['qwk']:.4f} "
          f"| n aktual {terbaik_acc['n_aktual']} prediksi {terbaik_acc['n_prediksi']} "
          "<- skema kolaps, QWK jatuh")
    print(f"  acuan 1,70/2,20  akurasi {acuan['akurasi']:.4f} | QWK {acuan['qwk']:.4f} "
          f"| n aktual {acuan['n_aktual']} prediksi {acuan['n_prediksi']}")
    # --- kandidat tetap yang dibahas di laporan
    kandidat = []
    for b1, b2, nama in [(1.70, 2.20, "dipakai"), (1.75, 2.20, "pembanding"),
                         (1.65, 2.20, "pembanding"), (1.75, 2.50, "gabung skema lama"),
                         (2.00, 3.00, "pita sama lebar")]:
        r = ukur(y, p_oof, b1, b2)
        r["nama"] = nama
        kandidat.append(r)
        print(f"  {nama:18s} {b1:.2f}/{b2:.2f}: akurasi {r['akurasi']:.4f} | "
              f"QWK {r['qwk']:.4f} | n {r['n_aktual']} | "
              f"recall {[round(v, 3) for v in r['recall']]}")

    # --- uji bersarang penuh: ambang dipilih lewat CV di dalam lipatan latih
    params = M.muat_params()
    luaran = M._daftar_fold(df, C.RANDOM_SEED)

    def latih_dan_pred(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
        xt, xv = M._encode(df.iloc[tr], df.iloc[va])
        return M.latih_pred(params, xt, y[tr], M.bobot(y[tr]), xv)

    n = len(y)
    pred_luar = np.full(n, np.nan)          # prediksi tiap siswa dari model yang tak melihatnya
    ambang_baris = np.zeros((n, 2))         # ambang yang dipilih untuk lipatan berisi siswa itu
    acc_lipat, qwk_lipat, pilihan = [], [], []
    for k, (tr, va) in enumerate(luaran, start=1):
        # (1) CV dalam: prediksi tiap baris latih dari model yang tak melihat baris itu
        inner = StratifiedKFold(n_splits=a.inner_folds, shuffle=True,
                                random_state=C.RANDOM_SEED)
        p_in = np.full(len(tr), np.nan, float)
        for itr, iva in inner.split(np.zeros(len(tr)), M._bin(y[tr])):
            p_in[iva] = latih_dan_pred(tr[itr], tr[iva])
        if np.isnan(p_in).any():
            _fail(f"CV dalam lipatan {k} tidak menutup semua baris latih")
        # (2) pilih ambang hanya dari baris latih
        pilih = max(sapu(y[tr], p_in, a.langkah, a.min_kelas),
                    key=lambda r: (r["qwk"], r["akurasi"]))
        # (3) latih di seluruh lipatan latih, nilai di lipatan uji
        pred_luar[va] = latih_dan_pred(tr, va)
        ambang_baris[va] = (pilih["b1"], pilih["b2"])
        r = ukur(y[va], pred_luar[va], pilih["b1"], pilih["b2"])
        acc_lipat.append(r["akurasi"])
        qwk_lipat.append(r["qwk"])
        pilihan.append([pilih["b1"], pilih["b2"]])
        print(f"  lipatan {k:>2}/{len(luaran)}: ambang {pilih['b1']:.2f}/{pilih['b2']:.2f} "
              f"-> akurasi {r['akurasi']:.4f} | QWK {r['qwk']:.4f}", flush=True)

    # (4) gabungan: kategori aktual dikunci di 1,70/2,20, prediksi memakai ambang lipatannya
    ka = kategori(y, 1.70, 2.20)
    kp = np.array([kategori([pred_luar[i]], *ambang_baris[i])[0] for i in range(n)])
    gabung = {
        "akurasi": float((ka == kp).mean()),
        "qwk": float(cohen_kappa_score(ka, kp, weights="quadratic")),
        "aktual_dikunci_di": [1.70, 2.20],
    }
    tersering = Counter(tuple(v) for v in pilihan)
    bersarang = {
        "n_lipatan": len(acc_lipat),
        "inner_folds": a.inner_folds,
        "definisi": {
            "rata_rata_per_lipatan": ("QWK rata-rata 30 lipatan; nilai aktual di lipatan uji "
                                      "dikategorikan dengan ambang lipatan itu sendiri"),
            "gabungan_306": ("QWK satu tabel 306 siswa; kategori aktual dikunci di 1,70/2,20, "
                             "prediksi memakai ambang lipatannya (sebanding dengan acuan)"),
        },
        "rata_rata_per_lipatan": {
            "akurasi": float(np.mean(acc_lipat)),
            "qwk": float(np.mean(qwk_lipat)),
        },
        "gabungan_306": gabung,
        "ambang_tersering": [[list(kk), v] for kk, v in tersering.most_common(3)],
    }
    print(f"\n[uji_ambang] uji bersarang penuh ({len(acc_lipat)} lipatan x "
          f"{a.inner_folds} lipatan dalam):")
    print(f"  rata-rata per lipatan : akurasi {np.mean(acc_lipat):.4f} | "
          f"QWK {np.mean(qwk_lipat):.4f} | tersering "
          f"{bersarang['ambang_tersering']}")
    print(f"  gabungan 306          : akurasi {gabung['akurasi']:.4f} | "
          f"QWK {gabung['qwk']:.4f} (aktual dikunci 1,70/2,20)")
    print(f"  acuan ambang tetap    : akurasi {acuan['akurasi']:.4f} | "
          f"QWK {acuan['qwk']:.4f}")

    # --- pembanding: skema empat tingkat lama
    t = [1.75, 2.50, 3.25]
    k4a, k4p = np.searchsorted(t, y, side="left"), np.searchsorted(t, p_oof, side="left")
    empat = {"akurasi": float((k4a == k4p).mean()),
             "qwk": float(cohen_kappa_score(k4a, k4p, weights="quadratic")),
             "n_aktual": [int((k4a == i).sum()) for i in range(4)]}
    print(f"  pembanding 4 tingkat 1,75/2,50/3,25: akurasi {empat['akurasi']:.4f} | "
          f"QWK {empat['qwk']:.4f} | n {empat['n_aktual']}")

    hasil = {
        "versi": "uji_ambang_v2",
        "masukan": {"prediksi": "outputs/model/oof.csv",
                    "target": "data/features.parquet (y_kontinu)",
                    "n_baris": int(n)},
        "grid": {"n_pasangan": len(grid), "langkah": a.langkah, "min_kelas": a.min_kelas,
                 "terbaik_qwk": terbaik_qwk, "terbaik_akurasi": terbaik_acc},
        "acuan_ambang_tetap": acuan,
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

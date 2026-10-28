"""Bandingkan model dengan regresi 9 faktor Bu Rita memakai protokol yang sama.

Spesifikasi di dokumennya (`2. Analisis Regresi Berganda.docx`): 9 prediktor —
6 faktor tekanan (akademik, keluarga, sosial, ekonomi, digital, masa depan),
coping, perilaku sehat, dan dukungan lingkungan — dengan R2 0,398 in-sample
pada N=306 dan F(9,296)=21,758.

Dua hal yang tidak dirinci di dokumennya: skor `coping` dan `dukungan
lingkungan` mana yang dipakai. Karena itu di sini diuji 16 kombinasi wajar
(3 subskala coping atau gabungannya) x (3 subskala dukungan atau gabungannya),
masing-masing:

* in-sample: R2, F, standard error of estimate, koefisien B;
* protokol sama dengan model: 30 lipatan (RepeatedStratifiedKFold 5x6 pada bin
  skor, seed 42), prediksi tiap ulangan dirata-ratakan seperti bag model, lalu
  MAE / RMSE / R2 dihitung pada 306 siswa.

Baris yang dipakai untuk perbandingan dipilih dengan aturan eksplisit: **MAE CV
terendah** — yaitu kombinasi yang paling menguntungkan regresi, sehingga
perbandingan tidak memihak model.

Catatan penting: tidak ada kombinasi yang mereproduksi R2 0,398 (rentang
0,47-0,59). Angka 0,398 karena itu diperlakukan sebagai angka dokumen, dan yang
dibandingkan adalah replika terdekat yang bisa dihitung dari data ini.

Keluaran: `outputs/model/perbandingan_regresi.json`.

Pakai: python src/bandingkan_regresi.py [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import config as C

N_SPLITS, N_REPEATS = 5, 6
TEKANAN = ["f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital", "f_masadepan"]
PILIHAN_COPING = {
    "f_coping_adaptif": "subskala adaptif (COPING1-12)",
    "f_coping_ekspresi": "subskala ekspresi (COPING13-15)",
    "f_coping_negatif": "subskala negatif (COPING16-18)",
    "coping_all": "gabungan (COPING1-18)",
}
PILIHAN_DUKUNGAN = {
    "f_duk_keluarga": "subskala keluarga (LINGKUNGAN1-4)",
    "f_duk_teman": "subskala teman (LINGKUNGAN5-8)",
    "f_duk_orang_dekat": "subskala orang dekat (LINGKUNGAN9-12)",
    "dukungan_all": "gabungan (LINGKUNGAN1-12)",
}
R2_DOKUMEN, SE_DOKUMEN = 0.398, 0.541


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(a, y, rcond=None)[0]


def in_sample(x: np.ndarray, y: np.ndarray) -> dict:
    b = ols(x, y)
    p = np.column_stack([np.ones(len(x)), x]) @ b
    n, k = len(y), x.shape[1]
    sisa = float(((y - p) ** 2).sum())
    r2 = _r2(y, p)
    return {
        "r2": r2,
        "f": float((r2 / k) / ((1 - r2) / (n - k - 1))),
        "se": float(np.sqrt(sisa / (n - k - 1))),
        "mae": float(np.abs(y - p).mean()),
        "koefisien_B": [round(float(v), 4) for v in b],
    }


def cv(x: np.ndarray, y: np.ndarray, bin_: np.ndarray, seed: int = C.RANDOM_SEED) -> dict:
    """Protokol model: 30 lipatan, prediksi 6 ulangan dirata-ratakan per siswa."""
    kf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    akum = np.zeros((N_REPEATS, len(y)))
    isi = np.zeros((N_REPEATS, len(y)), dtype=bool)
    for i, (tr, va) in enumerate(kf.split(x, bin_)):
        b = ols(x[tr], y[tr])
        akum[i // N_SPLITS, va] = np.column_stack([np.ones(len(va)), x[va]]) @ b
        isi[i // N_SPLITS, va] = True
    if not isi.all():
        _fail("ada siswa yang tidak pernah masuk data uji pada lipatan CV")
    p = akum.mean(axis=0)
    return {
        "mae": float(np.abs(y - p).mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "r2": _r2(y, p),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default=str(C.ROOT / "outputs" / "model" / "perbandingan_regresi.json"))
    a = p.parse_args()
    warnings.filterwarnings("ignore", message="The least populated class")

    if not C.DATA_FEATURES.is_file():
        _fail(f"tidak ada {C.DATA_FEATURES} — jalankan src/build_features.py lebih dulu")
    df = pd.read_parquet(C.DATA_FEATURES).sort_values("row_id").reset_index(drop=True)
    df["coping_all"] = df[[f"COPING{i}" for i in range(1, 19)]].mean(axis=1)
    df["dukungan_all"] = df[[f"LINGKUNGAN{i}" for i in range(1, 13)]].mean(axis=1)
    y = df[C.TARGET_CONT].to_numpy(float)
    bin_ = np.clip(np.digitize(y, [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]), 1, 10)

    baris = []
    for (kop, ket_kop), (duk, ket_duk) in product(PILIHAN_COPING.items(), PILIHAN_DUKUNGAN.items()):
        kolom = TEKANAN + ["f_perilaku_sehat", kop, duk]
        x = df[kolom].to_numpy(float)
        r = {"prediktor": kolom, "coping": ket_kop, "dukungan": ket_duk,
             "in_sample": in_sample(x, y), "cv": cv(x, y, bin_)}
        baris.append(r)
        print(f"  {ket_kop:<28} {ket_duk:<32} R2_in {r['in_sample']['r2']:.3f} | "
              f"CV MAE {r['cv']['mae']:.4f} R2 {r['cv']['r2']:.4f}")

    terbaik = min(baris, key=lambda r: r["cv"]["mae"])
    hasil = {
        "versi": "bandingkan_regresi_v1",
        "catatan": (
            "Regresi 9 faktor dengan protokol yang sama seperti model. Spesifikasi dokumen Bu Rita "
            "tidak merinci skor coping dan dukungan lingkungan, jadi 16 kombinasi diuji dan baris "
            "dengan MAE CV terendah dipakai sebagai pembanding (paling menguntungkan regresi)."),
        "spesifikasi_dokumen": {
            "n_prediktor": 9,
            "r2_in_sample": R2_DOKUMEN,
            "standard_error_of_estimate": SE_DOKUMEN,
            "reproduksi": (
                "tidak ada dari 16 kombinasi yang mereproduksi R2 0,398 maupun SE 0,541; rentang "
                "R2 in-sample di bawah ini adalah replika dari data yang sama, bukan angka dokumen"),
        },
        "protokol": {
            "lipatan": f"RepeatedStratifiedKFold {N_SPLITS}x{N_REPEATS} pada bin skor, "
                       f"seed {C.RANDOM_SEED}",
            "agregasi": "prediksi 6 ulangan dirata-ratakan per siswa, seperti bag model",
            "n_baris": int(len(y)),
        },
        "kandidat": baris,
        "dipakai": {
            "aturan": "MAE CV terendah di antara 16 kandidat",
            "catatan": (
                "Memilih yang terbaik dari 16 varian pada lipatan yang sama menguntungkan regresi "
                "(optimistis untuk pihak pembanding), jadi selisih yang dilaporkan bersifat "
                "konservatif terhadap model ini."),
            "coping": terbaik["coping"],
            "dukungan": terbaik["dukungan"],
            "prediktor": terbaik["prediktor"],
            "in_sample": terbaik["in_sample"],
            "cv": terbaik["cv"],
        },
        "pembanding_model": json.loads((C.ROOT / "outputs" / "model" / "cv.json").read_text())["mae"],
    }
    with open(a.output, "w") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n[bandingkan] dipakai: {terbaik['coping']} + {terbaik['dukungan']}")
    print(f"[bandingkan] in-sample R2 {terbaik['in_sample']['r2']:.3f} "
          f"SE {terbaik['in_sample']['se']:.3f} | CV MAE {terbaik['cv']['mae']:.4f} "
          f"RMSE {terbaik['cv']['rmse']:.4f} R2 {terbaik['cv']['r2']:.4f}")
    print(f"[bandingkan] -> {a.output}")
    return 0


def _fail(pesan: str) -> None:
    print(f"[bandingkan] {pesan}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

"""Angka pendukung laporan yang tidak dapat dihitung pembaca dari `cv.json`.

Semua angka yang dilaporkan di laporan tetapi tidak tersimpan di artefak lain
dihitung di sini dan ditulis ke `outputs/model/uji_tambahan.json`:

1. MAE dan QWK per ulangan (model tunggal), rata-rata dan rentangnya;
2. RMSE bag, kemiringan kalibrasi linear, MAE setelah kalibrasi;
3. selang kepercayaan bootstrap untuk MAE, QWK, dan akurasi (2000 resample);
4. selisih MAE berpasangan terhadap regresi 9 faktor + CI95 (10000 resample);
5. empati varian SOSIAL3: baseline, tanpa pembalikan, tanpa komposisi `f_sosial`,
   tanpa komposisi dan tanpa tiga itemnya;
6. statistik regresi pembanding in-sample: R2, adjusted R2, F, p, VIF 9 prediktor.

Butuh data privat (`data/features.parquet`, `data/raw/*.sav`) seperti skrip lain.
Deterministik: seluruh langkah memakai `config.RANDOM_SEED`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import train as M

TUJUAN = Path(__file__).resolve().parent.parent / "outputs" / "model" / "uji_tambahan.json"
PREDIKTOR_REG = ["f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital",
                 "f_masadepan", "f_coping_negatif", "f_perilaku_sehat", "f_duk_teman"]
ITEM_SOSIAL = ["SOSIAL1", "SOSIAL2", "SOSIAL3"]


def kategori(v: np.ndarray) -> np.ndarray:
    """Tiga tingkat: <=1,70 / 1,70-2,20 / >2,20 (nilai tepat di batas masuk bawah)."""
    return np.searchsorted([1.70, 2.20], v, side="left")


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def cv_prediksi(df: pd.DataFrame, y: np.ndarray, params: dict, seed: int,
                buang: list[str] | None = None) -> np.ndarray:
    """Prediksi out-of-fold gabungan 6 ulangan; `buang` mengeluarkan kolom dari daftar fitur.

    `train._encode` memilih kolom dari `config.NUMERIC_FEATURES`, jadi kolom yang
    dibuang harus juga dikeluarkan dari daftar itu selama perhitungan, lalu dipulihkan.
    """
    pakai = df.drop(columns=buang) if buang else df
    asli_num, asli_item = list(C.NUMERIC_FEATURES), list(M.ITEM_COLS)
    if buang:
        C.NUMERIC_FEATURES = [c for c in asli_num if c not in buang]
        M.ITEM_COLS = [c for c in asli_item if c not in buang]
    try:
        folds = M._daftar_fold(df, seed)
        oof = np.full((M.N_REPEATS, len(df)), np.nan)
        for k, (tr, va) in enumerate(folds):
            xt, xv = M._encode(pakai.iloc[tr], pakai.iloc[va])
            oof[k // M.N_SPLITS, va] = M.latih_pred(params, xt, y[tr], M.bobot(y[tr]), xv)
    finally:
        C.NUMERIC_FEATURES, M.ITEM_COLS = asli_num, asli_item
    return oof


def main() -> int:
    df = pd.read_parquet(C.DATA_FEATURES).sort_values("row_id").reset_index(drop=True)
    y = df[C.TARGET_CONT].to_numpy(float)
    params = M.muat_params()
    hasil: dict = {"seed": C.RANDOM_SEED, "n": int(len(df))}

    # 1-2. per ulangan, bag, kalibrasi
    oof = cv_prediksi(df, y, params, C.RANDOM_SEED)
    per = []
    for r in range(M.N_REPEATS):
        p = oof[r]
        per.append({"mae": round(mae(y, p), 4),
                    "qwk": round(float(cohen_kappa_score(kategori(y), kategori(p),
                                                         weights="quadratic")), 4)})
    bag = oof.mean(axis=0)
    kal = LinearRegression().fit(bag.reshape(-1, 1), y)
    hasil["per_ulangan"] = per
    # pembanding sederhana: tebak rata-rata
    hasil["tebak_rata2"] = {"mae": round(mae(y, np.full_like(y, y.mean())), 4)}
    hasil["bag"] = {
        "mae": round(mae(y, bag), 4),
        "rmse": round(float(np.sqrt(((y - bag) ** 2).mean())), 4),
        "r2": round(float(1 - ((y - bag) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "qwk": round(float(cohen_kappa_score(kategori(y), kategori(bag), weights="quadratic")), 4),
        "mae_rata2_satu_model": round(float(np.mean([v["mae"] for v in per])), 4),
        "mae_min_satu_model": min(v["mae"] for v in per),
        "mae_maks_satu_model": max(v["mae"] for v in per),
        "kalibrasi_slope": round(float(kal.coef_[0]), 3),
        "kalibrasi_intercept": round(float(kal.intercept_), 3),
        "mae_setelah_kalibrasi": round(mae(y, kal.predict(bag.reshape(-1, 1))), 4),
        "n_tepat_di_batas_1_70": int((y == 1.70).sum()),
        "n_tepat_di_batas_2_20": int((y == 2.20).sum()),
    }

    # 2b. sensitivitas duplikat: dua baris identik (row_id 53 dan 55) dibuang dari penilaian
    kembar = [53, 55]
    sisa = np.setdiff1d(np.arange(len(y)), kembar)
    terpisah = 0
    for tr, va in M._daftar_fold(df, C.RANDOM_SEED):
        if set(kembar) & set(tr.tolist()) and set(kembar) & set(va.tolist()):
            terpisah += 1
    hasil["duplikat"] = {
        "row_id": kembar,
        "n_lipatan_kembar_terpisah": int(terpisah),
        "n_lipatan": int(len(M._daftar_fold(df, C.RANDOM_SEED))),
        "mae_tanpa_duplikat": round(mae(y[sisa], bag[sisa]), 4),
        "mae_dengan_duplikat": round(mae(y, bag), 4),
        "selisih": round(mae(y, bag) - mae(y[sisa], bag[sisa]), 4),
    }

    # 3. bootstrap MAE/QWK/akurasi
    rng = np.random.default_rng(C.RANDOM_SEED)
    ka, kp = kategori(y), kategori(bag)
    boot = {"mae": [], "qwk": [], "akurasi": []}
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        boot["mae"].append(mae(y[i], bag[i]))
        boot["qwk"].append(float(cohen_kappa_score(ka[i], kp[i], weights="quadratic")))
        boot["akurasi"].append(float((ka[i] == kp[i]).mean()))
    hasil["bootstrap_2000_ci95"] = {
        k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        for k, v in boot.items()
    }

    # 4. regresi pembanding: in-sample, VIF, selisih berpasangan
    X = df[PREDIKTOR_REG].to_numpy(float)
    lr = LinearRegression().fit(X, y)
    n, k = len(y), len(PREDIKTOR_REG)
    r2 = float(lr.score(X, y))
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    F = (r2 / k) / ((1 - r2) / (n - k - 1))
    hasil["regresi"] = {
        "r2_in_sample": round(r2, 4),
        "r2_adjusted": round(float(adj), 4),
        "F": round(float(F), 2),
        "df": [k, n - k - 1],
        "p": float(sps.f.sf(F, k, n - k - 1)),
        "rmse_in_sample": round(float(np.sqrt(((y - lr.predict(X)) ** 2).mean())), 4),
    }
    vif = {}
    for i, nama in enumerate(PREDIKTOR_REG):
        lain = np.delete(X, i, axis=1)
        r2_i = LinearRegression().fit(lain, X[:, i]).score(lain, X[:, i])
        vif[nama] = round(float(1.0 / (1.0 - r2_i)), 2)
    hasil["regresi"]["vif"] = vif

    folds = M._daftar_fold(df, C.RANDOM_SEED)
    oof_reg = np.full((M.N_REPEATS, len(df)), np.nan)
    for kk, (tr, va) in enumerate(folds):
        lr_cv = LinearRegression().fit(X[tr], y[tr])
        oof_reg[kk // M.N_SPLITS, va] = lr_cv.predict(X[va])
    p_reg = oof_reg.mean(axis=0)
    selisih = np.abs(y - bag) - np.abs(y - p_reg)
    rng2 = np.random.default_rng(C.RANDOM_SEED)
    boot_sel = [float(selisih[rng2.integers(0, n, n)].mean()) for _ in range(10000)]
    hasil["regresi"]["cv"] = {
        "mae": round(mae(y, p_reg), 4),
        "rmse": round(float(np.sqrt(((y - p_reg) ** 2).mean())), 4),
        "r2": round(float(1 - ((y - p_reg) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "selisih_mae_model_min_regresi": round(float(mae(y, p_reg) - mae(y, bag)), 4),
        "ci95_selisih_bootstrap_10000": [round(float(np.percentile(boot_sel, 2.5)), 4),
                                         round(float(np.percentile(boot_sel, 97.5)), 4)],
    }

    # 5. varian SOSIAL3
    varian = {}
    for nama, buang in [
        ("tanpa_pembalikan", None),          # ditangani khusus di bawah
        ("tanpa_komposisi_f_sosial", ["f_sosial"]),
        ("tanpa_f_sosial_dan_item", ["f_sosial", *ITEM_SOSIAL]),
    ]:
        if nama == "tanpa_pembalikan":
            continue
        p = cv_prediksi(df, y, params, C.RANDOM_SEED, buang=buang).mean(axis=0)
        varian[nama] = {"mae": round(mae(y, p), 4),
                        "qwk": round(float(cohen_kappa_score(kategori(y), kategori(p),
                                                             weights="quadratic")), 4)}
    # tanpa pembalikan: bangun ulang kolom item + f_sosial dari data mentah
    import pyreadstat
    raw, _ = pyreadstat.read_sav(str(C.DATA_RAW))
    df2 = df.copy()
    mentah = raw["SOSIAL3"].to_numpy(float)
    df2["SOSIAL3"] = mentah
    df2["f_sosial"] = df2[ITEM_SOSIAL].mean(axis=1)
    p = cv_prediksi(df2, y, params, C.RANDOM_SEED).mean(axis=0)
    varian["tanpa_pembalikan"] = {"mae": round(mae(y, p), 4),
                                  "qwk": round(float(cohen_kappa_score(kategori(y), kategori(p),
                                                                       weights="quadratic")), 4)}
    hasil["varian_sosial3"] = varian

    TUJUAN.parent.mkdir(parents=True, exist_ok=True)
    TUJUAN.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n")
    print(f"[uji_tambahan] ditulis ke {TUJUAN}")
    print(f"  MAE per ulangan : {[v['mae'] for v in per]}")
    print(f"  bag             : MAE {hasil['bag']['mae']} | RMSE {hasil['bag']['rmse']} | "
          f"QWK {hasil['bag']['qwk']}")
    print(f"  CI QWK          : {hasil['bootstrap_2000_ci95']['qwk']}")
    print(f"  tebak rata-rata : MAE {hasil['tebak_rata2']['mae']}")
    print(f"  duplikat        : {hasil['duplikat']}")
    print(f"  varian SOSIAL3  : {varian}")
    print(f"  regresi         : F({k},{n-k-1}) {hasil['regresi']['F']} | adj R2 "
          f"{hasil['regresi']['r2_adjusted']} | VIF maks {max(vif.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

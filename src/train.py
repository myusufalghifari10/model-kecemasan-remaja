"""Model risiko kecemasan remaja: PLS1 sebagai tulang punggung linier + XGBoost pada residual.

Konstruksi:
  * Base = StandardScaler + PLSRegression(1 komponen), DI-FIT DI DALAM LIPATAN pada data
    latih lipatan saja. Satu komponen PLS adalah satu arah yang memaksimalkan kovarians
    dengan y: poros "distres umum" yang stabil pada 306 siswa dengan 88 fitur Likert yang
    saling berkorelasi. Dengan begitu pohon hanya perlu mempelajari SIMPANGAN dari tulang
    linier itu, bukan memprediksi y dari nol.
  * XGBoost (objective MAE, bobot per bin skor) dilatih pada residual y - base.
    Prediksi akhir = base + XGB.

PEMILIHAN KOMPONEN: jumlah komponen PLS (1) dipilih dari 5 kandidat
{PLS1, PLS2, PLS3, ridge alpha 300, ridge alpha 1000} memakai CV di 306 siswa, bukan data
uji. Konstanta PLS_COMPONENTS menyimpan pilihan itu; tidak dicari ulang di sini.

DUA ANGKA YANG TIDAK BOLEH DICAMPUR:
  * --mode cv   : memakai SELURUH 306 siswa (RepeatedStratifiedKFold 5x6, stratifikasi per
                  bin). Karena himpunan evaluasi = populasi, angkanya estimasi POPULASI;
                  68 siswa uji ikut di dalam CV, jadi ini BUKAN angka uji.
  * --mode test : melatih hanya di 238 siswa split v1 lalu menguji di 68 siswa. Angka ini
                  dilaporkan sekali sebagai konfirmasi dan tidak dipakai memilih apa pun.

Mode:
  --mode check-split : verifikasi split terkunci (sha 1d5f63efb7d61488, 238/68)
  --mode self-test   : uji fungsi murni tanpa menulis berkas
  --mode cv          : CV 5x6 di 306 -> cv.json, oof.csv, per_bin.csv
  --mode test        : latih 238 -> uji 68 -> test_<sha8>.json
  --mode final       : BAG 6 model di 306 -> model_bag.joblib + produksi.json
  --mode predict     : data siswa baru -> skor + kategori (Rendah/Sedang/Tinggi/Sangat
                       Tinggi) + fitur paling berpengaruh (SHAP, selalu)
Tanpa --mode hanya mencetak bantuan (tidak menjalankan apa pun).

Urutan pemakaian:
  python src/train.py --mode check-split
  python src/train.py --mode self-test
  python src/train.py --mode cv
  python src/train.py --mode test
  python src/train.py --mode final
  python src/train.py --mode predict --input siswa_baru.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SEED = C.RANDOM_SEED
N_SPLITS, N_REPEATS = 5, 6
N_FOLD = N_SPLITS * N_REPEATS
N_BOOT = 2000
N_POP = 306
N_BAG = N_REPEATS
PLS_COMPONENTS = 1
# n_jobs DIPAKSA 1. Mesin ini sering memuat llama-server + beberapa proses pi; dengan
# n_jobs=-1 satu fit pada 238 baris terukur 242-332 detik karena rebutan thread, sedang
# dengan 1 thread 0,10 detik. Angka hasil tidak berubah, hanya waktunya.
N_JOBS = 1

SPLIT_JSON = C.DATA_SPLIT
SHA_V1 = "1d5f63efb7d61488"      # 16 karakter pertama sha256_test_row_id split terkunci

OUT_DIR = C.ROOT / "outputs" / "model"
CV_JSON = OUT_DIR / "cv.json"
OOF_CSV = OUT_DIR / "oof.csv"
PER_BIN_CSV = OUT_DIR / "per_bin.csv"
PRODUKSI_JSON = OUT_DIR / "produksi.json"
MODEL_JOBLIB = OUT_DIR / "model_bag.joblib"
PREDIKSI_CSV = OUT_DIR / "prediksi.csv"

# Batas kategori mengikuti skema pada analisis sebelumnya: Rendah <= 1,75;
# Sedang 1,75-2,5; Tinggi 2,5-3,25; Sangat Tinggi > 3,25. Karena itu "Tinggi"
# (kemungkinan stress) mulai DI ATAS 2,5, bukan dari 2,5.
# Empat kategori kecemasan: angkanya TIDAK tertulis di dokumen analisis sebelumnya, hanya
# jumlah per kategori (115/153/35/3 dari 306). Ambang di bawah adalah satu-satunya potongan
# berspasi sama pada skala 1-4 yang mereproduksi jumlah itu persis, jadi statusnya
# rekonstruksi, bukan kutipan.
KATEGORI_AMBANG = (1.75, 2.50, 3.25)
KATEGORI_LABEL = ("Rendah", "Sedang", "Tinggi", "Sangat Tinggi")
N_PENJELASAN = 5      # jumlah fitur teratas yang dilaporkan per siswa

DISCLOSURE = {
    "cv_bukan_test": (
        "--mode cv memakai SELURUH 306 siswa, jadi 68 siswa uji ikut di dalam CV. Angka "
        "itu estimasi populasi, BUKAN angka uji."),
    "test_konfirmasi": (
        "Split v1 (68 siswa) dipakai sekali sebagai konfirmasi pada --mode test, tidak "
        "dipakai memilih model, base, atau parameter apa pun."),
    "seleksi_base": (
        "Jumlah komponen PLS dipilih dari 5 kandidat {PLS1, PLS2, PLS3, ridge300, "
        "ridge1000} memakai CV di 306 siswa, bukan data uji."),
    "bin_tipis": (
        "Bin 10 hanya 3 siswa (bin 8: 9, bin 9: 6). Tabel per bin dilaporkan dengan flag "
        "n<=5 dan bin tersebut tidak boleh diranking atau diklaim."),
    "privacy": (
        "oof.csv memuat y per siswa dan bersifat lokal: jangan di-commit. Pola .gitignore "
        "yang dipakai: outputs/model/oof.csv"),
}


def _fail(msg: str) -> None:
    raise SystemExit(f"[model] {msg}")


# ---------------------------------------------------------------- konstanta model
# Semua helper ada di berkas ini supaya bisa dibaca dan dijalankan tanpa modul lain.
# Semantiknya tidak diubah; jangan "merapikan" tanpa menguji ulang angka 0,2165.

BIN_EDGES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
WEIGHT_CAP = 3.0
OBJECTIVE = "reg:absoluteerror"

DATA_FEATURES = C.DATA_FEATURES
FACTOR_COLS = [c for cols in C.FACTOR_ITEMS.values() for c in cols]
ITEM_COLS = FACTOR_COLS + sorted(C.COPING_ITEMS) + C.PERILAKU_ITEMS + C.LINGKUNGAN_ITEMS


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _encode(train: pd.DataFrame, test: pd.DataFrame, fitur: str):
    """Encoder kategorikal di-fit HANYA di `train`; `fitur`="76" berarti item ikut masuk."""
    num = C.NUMERIC_FEATURES + (ITEM_COLS if fitur == "76" else [])
    cat = C.CATEGORICAL_FEATURES
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    xtr = np.hstack([train[num].to_numpy(float), enc.fit_transform(train[cat])])
    xte = np.hstack([test[num].to_numpy(float), enc.transform(test[cat])])
    return xtr, xte


def bobot(y_train: np.ndarray) -> np.ndarray:
    """w = min(CAP, sqrt(ref/n_bin)) per bin; ref = median bin pada y_train; rata-rata 1."""
    y = np.asarray(y_train, float)
    counts, _ = np.histogram(y, bins=BIN_EDGES)
    aktif = counts > 0
    assert aktif.any(), "semua bin kosong di fold latih"
    ref = float(np.median(counts[aktif]))
    w_bin = np.ones(len(counts), float)
    w_bin[aktif] = np.minimum(WEIGHT_CAP, np.sqrt(ref / counts[aktif]))
    idx = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, len(counts) - 1)
    w = w_bin[idx]
    return w / w.mean()


def muat_data() -> pd.DataFrame:
    """Baca matriks 76 fitur (306 baris) lalu pastikan row_id 0..305 berurutan."""
    if not DATA_FEATURES.exists():
        _fail(f"tidak ada {DATA_FEATURES} — matriks fitur lokal belum ada "
              "(arsip data, tidak ikut repo)")
    df = pd.read_parquet(DATA_FEATURES).sort_values("row_id").reset_index(drop=True)
    if len(df) != 306:
        _fail(f"features.parquet harus 306 baris, ada {len(df)}")
    if list(df["row_id"].astype(int)) != list(range(306)):
        _fail("row_id features.parquet tidak 0..305 berurutan")
    return df


def _bin(y: np.ndarray) -> np.ndarray:
    """Bin 1-10 (lebar 0,25) — definisi tetap."""
    return np.clip(np.digitize(np.asarray(y, float), BIN_EDGES), 1, 10)


def muat_split(df: pd.DataFrame) -> dict:
    """Baca split terkunci dan verifikasi hash + ukuran + keterpisahan."""
    if not SPLIT_JSON.exists():
        _fail(f"split terkunci tidak ada: {SPLIT_JSON} — jalankan src/split_universal.py --tulis")
    d = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    test_ids = sorted(int(r) for r in d["test_row_id"])
    train_ids = sorted(int(r) for r in d["train_row_id"])

    sha_ids = hashlib.sha256(json.dumps(test_ids).encode()).hexdigest()
    if sha_ids != d["sha256_test_row_id"]:
        _fail("sha256_test_row_id tidak cocok dengan daftar row_id — file split diubah manual")

    y_map = dict(zip(df["row_id"].astype(int), df[C.TARGET_CONT].astype(float)))
    hilang = [r for r in test_ids + train_ids if r not in y_map]
    if hilang:
        _fail(f"row_id split tidak ada di features: {hilang[:5]}")
    sha_y = hashlib.sha256(
        json.dumps([round(float(y_map[r]), 6) for r in test_ids]).encode()
    ).hexdigest()
    if sha_y != d["sha256_test_y"]:
        _fail("sha256_test_y tidak cocok — data y berubah sejak split dikunci")

    if len(test_ids) != 68 or len(train_ids) != 238:
        _fail(f"ukuran split salah: test={len(test_ids)} train={len(train_ids)} (harus 68/238)")
    if set(test_ids) & set(train_ids):
        _fail("test dan train beririsan")
    if len(set(test_ids) | set(train_ids)) != 306:
        _fail("gabungan test+train bukan 306 siswa")

    # Bobot populasi dihitung dari per_bin di JSON — angka itu WAJIB cocok dengan isi
    # test_row_id yang sebenarnya, kalau tidak MAE berbobot bisa menyimpang senyap.
    n_test_bin = {b: 0 for b in range(1, 11)}
    for r in test_ids:
        n_test_bin[int(_bin(y_map[r]))] += 1
    for b in range(1, 11):
        n_json = int(d["per_bin"][str(b)]["n_test"])
        if n_json != n_test_bin[b]:
            _fail(f"per_bin bin {b} di split JSON = {n_json}, aktual di test_row_id = {n_test_bin[b]}")
    if sum(int(d["per_bin"][str(b)]["n_test"]) for b in range(1, 11)) != len(test_ids):
        _fail("jumlah per_bin.n_test tidak sama dengan jumlah test_row_id")

    d["_test_ids"] = test_ids
    d["_train_ids"] = train_ids
    d["_sha_ids"] = sha_ids
    return d


def _bobot_populasi(split: dict) -> dict[int, float]:
    """Bobot per bin = proporsi populasi / proporsi test, dinormalkan rata-rata 1.

    Test meng-over-represent bin tipis (keputusan 2026-10-17); tanpa bobot ini
    MAE mentah bias ke atas.
    """
    n_test = split["n_test"]
    w = {}
    for b in range(1, 11):
        pb = split["per_bin"][str(b)]
        w[b] = float((pb["n_populasi"] / 306) / (pb["n_test"] / n_test))
    # rata-rata berbobot porsi test harus 1 (populasi 306, test 68) — normalisasi jaga-jaga
    rata = float(sum(w[b] * split["per_bin"][str(b)]["n_test"] / n_test for b in w))
    if not 0.5 <= rata <= 2.0:
        _fail(f"bobot populasi tidak wajar (rata-rata berbobot {rata:.3f})")
    return {b: w[b] / rata for b in w}


def _mae_berbobot(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, float)
    if w.sum() <= 0:
        _fail("bobot populasi berjumlah 0")
    return float(np.sum(w * np.abs(np.asarray(y, float) - np.asarray(p, float))) / np.sum(w))


def _preprocessor(df_train: pd.DataFrame) -> dict:
    """Encoder kategorikal di-fit di 238 train saja (disimpan untuk deployment)."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df_train[C.CATEGORICAL_FEATURES])
    return {"encoder": enc,
            "num": C.NUMERIC_FEATURES + ITEM_COLS,
            "cat": list(C.CATEGORICAL_FEATURES),
            "fitur": "76"}


def _transform(pre: dict, df: pd.DataFrame) -> np.ndarray:
    return np.hstack([df[pre["num"]].to_numpy(float),
                      pre["encoder"].transform(df[pre["cat"]])])


# ---------------------------------------------------------------- parameter & fold

# Sembilan hyperparameter XGB, ditetapkan lewat pencarian hyperparameter terpisah yang
# memakai lipatan CV pada data latih saja — tidak pernah dari data uji. Disimpan sebagai
# litera supaya berkas ini mandiri dan angkanya tidak bergeser tanpa ketahuan.
PARAM_MEDIAN = {
    "n_estimators": 1282,
    "max_depth": 6,
    "learning_rate": 0.0366866067282647,
    "min_child_weight": 8.520630926540825,
    "subsample": 0.6426773101653227,
    "colsample_bytree": 0.6009128736638583,
    "gamma": 0.041026056939631,
    "reg_alpha": 0.000918535444633,
    "reg_lambda": 0.3362195219220946,
}


def muat_params() -> dict:
    """9 hyperparameter XGB (lihat PARAM_MEDIAN)."""
    p = dict(PARAM_MEDIAN)
    if p.get("max_depth") is None or p.get("n_estimators") is None:
        _fail("parameter median tidak lengkap")
    return p


def _daftar_fold(df: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """30 fold (5x6) terstratifikasi per bin; deterministik untuk seed yang sama.

    Bin 10 hanya 3 siswa sehingga lebih kecil dari n_splits — sklearn memperingatkan itu;
    peringatan itulah yang disenyapkan di sini (bukan peringatan lain).
    """
    b = _bin(df[C.TARGET_CONT].to_numpy(float))
    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        folds = list(rkf.split(df, b))
    if len(folds) != N_FOLD:
        _fail(f"jumlah fold salah: {len(folds)} (harus {N_FOLD})")
    return folds


# ---------------------------------------------------------------- model dua tahap

def latih(params: dict, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Latih base PLS1 + XGB pada residual. Semua fitting hanya pada (x, y, w) ini."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape[0] != len(y) or len(y) != len(w):
        _fail(f"bentuk latih tidak cocok: x {x.shape}, y {len(y)}, w {len(w)}")
    if len(y) < 30:
        _fail(f"baris latih terlalu sedikit ({len(y)}) untuk base PLS")

    scaler = StandardScaler().fit(x)
    pls = PLSRegression(n_components=PLS_COMPONENTS, scale=False).fit(scaler.transform(x), y)
    base_tr = np.asarray(pls.predict(scaler.transform(x)), float).ravel()
    if base_tr.shape != (len(y),):
        _fail(f"bentuk prediksi base salah: {base_tr.shape} (harus ({len(y)},))")
    resid = y - base_tr
    if not np.isfinite(resid).all():
        _fail("residual base mengandung NaN/inf")

    xgb = xgboost.XGBRegressor(**{**C.BASE_XGB, "n_jobs": N_JOBS},
                               objective=OBJECTIVE, **params)
    xgb.fit(x, resid, sample_weight=np.asarray(w, float))
    return {"scaler": scaler, "pls": pls, "xgb": xgb}


def pred(model: dict, x: np.ndarray) -> np.ndarray:
    """pred == base(x) + xgb(x). Dipakai cv, test, final, dan self-test."""
    x = np.asarray(x, float)
    base = np.asarray(model["pls"].predict(model["scaler"].transform(x)), float).ravel()
    sisa = np.asarray(model["xgb"].predict(x), float)
    out = base + sisa
    if out.shape != (x.shape[0],) or not np.isfinite(out).all():
        _fail(f"prediksi tidak wajar: bentuk {out.shape}, finite {np.isfinite(out).all()}")
    return out


def latih_pred(params: dict, xt: np.ndarray, yt: np.ndarray, wt: np.ndarray,
               xv: np.ndarray) -> np.ndarray:
    return pred(latih(params, xt, yt, wt), xv)


# ---------------------------------------------------------------- akumulasi bag

def akumulasi_bag(folds: list[tuple[np.ndarray, np.ndarray]], n: int,
                  pred_fold) -> np.ndarray:
    """Matriks (N_REPEATS, n) prediksi OOF: tiap baris diuji tepat sekali per repeat.

    `pred_fold(tr, va)` mengembalikan prediksi untuk indeks `va`. Ada NaN tersisa berarti
    ada baris yang tidak pernah masuk lipatan uji — itu bug, bukan angka.
    """
    oof = np.full((N_REPEATS, n), np.nan, float)
    for k, (tr, va) in enumerate(folds):
        r = k // N_SPLITS
        if not np.isnan(oof[r, va]).any():
            _fail(f"baris {va[:3].tolist()} diuji dua kali pada repeat {r}")
        p = np.asarray(pred_fold(tr, va), float)
        if p.shape != (len(va),):
            _fail(f"bentuk prediksi fold {k} salah: {p.shape} (harus ({len(va)},))")
        oof[r, va] = p
    if np.isnan(oof).any():
        rusak = np.where(np.isnan(oof).any(axis=0))[0]
        _fail(f"bag OOF tidak menutup semua baris; bermasalah: {rusak[:5].tolist()}")
    return oof


def _ci_mae(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    sim = np.array([_mae(y[i], p[i]) for i in (rng.integers(0, len(y), len(y))
                                               for _ in range(n_boot))])
    return [float(np.percentile(sim, 2.5)), float(np.percentile(sim, 97.5))]


def _metrik_kategori(y: np.ndarray, pred: np.ndarray) -> dict:
    """Akurasi kategori 4 tingkat + tabel silang + rincian per kategori."""
    ka = _kategori(y)
    ks = _kategori(pred)
    pos = {k: i for i, k in enumerate(KATEGORI_LABEL)}
    ia = np.array([pos[v] for v in ka])
    ib = np.array([pos[v] for v in ks])
    selisih = np.abs(ia - ib)
    silang = {k: {k2: int(((ka == k) & (ks == k2)).sum()) for k2 in KATEGORI_LABEL}
              for k in KATEGORI_LABEL}
    rinci = []
    for k in KATEGORI_LABEL:
        m = ka == k
        n = int(m.sum())
        if n == 0:
            continue
        rinci.append({
            "kategori": k, "n": n,
            "tepat": int((ks[m] == k).sum()),
            "recall": float((ks[m] == k).mean()),
            "bias": float((pred[m] - y[m]).mean()),
            "mae": float(np.abs(pred[m] - y[m]).mean()),
        })
    return {
        "akurasi": float((ia == ib).mean()),
        "qwk": float(cohen_kappa_score(ia, ib, weights="quadratic")),
        "meleset_1_tingkat": int((selisih == 1).sum()),
        "meleset_2_tingkat_atau_lebih": int((selisih >= 2).sum()),
        "tabel_silang": silang,
        "per_kategori": rinci,
    }


def _metrik_skrining(y: np.ndarray, pred: np.ndarray) -> dict:
    """Pandangan biner: 'Tinggi atau lebih' (skor > 2,50) versus sisanya."""
    t = KATEGORI_AMBANG[1]
    ya = (y > t).astype(int)
    pa = (pred > t).astype(int)
    tp = int(((pa == 1) & (ya == 1)).sum())
    fn = int(((pa == 0) & (ya == 1)).sum())
    fp = int(((pa == 1) & (ya == 0)).sum())
    return {
        "ambang": float(t),
        "positif_aktual": int(ya.sum()),
        "ditandai": int(pa.sum()),
        "auc": float(roc_auc_score(ya, pred)),
        "sensitivitas": float(tp / (tp + fn)) if tp + fn else None,
        "spesifisitas": float(1 - fp / int((ya == 0).sum())),
        "ppv": float(tp / (tp + fp)) if tp + fp else None,
    }


def _per_bin(y: np.ndarray, p: np.ndarray, b: np.ndarray) -> list[dict]:
    out = []
    for k in range(1, 11):
        m = b == k
        if not m.any():
            out.append({"bin": k, "n": 0, "boleh_diranking": False,
                        "catatan": "tidak ada baris"})
            continue
        out.append({"bin": k, "n": int(m.sum()),
                    "mae": float(_mae(y[m], p[m])),
                    "bias": float((y[m] - p[m]).mean()),
                    "y_mean": float(y[m].mean()),
                    "pred_mean": float(p[m].mean()),
                    "boleh_diranking": bool(m.sum() > 5)})
    return out


# ---------------------------------------------------------------- check-split

def run_check_split(df: pd.DataFrame) -> None:
    """Verifikasi split terkunci; gagal keras bila bergeser."""
    if len(df) != N_POP:
        _fail(f"data harus {N_POP} baris, ada {len(df)}")
    if not SPLIT_JSON.exists():
        _fail(f"{SPLIT_JSON} tidak ada — split terkunci hilang")
    sp = muat_split(df)             # memverifikasi sha, 238/68, dan keterpisahan
    sha = str(sp["_sha_ids"])
    if sha[:len(SHA_V1)] != SHA_V1:
        _fail(f"sha split berubah: {sha[:16]} (harus {SHA_V1})")
    y = df[C.TARGET_CONT].to_numpy(float)
    print(f"[model] split v1 utuh — sha {sha[:16]} | train {len(sp['_train_ids'])} / "
          f"test {len(sp['_test_ids'])}")
    print(f"[model] data {len(df)} siswa | y {y.min():.2f}..{y.max():.2f} "
          f"rata-rata {y.mean():.3f}")
    print(f"[model] base = StandardScaler + PLS({PLS_COMPONENTS}) lalu XGB pada residual")


# ---------------------------------------------------------------- cv

def run_cv(df: pd.DataFrame, seed: int) -> None:
    """CV 5x6 di seluruh 306 siswa: estimasi populasi (MAE, R2, tabel per bin)."""
    params = muat_params()
    y = df[C.TARGET_CONT].to_numpy(float)
    b = _bin(y)
    n = len(df)
    folds = _daftar_fold(df, seed)
    print(f"[model] CV {N_SPLITS}x{N_REPEATS} di {n} siswa (seed {seed}) — "
          f"base PLS{PLS_COMPONENTS} + XGB residual", flush=True)

    t0 = time.time()

    def fp_model(tr, va):
        # Encoder di-fit hanya pada data latih lipatan (lihat _encode).
        xt, xv = _encode(df.iloc[tr], df.iloc[va], "76")
        return latih_pred(params, xt, y[tr], bobot(y[tr]), xv)

    oof = akumulasi_bag(folds, n, fp_model)
    print(f"  selesai ({time.time() - t0:.0f}s)", flush=True)

    pooled = oof.mean(axis=0)
    mae_model = float(_mae(y, pooled))
    r2 = float(1 - np.sum((y - pooled) ** 2) / np.sum((y - y.mean()) ** 2))
    tabel_bin = _per_bin(y, pooled, b)

    ringkas = {
        "versi": "cv_v1",
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "model": {"nama": "PLS1 + XGB residual", "pls_components": PLS_COMPONENTS,
                  "objective": OBJECTIVE, "parameter": params,
                  "sumber_parameter": "PARAM_MEDIAN (litera di berkas ini)"},
        "seed": int(seed),
        "n_baris": int(n),
        "n_fold": int(len(folds)),
        "metode": "RepeatedStratifiedKFold 5x6 di seluruh 306; bag = rata-rata 6 repeat",
        "mae": mae_model,
        "mae_ci95": _ci_mae(y, pooled),
        "rmse": float(np.sqrt(np.mean((y - pooled) ** 2))),
        "r2": r2,
        "metrik_kategori": _metrik_kategori(y, pooled),
        "metrik_skrining": _metrik_skrining(y, pooled),
        "per_bin": tabel_bin,
        "catatan_metrik": (
            "Himpunan eval = populasi 306, jadi MAE mentah di sini sudah sama dengan MAE "
            "berbobot populasi. Angka ini BUKAN angka test (68 siswa ikut di dalam CV)."),
        "elapsed_detik": round(time.time() - t0, 1),
        "disclosure": DISCLOSURE,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": df["row_id"].astype(int), "y": y, "bin": b,
                  "pred": pooled}).to_csv(OOF_CSV, index=False)
    pd.DataFrame(tabel_bin).to_csv(PER_BIN_CSV, index=False)
    CV_JSON.write_text(json.dumps(ringkas, indent=2), encoding="utf-8")

    print(f"\n[model] MAE populasi 306 : {mae_model:.4f} "
          f"CI95 [{ringkas['mae_ci95'][0]:.3f}, {ringkas['mae_ci95'][1]:.3f}] | R2 {r2:.4f}")
    for r in tabel_bin:
        if r["n"]:
            flag = "" if r["boleh_diranking"] else "  [n<=5: tidak boleh diranking]"
            print(f"     bin {r['bin']:>2} n={r['n']:>3} mae {r['mae']:.3f} "
                  f"bias {r['bias']:+.3f}{flag}")


# ---------------------------------------------------------------- test (238 -> 68)

def run_test(df: pd.DataFrame, force: bool = False) -> None:
    """Satu-satunya tempat model menyentuh 68 siswa test. Menolak jalan dua kali."""
    lama = sorted(OUT_DIR.glob("test_*.json"))
    if lama and not force:
        _fail(f"{lama[0].name} sudah ada — evaluasi test hanya boleh sekali; "
              "pakai --force hanya bila run sebelumnya gagal/rusak")

    params = muat_params()
    sp = muat_split(df)
    if str(sp["_sha_ids"])[:len(SHA_V1)] != SHA_V1:
        _fail("sha split berubah — jangan evaluasi test")
    train = df[df["row_id"].isin(sp["_train_ids"])].reset_index(drop=True)
    test = df[df["row_id"].isin(sp["_test_ids"])].reset_index(drop=True)
    if len(train) != 238 or len(test) != 68:
        _fail(f"ukuran split salah: train {len(train)} / test {len(test)}")

    ytr = train[C.TARGET_CONT].to_numpy(float)
    yte = test[C.TARGET_CONT].to_numpy(float)
    bte = _bin(yte)
    xt, xv = _encode(train, test, "76")      # encoder hanya di-fit di 238
    p = latih_pred(params, xt, ytr, bobot(ytr), xv)

    w_bin = _bobot_populasi(sp)
    w_baris = np.array([w_bin[int(k)] for k in bte], float)
    mae_raw = float(_mae(yte, p))
    mae_w = _mae_berbobot(yte, p, w_baris)

    hasil = {
        "versi": "test_v1",
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "model": {"nama": "PLS1 + XGB residual", "pls_components": PLS_COMPONENTS,
                  "parameter": params, "objective": OBJECTIVE},
        "split": {"sumber": str(SPLIT_JSON.relative_to(C.ROOT)),
                  "sha256_test_row_id": str(sp["_sha_ids"]),
                  "n_train": int(len(train)), "n_test": int(len(test))},
        "n_fitur": int(xt.shape[1]),
        "mae_raw": mae_raw,
        "mae_berbobot": mae_w,
        "mae_ci95_raw": _ci_mae(yte, p),
        "rmse": float(np.sqrt(np.mean((yte - p) ** 2))),
        "r2": float(1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)),
        "metrik_kategori": _metrik_kategori(yte, p),
        "metrik_skrining": _metrik_skrining(yte, p),
        "per_bin": _per_bin(yte, p, bte),
        "disclosure": DISCLOSURE,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"test_{str(sp['_sha_ids'])[:8]}.json"
    path.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print(f"\n[model] TEST 238 -> 68: MAE mentah {mae_raw:.4f} | berbobot populasi {mae_w:.4f}")
    print(f"     R2 {hasil['r2']:.4f}")
    print(f"     ditulis ke {path.relative_to(C.ROOT)}")


# ---------------------------------------------------------------- final (bag di 306)

def run_final(df: pd.DataFrame, force: bool = False) -> None:
    if MODEL_JOBLIB.exists() and not force:
        _fail(f"{MODEL_JOBLIB.name} sudah ada — pakai --force hanya bila run sebelumnya "
              "gagal/rusak")
    if not CV_JSON.exists():
        _fail(f"{CV_JSON.name} belum ada — jalankan --mode cv dulu (artefak harus "
              "membawa angka CV yang sah)")
    cv = json.loads(CV_JSON.read_text(encoding="utf-8"))
    params = muat_params()
    y = df[C.TARGET_CONT].to_numpy(float)
    pre = _preprocessor(df)
    x = _transform(pre, df)
    x_cek, _ = _encode(df, df, "76")
    if x.shape != x_cek.shape or not np.allclose(x, x_cek):
        _fail("preprocessor tersimpan tidak identik dengan _encode — jangan lanjut")

    # BAG: satu model per repeat, masing-masing dilatih pada 5/6 siswa dari pengacakan
    # berbeda, lalu prediksi dirata-rata. Konstruksi ini sama dengan bag di --mode cv.
    folds = _daftar_fold(df, SEED)
    models = []
    for r in range(N_BAG):
        tr = folds[r * N_SPLITS + 0][0]
        models.append(latih(params, x[tr], y[tr], bobot(y[tr])))
    pred_bag = np.mean(np.vstack([pred(m, x) for m in models]), axis=0)
    if not np.isfinite(pred_bag).all():
        _fail("prediksi bag mengandung NaN/inf")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "preprocessor": pre, "parameter": params,
                 "pls_components": int(PLS_COMPONENTS), "n_bag": int(N_BAG),
                 "seed": int(SEED), "versi": "bag_v1"}, MODEL_JOBLIB)

    simpan = joblib.load(MODEL_JOBLIB)
    if len(simpan["models"]) != N_BAG or simpan["pls_components"] != PLS_COMPONENTS:
        _fail("isi joblib tidak sesuai")
    pred_muat = np.mean(np.vstack([pred(m, x) for m in simpan["models"]]), axis=0)
    selisih = float(np.max(np.abs(pred_muat - pred_bag)))
    if selisih > 1e-9:
        _fail(f"round-trip joblib meleset {selisih:.2e} > 1e-9")

    hasil = {
        "versi": "produksi_v1",
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "model": {"nama": "PLS1 + XGB residual", "pls_components": int(PLS_COMPONENTS),
                  "parameter": params, "objective": OBJECTIVE, "n_bag": int(N_BAG)},
        "artefak": MODEL_JOBLIB.name,
        "n_baris": int(len(df)),
        "n_fitur": int(x.shape[1]),
        "kategori_ambang": list(KATEGORI_AMBANG),
        "kategori_label": list(KATEGORI_LABEL),
        "mae_cv_bag": cv["mae"],
        "mae_cv_ci95": cv["mae_ci95"],
        "prediksi": {"in_sample": True,
                     "catatan": ("prediksi bag pada 306 baris yang SAMA dengan data latih; "
                                 "estimasi out-of-sample = mae_cv_bag"),
                     "mean": float(pred_bag.mean()), "sd": float(pred_bag.std()),
                     "min": float(pred_bag.min()), "max": float(pred_bag.max())},
        "round_trip_selisih_maks": selisih,
        "versi_paket": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__},
        "disclosure": DISCLOSURE,
    }
    PRODUKSI_JSON.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print(f"\n[model] model produksi: BAG {N_BAG} model, {len(df)} siswa x {x.shape[1]} fitur")
    print(f"     MAE CV (populasi 306) {cv['mae']:.4f} | R2 {cv['r2']:.4f}")
    print(f"     round-trip joblib {selisih:.1e} | {MODEL_JOBLIB.relative_to(C.ROOT)}")


# ---------------------------------------------------------------- predict (data baru)

def _kontribusi(m: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kontribusi tiap fitur terhadap prediksi satu model: PLS (linier) + TreeSHAP XGB.

    pred = base + xgb, dan keduanya aditif per fitur:
      base  = intercept_PLS + sum_j koef_j * z_j        (z = fitur terstandardisasi)
      xgb   = bias_TreeSHAP + sum_j shap_j
    Jadi kontribusi = koef_j * z_j + shap_j, dan total + bias = prediksi persis.
    """
    z = m["scaler"].transform(x)
    linier = z * np.asarray(m["pls"].coef_, float).ravel()
    shap = m["xgb"].get_booster().predict(xgboost.DMatrix(x), pred_contribs=True)
    return linier + shap[:, :-1], float(np.asarray(m["pls"].intercept_).ravel()[0]) + shap[:, -1]


def _kategori(skor: np.ndarray) -> np.ndarray:
    """Rendah / Sedang / Tinggi / Sangat Tinggi dari skor (skala 1-4)."""
    return np.array(KATEGORI_LABEL, dtype=object)[np.searchsorted(KATEGORI_AMBANG, skor, side="left")]


def _siapkan_input(pre: dict, path_input: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Baca dan validasi data siswa baru, kembalikan (tabel asli, matriks 88 kolom)."""
    if not path_input.exists():
        _fail(f"berkas input tidak ada: {path_input}")
    baru = (pd.read_parquet(path_input) if path_input.suffix.lower() == ".parquet"
            else pd.read_csv(path_input))
    if len(baru) == 0:
        _fail("input kosong")
    wajib = pre["num"] + pre["cat"]
    kurang = [c for c in wajib if c not in baru.columns]
    if kurang:
        _fail(f"{len(kurang)} kolom wajib tidak ada di input, mis. {kurang[:5]}")
    kosong = [c for c in wajib if baru[c].isna().any()]
    if kosong:
        _fail(f"ada nilai kosong di kolom: {kosong[:5]}")
    return baru, _transform(pre, baru)


def run_predict(path_input: Path, path_output: Path | None = None,
                top: int = N_PENJELASAN) -> None:
    """Skor + label kemungkinan stress + alasan per siswa, dalam satu keluaran.

    Alasan memakai kontribusi fitur yang aditif (PLS linier + TreeSHAP XGB), jadi
    jumlah kontribusi + bias = skor yang dilaporkan; bobotnya dirata-ratakan atas
    6 model bag.
    """
    if not MODEL_JOBLIB.exists():
        _fail(f"{MODEL_JOBLIB.name} belum ada — jalankan --mode final dulu")
    if top < 1:
        _fail("--top minimal 1")

    art = joblib.load(MODEL_JOBLIB)
    pre = art["preprocessor"]
    baru, x = _siapkan_input(pre, path_input)
    names = list(pre["num"]) + list(pre["encoder"].get_feature_names_out(pre["cat"]))

    skor = np.mean(np.vstack([pred(m, x) for m in art["models"]]), axis=0)
    if not np.isfinite(skor).all():
        _fail("prediksi mengandung NaN/inf")

    kontribusi, bias = [], []
    for m in art["models"]:
        k, b = _kontribusi(m, x)
        kontribusi.append(k)
        bias.append(b)
    K = np.mean(np.stack(kontribusi, axis=0), axis=0)      # (6 model, n, 88) -> (n, 88)
    galat = float(np.max(np.abs(K.sum(axis=1) + np.mean(bias) - skor)))
    if galat > 1e-6:
        _fail(f"kontribusi tidak menjumlah ke skor (selisih {galat:.2e})")

    urut = np.argsort(-np.abs(K), axis=1)[:, :top]
    hasil = pd.DataFrame({
        "row_id" if "row_id" in baru.columns else "baris":
            (baru["row_id"].to_numpy() if "row_id" in baru.columns
             else np.arange(1, len(baru) + 1)),
        "skor": np.round(skor, 4),
        "kategori": _kategori(skor),
    })
    for i in range(top):
        hasil[f"fitur_{i + 1}"] = [names[j] for j in urut[:, i]]
        hasil[f"kontribusi_{i + 1}"] = np.round(K[np.arange(len(K)), urut[:, i]], 4)

    out = path_output or PREDIKSI_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    hasil.to_csv(out, index=False)

    kat = _kategori(skor)
    sebaran = pd.Series(kat).value_counts().reindex(KATEGORI_LABEL, fill_value=0)
    print(f"[model] {len(baru)} siswa | bag {len(art['models'])} model | {top} fitur teratas")
    print(f"     skor {skor.min():.3f}..{skor.max():.3f} | rata-rata {skor.mean():.3f}")
    print("     kategori: " + " | ".join(f"{k} {v}" for k, v in sebaran.items()))
    print(f"     jumlah kontribusi = skor (selisih {galat:.1e})")
    for i in range(len(baru)):
        print(f"\n  baris {hasil.iloc[i, 0]} | skor {skor[i]:.3f} | {kat[i]}")
        for j in urut[i]:
            print(f"     {names[j]:<34} {K[i, j]:+.3f}")
    print(f"\n  ditulis ke {out}")


# ---------------------------------------------------------------- self-test

def _self_test() -> None:
    """Uji fungsi murni + check-split. Tidak menulis berkas apa pun."""
    rs = np.random.default_rng(42)
    xs = rs.normal(size=(60, 6))
    ys = xs @ np.array([1.0, -0.5, 0.3, 0.0, 0.2, -0.1]) + rs.normal(scale=0.2, size=60)
    ws = np.ones(60)
    # Tanpa random_state/verbosity: keduanya sudah ada di C.BASE_XGB yang dipakai `latih`.
    params = {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1}

    # (a) bentuk prediksi base (n,) setelah ravel
    m = latih(params, xs, ys, ws)
    base = np.asarray(m["pls"].predict(m["scaler"].transform(xs)), float)
    # sklearn 1.9 mengembalikan (n,) untuk target tunggal; versi lama (n,1). Keduanya sah,
    # yang wajib adalah hasil ravel berbentuk (n,).
    assert base.shape in ((60,), (60, 1)), f"bentuk PLSR tak terduga: {base.shape}"
    assert base.ravel().shape == (60,), "ravel base salah"
    p = pred(m, xs)
    assert p.shape == (60,), f"prediksi harus (60,), dapat {p.shape}"

    # (b) pred == base + xgb
    manual = (base.ravel()
              + np.asarray(m["xgb"].predict(xs), float))
    assert np.allclose(p, manual), "pred != base + xgb"
    # dan base + xgb benar-benar memakai residual (bukan y mentah)
    assert not np.allclose(np.asarray(m["xgb"].predict(xs), float), ys, atol=1e-9), \
        "XGB seharusnya belajar residual, bukan y mentah"

    # (c) bag OOF menutup semua baris tanpa NaN (folds buatan, n kecil)
    #     n=10 dengan 5 lipatan -> tiap lipatan uji 2 baris, latih 8 baris (konstan),
    #     sehingga rata-rata bag per baris pasti 8,0.
    n = 10
    fake = []
    for r in range(N_REPEATS):
        for f in range(N_SPLITS):
            idx = np.arange(n)
            va = idx[idx % N_SPLITS == f]
            tr = idx[idx % N_SPLITS != f]
            fake.append((tr, va))
    oof = akumulasi_bag(fake, n, lambda tr, va: np.full(len(va), float(len(tr))))
    assert oof.shape == (N_REPEATS, n), f"bentuk bag salah: {oof.shape}"
    assert not np.isnan(oof).any(), "bag OOF mengandung NaN"
    assert np.allclose(oof.mean(axis=0), np.full(n, 8.0)), "rata-rata bag salah"

    # (d) determinisme: PLS1 diulang pada data sama -> prediksi identik
    m2 = latih(params, xs, ys, ws)
    assert np.max(np.abs(pred(m, xs) - pred(m2, xs))) < 1e-12, "model tidak deterministik"

    # (e) split terkunci
    df = muat_data()
    run_check_split(df)

    print("SELF-TEST OK: bentuk base (n,), pred == base + xgb, XGB pada residual, "
          "bag OOF menutup semua baris, determinisme PLS1, split sha " + SHA_V1)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Model risiko kecemasan: PLS1 + XGB residual (evaluasi CV populasi 306 + konfirmasi test).")
    ap.add_argument("--mode",
                    choices=["check-split", "self-test", "cv", "test", "final", "predict"],
                    help="mode yang dijalankan; tanpa --mode hanya menampilkan bantuan")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"seed RepeatedStratifiedKFold untuk --mode cv (default {SEED})")
    ap.add_argument("--input", type=Path,
                    help="berkas CSV/parquet data siswa baru (wajib untuk --mode predict)")
    ap.add_argument("--output", type=Path,
                    help="tujuan CSV hasil --mode predict (default outputs/model/prediksi.csv)")
    ap.add_argument("--top", type=int, default=N_PENJELASAN,
                    help=f"jumlah fitur teratas yang dilaporkan (default {N_PENJELASAN})")
    ap.add_argument("--force", action="store_true",
                    help="izinkan --mode test/final menimpa artefak yang sudah ada "
                         "(hanya bila run sebelumnya gagal/rusak)")
    args = ap.parse_args(argv)

    if args.mode is None:
        ap.print_help()
        print("\nMode tersedia: check-split | self-test | cv | test | final | predict\n"
              "Urutan: check-split -> self-test -> cv -> test -> final.\n"
              "cv memakai SELURUH 306 siswa (estimasi populasi, bukan angka test);\n"
              "test memakai split v1 (238 -> 68) yang sudah terpakai dan hanya konfirmasi.")
        return

    if args.mode == "self-test":
        _self_test()
        return

    if args.mode == "predict":
        if args.input is None:
            _fail("--mode predict butuh --input <berkas.csv|.parquet>")
        run_predict(args.input, args.output, top=args.top)
        return

    df = muat_data()
    if args.mode == "check-split":
        run_check_split(df)
    elif args.mode == "cv":
        run_cv(df, args.seed)
    elif args.mode == "test":
        run_test(df, force=args.force)
    elif args.mode == "final":
        run_final(df, force=args.force)


if __name__ == "__main__":
    main()

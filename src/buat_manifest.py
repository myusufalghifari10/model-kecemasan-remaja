"""Tulis manifest: sha256 tiap berkas + angka utama yang dilaporkan.

Gunanya: angka di laporan/README bisa diikat ke berkas yang persis menghasilkannya.
Berkas privat (data mentah, `features.parquet`, `oof.csv`) ikut di-hash bila ada —
sha256 tidak membocorkan isi, jadi orang lain bisa mencocokkan berkas yang mereka
punya dengan yang dipakai di sini.

Keluaran deterministik (tanpa cap waktu): jalankan dua kali, hasilnya identik.
Kalau ada angka yang berubah tanpa manifest ikut berubah, berarti artefaknya
diedit tangan.

Pakai: python src/buat_manifest.py [--output PATH]
       python src/buat_manifest.py --check   # cocokkan berkas sekarang dengan manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import config as C

PUBLIK = [
    "src/train.py", "src/config.py", "src/build_features.py", "src/bandingkan_regresi.py",
    "src/buat_manifest.py", "requirements.txt", "README.md", "docs/codebook.md",
    "docs/Laporan-Model-Kecemasan-Remaja.docx",
    "data/split_v1.json",
    "outputs/model/cv.json", "outputs/model/per_bin.csv",
    "outputs/model/test_1d5f63ef.json", "outputs/model/produksi.json",
    "outputs/model/perbandingan_regresi.json", "outputs/model/model_bag.joblib",
]
PRIVAT = ["data/features.parquet", "outputs/model/oof.csv", "data/raw/INPUT DATA  (1).sav"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def baca(*bagian: str) -> dict:
    return json.loads((C.ROOT.joinpath(*bagian)).read_text())


def periksa() -> int:
    """Cocokkan sha256 berkas sekarang dengan manifest yang tersimpan."""
    simpan = baca("outputs", "model", "manifest.json")
    beda = 0
    for rel, catat in simpan["berkas"].items():
        f = C.ROOT / rel
        if not f.is_file():
            print(f"  HILANG {rel}")
            beda += 1
        elif (h := sha256(f)) != catat["sha256"]:
            print(f"  BEDA   {rel} (manifest {catat['sha256'][:12]}, sekarang {h[:12]})")
            beda += 1
    print(f"[manifest] publik: {len(simpan['berkas']) - beda}/{len(simpan['berkas'])} cocok")
    for rel, catat in simpan.get("privat", {}).items():
        f = C.ROOT / rel
        if not f.is_file():
            print(f"  privat {rel}: tidak ada di mesin ini (dilewati)")
        elif (h := sha256(f)) != catat["sha256"]:
            print(f"  BEDA   {rel} (privat)")
            beda += 1
    return 0 if beda == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default=str(C.ROOT / "outputs" / "model" / "manifest.json"))
    p.add_argument("--check", action="store_true",
                   help="cocokkan berkas sekarang dengan manifest tersimpan, jangan menulis")
    a = p.parse_args()

    if a.check:
        return periksa()

    cv, te, pr = baca("outputs", "model", "cv.json"), baca("outputs", "model", "test_1d5f63ef.json"), \
        baca("outputs", "model", "produksi.json")
    reg = baca("outputs", "model", "perbandingan_regresi.json")

    berkas, hilang = {}, []
    for rel in PUBLIK:
        f = C.ROOT / rel
        if f.is_file():
            berkas[rel] = {"sha256": sha256(f), "byte": f.stat().st_size}
        else:
            hilang.append(rel)
    privat = {}
    for rel in PRIVAT:
        f = C.ROOT / rel
        if f.is_file():
            privat[rel] = {"sha256": sha256(f), "byte": f.stat().st_size}
    if hilang:
        print(f"[manifest] BERKAS HILANG: {hilang}", file=sys.stderr)
        return 1

    hasil = {
        "versi": "manifest_v1",
        "catatan": ("sha256 berkas publik + privat (bila ada) dan angka utama yang dilaporkan. "
                    "Kalau angka berubah, sha256 berkas sumbernya harus ikut berubah."),
        "angka_utama": {
            "cv": {"n_baris": cv["n_baris"], "mae": cv["mae"], "mae_ci95": cv["mae_ci95"],
                   "r2": cv["r2"], "akurasi_kategori": cv["kategori"]["akurasi"],
                   "qwk": cv["kategori"]["qwk"]},
            "test_68": {"mae_raw": te["mae_raw"], "mae_berbobot": te["mae_berbobot"],
                        "r2": te["r2"], "akurasi_kategori": te["kategori"]["akurasi"],
                        "qwk": te["kategori"]["qwk"]},
            "produksi": {"n_bag": len(pr["pemilihan_model"]["indeks_0based"]), "mae_in_sample_bag": pr["mae_in_sample_bag"],
                         "r2_in_sample_bag": pr["r2_in_sample_bag"],
                         "kategori_ambang": pr["kategori_ambang"]},
            "regresi_9_faktor_protokol_sama": {
                "mae_cv": reg["dipakai"]["cv"]["mae"], "r2_cv": reg["dipakai"]["cv"]["r2"],
                "r2_in_sample": reg["dipakai"]["in_sample"]["r2"],
                "aturan_pemilihan": reg["dipakai"]["aturan"]},
        },
        "berkas": berkas,
        "privat": privat,
    }
    if not privat:
        hasil["catatan_privat"] = ("berkas privat tidak ada di mesin ini; angka tetap bisa dicek "
                                  "terhadap cv.json / test / produksi.json yang ikut repo")
    Path(a.output).write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n")
    print(f"[manifest] {len(berkas)} berkas publik"
          f"{f', {len(privat)} privat' if privat else ''} -> {a.output}")
    print(f"[manifest] CV MAE {cv['mae']:.16f} | test {te['mae_raw']:.6f} | "
          f"in-sample {pr['mae_in_sample_bag']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

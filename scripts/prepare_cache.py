"""
prepare_cache.py
~~~~~~~~~~~~~~~~

Build a pre-resized copy of the dataset. Decoding 4000x4000 JPEGs every epoch
is the single biggest cost in this pipeline; resizing once turns an 11 minute
epoch into a couple of minutes.

    python scripts/prepare_cache.py --src data --dst data_256 --size 256

The output mirrors the input tree (images resized, labels copied verbatim), so
you can point the trainer straight at it:

    python main.py --data data_256 --arch resnet

Disk cost: roughly 25-40 KB per image at size 256.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def resize_one(args):
    src, dst, size, quality = args
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return None
    except Exception as exc:                       # noqa: BLE001
        return (str(src), str(exc)[:60])


def main():
    parser = argparse.ArgumentParser(description="Pre-resize the dataset once.")
    parser.add_argument("--src", default="data", help="source dataset folder")
    parser.add_argument("--dst", default="data_256", help="output folder")
    parser.add_argument("--size", type=int, default=256,
                        help="short side of the output images")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source not found: {src}")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"destination already exists and is not empty: {dst}")

    total, failures = 0, []
    for split in SPLITS:
        images_dir = src / split / "images"
        if not images_dir.is_dir():
            continue

        out_images = dst / split / "images"
        out_images.mkdir(parents=True, exist_ok=True)

        # labels and anything else are copied untouched
        labels_dir = src / split / "labels"
        if labels_dir.is_dir():
            shutil.copytree(labels_dir, dst / split / "labels", dirs_exist_ok=True)

        work = [
            (p, out_images / (p.stem + ".jpg"), args.size, args.quality)
            for p in sorted(images_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXT
        ]
        print(f"{split}: resizing {len(work)} images -> {out_images}")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, failure in enumerate(pool.map(resize_one, work, chunksize=16), 1):
                if failure:
                    failures.append(failure)
                if i % 1000 == 0:
                    print(f"   {i}/{len(work)}", flush=True)
        total += len(work)

    print(f"\ndone: {total} images -> {dst}")
    if failures:
        print(f"skipped {len(failures)} unreadable image(s):")
        for name, err in failures[:5]:
            print(f"   {name}: {err}")


if __name__ == "__main__":
    main()

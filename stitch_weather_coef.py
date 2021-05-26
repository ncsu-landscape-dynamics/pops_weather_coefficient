# %%
import argparse
from osgeo import gdal
from pathlib import Path

# %%
GDAL_CO = [
    "BIGTIFF=YES",
    "BLOCKXSIZE=1024",
    "BLOCKYSIZE=1024",
    "TILED=YES",
    "COMPRESS=LZW",
    "NUM_THREADS=ALL_CPUS",
    # "PREDICTOR=3",
    # "ZSTD_LEVEL=1",
]

# %%
def main(src_dir, dst_dir, dst_srs=None):
    # %%
    tif_files = list(src_dir.glob("*.tif"))
    tif_files
    # %%
    # tile_ids = [x.stem.split("-")[-2:] for x in tif_files]
    # tile_ids = ["-".join(x) for x in tile_ids]
    # print(tile_ids)
    unique_ids = [x.stem.split("-")[:-2] for x in tif_files]
    unique_ids = ["-".join(x) for x in unique_ids]
    unique_ids = set(unique_ids)
    # unique_ids

    # %%
    for uid in unique_ids:
        files_to_merge = list(src_dir.glob(f"{uid}*.tif"))
        dst_tif_path = dst_dir / f"{uid}.tif"
        print(f"mosaicking image {dst_tif_path}...", end="")
        my_ras = gdal.Warp(
            str(dst_tif_path),
            [str(x) for x in files_to_merge],
            options=gdal.WarpOptions(dstSRS=dst_srs, creationOptions=GDAL_CO),
        )
        my_ras = None
        print(" done.")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_dir", help="directory containing images to be mosaicked", type=Path
    )
    parser.add_argument("dst_dir", help="directory to save mosaicked images", type=Path)
    parser.add_argument(
        "-dst_srs",
        help="destination spatial reference system",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args.src_dir, args.dst_dir, args.dst_srs)

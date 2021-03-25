# ======================================================================================
# name:         stack_weather_coef_tifs.py
# author:       Nick kruskamp
# email:        nfkruska@ncsu.edu
# date:         2021-03-03
# description:  Basic code to stack time-series precipitation and temperature weather
#               coefficients exported from google earth engine into raster stacks as
#               expected by the rpops model. This can potentially create huge files.
#
# ======================================================================================

# %%
import rasterio
from rasterio.merge import merge
from pathlib import Path
from tqdm import tqdm
import numpy as np

# %%
def stack_rasters(src_files, dst_file):
    with rasterio.open(src_files[0]) as src:
        profile = src.profile
        n_bands = src.count

    n_dst_bands = len(src_files) * n_bands
    profile.update({"count": n_dst_bands, "BIGTIFF": "YES"})

    with rasterio.open(dst_file, "w", **profile) as dst:
        for i, file, in zip(
            range(1, n_dst_bands, n_bands), tqdm(src_files, desc="stacking files")
        ):
            with rasterio.open(file) as src:
                data = src.read()
                dst.write(data, indexes=range(i, i + n_bands))

    return None

# %%

src_dir = Path("Q:/My Drive/PoPS_weather_data")
dst_dir = Path("Q:/Shared drives/Data/Raster/Regional/SOD_CA/0030m/weather_coefs")
if not dst_dir.exists():
    dst_dir.mkdir(parents=True)

coef_types = ["prcp_coef", "temp_coef"]

# %%
all_tif_files = list(src_dir.glob("*.tif"))
years = [x.stem.split("_")[0] for x in all_tif_files]
years = np.unique(years)
years
# %%

with tqdm(
    total=(len(years) * len(coef_types)), desc="merging yearly coef files") as pbar:
    for coef_type in coef_types:
        for year in years:
            dst_path = dst_dir / "{}_{}_coef.tif".format(year, coef_type)
            if dst_path.exists():
                pbar.update(1)
                continue
            tif_files = list(src_dir.glob("{}*{}*.tif".format(year, coef_type)))
            try:
                tmp = merge(
                    tif_files,
                    dst_path=dst_path,
                    dst_kwds={
                        "compress": "lzw",
                        "BIGTIFF": "YES"
                    })
            except:
                print("issue with {}".format(dst_path.stem))
            pbar.update(1)
# %%
for coef_type in tqdm(coef_types, desc="stacking coef files"):
    tif_files = list(dst_dir.glob("*{}*.tif".format(coef_type)))
    dst_path = dst_dir / "{}_stack.tif".format(coef_type)
    print(dst_path)
    if dst_path.exists():
        continue
    stack_rasters(tif_files, dst_path)
# %%
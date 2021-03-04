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
from pathlib import Path
from tqdm import tqdm

# %%


def stack_weather_coefs(src_files, dst_file):
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
dst_dir = Path("D:/nfkruska/ch2_data/BS_weather_coefs")
if not dst_dir.exists():
    dst_dir.mkdir()

prcp_files = list(src_dir.glob("*prcp*100m.tif"))
temp_files = list(src_dir.glob("*temp*100m.tif"))

dst_prcp_file = dst_dir / "prcp_stack.tif"
dst_temp_file = dst_dir / "temp_stack.tif"

stack_weather_coefs(prcp_files, dst_prcp_file)
stack_weather_coefs(temp_files, dst_temp_file)
# %%

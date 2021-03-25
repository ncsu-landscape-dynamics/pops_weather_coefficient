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
from tqdm.autonotebook import tqdm
import numpy as np

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
dst_dir = Path("Q:/Shared drives/Data/Raster/Regional/SOD_CA/0030m/weather_coefs")
if not dst_dir.exists():
    dst_dir.mkdir(parents=True)

coef_types = ["prcp_coef", "temp_coef"]
coef_file_dict = {}
for coef_type in coef_types:
    coef_file_dict[coef_type] = list(src_dir.glob("*{}*.tif".format(coef_type)))
coef_file_dict

# %%
years = [x.stem.split("_")[0] for x in coef_file_dict[coef_types[0]]]
years = np.unique(years)
years
# %%

for coef_type, tif_files in coef_file_dict.items():
    for year in tqdm(years, desc="merging {} files".format(coef_type)):
        year_files = [x for x in tif_files if x.match("*{}*".format(year))]
        dst_path = dst_dir / "{}_{}_coef.tif".format(year, coef_type)
        # print(dst_path)
        tmp = merge(year_files, dst_path=dst_path, dst_kwds={"compress": "lzw"})



# # %%
# dst_prcp_file = dst_dir / "prcp_stack.tif"
# dst_temp_file = dst_dir / "temp_stack.tif"
# # %%
# stack_weather_coefs(prcp_files, dst_prcp_file)
# stack_weather_coefs(temp_files, dst_temp_file)
# # %%

# %%

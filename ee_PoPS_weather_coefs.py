# ======================================================================================
# name: ee_PoPS_weather_coefs.py
# authors: Chris Jones, Nick Kruskamp, (WHO ELSE?)
# email: cmjone25@ncsu.edu
# date: 2021-02-26
#
# North Carolina State University
# Center for Geospatial Analytics
#
# ======================================================================================
# %%
import datetime
import time

import ee
import geopandas as gpd
import numpy as np

# from tqdm import tqdm

# Kruskamp stuff for getting aoi files
from nfkruska.my_stuff import *

ee.Initialize()
# %%
"""VARIABLES TO BE SET BY USER"""
# daymet starts on 1980-01-01 and PRISM starts on 1981-01-01
start_year = 2006
# NOTE: end_year is EXCLUSIVE
end_year = 2016

# time step for model
# options: ["weekly", "monthly"]
time_step = "weekly"

# weather data to be processed
# options: ["daymet", "prism"]
data_selector = "daymet"

# function to process data
# options ["reclass", "polynomial"]
temp_method = "reclass"
prcp_method = "reclass"
rh_method = "reclass"

# data to be processed
precipitation = True
temperature = True
relative_humidity = False
lethal_temperature = False

# your google drive folder for export
# NOTE: seem to only support one directory level (e.g. not my_folder/another_folder)
output_folder = "PoPS_weather_data"

# options: ["Late Blight", "SOD", "SLF"]
pest_or_path = "SOD"

# adjust GEE pixel limit for export and analysis
max_pix = 1e10

# for GEE export and analysis
output_scale = 100
# %%
""" SET AOI"""

"""GET A LOCAL GEOFILE TO USE AS AOI"""
# read the local file
ls_scenes = gpd.read_file(CH1_GPKG, layer="WRS2_scenes_2")
# reproject
# NOTE: must be 4236!
ls_scenes = ls_scenes.to_crs(epsg=4326)
# limit to single feature
# NOTE: I *think* it needs to be a single feature, or else you can use unary_union() to
# union features together into one big feature
big_sur_aoi = ls_scenes[(ls_scenes["PATH"] == 43) & (ls_scenes["ROW"] == 35)]
# must use __geo_interface__ to pass feature to EE.
feature = big_sur_aoi.__geo_interface__["features"][0]
ee_aoi_feat = ee.Feature(feature)
# create feature collection
ee_aoi_fc = ee.FeatureCollection(ee_aoi_feat)

"""USE EXISTING GEE ASSET"""
# ee_aoi_fc = (ee.FeatureCollection("TIGER/2016/States")
#     .filter(
#         ee.Filter.eq("NAME", "California")
#     )
# )

# ee_aoi_fc = (ee.FeatureCollection("TIGER/2016/States")
#     .filter(ee.Filter.or(
#         ee.Filter.eq("NAME", "Kentucky"),
#         ee.Filter.eq("NAME", "Tennessee"),
#         ee.Filter.eq("NAME", "North Carolina"),
#         ee.Filter.eq("NAME", "Virginia"),
#         ee.Filter.eq("NAME", "West Virginia"),
#         ee.Filter.eq("NAME", "Maryland"),
#         ee.Filter.eq("NAME", "District of Columbia"),
#         ee.Filter.eq("NAME", "Delaware"),
#         ee.Filter.eq("NAME", "Ohio"),
#         ee.Filter.eq("NAME", "Pennsylvania"),
#         ee.Filter.eq("NAME", "New Jersey"),
#         ee.Filter.eq("NAME", "New York"),
#         ee.Filter.eq("NAME", "Vermont"),
#         ee.Filter.eq("NAME", "New Hampshire"),
#         ee.Filter.eq("NAME", "Massachusetts"),
#         ee.Filter.eq("NAME", "Rhode Island"),
#         ee.Filter.eq("NAME", "Connecticut"))
#     )
# )

# ee_aoi_fc = (ee.FeatureCollection("TIGER/2016/States")
#     .filter(ee.Filter.or(
#         ee.Filter.eq("NAME", "Wisconsin"),
#         ee.Filter.eq("NAME", "Michigan"),
#         ee.Filter.eq("NAME", "Illinois"),
#         ee.Filter.eq("NAME", "Indiana"),
#         ee.Filter.eq("NAME", "Kentucky"),
#         ee.Filter.eq("NAME", "Tennessee"),
#         ee.Filter.eq("NAME", "Mississippi"),
#         ee.Filter.eq("NAME", "Alabama"),
#         ee.Filter.eq("NAME", "Georgia"),
#         ee.Filter.eq("NAME", "Florida"),
#         ee.Filter.eq("NAME", "South Carolina"),
#         ee.Filter.eq("NAME", "North Carolina"),
#         ee.Filter.eq("NAME", "Virginia"),
#         ee.Filter.eq("NAME", "West Virginia"),
#         ee.Filter.eq("NAME", "Ohio"),
#         ee.Filter.eq("NAME", "Maryland"),
#         ee.Filter.eq("NAME", "District of Columbia"),
#         ee.Filter.eq("NAME", "Delaware"),
#         ee.Filter.eq("NAME", "Ohio"),
#         ee.Filter.eq("NAME", "Pennsylvania"),
#         ee.Filter.eq("NAME", "New Jersey"),
#         ee.Filter.eq("NAME", "New York"),
#         ee.Filter.eq("NAME", "Vermont"),
#         ee.Filter.eq("NAME", "New Hampshire"),
#         ee.Filter.eq("NAME", "Maine"),
#         ee.Filter.eq("NAME", "Massachusetts"),
#         ee.Filter.eq("NAME", "Rhode Island"),
#         ee.Filter.eq("NAME", "Connecticut"))
#     )
# )

# %%
"""FUNCTIONS / CONSTANTS / VARIABLES"""
# get aoi geometry
ee_aoi_geom = ee_aoi_fc.geometry()
# get aoi bounds
ee_aoi_bounds = ee_aoi_geom.bounds()

# how many seconds to wait when checking export task statuses
WAIT_TIME = 60

TEMP_MAT_DICT = {
    "SOD": [
        [-100, 0, 0.0],
        [0, 1, 0.024],
        [1, 2, 0.06],
        [2, 3, 0.099],
        [3, 4, 0.112],
        [4, 5, 0.137],
        [5, 6, 0.175],
        [6, 7, 0.22],
        [7, 8, 0.278],
        [8, 9, 0.34],
        [9, 10, 0.407],
        [10, 11, 0.476],
        [11, 12, 0.546],
        [12, 13, 0.615],
        [13, 14, 0.68],
        [14, 15, 0.744],
        [15, 16, 0.80],
        [16, 17, 0.847],
        [17, 18, 0.88],
        [18, 19, 0.91],
        [19, 20, 0.923],
        [20, 21, 0.92],
        [21, 22, 0.90],
        [22, 23, 0.86],
        [23, 24, 0.80],
        [24, 25, 0.72],
        [25, 26, 0.61],
        [26, 27, 0.48],
        [27, 28, 0.32],
        [28, 29, 0.13],
        [29, 30, 0.012],
        [31, 100, 0],
    ],
    "Late Blight": [
        [-100, 3, 0.0],
        [3, 4.33, 0.1],
        [4.33, 5.66, 0.2],
        [5.66, 7.00, 0.3],
        [7.00, 8.33, 0.4],
        [8.33, 9.66, 0.5],
        [9.66, 11.00, 0.6],
        [11.00, 12.33, 0.7],
        [12.33, 13.66, 0.8],
        [13.66, 15.00, 0.9],
        [15.0, 18, 1.0],
        [18, 18.77, 0.9],
        [18.77, 19.55, 0.8],
        [19.55, 20.33, 0.7],
        [20.33, 21.11, 0.6],
        [21.11, 21.88, 0.5],
        [21.88, 22.66, 0.4],
        [22.66, 23.44, 0.3],
        [23.44, 24.22, 0.2],
        [24.22, 25, 0.1],
        [25, 100, 0.0],
    ],
    "SLF": [
        [-100, 5, 0.0],
        [5, 7.5, 0.05],
        [7.5, 10, 0.1],
        [10, 12.5, 0.2],
        [12.5, 15, 0.3],
        [15, 16, 0.4],
        [16, 17, 0.5],
        [17, 18, 0.6],
        [18, 19, 0.7],
        [19, 20, 0.85],
        [20, 24, 0.95],
        [24, 26, 0.9],
        [26, 27, 0.8],
        [27, 29, 0.7],
        [29, 30, 0.6],
        [30, 31, 0.5],
        [31, 32, 0.4],
        [32, 33, 0.3],
        [33, 34, 0.2],
        [34, 35, 0.1],
        [35, 100, 0.0],
    ],
}

if precipitation:
    if prcp_method == "polynomial":
        prcp_a0 = 1
        prcp_a1 = 1
        prcp_a2 = 0
        prcp_a3 = 0
        prcp_x1 = 0
        prcp_x2 = 0
        prcp_x3 = 0
    elif prcp_method == "reclass":
        prcp_matrix = [[0.0, 2.5, 0.0], [2.5, 20000, 1]]

if relative_humidity:
    if rh_method == "polynomial":
        rh_a0 = 1
        rh_a1 = 1
        rh_a2 = 0
        rh_a3 = 0
        rh_x1 = 0
        rh_x2 = 0
        rh_x3 = 0
    elif rh_method == "reclass":
        rh_matrix = [[0.0, 2.5, 0.0], [2.5, 20000, 1]]

if temperature:
    if temp_method == "reclass":
        temp_matrix = TEMP_MAT_DICT[pest_or_path]
    elif temp_method == "polynomial":
        temp_a0 = 1
        temp_a1 = 1
        temp_a2 = 0
        temp_a3 = 0
        temp_x1 = 0
        temp_x2 = 0
        temp_x3 = 0


def rename_prism_ppt(image):
    bands = image.bandNames()
    new_bands = ["prcp", "tavg", "tmin", "tmax", "tdmean", "vpdmin", "vpdmax"]
    return image.select(bands).rename(new_bands)


def clipper(image):
    return image.clip(ee_aoi_fc)


def rh_reclass(image):
    coefficient = image
    for i in range(len(rh_matrix)):
        small = image.gte(rh_matrix[i][0])
        large = image.lt(rh_matrix[i][1])
        small_large = small.add(large)
        coefficient = coefficient.where(small_large.eq(2), rh_matrix[i][2])
    return coefficient


def rh_poly(image):
    return image.expression(
        "a0+(a1*(p+x1))+(a2*(p+x2))+(a3*(p+x3))",
        {
            "p": image.select("prcp"),
            "a0": rh_a0,
            "a1": rh_a1,
            "a2": rh_a2,
            "a3": rh_a3,
            "x1": rh_x1,
            "x2": rh_x2,
            "x3": rh_x3,
        },
    )


def prcp_reclass(image):
    coefficient = image
    for i in range(len(prcp_matrix)):
        small = image.gte(prcp_matrix[i][0])
        large = image.lt(prcp_matrix[i][1])
        small_large = small.add(large)
        coefficient = coefficient.where(small_large.eq(2), prcp_matrix[i][2])
    return coefficient


def prcp_poly(image):
    return image.expression(
        "a0+(a1*(p+x1))+(a2*(p+x2))+(a3*(p+x3))",
        {
            "p": image.select("prcp"),
            "a0": prcp_a0,
            "a1": prcp_a1,
            "a2": prcp_a2,
            "a3": prcp_a3,
            "x1": prcp_x1,
            "x2": prcp_x2,
            "x3": prcp_x3,
        },
    )


def temp_reclass(image):
    coefficient = image
    for i in range(len(temp_matrix)):
        small = image.gte(temp_matrix[i][0])
        large = image.lt(temp_matrix[i][1])
        small_large = small.add(large)
        coefficient = coefficient.where(small_large.eq(2), temp_matrix[i][2])
    return coefficient


def temp_poly(image):
    return image.expression(
        "a0+(a1*(t+x1))+(a2*(t+x2))+(a3*(t+x3))",
        {
            "t": image.select("tavg"),
            "a0": temp_a0,
            "a1": temp_a1,
            "a2": temp_a2,
            "a3": temp_a3,
            "x1": temp_x1,
            "x2": temp_x2,
            "x3": temp_x3,
        },
    )


def RHavg(image):
    rhavg = image.expression(
        "100*(vp*0.001)/(0.611*exp((17.27*t)/(t+237.3)))",
        {"t": image.select("tavg"), "vp": image.select("vp")},
    )
    rhavg = rhavg.rename("rhavg")
    return image.addBands(rhavg)


def RHmax(image):
    rhmax = image.expression(
        "100*((0.611*exp((17.27*t)/(t+237.3)))-(vpd*0.1))/(0.611*exp((17.27*t)/(t+237.3)))",
        {"t": image.select("tmin"), "vpd": image.select("vpdmin")},
    )
    rhmax = rhmax.rename("rhmax")
    return image.addBands(rhmax)


def RHmin(image):
    rhmin = image.expression(
        "100*((0.611*exp((17.27*t)/(t+237.3)))-(vpd*0.1))/(0.611*exp((17.27*t)/(t+237.3)))",
        {"t": image.select("tmax"), "vpd": image.select("vpdmax")},
    )
    rhmin = rhmin.rename("rhmin")
    return image.addBands(rhmin)


def meanTemp(image):
    tavg = image.expression(
        "(max + min)/x",
        {"min": image.select("tmin"), "max": image.select("tmax"), "x": 2},
    )
    tavg = tavg.rename("tavg")
    return image.addBands(tavg)


def RHmean(image):
    rhavg = image.expression(
        "(max + min)/x",
        {"min": image.select("rhmin"), "max": image.select("rhmax"), "x": 2},
    )
    rhavg = rhavg.rename("rhavg")
    return image.addBands(rhavg)


def ExportCol(
    col,
    folder=output_folder,
    scale=output_scale,
    type=float,
    nimg=500,
    maxPixels=max_pix,
    region=None,
):
    colList = col.toList(nimg)
    n = colList.size().getInfo()

    for i in range(n):
        img = ee.Image(colList.get(i))
        id = img.id().getInfo()
        if region is None:
            region = img.geometry().bounds().getInfo().coordinates

    imgtype = {
        "float": img.toFloat(),
        "byte": img.toByte(),
        "int": img.toInt(),
        "double": img.toDouble(),
    }

    task = ee.batch.Export.image.toDrive(
        image=imgtype[type],
        description=id,
        folder=folder,
        fileNamePrefix=id,
        region=region,
        scale=scale,
        maxPixels=maxPixels,
    )
    task.start()


def create_date_list(y, time_step):
    y = int(y)
    start_date = datetime.datetime(y, 1, 1)
    end_date = datetime.datetime(y + 1, 1, 1)
    time_step_list = [start_date]
    if time_step == "weekly":
        for i in range(1, 52):
            new_date = start_date + datetime.timedelta(weeks=i)
            time_step_list.append(new_date)
    elif time_step == "monthly":
        for i in range(2, 13):
            new_date = datetime.datetime(start_year, i, 1)
            time_step_list.append(new_date)
    else:
        print('not a recognized time step. Most be one of ["weekly", "monthly"]')
        return None
    time_step_list.append(end_date)
    return time_step_list


if data_selector == "prism":
    weather_data = ee.ImageCollection("OREGONSTATE/PRISM/AN81d")
    weather_data = weather_data.map(rename_prism_ppt)
    weather_data = weather_data.map(RHmin)
    weather_data = weather_data.map(RHmax)
    weather_data = weather_data.map(RHmean)
elif data_selector == "daymet":
    weather_data = ee.ImageCollection("NASA/ORNL/DAYMET_V3")
    weather_data = weather_data.map(meanTemp)
    weather_data = weather_data.map(RHavg)

if temp_method == "reclass":
    temp_fun = temp_reclass
elif temp_method == "polynomial":
    temp_fun = temp_poly

if prcp_method == "reclass":
    prcp_fun = prcp_reclass
elif prcp_method == "polynomial":
    prcp_fun = prcp_poly

# %%
weather_data = weather_data.map(clipper)
# means = weather_data.mean()
task_status_list = []
year_list = range(start_year, end_year)
for y in year_list:
    print("processing year {}".format(y))
    date_list = create_date_list(y, "weekly")
    temp_file_name = "{}_temp_coef_{}m".format(y, output_scale)
    prcp_file_name = "{}_prcp_coef_{}m".format(y, output_scale)
    for i in range(len(date_list) - 1):
        date_1 = date_list[i]
        date_2 = date_list[(i + 1)]
        date_1_name = date_1.strftime("%Y-%m-%d")
        weather_datas = weather_data.filter(ee.Filter.date(date_1, date_2))

        if temperature:
            temp = weather_datas.select("tavg").map(temp_fun).mean().rename(date_1_name)
            if i == 0:
                temp_coef_stack = temp
            else:
                temp_coef_stack = temp_coef_stack.addBands(temp)

        if precipitation:
            prcp = weather_datas.select("prcp").map(prcp_fun).mean().rename(date_1_name)
            if i == 0:
                prcp_coef_stack = prcp
            else:
                prcp_coef_stack = prcp_coef_stack.addBands(prcp)

    if temperature:
        tasks = ee.batch.Export.image.toDrive(
            image=temp_coef_stack,
            description=temp_file_name,
            folder=output_folder,
            region=ee_aoi_bounds,
            scale=output_scale,
            maxPixels=max_pix,
        )
        tasks.start()
        task_status_list.append(tasks.list()[0])

    if precipitation:
        tasks = ee.batch.Export.image.toDrive(
            image=prcp_coef_stack,
            description=prcp_file_name,
            folder=output_folder,
            region=ee_aoi_bounds,
            scale=output_scale,
            maxPixels=max_pix,
        )
        tasks.start()
        task_status_list.append(tasks.list()[0])


# %%
# NOTE: quick and dirty method to keep track of export task states and know which
# completed or which failed.
# TODO:
# 1. write a log file of final tasks statuses
# 2. write a progress bar instead of printing everything

# ready_bar = tqdm(total=len(task_status_list))
# running_bar = tqdm(total=len(task_status_list))
# complete_bar = tqdm(total=len(task_status_list))

while True:
    task_statuses = [x.status() for x in task_status_list]
    for x in task_statuses:
        states_list = [x["state"] for x in task_statuses]
        print("{}: {}".format(x["description"], x["state"]))
    print()
    all_done = np.isin(np.unique(states_list), ["COMPLETED", "FAILED"]).all()
    if all_done:
        break
    print("will check again in {} seconds...".format(WAIT_TIME))
    time.sleep(WAIT_TIME)

print("done.")
# %%

# weather_datas = weather_data.filter(ee.Filter.date(start_date, end_date))
# temp = weather_datas.select("tavg")
# temp_coeffs = temp.map(temp_reclass).mean()
# stats = temp_coeffs.reduceRegion(
#     reducer=ee.Reducer.mean(), geometry=ee_aoi_geom, scale=output_scale
# )
# pp(stats.getInfo())
# # %%

# image1 = weather_data.first()
# bandNames = weather_data.first().bandNames()
# print("Band names: ", bandNames.getInfo())
## add export here weather_data
## add export for temp_coef_stack, prcp_coef_stack, and rh_coeff
# %%

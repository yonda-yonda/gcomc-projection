from scipy import interpolate
import os
import io
import re
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy.interpolate import griddata
from osgeo import gdal
from osgeo import osr

defalut_qa_masks = {
    'LST': [
        {'bit': 0, 'mask': 1},
        {'bit': 1, 'mask': 1},
        {'bit': 4, 'mask': 1},
        {'bit': 11, 'mask': 1},
        {'bit': 12, 'mask': 1},
        {'bit': 13, 'mask': 1},
        {'bit': 14, 'mask': 1},
        {'bit': 15, 'mask': 1}
    ]
}


class Reader:
    def __init__(self, hdf5, product='LST', qa_masks=None):
        # HDF読込,情報取得
        with h5py.File(hdf5, 'r') as f:
            self.data = f['Image_data'][product][:]
            self.qa_flags = f['Image_data']['QA_flag'][:]
            self.error_dn = f['Image_data'][product].attrs['Error_DN'][0]
            self.min_valid_dn = f['Image_data'][product].attrs['Minimum_valid_DN'][0]
            self.max_valid_dn = f['Image_data'][product].attrs['Maximum_valid_DN'][0]
            self.slope = f['Image_data'][product].attrs['Slope'][0]
            self.offset = f['Image_data'][product].attrs['Offset'][0]
            self.lines = f['Image_data'].attrs['Number_of_lines'][0]
            self.pixels = f['Image_data'].attrs['Number_of_pixels'][0]
            self.grid_interval_num = f['Image_data'].attrs['Grid_interval'][0]

        self.filename = os.path.splitext(os.path.basename(hdf5))[0]

        if (len(self.filename.split('_')) < 3 or not re.match('^T\d{4}', self.filename.split('_')[2])):
            raise Exception('This file name don\'t include tile num.')

        tile_position = self.filename.split('_')[2]
        self.vtile = int(tile_position[1:3])
        self.htile = int(tile_position[3:5])
        self.nodata_value = np.NaN

        self.qa_masks = qa_masks if (
            type(qa_masks) is list) else defalut_qa_masks[product]

        # 物理量変換
        self.data = self.slope * self.data + self.offset

        # マスク
        self.data[self.data == self.error_dn] = self.nodata_value
        self.data[(self.data < self.slope * self.min_valid_dn + self.offset) | (
            self.data > self.slope * self.max_valid_dn + self.offset)] = self.nodata_value

        if len(self.qa_masks) > 0:
            # bitが0だったときマスク
            masked_bits = [qa_mask['bit']
                           for qa_mask in self.qa_masks if qa_mask['mask'] == 0]
            if(len(masked_bits)):
                mask_0 = np.bitwise_and(~self.qa_flags, np.sum(
                    np.power(2, masked_bits))).astype(np.bool)
            else:
                mask_0 = np.full_like(self.data, False, dtype=np.bool)

            # bitが1だったときマスク
            masked_bits = [qa_mask['bit']
                           for qa_mask in self.qa_masks if qa_mask['mask'] == 1]
            if(len(masked_bits)):
                mask_1 = np.bitwise_and(self.qa_flags, np.sum(
                    np.power(2, masked_bits))).astype(np.bool)
            else:
                mask_0 = np.full_like(self.data, False, dtype=np.bool)
            self.data[mask_0 | mask_1] = self.nodata_value
        return

    def calc_lonlat(self, trim_lines, trim_pixels):
        trim_line_length = trim_lines[1] - trim_lines[0]
        trim_pixel_length = trim_pixels[1] - trim_pixels[0]
        lons = np.zeros((trim_line_length, trim_pixel_length), np.float32)
        lats = np.zeros((trim_line_length, trim_pixel_length), np.float32)

        lin_tile = self.lines
        col_tile = self.pixels
        vtile_num = 18
        # htile_num = 36
        v_pixel = int(lin_tile * vtile_num)
        d = 180 / v_pixel
        NL = v_pixel  # 180 / d
        NP0 = 2 * NL

        for i in range(trim_line_length):
            lin_total = (i + trim_lines[0]) + (self.vtile * lin_tile)

            for j in range(trim_pixel_length):
                col_total = (j + trim_pixels[0]) + (self.htile * col_tile)
                lat = 90 - (lin_total + 0.5) * d
                NPi = int(round(NP0 * math.cos(math.radians(lat)), 0))
                lon = 360 / NPi * (col_total - NP0 / 2 + 0.5)
                lons[i, j] = lon
                lats[i, j] = lat

        # nint int(round(f, 0))

        return (lons, lats)

    def _calc_data(self, trim_x, trim_y, error_value, normalize_func):
        if trim_x is None:
            trim_x = [0, self.lines]
        if trim_x[0] < 0 or trim_x[1] > self.lines:
            raise Exception('trim_x is out of range.')
        if trim_y is None:
            trim_y = [0, self.pixels]
        if trim_y[0] < 0 or trim_y[1] > self.pixels:
            raise Exception('trim_y is out of range.')

        # 緯度経度
        lonlat = self.calc_lonlat(trim_x, trim_y)
        longitudes = lonlat[0]
        latitudes = lonlat[1]

        # 日付変更線処理
        # 範囲が180度を越えるシーンは無い前提
        is_crossing_meridian = longitudes.max() * longitudes.min() < 0 and (
            longitudes.max() - longitudes.min()) > 180
        if(is_crossing_meridian):
            longitudes = np.copy(longitudes)

        lat_lim = [latitudes.min(), latitudes.max()]
        lon_lim = [longitudes.min(), longitudes.max()]

        # 投影後緯度経度グリッドの作成
        [lat_grid, lon_grid] = np.mgrid[lat_lim[1]:lat_lim[0]:-1*self.grid_interval_num,
                                        lon_lim[0]:lon_lim[1]:self.grid_interval_num]

        # 一部のデータを抽出,グリッドに整形
        points = np.stack([latitudes.flatten(), longitudes.flatten()], 1)
        values = self.data[trim_x[0]: trim_x[1],
                           trim_y[0]: trim_y[1]].flatten()
 
        grid_array = griddata(
            points, values, (lat_grid, lon_grid), method='linear')

        # 正規化
        if(callable(normalize_func)):
            grid_array = normalize_func(grid_array)
        grid_array[np.isnan(grid_array)] = error_value

        output_data = []

        if (is_crossing_meridian):
            merdian_left_index = max(
                (i for i in range(lon_grid.shape[1]) if lon_grid[0, i] <= 180), default=0)
            merdian_right_index = min(
                (i for i in range(lon_grid.shape[1]) if lon_grid[0, i] > 180), default=-1)

            if (merdian_left_index != 0):
                lon_grid_left = lon_grid[:, :merdian_left_index]
                lat_grid_left = lat_grid[:, :merdian_left_index]
                grid_array_left = grid_array[:, :merdian_left_index]
                output_data.append({
                    'dst_path': self.filename + '_left.tif',
                    'grid_array': grid_array_left,
                    'lon_grid': lon_grid_left,
                    'lat_grid': lat_grid_left
                })
            if(merdian_left_index != -1):
                lon_grid_right = lon_grid[:, merdian_right_index:] - 360
                lat_grid_right = lat_grid[:, merdian_right_index:]
                grid_array_right = grid_array[:, merdian_right_index:]
                output_data.append({
                    'dst_path': self.filename + '_right.tif',
                    'grid_array': grid_array_right,
                    'lon_grid': lon_grid_right,
                    'lat_grid': lat_grid_right,
                })
        else:
            output_data.append({
                'dst_path': self.filename + '.tif',
                'grid_array': grid_array,
                'lon_grid': lon_grid,
                'lat_grid': lat_grid
            })
        return output_data

    def save_tiff(self, trim_x=None, trim_y=None, error_value=-9999.0, normalize_func=None, dtype=gdal.GDT_Float32):
        output_data = self._calc_data(trim_x, trim_y, error_value, normalize_func)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        for data in output_data:
            driver = gdal.GetDriverByName('GTiff')
            output = driver.Create(data['dst_path'], data['grid_array'].shape[1],
                                   data['grid_array'].shape[0], 1, dtype)

            output.GetRasterBand(1).WriteArray(data['grid_array'])
            output.GetRasterBand(1).SetNoDataValue(error_value)
            output.SetGeoTransform(
                [data['lon_grid'].min(), self.grid_interval_num, 0, data['lat_grid'].max(), 0, -1*self.grid_interval_num])
            output.SetProjection(srs.ExportToWkt())
            output.FlushCache()
        return

    def save_color_tiff(self, trim_x=None, trim_y=None, normalize_func=None, cmap='jet'):
        error_value=np.nan
        output_data = self._calc_data(trim_x, trim_y, error_value, normalize_func)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        for data in output_data:
            arr = data['grid_array']

            temp = io.BytesIO()
            plt.imsave(temp, arr, vmin=0, vmax=255, cmap=cmap)
            temp_img = np.array(Image.open(temp))
            for i in range(0,4):
                temp_img[:,:,i][np.isnan(arr)] = 0

            driver = gdal.GetDriverByName('GTiff')
            output = driver.Create(data['dst_path'], data['grid_array'].shape[1],
                                   data['grid_array'].shape[0], 4)

            for i in range(0,4):
                output.GetRasterBand(i + 1).WriteArray(temp_img[:,:, i])

            output.SetGeoTransform(
                [data['lon_grid'].min(), self.grid_interval_num, 0, data['lat_grid'].max(), 0, -1*self.grid_interval_num])
            output.SetProjection(srs.ExportToWkt())
            output.FlushCache()
        return
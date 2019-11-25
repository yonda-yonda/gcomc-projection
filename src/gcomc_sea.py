from scipy import interpolate
import os
import io
import h5py
import numpy as np
from scipy.interpolate import griddata
from tifffile import imsave as tifsave
from osgeo import gdal
from osgeo import osr


def _interp2d_biliner(data, interval):
    data = np.concatenate((data, data[-1].reshape(1, -1)), axis=0)
    data = np.concatenate(
        (data, data[:, -1].reshape(-1, 1)), axis=1)
    ratio_horizontal = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
                               (data.shape[0]*interval, data.shape[1] - 1))
    ratio_vertical = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
                             (data.shape[0] - 1, (data.shape[1] - 1)*interval))
    repeat_data = np.repeat(data, interval, axis=0)
    repeat_data = np.repeat(repeat_data, interval, axis=1)
    horizontal_interp = (1. - ratio_horizontal) * \
        repeat_data[:, :-interval] + \
        ratio_horizontal * repeat_data[:, interval:]
    ret = (1. - ratio_vertical) * \
        horizontal_interp[:-interval, :] + \
        ratio_vertical * horizontal_interp[interval:, :]
    return ret


class sst:
    def __init__(self, hdf5, qa_masks=None):
        product = 'SST'
        # HDF読込,情報取得
        with h5py.File(hdf5, 'r') as f:
            self.data = f['Image_data'][product][:]
            self.latitudes = f['Geometry_data']['Latitude'][:]
            self.longitudes = f['Geometry_data']['Longitude'][:]
            self.qa_flags = f['Image_data']['QA_flag'][:]
            self.latitude_resampling = f['Geometry_data']['Latitude'].attrs['Resampling_interval'][0]
            self.longitude_resampling = f['Geometry_data']['Longitude'].attrs['Resampling_interval'][0]
            self.error_dn = f['Image_data'][product].attrs['Error_DN'][0]
            self.min_valid_dn = f['Image_data'][product].attrs['Minimum_valid_DN'][0]
            self.max_valid_dn = f['Image_data'][product].attrs['Maximum_valid_DN'][0]
            self.slope = f['Image_data'][product].attrs['Slope'][0]
            self.offset = f['Image_data'][product].attrs['Offset'][0]
            self.lines = f['Image_data'].attrs['Number_of_lines'][0]
            self.pixels = f['Image_data'].attrs['Number_of_pixels'][0]
            self.grid_interval_num = f['Image_data'].attrs['Grid_interval'][0]

        self.filename = os.path.splitext(os.path.basename(hdf5))[0]
        self.nodata_value = np.NaN
        if (self.grid_interval_num == 250):
            # 40,073,834[m] /360 * 10/4800 = 230[m]
            self.ddeg = [10 / 4800, 10 / 4800]

        elif(self.grid_interval_num == 1000):
            # 40,073,834[m] /360 * 4* 10/4800 = 930[m]
            self.ddeg = [4*10 / 4800, 4*10 / 4800]
        else:
            raise Exception
        self.qa_masks = qa_masks if (type(qa_masks) is list) else [
            {'bit': 0, 'mask': 1},
            {'bit': 1, 'mask': 1},
            {'bit': 2, 'mask': 1},
            {'bit': 3, 'mask': 1},
            {'bit': 4, 'mask': 1},
            {'bit': 5, 'mask': 1},
            {'bit': 11, 'mask': 1},
            {'bit': 12, 'mask': 1},
            {'bit': 15, 'mask': 0}
        ]

        # 物理量変換
        self.data = self.slope * self.data + self.offset

        # マスク
        self.data[self.data == self.error_dn] = self.nodata_value
        self.data[(self.data < self.min_valid_dn) | (
            self.data > self.max_valid_dn)] = self.nodata_value

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

    def save_tiff(self, trim_x=None, trim_y=None, error_value=-9999.0, normalize_func=None, dtype=gdal.GDT_Float32):
        if trim_x is None:
            trim_x = [0, self.lines]
        if trim_y is None:
            trim_y = [0, self.pixels]

        # 日付変更線処理
        is_crossing_meridian = (
            self.longitudes.max() - self.longitudes.min()) > 180

        # 緯度経度リサンプリングデータ補間,一部の領域を抽出
        lat = _interp2d_biliner(self.latitudes, self.latitude_resampling)[
            trim_x[0]: trim_x[1], trim_y[0]: trim_y[1]]

        if(is_crossing_meridian):
            longitudes = np.copy(self.longitudes)
            longitudes[longitudes < 0] += 360
        else:
            longitudes = self.longitudes

        lon = _interp2d_biliner(longitudes, self.longitude_resampling)[
            trim_x[0]: trim_x[1], trim_y[0]: trim_y[1]]

        # 投影後緯度経度グリッドの作成
        [lat_grid, lon_grid] = np.mgrid[lat.max():lat.min():-1*self.ddeg[0],
                                        lon.min(): lon.max(): self.ddeg[1]]

        # 一部のデータを抽出,グリッドに整形
        points = np.stack([lat.flatten(), lon.flatten()], 1)
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

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        for data in output_data:
            driver = gdal.GetDriverByName('GTiff')
            output = driver.Create(data['dst_path'], data['grid_array'].shape[1],
                                   data['grid_array'].shape[0], 1, dtype)
            output.GetRasterBand(1).WriteArray(data['grid_array'])
            output.GetRasterBand(1).SetNoDataValue(error_value)
            output.SetGeoTransform(
                [data['lon_grid'].min(), self.ddeg[0], 0, data['lat_grid'].max(), 0, -self.ddeg[1]])
            output.SetProjection(srs.ExportToWkt())
            output.FlushCache()
        return
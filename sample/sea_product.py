import numpy as np
import sys
from osgeo import gdal
sys.path.append('../src')
import gcomc_sea
# download from https://suzaku.eorc.jaxa.jp/GCOM_C/data/product_std_j.html

hdf5 = 'GC1SG1_201809050115H04610_L2SG_SSTDQ_0170.h5'

def normalize_sst(data):
    normalize_range = [24, 31]
    result = (data - normalize_range[0]) / \
        (normalize_range[1] - normalize_range[0])
    with np.errstate(invalid='ignore'):
        result[result > 1] = 1
        result[result < 0] = 0
        result = result * 254 + 1
        result = result.astype(np.uint8)
        return result


product_sst = gcomc_sea.sst(hdf5, qa_masks=[{'bit': 0, 'mask': 1},
                                {'bit': 1, 'mask': 1}])
product_sst.save_tiff([4400, 5400], [950, 1950], error_value=0,
                    normalize_func=normalize_sst, dtype=gdal.GDT_UInt16)

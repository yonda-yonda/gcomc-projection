import sys
sys.path.append('../src')
from osgeo import gdal
import numpy as np
import gcomc_sea

sst_hdf5 = 'GC1SG1_201904040133J05110_L2SG_SSTDQ_1002.h5'
chla_hdf5 = 'GC1SG1_201904040133J05110_L2SG_IWPRQ_1000.h5'

def normalize_sst(data):
    normalize_range = [0, 20]
    result = (data - normalize_range[0]) / \
        (normalize_range[1] - normalize_range[0])
    with np.errstate(invalid='ignore'):
        result[result > 1] = 1
        result[result < 0] = 0
        result = result * 254 + 1
        result = result.astype(np.uint8)
        return result


product_sst = gcomc_sea.Reader(sst_hdf5, product = 'SST', qa_masks=[{'bit': 0, 'mask': 1},
                                            {'bit': 1, 'mask': 1}])
product_sst.save_tiff([4400, 5400], [2500, 3500], error_value=0, normalize_func=normalize_sst, dtype=gdal.GDT_UInt16)


product_chla = gcomc_sea.Reader(chla_hdf5, product = 'CHLA', qa_masks=[{'bit': 0, 'mask': 1},
                                            {'bit': 1, 'mask': 1},
                                            {'bit': 2, 'mask': 1}])
product_chla.save_tiff([4400, 5400], [2500, 3500])
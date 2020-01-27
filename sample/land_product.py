import sys
sys.path.append('./')
from osgeo import gdal
import numpy as np
import gcomc_land

lst_hdf5 = 'GC1SG1_20200101D01D_T0529_L2SG_LST_Q_1000.h5'


product_lst = gcomc_land.Reader(lst_hdf5, product = 'LST')
product_lst.save_tiff([2000, 3000], [1000, 2000], error_value=0, dtype=gdal.GDT_Float32)

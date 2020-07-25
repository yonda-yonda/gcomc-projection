import sys
sys.path.append('../src')
from osgeo import gdal
import numpy as np
import gcomc_land

lst_hdf5 = 'GC1SG1_20200101D01D_T0529_L2SG_LST_Q_1000.h5'

def normalize_func(data):
	min = 203.15
	max = 343.15
	normalized = (data - min) / (max - min)

	with np.errstate(invalid='ignore'):
		normalized[normalized > 1] = 1
		normalized[normalized < 0] = 0
	normalized = (normalized * 255)
	return normalized


product_lst = gcomc_land.Reader(lst_hdf5, product = 'LST')
product_lst.save_color_tiff([2000, 2800], [1000, 1800], normalize_func=normalize_func)
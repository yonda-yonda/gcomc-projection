# GCOM-C Projection
GCOM-Cの観測データをEPSG:4326上に投影する。

## Product
### 海洋(gcomc_sea.py)
* SST
* クロロフィルa

### 陸域(gcomc_land.py)
* 地表面温度

## Sample
* [sea](sample/sea_product.py)
* [land](sample/land_product.py)

## GDAL
[docker](https://github.com/OSGeo/gdal/tree/master/gdal/docker)

## thanks
内挿処理の一部は[SGLI_Python_Open_Tool](https://github.com/K0gata/SGLI_Python_Open_Tool)を参考にさせていただきました。

## Reference
[GCOM-Cプロダクト利用の手引き（入門編）](https://gportal.jaxa.jp/gpr/assets/mng_upload/GCOM-C/GCOM-C_Products_Users_Guide_entrylevel_jp.pdf)


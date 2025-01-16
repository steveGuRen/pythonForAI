##
# deepin 系统
# sudo apt install python3-tk
# Mac OS
# brew install python-tk



import geopandas as gpd
import matplotlib.pyplot as plt

# 读取Shapefile文件（假设文件是.shp格式）
gdf = gpd.read_file('province.shp')

# 绘制地图
gdf.plot()
plt.show()

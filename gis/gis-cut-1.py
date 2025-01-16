import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

# 读取 shapefile
gdf = gpd.read_file("province.shp")

# 检查并设置坐标参考系
if gdf.crs is None:
    gdf.set_crs("EPSG:4326", inplace=True)  # 假设原始坐标系是 WGS 84

# 设置广东省的最南和最北的坐标范围 (经纬度)
min_lat, max_lat = 20.0, 25.5  # 纬度范围
min_lon, max_lon = 109.5, 117.0  # 经度范围

# 使用 shapely 的 box 创建边界框
bounding_box = box(min_lon, min_lat, max_lon, max_lat)

# 将边界框转换为 GeoDataFrame，设置坐标参考系为 WGS 84 (EPSG:4326)
bbox_gdf = gpd.GeoDataFrame({"geometry": [bounding_box]}, crs="EPSG:4326")

# 检查数据的坐标系并重新投影到 WGS 84 (如果需要)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# 过滤：只保留与边界框相交的几何形状
filtered_gdf = gdf[gdf.geometry.intersects(bounding_box)]

# 绘制过滤后的地图
ax = filtered_gdf.plot(color="lightblue", edgecolor="black", figsize=(8, 6))
bbox_gdf.boundary.plot(ax=ax, edgecolor="red", linestyle="--", alpha=0.7)  # 可视化边界框
plt.title("Filtered Map of Guangdong Province")
plt.show()

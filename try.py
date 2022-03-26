import transbigdata as tbd
import osmnx as ox
from time import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Pool


t1 = time()
# 获取路网

r = ox.graph_from_place("成都市,四川省，中国",network_type="drive")
t2 = time()

G_p = ox.project_graph(r, to_crs=2416)

nodes, edges = ox.graph_to_gdfs(G_p, nodes=True, edges=True)



print(t2 - t1) #下载地图所用时间


t3 = time()

# edges['lon'] = edges.centroid.x
# edges['lat'] = edges.centroid.y
def cen_x():
    edges['lon'] = edges.centroid.x

    edges['lat'] = edges.centroid.y


with ProcessPoolExecutor() as pool:
    pool.map(cen_x())
t4 = time()

nodes_p, edges_p = ox.graph_to_gdfs(G_p, nodes=True, edges=True)


print(t4 - t3) #获取道路中心点坐标的运行时间
print(t4 - t1) #总共运行时间

from leuvenmapmatching.map.inmem import InMemMap
import osmread
from pathlib import Path
from leuvenmapmatching import visualization
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching import visualization as mmviz
import pandas as pd
import geopandas as gpd

# 将路网转化为网络
map_con = InMemMap(name="myosm", use_latlon=False, use_rtree=True, index_edges=True)

# Create GeoDataFrames

wl_start = time()
# 构建网络

# for node_id, row in nodes_p.iterrows():
#     map_con.add_node(node_id, (row['y'], row['x']))
# for node_id_1, node_id_2, _ in G_p.edges:
#     map_con.add_edge(node_id_1, node_id_2)

def wangluo():
    for node_id, row in nodes_p.iterrows():
        map_con.add_node(node_id, (row['y'], row['x']))


    for node_id_1, node_id_2, _ in G_p.edges:
        map_con.add_edge(node_id_1, node_id_2)
with ProcessPoolExecutor() as pool:
    pool.map(wangluo())
wl_end = time()
print(wl_end - wl_start)  # 16.712316036224365

#读取数据------------------------------------------
import transbigdata as tbd
data = pd.read_csv(r'E:\essay\gps_data\1\cut_pu\ct_cc2d380a0f62fb268512c6565e2f5c4d',sep='\t')
data.columns=['order','time','Lng','Lat']

#转换轨迹的坐标系为地理坐标系
data['geometry'] = gpd.points_from_xy(data['Lng'],data['Lat'])
data = gpd.GeoDataFrame(data)
data.crs = ('epsg:4326')
data = data.to_crs(2416)

#获得轨迹点
path = list(zip(data.geometry.y, data.geometry.x))


# 构建地图匹配工具
matcher = DistanceMatcher(map_con,
                          max_dist=500,
                          max_dist_init=170,
                          min_prob_norm=0.0001,
                          non_emitting_length_factor=0.95,
                          obs_noise=50,
                          obs_noise_ne=50,
                          dist_noise=50,
                          max_lattice_width=20,
                          non_emitting_states=True)

pp_start = time()
#进行地图匹配

# 将path列表分割成若干份
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
father_list = split(path,4)

states,_ = matcher.match(path, unique=False)


"""
matcher.match(path, unique=False)  返回值是一个元组（列表，整型）,
如([(4220527834, 2329094579),(4220527834, 2329094579)],1246)
"""
list0 = []
def pp(path,list0):
    states, _ = matcher.match(path, unique=False)
    list0 = list0 + states

size=4
pool=Pool(size) #设置
"""
father_list是 一个大列表 --> [[列表1]，[列表2]，[列表3]，[列表4]],
我想把这里面4个子列表用四个进程同时运行，并将他们的结果合并到一起
"""
for i in father_list:
    pool.apply_async(matcher.match,args=(i,))
pool.close()
pool.join()

pp_stop = time()

print(pp_stop - pp_start)

#绘制底图匹配结果
mmviz.plot_map(map_con, matcher=matcher,
#                use_osm=True,
#                zoom_path=True,
               show_labels=False, show_matching=True,#show_graph=True,
               filename=None)

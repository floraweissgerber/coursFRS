import numpy as np
import matplotlib.pyplot as plt
import rasterio.windows
plt.close('all')
import function as fct
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
from lxml import etree

import rasterio
from rasterio.windows import Window

from pyproj import CRS
from pyproj import Transformer

import sys
sys.path.insert(1, "/d/fweissge/LABSARV42/")
from satellites import pydecoupe_Sentinel1_V42 as SENT
from satellites import pydecoupe_TSX_V42 as TSX
from sardecoupe import pydecoupe_V42 as pydec

import sarimages.sarimages.sar_display as sard
import sarimages.sarimages.read_s1 as S1
import sarimages.sarimages.projectionSAR as projSAR


path_dir = '/scratcht/fweissge/cours_FRS/'
dir_IW_1 = '/S1A_IW_SLC__1SDV_20241116T014710_20241116T014739_056573_06EFF8_2171.SAFE/'
dir_IW_2 = '/S1A_IW_SLC__1SDV_20241116T014737_20241116T014805_056573_06EFF8_A62A.SAFE/'
file_sar_1 = path_dir + dir_IW_1
file_sar_2 = path_dir + dir_IW_2

lat_ref = -21.2430556
delta_latref =0.05
lon_ref = 55.70722222222223
alt_ref =  2632
point_ref = [lon_ref, lat_ref+delta_latref,  alt_ref]

list_xml_1 = S1.getXMLFromSAFE(file_sar_1)
list_xml_2 = S1.getXMLFromSAFE(file_sar_2)
print('xml \n')
print(list_xml_2)
list_image_1 = S1.getImageFromSAFE(file_sar_1)
list_image_2 = S1.getImageFromSAFE(file_sar_2)
print('image \n')
print(list_image_2)

#%% Remove labsar from projectionSAR


product_master = SENT.charger(file_sar_2, point_ref)
Nazi = 1600
Nrange = 2000
image_1 = product_master.lirecoordcrop(point_ref, Nrange, Nazi)
fig, ax = plt.subplots()
ax.imshow(sard.threshSAR(image_1[0]), cmap ='gray')



crs_4326 = CRS.from_epsg(4326) #latlon
crs_ECEF = CRS.from_epsg(4978) 
transformer_latlon_to_ECEF = Transformer.from_crs(crs_4326, crs_ECEF)
point_ref_ecef = transformer_latlon_to_ECEF.transform(lat_ref, lon_ref, alt_ref)
dem_ecef = np.array([[[point_ref_ecef[0], point_ref_ecef[1], point_ref_ecef[2]]]])
mat_rab_labsar, mat_pos_sat_labsar= projSAR.project_MNT_poivron_sentinel_labsar(product_master, dem_ecef, N_interp=0)

#%%

ind_swath_1 = S1.select_swath(list_xml_1, dem_ecef)
ind_swath_2 = S1.select_swath(list_xml_2, dem_ecef)

#%%

traj_pos = product_master.tabtrajectoire[1]
traj_time=product_master.tabtrajectoire[0]
vit_lum=product_master.celeritelum
time_init_azi=product_master.tempslongstart
time_init_range=product_master.tempscourtstart #pour sentinel, pour terraSAR c'est dans tabcapteur[11]

prf=product_master.prf
rsf=product_master.rsf
array_burst = np.array(product_master.burst)
time_burst_start = array_burst[:,1]
mat_rab_paramlabsar,  mat_pos_sat_paramlabsar= projSAR.project_MNT_poivron_sentinel(traj_pos, traj_time, vit_lum, time_burst_start, time_init_range, prf, rsf, dem_ecef, N_interp=0)

dict_swath_information = S1.getSwathInformation(list_xml_2[3])
state_vector_wtlabsar = np.array(dict_swath_information["state_vector"])
time_burst_start_wtlabsar = np.array(dict_swath_information['burst_azimuth_time'])
time_init_range_wtlabsar = np.array(dict_swath_information['time_init_range'])
prf_wtlabsar = np.array(dict_swath_information['prf'])
rsf_wtlabsar = np.array(dict_swath_information['rsf'])
mat_rab_wtlabsar,  mat_pos_wtlabsar = projSAR.project_MNT_poivron_sentinel(state_vector_wtlabsar[:,1:4], state_vector_wtlabsar[:,0], S1.lightSpeed, time_burst_start_wtlabsar, time_init_range, prf, rsf, dem_ecef, N_interp=10000)

mat_rab_wtlabsar,  mat_pos_wtlabsar = projSAR.project_MNT_poivron_sentinel(state_vector_wtlabsar[:,1:4], state_vector_wtlabsar[:,0], S1.lightSpeed, time_burst_start_wtlabsar, time_init_range_wtlabsar, prf_wtlabsar, rsf_wtlabsar, dem_ecef, N_interp=10000)

#%%

dict_swath_information_IW1 = S1.getSwathInformation(list_xml_2[0])
state_vecto_IW1= np.array(dict_swath_information_IW1["state_vector"])
time_burst_start_IW1 = np.array(dict_swath_information_IW1['burst_azimuth_time'])
time_init_range_IW1 = np.array(dict_swath_information_IW1['time_init_range'])
prf_IW1 = np.array(dict_swath_information_IW1['prf'])
rsf_IW1 = np.array(dict_swath_information_IW1['rsf'])
first_valid_sample_IW1 = np.array(dict_swath_information_IW1['burst_first_valid_sample'])
last_valid_sample_IW1 = np.array(dict_swath_information_IW1['burst_last_valid_sample'])
mat_rab_IW1,  mat_pos_IW1 = projSAR.project_MNT_poivron_sentinel(state_vecto_IW1[:,1:4], state_vecto_IW1[:,0], S1.lightSpeed, time_burst_start_IW1, time_init_range_IW1, prf_IW1, rsf_IW1, dem_ecef, N_interp=10000)
burst = int(mat_rab_IW1[0,0,2])
rangeValue = mat_rab_IW1[0,0,0]
test_IW1 = (rangeValue>first_valid_sample_IW1[burst]) & (rangeValue<last_valid_sample_IW1[burst])


dict_swath_information_IW2 = S1.getSwathInformation(list_xml_2[3])
state_vector_IW2= np.array(dict_swath_information_IW2["state_vector"])
time_burst_start_IW2 = np.array(dict_swath_information_IW2['burst_azimuth_time'])
time_init_range_IW2 = np.array(dict_swath_information_IW2['time_init_range'])
prf_IW2 = np.array(dict_swath_information_IW2['prf'])
rsf_IW2 = np.array(dict_swath_information_IW2['rsf'])
first_valid_sample_IW2 = np.array(dict_swath_information_IW2['burst_first_valid_sample'])
last_valid_sample_IW2 = np.array(dict_swath_information_IW2['burst_last_valid_sample'])
mat_rab_IW2,  mat_pos_IW2 = projSAR.project_MNT_poivron_sentinel(state_vector_IW2[:,1:4], state_vector_IW2[:,0], S1.lightSpeed, time_burst_start_IW2, time_init_range_IW2, prf_IW2, rsf_IW2, dem_ecef, N_interp=10000)
burst = int(mat_rab_IW2[0,0,2])
rangeValue = mat_rab_IW2[0,0,0]
test_IW2 = (rangeValue>first_valid_sample_IW2[burst]) & (rangeValue<last_valid_sample_IW2[burst])

dict_swath_information_IW3 = S1.getSwathInformation(list_xml_2[5])
state_vector_IW3= np.array(dict_swath_information_IW3["state_vector"])
time_burst_start_IW3 = np.array(dict_swath_information_IW3['burst_azimuth_time'])
time_init_range_IW3 = np.array(dict_swath_information_IW3['time_init_range'])
prf_IW3 = np.array(dict_swath_information_IW3['prf'])
rsf_IW3 = np.array(dict_swath_information_IW3['rsf'])
first_valid_sample_IW3 = np.array(dict_swath_information_IW3['burst_first_valid_sample'])
last_valid_sample_IW3 = np.array(dict_swath_information_IW3['burst_last_valid_sample'])
mat_rab_IW3,  mat_pos_IW3 = projSAR.project_MNT_poivron_sentinel(state_vector_IW3[:,1:4], state_vector_IW3[:,0], S1.lightSpeed, time_burst_start_IW3, time_init_range_IW3, prf_IW3, rsf_IW3, dem_ecef, N_interp=10000)
burst = int(mat_rab_IW3[0,0,2])
rangeValue = mat_rab_IW3[0,0,0]
test_IW3 = (rangeValue>first_valid_sample_IW3[burst]) & (rangeValue<last_valid_sample_IW3[burst])


plt.show()
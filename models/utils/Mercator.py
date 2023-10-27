import numpy as np
import os.path
import torch
import math

TILE_SIZE = 256
SAT_SIZE = 640
    
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the real distance between two locations using their GPS coordinates.
    :param lat1: Latitude of location 1
    :param lon1: Longitude of location 1
    :param lat2: Latitude of location 2
    :param lon2: Longitude of location 2
    :return: Real distance between the two locations (in meters)
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Use Haversine Equation 
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371393 # Earth's radius (in meters)
    distance = c * r

    return distance

# Use global x,y and zoom get GPS 
def get_real_lonlat(x, y, scale):
    lon_rel = (x/TILE_SIZE/scale-0.5)*360
    temp = np.exp((0.5-y/TILE_SIZE/scale)*4*np.pi)
    lat_rel = np.arcsin((temp-1)/(temp+1))/np.pi*180
    return lat_rel, lon_rel

# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y

# Get GPS of a pixel based on the current satellite image's GPS and pixel coordinates
def get_loc_lonlat(lat1, lon1, u, v, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)

    ul_pixel_x = int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = int(ul_tile_y * TILE_SIZE)

    lat1_rel,lon1_rel = get_real_lonlat(ul_pixel_x + u,ul_pixel_y+v,scale)
    return lat1_rel,lon1_rel

# Get pixel coordinates on a specific satellite image based on a pixel's GPS location
def get_rel_loc(lat1, lon1, lat, lon, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)

    ul_pixel_x = int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = int(ul_tile_y * TILE_SIZE)

    br_proj_x, br_proj_y = project_with_scale(lat, lon, scale)

    br_pixel_x = br_proj_x * TILE_SIZE
    br_pixel_y = br_proj_y * TILE_SIZE

    return br_pixel_x-ul_pixel_x,br_pixel_y-ul_pixel_y

# Function to calculate GPS coordinates based on given parameters
def get_latlon(lat_real, lon_real, u, v, zoom, sat_size = SAT_SIZE):
    c_u, c_v = get_rel_loc(lat_real, lon_real,lat_real, lon_real, zoom)
    return get_loc_lonlat(lat_real, lon_real,c_u-sat_size/2+u, c_v-sat_size/2+v, zoom)

# Function to calculate pixel coordinates on a satellite image based on given GPS coordinates
def get_pixel(lat_sat, lon_sat, lat, lon, zoom, sat_size = SAT_SIZE):
    c_u, c_v = get_rel_loc(lat_sat, lon_sat,lat_sat, lon_sat, zoom)
    g_u, g_v = get_rel_loc(lat_sat, lon_sat, lat, lon, zoom)
    g_u = g_u- c_u+sat_size/2
    g_v = g_v- c_v+sat_size/2
    return g_u, g_v


############################################
def get_real_lonlat_tensor(x, y, scale):
    lon_rel = (x.double()/TILE_SIZE/scale-0.5)*360
    temp = torch.exp((0.5-y.double()/TILE_SIZE/scale)*4*torch.pi)
    lat_rel = torch.arcsin((temp-1)/(temp+1))/torch.pi*180
    return lat_rel, lon_rel

def project_with_scale_tensor(lat, lon, scale):
    lat = torch.deg2rad(lat.double())
    siny = torch.sin(lat)
    # siny = min(max(siny, -0.9999), 0.9999)    
    x = scale * (0.5 + lon.double()  / 360)
    y = scale * (0.5 - torch.log((1 + siny) / (1 - siny)) / (4 * torch.pi))
    return x, y


def get_loc_lonlat_tensor(lat1, lon1, u, v, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale_tensor(lat1, lon1, scale)
    ul_tile_x = ul_proj_x//1
    ul_tile_y = ul_proj_y//1

    ul_pixel_x = ul_tile_x * TILE_SIZE
    ul_pixel_y = ul_tile_y * TILE_SIZE

    # print(ul_pixel_x + u,ul_pixel_y+v)
    lat1_rel,lon1_rel = get_real_lonlat_tensor(ul_pixel_x + u,ul_pixel_y+v,scale)
    return lat1_rel,lon1_rel

def get_rel_loc_tensor(lat1, lon1, lat, lon, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale_tensor(lat1, lon1, scale)
    # print(ul_proj_x * TILE_SIZE, ul_proj_y* TILE_SIZE)
    ul_tile_x = ul_proj_x//1
    ul_tile_y = ul_proj_y//1

    ul_pixel_x = ul_tile_x * TILE_SIZE # int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = ul_tile_y * TILE_SIZE # int(ul_tile_y * TILE_SIZE)

    br_proj_x, br_proj_y = project_with_scale_tensor(lat, lon, scale)

    br_pixel_x = (br_proj_x * TILE_SIZE) # int(br_proj_x * TILE_SIZE)
    br_pixel_y = (br_proj_y * TILE_SIZE) # int(br_proj_y * TILE_SIZE)
    # print(br_pixel_x, br_pixel_y)

    return br_pixel_x-ul_pixel_x,br_pixel_y-ul_pixel_y

# Function to calculate pixel coordinates on a satellite image based on given GPS coordinates
def get_latlon_tensor(lat_sat, lon_sat, u, v, zoom, sat_size = SAT_SIZE):
    c_u, c_v = get_rel_loc_tensor(lat_sat, lon_sat,lat_sat, lon_sat, zoom)
    return get_loc_lonlat_tensor(lat_sat, lon_sat,c_u-sat_size/2+u, c_v-sat_size/2+v, zoom)

def get_pixel_tensor(lat_sat, lon_sat, lat, lon, zoom, sat_size = SAT_SIZE):
    c_u, c_v = get_rel_loc_tensor(lat_sat, lon_sat,lat_sat, lon_sat, zoom)
    g_u, g_v = get_rel_loc_tensor(lat_sat, lon_sat, lat, lon, zoom)
    g_u = g_u- c_u+sat_size/2
    g_v = g_v- c_v+sat_size/2
    return g_u.float(), g_v.float()

def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = torch.deg2rad(Lat_A.double()) #Lat_A * torch.pi/180.
    lat_B = torch.deg2rad(Lat_B.double())
    lng_A = torch.deg2rad(Lng_A.double())
    lng_B = torch.deg2rad(Lng_B.double())
    R = torch.tensor(6371004.).to(Lat_A.device)  # Earth's radius in meters
    C = torch.sin(lat_A) * torch.sin(lat_B) + torch.cos(lat_A) * torch.cos(lat_B) * torch.cos(lng_A - lng_B)
    C = torch.clamp(C, min=-1.0, max=1.0)
    distance = R * torch.acos(C)
    return distance



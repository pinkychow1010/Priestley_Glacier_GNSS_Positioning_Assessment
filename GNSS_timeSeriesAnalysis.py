# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:15:42 2020
@author: Ka Hei

This script includes the processing of GNSS data in all five stations under pandas dataframe.
Butterworth low pass filter is implemented to reduce noise. Dynamic Time Warping is 
performed on the data and SSA is implemented to discompose the periodical signals.

"""
#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
from scipy.signal import butter,filtfilt
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy import fftpack
from fastdtw import fastdtw
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from pymssa import MSSA
from datetime import datetime, timedelta
#%%
plt.ioff() #turn off plot pop-ups
sns.set() #set up seabornisualization style
#%%
"""Define Function"""
#this function parses DOY into panda time stamps
def parse_doy2date(doy):
    epoch = datetime(datetime.now().year - 3, 12, 31)
    result = epoch + timedelta(days=float(doy))
    return pd.Timestamp(result)   
#this function return scientific values
def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.0E' % x
#this function  set up axis format
def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    for ax in fig.get_axes():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)        
#This is the function for butterworth low pass filter to filter out noise from the GNSS data
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
#this is the function transforming latitude, longitude to decimal degree
def dms2dd(vec):
    """Convert lat/lon from DMS to DD"""
    dd = float(abs(vec[0])) + float(vec[1])/60 + float(vec[2])/(60*60);
    if vec[0] < 0:
        dd *= -1
    return dd;
#this function parse date columns to panda time stamps
def parse_date(year, month, day, hour, minute, sec):
    return pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(hour), \
                        minute=int(minute), second=int(sec))    
#this function transform latitude, longitude from decimal degree to meters
def polar_lonlat_to_xy(vec):
    """Convert from geodetic longitude and latitude to Polar Stereographic
    (X, Y) coordinates in m.
    Args:
        longitude (float): longitude or longitude array in degrees
        latitude (float): latitude or latitude array in degrees (positive)
        true_scale_lat (float): true-scale latitude in degrees
        re (float): Earth radius in km
        e (float): Earth eccentricity
        hemisphere (1 or -1): Northern or Southern hemisphere
    Returns:
        If longitude and latitude are scalars then the result is a
        two-element list containing [X, Y] in m.
        If longitude and latitude are numpy arrays then the result will be a
        two-element list where the first element is a numpy array containing
        the X coordinates and the second element is a numpy array containing
        the Y coordinates.
    """
    #Parameters for WGS84 (the parameters used to transform longitude and latitude to Polar Stereographic coordinates
    #https://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf
    re = 6378.137 #km
    hemisphere = -1 #southern
    e = 0.081819190842622
    true_scale_lat = 71
    
    lat = abs(vec[0]) * np.pi / 180
    lon = vec[1] * np.pi / 180
    slat = true_scale_lat * np.pi / 180
    
    e2 = e * e
    # Snyder (1987) p. 161 Eqn 15-9
    t = np.tan(np.pi / 4 - lat / 2) /  ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2)
    if abs(90 - true_scale_lat) < 1e-5:
        # Snyder (1987) p. 161 Eqn 21-33
        rho = 2 * re * t / np.sqrt((1 + e) ** (1 + e) * (1 - e) ** (1 - e))
    else:
        # Snyder (1987) p. 161 Eqn 21-34
        tc = np.tan(np.pi / 4 - slat / 2) / ((1 - e * np.sin(slat)) / (1 + e * np.sin(slat))) ** (e / 2)
        mc = np.cos(slat) / np.sqrt(1 - e2 * (np.sin(slat) ** 2))
        rho = re * mc * t / tc
    x = rho * hemisphere * np.sin(hemisphere * lon) * 1000
    y = -rho * hemisphere * np.cos(hemisphere * lon) * 1000
    return [x, y]

#%%
"""Station: Base"""
"""Collect files""" #read multiple data file at the same time with glab function and sort them
MAGICf = sorted(glob.glob('magicGNSS_base*.txt'))
CSRSf = sorted(glob.glob('CSRS_base*.pos'))
RTKf = sorted(glob.glob('PPP_RTK_base*.pos'))
magic_full_base=pd.DataFrame()
csrs_full_base=pd.DataFrame()
rtk_full_base=pd.DataFrame()
dataperiod = len(RTKf)
headers = ["year", "month", "day", "hour", "min", "sec", "ddd_lat", "mm_lat", "ss_lat",
           "ddd_lon", "mm_lon", "ss_lon", "h", "lat_sigma", "lon_sigma", "h_sigma"]   
parse_dates = ["year", "month", "day", "hour", "min", "sec"]

for i in range(dataperiod):
    """Input MagicGNSS data"""
    data_m = pd.read_csv(MAGICf[i], delim_whitespace=True, comment='#', usecols=
                         [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18], header=None, names=headers,  
                         parse_dates={'datetime': ['year', 'month', 'day', 'hour', 'min', 'sec']}, 
                         date_parser=parse_date)
    data_m = data_m.set_index(pd.DatetimeIndex(data_m['datetime']))
    #DMS converted to DD
    data_m["latitude_DD"] = data_m[['ddd_lat','mm_lat','ss_lat']].apply(dms2dd, axis=1)
    data_m["longitude_DD"] = data_m[['ddd_lon','mm_lon','ss_lon']].apply(dms2dd, axis=1)
    data_m.columns = ["datetime","ddd_lat", "mm_lat", "ss_lat", "ddd_lon", "mm_lon", "ss_lon"
                      , "h", "lat_sigma", "lon_sigma", "h_sigma", "latitude_DD", "longitude_DD"]
    magic_full_base = magic_full_base.append(data_m, ignore_index=True)

    """Input CSRS data"""
    data_c = pd.read_csv(CSRSf[i],delim_whitespace=True, comment='#', 
                             header=6, usecols=[4,5,10,11,12,15,16,17,20,21,22,23,24,25,26],
                             parse_dates=[['YEAR-MM-DD', 'HR:MN:SS.SS']])
    data_c.rename(columns={'YEAR-MM-DD_HR:MN:SS.SS':'datetime'}, inplace=True)
    data_c = data_c.set_index(pd.DatetimeIndex(data_c['datetime']))
    data_c["latitude_DD"] = data_c[['LATDD','LATMN','LATSS']].apply(dms2dd, axis=1)
    data_c["longitude_DD"] = data_c[['LONDD','LONMN','LONSS']].apply(dms2dd, axis=1)
    data_c.columns = ['datetime', 'DLAT(m)', 'DLON(m)', 'DHGT(m)', 'SDLAT(95%)', 'SDLON(95%)',
                      'SDHGT(95%)', 'LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS', 'HGT(m)',
                      'latitude_DD', 'longitude_DD']
    csrs_full_base = csrs_full_base.append(data_c, ignore_index=True)
    
    """Input RTKLIB data"""
    data_r = pd.read_csv(RTKf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], 
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],
                                             parse_dates=[['date', 'time']])
    data_r.rename(columns={'date_time':'datetime'}, inplace=True)
    data_r = data_r.set_index(pd.DatetimeIndex(data_r['datetime']))
    data_r.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtk_full_base = rtk_full_base.append(data_r, ignore_index=True)
    
#Drop excess columns
magic_full_base = magic_full_base.drop(['ddd_lat', 'mm_lat', 'ss_lat', 'ddd_lon', 'mm_lon', 'ss_lon'], axis=1)
csrs_full_base = csrs_full_base.drop(['LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS'], axis=1)

#Reset index
magic_full_base = magic_full_base.set_index(pd.DatetimeIndex(magic_full_base['datetime']))
csrs_full_base = csrs_full_base.set_index(pd.DatetimeIndex(csrs_full_base['datetime']))
rtk_full_base = rtk_full_base.set_index(pd.DatetimeIndex(rtk_full_base['datetime']))

#1D interpolation
#RTK
new_range_r = pd.date_range(rtk_full_base.datetime[0], rtk_full_base.datetime.values[-1], freq='S')
rtk_full_base = rtk_full_base[~rtk_full_base.index.duplicated()]
rtk_full_base.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#CSRS
new_range_c = pd.date_range(csrs_full_base.datetime[0], csrs_full_base.datetime.values[-1], freq='S')
csrs_full_base = csrs_full_base[~csrs_full_base.index.duplicated()]
csrs_full_base.set_index('datetime').reindex(new_range_c).interpolate(method='time').reset_index()
#MagicGNSS
new_range = pd.date_range(magic_full_base.datetime[0], magic_full_base.datetime.values[-1], freq='S')
magic_full_base = magic_full_base[~magic_full_base.index.duplicated()]
magic_full_base.set_index('datetime').reindex(new_range).interpolate(method='time').reset_index()

#%%
"""Changing x and y from degrees to meters by applying the function"""
magic_full_base[['latitude_M','longitude_M']] = magic_full_base[['latitude_DD','longitude_DD']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

csrs_full_base[['latitude_M','longitude_M']] = csrs_full_base[['latitude_DD','longitude_DD']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_base[['latitude_M','longitude_M']] = rtk_full_base[['latitude(deg)','longitude(deg)']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

#%%
plt.clf()
plt.cla()
plt.close('all') #close plots
#%%
"""Spike Removal"""
rtk_full_base["lat_diff"] = rtk_full_base["latitude_M"].diff().abs()
rtk_lat_cutoff = rtk_full_base["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtk_full_base[(rtk_full_base.index.hour == 23) | (rtk_full_base.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtk_full_base["latitude_M"].loc[daymargin].loc[(rtk_full_base["lat_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_base["lon_diff"] = rtk_full_base["longitude_M"].diff().abs()
rtk_lon_cutoff = rtk_full_base["lon_diff"].median()

for i in range(-5,5):
    rtk_full_base["longitude_M"].loc[daymargin].loc[(rtk_full_base["lon_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_base["h_diff"] = rtk_full_base["height(m)"].diff().abs()
rtk_h_cutoff = rtk_full_base["h_diff"].median()

for i in range(-5,5):
    rtk_full_base["height(m)"].loc[daymargin].loc[(rtk_full_base["h_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtk_full_base["latitude_M"].loc[(rtk_full_base["lat_diff"].shift(i) > rtk_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_base["longitude_M"].loc[(rtk_full_base["lon_diff"].shift(i) > rtk_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_base["height(m)"].loc[(rtk_full_base["h_diff"].shift(i) > rtk_h_cutoff*10)] = np.nan

rtk_full_base = rtk_full_base.interpolate(method='time', axis=0).ffill().bfill()

#%%
"""Demean and Detrend the positioning signals"""
magic_full_base['Demean_detrend_Latitude'] = magic_full_base['latitude_M']
magic_full_base['Demean_detrend_Longitude'] = magic_full_base['longitude_M']
magic_full_base['Demean_detrend_Height'] = magic_full_base['h']

csrs_full_base['Demean_detrend_Latitude'] = csrs_full_base['latitude_M']
csrs_full_base['Demean_detrend_Longitude'] = csrs_full_base['longitude_M']
csrs_full_base['Demean_detrend_Height'] = csrs_full_base['HGT(m)']

rtk_full_base['Demean_detrend_Latitude'] = rtk_full_base['latitude_M']
rtk_full_base['Demean_detrend_Longitude'] = rtk_full_base['longitude_M']
rtk_full_base['Demean_detrend_Height'] = rtk_full_base['height(m)']

#MagicGNSS
magic_full_base['Demean_detrend_Latitude'] = signal.detrend(magic_full_base['Demean_detrend_Latitude'].sub(magic_full_base['Demean_detrend_Latitude'].mean()))
magic_full_base['Demean_detrend_Longitude'] = signal.detrend(magic_full_base['Demean_detrend_Longitude'].sub(magic_full_base['Demean_detrend_Longitude'].mean()))
magic_full_base['Demean_detrend_Height'] = signal.detrend(magic_full_base['Demean_detrend_Height'].sub(magic_full_base['Demean_detrend_Height'].mean()))
#CSRS
csrs_full_base['Demean_detrend_Latitude'] = signal.detrend(csrs_full_base['Demean_detrend_Latitude'].sub(csrs_full_base['Demean_detrend_Latitude'].mean()))
csrs_full_base['Demean_detrend_Longitude'] = signal.detrend(csrs_full_base['Demean_detrend_Longitude'].sub(csrs_full_base['Demean_detrend_Longitude'].mean()))
csrs_full_base['Demean_detrend_Height'] = signal.detrend(csrs_full_base['Demean_detrend_Height'].sub(csrs_full_base['Demean_detrend_Height'].mean()))
#RTK
rtk_full_base['Demean_detrend_Latitude'] = signal.detrend(rtk_full_base['Demean_detrend_Latitude'].sub(rtk_full_base['Demean_detrend_Latitude'].mean()))
rtk_full_base['Demean_detrend_Longitude'] = signal.detrend(rtk_full_base['Demean_detrend_Longitude'].sub(rtk_full_base['Demean_detrend_Longitude'].mean()))
rtk_full_base['Demean_detrend_Height'] = signal.detrend(rtk_full_base['Demean_detrend_Height'].sub(rtk_full_base['Demean_detrend_Height'].mean()))
#%%
"""Fast Fourier Transform for checking signal frequency"""
FFT, axes = plt.subplots(nrows=3,ncols=3,sharey=True,sharex=True,figsize=(30,24))
#ax0 = FFT.add_subplot(331)

axes[0][0].set_xlim(0, 0.006)
axes[0][0].set_ylim(0, 0.15)

FFT.text(0.5,0.04, "Signal Frequency (Hz)", ha="center", va="center", fontsize=16)
FFT.text(0.07,0.5, "Amplitude (m)", ha="center", va="center", rotation=90, fontsize=16)
FFT.text(0.09,0.5, "z                              \
         y                                \
                x", ha="center", va="center", rotation=90, fontsize=16)
#X
#FFT: RTK
rtk_full_base['Latitude_fft'] = fftpack.fft(rtk_full_base['Demean_detrend_Latitude'].values)
rtk_full_base['sig_noise_amp'] = 2 / rtk_full_base['datetime'].size * np.abs(rtk_full_base['Latitude_fft'])
rtk_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(rtk_full_base['datetime'].size, 1))
rtk_full_base.plot(ax=axes[0][0], x='sig_noise_freq', y='sig_noise_amp',legend=False)
#ax1 = FFT.add_subplot(332)

#FFT: Magic
magic_full_base['Latitude_fft'] = fftpack.fft(magic_full_base['Demean_detrend_Latitude'].values)
magic_full_base['sig_noise_amp'] = 2 / magic_full_base['datetime'].size * np.abs(magic_full_base['Latitude_fft'])
magic_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(magic_full_base['datetime'].size, 1))
magic_full_base.plot(ax=axes[0][1], x='sig_noise_freq', y='sig_noise_amp',legend=False)
#ax2 = FFT.add_subplot(333)

#FFT: CSRS
csrs_full_base['Latitude_fft'] = fftpack.fft(csrs_full_base['Demean_detrend_Latitude'].values)
csrs_full_base['sig_noise_amp'] = 2 / csrs_full_base['datetime'].size * np.abs(csrs_full_base['Latitude_fft'])
csrs_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(csrs_full_base['datetime'].size, 1))
csrs_full_base.plot(ax=axes[0][2], x='sig_noise_freq', y='sig_noise_amp',legend=False)

#Y
#ax3 = FFT.add_subplot(334)
#FFT: RTK
rtk_full_base['Longitude_fft'] = fftpack.fft(rtk_full_base['Demean_detrend_Longitude'].values)
rtk_full_base['sig_noise_amp'] = 2 / rtk_full_base['datetime'].size * np.abs(rtk_full_base['Longitude_fft'])
rtk_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(rtk_full_base['datetime'].size, 1))
rtk_full_base.plot(ax=axes[1][0], x='sig_noise_freq', y='sig_noise_amp',legend=False)

#ax4 = FFT.add_subplot(335)
#FFT: Magic
magic_full_base['Longitude_fft'] = fftpack.fft(magic_full_base['Demean_detrend_Longitude'].values)
magic_full_base['sig_noise_amp'] = 2 / magic_full_base['datetime'].size * np.abs(magic_full_base['Longitude_fft'])
magic_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(magic_full_base['datetime'].size, 1))
magic_full_base.plot(ax=axes[1][1], x='sig_noise_freq', y='sig_noise_amp',legend=False)

#ax5 = FFT.add_subplot(336)
#FFT: CSRS
csrs_full_base['Longitude_fft'] = fftpack.fft(csrs_full_base['Demean_detrend_Longitude'].values)
csrs_full_base['sig_noise_amp'] = 2 / csrs_full_base['datetime'].size * np.abs(csrs_full_base['Longitude_fft'])
csrs_full_base['sig_noise_freq'] = np.abs(fftpack.fftfreq(csrs_full_base['datetime'].size, 1))
csrs_full_base.plot(ax=axes[1][2], x='sig_noise_freq', y='sig_noise_amp',legend=False)

#Height
#ax6 = FFT.add_subplot(337)
#FFT: RTK
rtk_full_base['Height_fft'] = fftpack.fft(rtk_full_base['Demean_detrend_Height'].values)
rtk_full_base['sig_noise_amp'] = 2 / rtk_full_base['datetime'].size * np.abs(rtk_full_base['Height_fft'])
rtk_full_base['RTKLIB/PPP'] = np.abs(fftpack.fftfreq(rtk_full_base['datetime'].size, 1))
rtk_full_base.plot(ax=axes[2][0], x='RTKLIB/PPP', y='sig_noise_amp',legend=False)

#ax7 = FFT.add_subplot(338)
#FFT: Magic
magic_full_base['Height_fft'] = fftpack.fft(magic_full_base['Demean_detrend_Height'].values)
magic_full_base['sig_noise_amp'] = 2 / magic_full_base['datetime'].size * np.abs(magic_full_base['Height_fft'])
magic_full_base['magicGNSS/PPP'] = np.abs(fftpack.fftfreq(magic_full_base['datetime'].size, 1))
magic_full_base.plot(ax=axes[2][1], x='magicGNSS/PPP', y='sig_noise_amp',legend=False)

#ax8 = FFT.add_subplot(339)
#FFT: CSRS
csrs_full_base['Height_fft'] = fftpack.fft(csrs_full_base['Demean_detrend_Height'].values)
csrs_full_base['sig_noise_amp'] = 2 / csrs_full_base['datetime'].size * np.abs(csrs_full_base['Height_fft'])
csrs_full_base['CSRS/PPP'] = np.abs(fftpack.fftfreq(csrs_full_base['datetime'].size, 1))
csrs_full_base.plot(ax=axes[2][2], x='CSRS/PPP', y='sig_noise_amp',legend=False)

plt.subplots_adjust(wspace=0.1,hspace=0.15)
plt.show()


#%%
#Base on the results from FFT, desired cutoff frequencies are setted for all packages
"""Applying Butterworth Low Pass Filter"""
T = 60*60*24         # Sample Period
fs = 1      # sample rate, Hz
cutoff = 0.0025      # desired cutoff frequency
cutoff_rtk = 0.0008
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#Filtering Latitude
magic_full_base['Filtered_Latitude'] = butter_lowpass_filter(magic_full_base['Demean_detrend_Latitude'], cutoff, fs, order)
csrs_full_base['Filtered_Latitude'] = butter_lowpass_filter(csrs_full_base['Demean_detrend_Latitude'], cutoff, fs, order)
rtk_full_base['Filtered_Latitude'] = butter_lowpass_filter(rtk_full_base['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
#Filtering Longitude
magic_full_base['Filtered_Longitude'] = butter_lowpass_filter(magic_full_base['Demean_detrend_Longitude'], cutoff, fs, order)
csrs_full_base['Filtered_Longitude'] = butter_lowpass_filter(csrs_full_base['Demean_detrend_Longitude'], cutoff, fs, order)
rtk_full_base['Filtered_Longitude'] = butter_lowpass_filter(rtk_full_base['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
#Filtering Height
magic_full_base['Filtered_Height'] = butter_lowpass_filter(magic_full_base['Demean_detrend_Height'], cutoff, fs, order)
csrs_full_base['Filtered_Height'] = butter_lowpass_filter(csrs_full_base['Demean_detrend_Height'], cutoff, fs, order)
rtk_full_base['Filtered_Height'] = butter_lowpass_filter(rtk_full_base['Demean_detrend_Height'], cutoff_rtk, fs, order)

#%%
"""Plotting filtered signals all together"""
fig9 = plt.figure(figsize=(18,14))
ax1 = fig9.add_subplot(311)
ax1.plot(magic_full_base['Filtered_Latitude'], lw=2)
ax1.plot(csrs_full_base['Filtered_Latitude'], lw=2)
ax1.plot(rtk_full_base['Filtered_Latitude'], lw=2)
ax1.set_ylabel('Latitude (m)', fontsize=12)
ax1.set_title('base (1105-14): Filtered Positioning', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.plot(magic_full_base['Filtered_Longitude'], lw=2)
ax2.plot(csrs_full_base['Filtered_Longitude'], lw=2)
ax2.plot(rtk_full_base['Filtered_Longitude'], lw=2)
ax2.set_ylabel('Longitude (m)', fontsize=12)

ax3 = fig9.add_subplot(313)
ax3.plot(magic_full_base['Filtered_Height'], lw=2)
ax3.plot(csrs_full_base['Filtered_Height'], lw=2)
ax3.plot(rtk_full_base['Filtered_Height'], lw=2)
ax3.set_ylabel('Datetime', fontsize=12)
ax3.set_ylabel('Height (m)', fontsize=12)
ax3.legend(labels = ["magicGNSS","CSRS","RTKLIB"])

#%%
"""Calculating Velocity by differencing filtered x,y,z distance"""
magic_full_base['distx'] = magic_full_base['Filtered_Latitude'].diff().fillna(0.)
magic_full_base['disty'] = magic_full_base['Filtered_Longitude'].diff().fillna(0.)
magic_full_base['distz'] = magic_full_base['Filtered_Height'].diff().fillna(0.)
magic_full_base['horizontal_velocity'] = (np.sqrt(magic_full_base['distx']**2 + magic_full_base['disty']**2)/15)*60*60
magic_full_base['vertical_velocity'] = (magic_full_base['distz']/15)*60*60

csrs_full_base['distx'] = csrs_full_base['Filtered_Latitude'].diff().fillna(0.)
csrs_full_base['disty'] = csrs_full_base['Filtered_Longitude'].diff().fillna(0.)
csrs_full_base['distz'] = csrs_full_base['Filtered_Height'].diff().fillna(0.)
csrs_full_base['horizontal_velocity'] = (np.sqrt(csrs_full_base['distx']**2 + csrs_full_base['disty']**2)/15)*60*60
csrs_full_base['vertical_velocity'] = (csrs_full_base['distz']/15)*60*60

rtk_full_base['distx'] = rtk_full_base['Filtered_Latitude'].diff().fillna(0.)
rtk_full_base['disty'] = rtk_full_base['Filtered_Longitude'].diff().fillna(0.)
rtk_full_base['distz'] = rtk_full_base['Filtered_Height'].diff().fillna(0.)
rtk_full_base['horizontal_velocity'] = (np.sqrt(rtk_full_base['distx']**2 + rtk_full_base['disty']**2)/15)*60*60
rtk_full_base['vertical_velocity'] = (rtk_full_base['distz']/15)*60*60
#unit of velocity is meters per hour

#As noise can still be seen after transforming the filtered position to velocity, low pass filter is applied again
magic_full_base['Filtered_vertical_velocity'] = butter_lowpass_filter(magic_full_base['vertical_velocity'], cutoff_v, fs, order)
csrs_full_base['Filtered_vertical_velocity'] = butter_lowpass_filter(csrs_full_base['vertical_velocity'], cutoff_v, fs, order)
rtk_full_base['Filtered_vertical_velocity'] = butter_lowpass_filter(rtk_full_base['vertical_velocity'], cutoff_v, fs, order)

magic_full_base['Filtered_horizontal_velocity'] = butter_lowpass_filter(magic_full_base['horizontal_velocity'], cutoff_v, fs, order)
csrs_full_base['Filtered_horizontal_velocity'] = butter_lowpass_filter(csrs_full_base['horizontal_velocity'], cutoff_v, fs, order)
rtk_full_base['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtk_full_base['horizontal_velocity'], cutoff_v, fs, order)

fig12 = plt.figure(figsize=(18,14))
ax1 = fig12.add_subplot(111)
ax1.plot(magic_full_base['Filtered_vertical_velocity'], lw=2)
ax1.plot(csrs_full_base['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtk_full_base['Filtered_vertical_velocity'], lw=2)
ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('Vertical velocity (mh^-1)', fontsize=12)
ax1.set_title('base (1105-14): Filtered Vertical Velocity', fontsize=18)
ax1.legend(labels = ["magicGNSS","CSRS","RTKLIB"])

#%%
"""Calculate the Raw Velocity to understand the real glacier movement"""
#we want to know what are the raw velocities calculated by different packages. The values represents 
#the real glacier movement measured by the GNSS stations.

"""Calculating Distance and Velocity"""
magic_full_base['distx'] = magic_full_base['latitude_M'].diff().fillna(0.)
magic_full_base['disty'] = magic_full_base['longitude_M'].diff().fillna(0.)
magic_full_base['distz'] = magic_full_base['h'].diff().fillna(0.)
magic_full_base['raw_horizontal_velocity'] = (np.sqrt(magic_full_base['distx']**2 + magic_full_base['disty']**2)/15)*60*60
magic_full_base['raw_vertical_velocity'] = (magic_full_base['distz']/15)*60*60

csrs_full_base['distx'] = csrs_full_base['latitude_M'].diff().fillna(0.)
csrs_full_base['disty'] = csrs_full_base['longitude_M'].diff().fillna(0.)
csrs_full_base['distz'] = csrs_full_base['HGT(m)'].diff().fillna(0.)
csrs_full_base['raw_horizontal_velocity'] = (np.sqrt(csrs_full_base['distx']**2 + csrs_full_base['disty']**2)/15)*60*60
csrs_full_base['raw_vertical_velocity'] = (csrs_full_base['distz']/15)*60*60

rtk_full_base['distx'] = rtk_full_base['latitude_M'].diff().fillna(0.)
rtk_full_base['disty'] = rtk_full_base['longitude_M'].diff().fillna(0.)
rtk_full_base['distz'] = rtk_full_base['height(m)'].diff().fillna(0.)
rtk_full_base['raw_horizontal_velocity'] = (np.sqrt(rtk_full_base['distx']**2 + rtk_full_base['disty']**2)/15)*60*60
rtk_full_base['raw_vertical_velocity'] = (rtk_full_base['distz']/15)*60*60
#unit of velocity is meters per hour

magic_full_base = magic_full_base.drop(columns=['distx','disty','distz'])
csrs_full_base = csrs_full_base.drop(columns=['distx','disty','distz'])
rtk_full_base = rtk_full_base.drop(columns=['distx','disty','distz'])

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Create multi-index dataframe for summarizing GNSS statistics"""
stat = {'CSRS' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']), 
      'RTK' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'Magic' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar'])} 
base_stat = pd.DataFrame(stat)

base_stat['Magic']['xmean'] = round(magic_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['Magic']['xsd'] = round(magic_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['Magic']['xvar'] = round(magic_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['RTK']['xmean'] = round(rtk_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['RTK']['xsd'] = round(rtk_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['RTK']['xvar'] = round(rtk_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['CSRS']['xmean'] = round(csrs_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['CSRS']['xsd'] = round(csrs_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['CSRS']['xvar'] = round(csrs_full_base['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['Magic']['ymean'] = round(magic_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['Magic']['ysd'] = round(magic_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['Magic']['yvar'] = round(magic_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['RTK']['ymean'] = round(rtk_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['RTK']['ysd'] = round(rtk_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['RTK']['yvar'] = round(rtk_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['CSRS']['ymean'] = round(csrs_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['CSRS']['ysd'] = round(csrs_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['CSRS']['yvar'] = round(csrs_full_base['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['Magic']['zmean'] = round(magic_full_base['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['Magic']['zsd'] = round(magic_full_base['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['Magic']['zvar'] = round(magic_full_base['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['RTK']['zmean'] = round(rtk_full_base['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['RTK']['zsd'] = round(rtk_full_base['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['RTK']['zvar'] = round(rtk_full_base['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

base_stat['CSRS']['zmean'] = round(csrs_full_base['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
base_stat['CSRS']['zsd'] = round(csrs_full_base['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
base_stat['CSRS']['zvar'] = round(csrs_full_base['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)
base_stat.to_csv('base_stat.csv', index=True)  

#%%
"""Plotting statistics"""
"""Base"""
m1 = abs(magic_full_base['latitude_M'].resample('D').mean().to_numpy()[0:10] - 490245.7601)
m2 = abs(rtk_full_base['latitude_M'].resample('D').mean().to_numpy()[0:10] - 490245.7601)
m3 = abs(csrs_full_base['latitude_M'].resample('D').mean().to_numpy()[0:10] - 490245.7601)
m11 = magic_full_base['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22 = rtk_full_base['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m33 = csrs_full_base['latitude_M'].resample('D').std().to_numpy()[0:10]/1000

m4 = abs(magic_full_base['longitude_M'].resample('D').mean().to_numpy()[0:10] - (-1648451.5837))
m5 = abs(rtk_full_base['longitude_M'].resample('D').mean().to_numpy()[0:10] - (-1648451.5837))
m6 = abs(csrs_full_base['longitude_M'].resample('D').mean().to_numpy()[0:10] - (-1648451.5837))
m44 = magic_full_base['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55 = rtk_full_base['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m66 = csrs_full_base['longitude_M'].resample('D').std().to_numpy()[0:10]/1000

m7 = magic_full_base['h'].resample('D').mean().to_numpy()[0:10]
m8 = rtk_full_base['height(m)'].resample('D').mean().to_numpy()[0:10]
m9 = csrs_full_base['HGT(m)'].resample('D').mean().to_numpy()[0:10]
m77 = magic_full_base['h'].resample('D').std().to_numpy()[0:10]/1000
m88 = rtk_full_base['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m99 = csrs_full_base['HGT(m)'].resample('D').std().to_numpy()[0:10]/1000
day = [309,310,311,312,313,314,315,316,317,318]

fig17, axes = plt.subplots(figsize=(18,18))
plt.suptitle("PPP Positioning Statistics (Base)",size=18,y=0.95)

ax01 = plt.subplot2grid((2,3),(0,0))
ax02 = plt.subplot2grid((2,3),(1,0))
ax03 = plt.subplot2grid((2,3),(0,1),colspan = 2)
ax04 = plt.subplot2grid((2,3),(1,1),colspan = 2)

#X coordinates
df1 = pd.DataFrame({'magic': m1, 'rtk': m2, 'csrs': m3}, index=day)
df1 = df1/1000
ax01 = df1[['magic','rtk','csrs']].plot(ax=ax01,kind='bar',grid=True,ylim=[11.148,11.158],width=1, yerr=[m11,m22,m33],legend=False,rot=None)
ax01.set_title("(a) X coordinate")
ax01.set(ylabel="km")

#Y coordinates
df2 = pd.DataFrame({'magic': m4, 'rtk': m5, 'csrs': m6}, index=day)
df2 = df2/1000
ax02 = df2[['magic','rtk','csrs']].plot(ax=ax02,kind='bar',grid=True,ylim=[23.639,23.645],width=1, yerr=[m44,m55,m66],legend=False,rot=None)
ax02.set_title("(b) Y coordinate")
ax02.set(ylabel="km")
ax02.set(xlabel="Day of the Year")

#Height
ax03 = magic_full_base['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03 = rtk_full_base['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03 = csrs_full_base['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03.set_title("(c) Average Height")
ax03.set_ylabel('meter', fontsize=12)

#Velocity
ax04 = magic_full_base['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04 = rtk_full_base['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04 = csrs_full_base['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04.set_title("(d) Vertical velocity")
ax04.set_ylabel('meter/hour', fontsize=12)

ax04.legend(labels = ["magicGNSS","RTKLIB","CSRS"],bbox_to_anchor=(0.8, 1), loc='upper left')
ax02.legend(labels = ["magicGNSS","RTKLIB","CSRS"],bbox_to_anchor=(0.6, 1), loc='upper left')

fig17.subplots_adjust(hspace=0.5)
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Station: Shirase""" #same process applied to Shirase
"""Collect files""" #read multiple data file at the same time with glab function and sort them
MAGICf = sorted(glob.glob('magicGNSS_shirase*.txt')) #PPP
CSRSf = sorted(glob.glob('CSRS_shirase*.pos')) #PPP
RTKf = sorted(glob.glob('PPP_RTK_shirase*.pos')) #DGNSS
RTKDf = sorted(glob.glob('DGNSS_RTK_shirase*.pos')) #DGNSS
magic_full_shirase=pd.DataFrame()
csrs_full_shirase=pd.DataFrame()
rtk_full_shirase=pd.DataFrame()
rtkd_full_shirase=pd.DataFrame()
dataperiod = len(RTKf)
headers = ["year", "month", "day", "hour", "min", "sec", "ddd_lat", "mm_lat", "ss_lat",\
           "ddd_lon", "mm_lon", "ss_lon", "h", "lat_sigma", "lon_sigma", "h_sigma"]   
parse_dates = ["year", "month", "day", "hour", "min", "sec"]

for i in range(dataperiod):
    """Input MagicGNSS data"""
    data_m = pd.read_csv(MAGICf[i], delim_whitespace=True, comment='#', usecols=\
                         [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18], header=None, names=headers,  \
                         parse_dates={'datetime': ['year', 'month', 'day', 'hour', 'min', 'sec']}, \
                         date_parser=parse_date)
    data_m = data_m.set_index(pd.DatetimeIndex(data_m['datetime']))
    #DMS converted to DD
    data_m["latitude_DD"] = data_m[['ddd_lat','mm_lat','ss_lat']].apply(dms2dd, axis=1)
    data_m["longitude_DD"] = data_m[['ddd_lon','mm_lon','ss_lon']].apply(dms2dd, axis=1)
    data_m.columns = ["datetime","ddd_lat", "mm_lat", "ss_lat", "ddd_lon", "mm_lon", "ss_lon"\
                      , "h", "lat_sigma", "lon_sigma", "h_sigma", "latitude_DD", "longitude_DD"]
    magic_full_shirase = magic_full_shirase.append(data_m, ignore_index=True)

    """Input CSRS data"""
    data_c = pd.read_csv(CSRSf[i],delim_whitespace=True, comment='#', \
                             header=6, usecols=[4,5,10,11,12,15,16,17,20,21,22,23,24,25,26],\
                             parse_dates=[['YEAR-MM-DD', 'HR:MN:SS.SS']])
    data_c.rename(columns={'YEAR-MM-DD_HR:MN:SS.SS':'datetime'}, inplace=True)
    data_c = data_c.set_index(pd.DatetimeIndex(data_c['datetime']))
    data_c["latitude_DD"] = data_c[['LATDD','LATMN','LATSS']].apply(dms2dd, axis=1)
    data_c["longitude_DD"] = data_c[['LONDD','LONMN','LONSS']].apply(dms2dd, axis=1)
    data_c.columns = ['datetime', 'DLAT(m)', 'DLON(m)', 'DHGT(m)', 'SDLAT(95%)', 'SDLON(95%)',\
                      'SDHGT(95%)', 'LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS', 'HGT(m)',\
                      'latitude_DD', 'longitude_DD']
    csrs_full_shirase = csrs_full_shirase.append(data_c, ignore_index=True)
    
    """Input RTKLIB PPP data"""
    data_r = pd.read_csv(RTKf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_r.rename(columns={'date_time':'datetime'}, inplace=True)
    data_r = data_r.set_index(pd.DatetimeIndex(data_r['datetime']))
    data_r.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtk_full_shirase = rtk_full_shirase.append(data_r, ignore_index=True)
    
    """Input RTKLIB DGNSS data"""
    data_rd = pd.read_csv(RTKDf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_rd.rename(columns={'date_time':'datetime'}, inplace=True)
    data_rd = data_rd.set_index(pd.DatetimeIndex(data_rd['datetime']))
    data_rd.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtkd_full_shirase = rtkd_full_shirase.append(data_rd, ignore_index=True)
    
#Drop excess columns
magic_full_shirase = magic_full_shirase.drop(['ddd_lat', 'mm_lat', 'ss_lat', 'ddd_lon', 'mm_lon', 'ss_lon'], axis=1)
csrs_full_shirase = csrs_full_shirase.drop(['LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS'], axis=1)

#Reset index
magic_full_shirase = magic_full_shirase.set_index(pd.DatetimeIndex(magic_full_shirase['datetime']))
csrs_full_shirase = csrs_full_shirase.set_index(pd.DatetimeIndex(csrs_full_shirase['datetime']))
rtk_full_shirase = rtk_full_shirase.set_index(pd.DatetimeIndex(rtk_full_shirase['datetime']))
rtkd_full_shirase = rtkd_full_shirase.set_index(pd.DatetimeIndex(rtkd_full_shirase['datetime']))

#1D interpolation
#RTK PPP
new_range_r = pd.date_range(rtk_full_shirase.datetime[0], rtk_full_shirase.datetime.values[-1], freq='S')
rtk_full_shirase = rtk_full_shirase[~rtk_full_shirase.index.duplicated()]
rtk_full_shirase.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#RTK DGNSS
new_range_rd = pd.date_range(rtkd_full_shirase.datetime[0], rtkd_full_shirase.datetime.values[-1], freq='S')
rtkd_full_shirase = rtkd_full_shirase[~rtkd_full_shirase.index.duplicated()]
rtkd_full_shirase.set_index('datetime').reindex(new_range_rd).interpolate(method='time').reset_index()
#CSRS
new_range_c = pd.date_range(csrs_full_shirase.datetime[0], csrs_full_shirase.datetime.values[-1], freq='S')
csrs_full_shirase = csrs_full_shirase[~csrs_full_shirase.index.duplicated()]
csrs_full_shirase.set_index('datetime').reindex(new_range_c).interpolate(method='time').reset_index()
#MagicGNSS
new_range = pd.date_range(magic_full_shirase.datetime[0], magic_full_shirase.datetime.values[-1], freq='S')
magic_full_shirase = magic_full_shirase[~magic_full_shirase.index.duplicated()]
magic_full_shirase.set_index('datetime').reindex(new_range).interpolate(method='time').reset_index()
#%%
"""Changing from degrees to meters"""
"""polar stereographic coordinates"""

magic_full_shirase[['latitude_M','longitude_M']] = magic_full_shirase[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

csrs_full_shirase[['latitude_M','longitude_M']] = csrs_full_shirase[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_shirase[['latitude_M','longitude_M']] = rtk_full_shirase[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtkd_full_shirase[['latitude_M','longitude_M']] = rtkd_full_shirase[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_shirase['latitude_M2'] = rtk_full_shirase['latitude_M']

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Spike Removal RTK PPP""" # extra filter implemented for RTK due to strong noise at day margins
rtk_full_shirase["lat_diff"] = rtk_full_shirase["latitude_M"].diff().abs()
rtk_lat_cutoff = rtk_full_shirase["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtk_full_shirase[(rtk_full_shirase.index.hour == 23) | (rtk_full_shirase.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtk_full_shirase["latitude_M"].loc[daymargin].loc[(rtk_full_shirase["lat_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_shirase["lon_diff"] = rtk_full_shirase["longitude_M"].diff().abs()
rtk_lon_cutoff = rtk_full_shirase["lon_diff"].median()

for i in range(-5,5):
    rtk_full_shirase["longitude_M"].loc[daymargin].loc[(rtk_full_shirase["lon_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_shirase["h_diff"] = rtk_full_shirase["height(m)"].diff().abs()
rtk_h_cutoff = rtk_full_shirase["h_diff"].median()

for i in range(-5,5):
    rtk_full_shirase["height(m)"].loc[daymargin].loc[(rtk_full_shirase["h_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtk_full_shirase["latitude_M"].loc[(rtk_full_shirase["lat_diff"].shift(i) > rtk_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_shirase["longitude_M"].loc[(rtk_full_shirase["lon_diff"].shift(i) > rtk_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_shirase["height(m)"].loc[(rtk_full_shirase["h_diff"].shift(i) > rtk_h_cutoff*10)] = np.nan

rtk_full_shirase = rtk_full_shirase.interpolate(method='time', axis=0).ffill().bfill()

"""Spike Removal RTK DGNSS""" #same for DGNSS
rtkd_full_shirase["lat_diff"] = rtkd_full_shirase["latitude_M"].diff().abs()
rtkd_lat_cutoff = rtkd_full_shirase["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtkd_full_shirase[(rtkd_full_shirase.index.hour == 23) | (rtkd_full_shirase.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtkd_full_shirase["latitude_M"].loc[daymargin].loc[(rtkd_full_shirase["lat_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_shirase["lon_diff"] = rtkd_full_shirase["longitude_M"].diff().abs()
rtkd_lon_cutoff = rtkd_full_shirase["lon_diff"].median()

for i in range(-5,5):
    rtkd_full_shirase["longitude_M"].loc[daymargin].loc[(rtkd_full_shirase["lon_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_shirase["h_diff"] = rtkd_full_shirase["height(m)"].diff().abs()
rtkd_h_cutoff = rtkd_full_shirase["h_diff"].median()

for i in range(-5,5):
    rtkd_full_shirase["height(m)"].loc[daymargin].loc[(rtkd_full_shirase["h_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtkd_full_shirase["latitude_M"].loc[(rtkd_full_shirase["lat_diff"].shift(i) > rtkd_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_shirase["longitude_M"].loc[(rtkd_full_shirase["lon_diff"].shift(i) > rtkd_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_shirase["height(m)"].loc[(rtkd_full_shirase["h_diff"].shift(i) > rtkd_h_cutoff*10)] = np.nan

rtkd_full_shirase = rtkd_full_shirase.interpolate(method='time', axis=0).ffill().bfill()

#%%
"""Demean and Detrend the positioning signals"""

magic_full_shirase['Demean_detrend_Latitude'] = magic_full_shirase['latitude_M']
magic_full_shirase['Demean_detrend_Longitude'] = magic_full_shirase['longitude_M']
magic_full_shirase['Demean_detrend_Height'] = magic_full_shirase['h']

csrs_full_shirase['Demean_detrend_Latitude'] = csrs_full_shirase['latitude_M']
csrs_full_shirase['Demean_detrend_Longitude'] = csrs_full_shirase['longitude_M']
csrs_full_shirase['Demean_detrend_Height'] = csrs_full_shirase['HGT(m)']

rtk_full_shirase['Demean_detrend_Latitude'] = rtk_full_shirase['latitude_M']
rtk_full_shirase['Demean_detrend_Longitude'] = rtk_full_shirase['longitude_M']
rtk_full_shirase['Demean_detrend_Height'] = rtk_full_shirase['height(m)']

rtkd_full_shirase['Demean_detrend_Latitude'] = rtkd_full_shirase['latitude_M']
rtkd_full_shirase['Demean_detrend_Longitude'] = rtkd_full_shirase['longitude_M']
rtkd_full_shirase['Demean_detrend_Height'] = rtkd_full_shirase['height(m)']

#MagicGNSS
magic_full_shirase['Demean_detrend_Latitude'] = signal.detrend(magic_full_shirase['Demean_detrend_Latitude'].sub(magic_full_shirase['Demean_detrend_Latitude'].mean()))
magic_full_shirase['Demean_detrend_Longitude'] = signal.detrend(magic_full_shirase['Demean_detrend_Longitude'].sub(magic_full_shirase['Demean_detrend_Longitude'].mean()))
magic_full_shirase['Demean_detrend_Height'] = signal.detrend(magic_full_shirase['Demean_detrend_Height'].sub(magic_full_shirase['Demean_detrend_Height'].mean()))
#CSRS
csrs_full_shirase['Demean_detrend_Latitude'] = signal.detrend(csrs_full_shirase['Demean_detrend_Latitude'].sub(csrs_full_shirase['Demean_detrend_Latitude'].mean()))
csrs_full_shirase['Demean_detrend_Longitude'] = signal.detrend(csrs_full_shirase['Demean_detrend_Longitude'].sub(csrs_full_shirase['Demean_detrend_Longitude'].mean()))
csrs_full_shirase['Demean_detrend_Height'] = signal.detrend(csrs_full_shirase['Demean_detrend_Height'].sub(csrs_full_shirase['Demean_detrend_Height'].mean()))
#RTK PPP
rtk_full_shirase['Demean_detrend_Latitude'] = signal.detrend(rtk_full_shirase['Demean_detrend_Latitude'].sub(rtk_full_shirase['Demean_detrend_Latitude'].mean()))
rtk_full_shirase['Demean_detrend_Longitude'] = signal.detrend(rtk_full_shirase['Demean_detrend_Longitude'].sub(rtk_full_shirase['Demean_detrend_Longitude'].mean()))
rtk_full_shirase['Demean_detrend_Height'] = signal.detrend(rtk_full_shirase['Demean_detrend_Height'].sub(rtk_full_shirase['Demean_detrend_Height'].mean()))
#RTK DGNSS
rtkd_full_shirase['Demean_detrend_Latitude'] = signal.detrend(rtkd_full_shirase['Demean_detrend_Latitude'].sub(rtkd_full_shirase['Demean_detrend_Latitude'].mean()))
rtkd_full_shirase['Demean_detrend_Longitude'] = signal.detrend(rtkd_full_shirase['Demean_detrend_Longitude'].sub(rtkd_full_shirase['Demean_detrend_Longitude'].mean()))
rtkd_full_shirase['Demean_detrend_Height'] = signal.detrend(rtkd_full_shirase['Demean_detrend_Height'].sub(rtkd_full_shirase['Demean_detrend_Height'].mean()))

#%%
#FFT: RTK/PPP
rtk_full_shirase['Latitude_fft'] = fftpack.fft(rtk_full_shirase['Demean_detrend_Latitude'].values)
rtk_full_shirase['sig_noise_amp'] = 2 / rtk_full_shirase['datetime'].size * np.abs(rtk_full_shirase['Latitude_fft'])
rtk_full_shirase['sig_noise_freq'] = np.abs(fftpack.fftfreq(rtk_full_shirase['datetime'].size, 1))
#rtk_full_shirase.plot(ax=axes[0][0], x='sig_noise_freq', y='sig_noise_amp', title='rtk Latitude: decomposition of noise frequency')
rtk_full_shirase.plot(x='sig_noise_freq', y='sig_noise_amp',legend=False)
plt.show()
#%%
#FFT: RTK/DGNSS
rtkd_full_shirase['Latitude_fft'] = fftpack.fft(rtkd_full_shirase['Demean_detrend_Latitude'].values)
rtkd_full_shirase['sig_noise_amp'] = 2 / rtkd_full_shirase['datetime'].size * np.abs(rtkd_full_shirase['Latitude_fft'])
rtkd_full_shirase['sig_noise_freq'] = np.abs(fftpack.fftfreq(rtkd_full_shirase['datetime'].size, 1))
#rtkd_full_shirase.plot(ax=axes[0][0], x='sig_noise_freq', y='sig_noise_amp', title='rtkd Latitude: decomposition of noise frequency')
rtkd_full_shirase.plot(x='sig_noise_freq', y='sig_noise_amp',legend=False)
plt.show()

#%%

"""Applying Butterworth Low Pass Filter on Shirase data"""
T = 60*60*24         # Sample Period
fs = 1      # sample rate, Hz
cutoff = 0.0025      # desired cutoff frequency
cutoff_rtk = 0.0008
cutoff_rtkd = 0.0005
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#Filtering Latitude
magic_full_shirase['Filtered_Latitude'] = butter_lowpass_filter(magic_full_shirase['Demean_detrend_Latitude'], cutoff, fs, order)
csrs_full_shirase['Filtered_Latitude'] = butter_lowpass_filter(csrs_full_shirase['Demean_detrend_Latitude'], cutoff, fs, order)
rtk_full_shirase['Filtered_Latitude'] = butter_lowpass_filter(rtk_full_shirase['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
rtkd_full_shirase['Filtered_Latitude'] = butter_lowpass_filter(rtkd_full_shirase['Demean_detrend_Latitude'], cutoff_rtkd, fs, order)
#Filtering Longitude
magic_full_shirase['Filtered_Longitude'] = butter_lowpass_filter(magic_full_shirase['Demean_detrend_Longitude'], cutoff, fs, order)
csrs_full_shirase['Filtered_Longitude'] = butter_lowpass_filter(csrs_full_shirase['Demean_detrend_Longitude'], cutoff, fs, order)
rtk_full_shirase['Filtered_Longitude'] = butter_lowpass_filter(rtk_full_shirase['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
rtkd_full_shirase['Filtered_Longitude'] = butter_lowpass_filter(rtkd_full_shirase['Demean_detrend_Longitude'], cutoff_rtkd, fs, order)
#Filtering Height
magic_full_shirase['Filtered_Height'] = butter_lowpass_filter(magic_full_shirase['Demean_detrend_Height'], cutoff, fs, order)
csrs_full_shirase['Filtered_Height'] = butter_lowpass_filter(csrs_full_shirase['Demean_detrend_Height'], cutoff, fs, order)
rtk_full_shirase['Filtered_Height'] = butter_lowpass_filter(rtk_full_shirase['Demean_detrend_Height'], cutoff_rtk, fs, order)
rtkd_full_shirase['Filtered_Height'] = butter_lowpass_filter(rtkd_full_shirase['Demean_detrend_Height'], cutoff_rtkd, fs, order)

#%%
plt.clf()
plt.cla()
#plt.close()
plt.close('all')
#%%
"""Plotting all filtered signals"""
fig9 = plt.figure(figsize=(18,14))
ax1 = fig9.add_subplot(311)
ax1.plot(magic_full_shirase['Filtered_Latitude'], lw=2)
ax1.plot(csrs_full_shirase['Filtered_Latitude'], lw=2)
ax1.plot(rtk_full_shirase['Filtered_Latitude'], lw=2)
ax1.plot(rtkd_full_shirase['Filtered_Latitude'], lw=2)
ax1.set_ylabel('Latitude (m)', fontsize=12)
ax1.set_title('shirase (1105-14): Filtered Positioning', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.plot(magic_full_shirase['Filtered_Longitude'], lw=2)
ax2.plot(csrs_full_shirase['Filtered_Longitude'], lw=2)
ax2.plot(rtk_full_shirase['Filtered_Longitude'], lw=2)
ax2.plot(rtkd_full_shirase['Filtered_Longitude'], lw=2)
ax2.set_ylabel('Longitude (m)', fontsize=12)

ax3 = fig9.add_subplot(313)
ax3.plot(magic_full_shirase['Filtered_Height'], lw=2)
ax3.plot(csrs_full_shirase['Filtered_Height'], lw=2)
ax3.plot(rtk_full_shirase['Filtered_Height'], lw=2)
ax3.plot(rtkd_full_shirase['Filtered_Height'], lw=2)
ax3.set_ylabel('Datetime', fontsize=12)
ax3.set_ylabel('Height (m)', fontsize=12)
ax3.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])

plt.show()

#%%
"""Calculating Velocity by differencing filtered x,y,z distance"""

magic_full_shirase['distx'] = magic_full_shirase['Filtered_Latitude'].diff().fillna(0.)
magic_full_shirase['disty'] = magic_full_shirase['Filtered_Longitude'].diff().fillna(0.)
magic_full_shirase['distz'] = magic_full_shirase['Filtered_Height'].diff().fillna(0.)
magic_full_shirase['horizontal_velocity'] = (np.sqrt(magic_full_shirase['distx']**2 + magic_full_shirase['disty']**2)/15)*60*60
magic_full_shirase['vertical_velocity'] = (magic_full_shirase['distz']/15)*60*60

csrs_full_shirase['distx'] = csrs_full_shirase['Filtered_Latitude'].diff().fillna(0.)
csrs_full_shirase['disty'] = csrs_full_shirase['Filtered_Longitude'].diff().fillna(0.)
csrs_full_shirase['distz'] = csrs_full_shirase['Filtered_Height'].diff().fillna(0.)
csrs_full_shirase['horizontal_velocity'] = (np.sqrt(csrs_full_shirase['distx']**2 + csrs_full_shirase['disty']**2)/15)*60*60
csrs_full_shirase['vertical_velocity'] = (csrs_full_shirase['distz']/15)*60*60

rtk_full_shirase['distx'] = rtk_full_shirase['Filtered_Latitude'].diff().fillna(0.)
rtk_full_shirase['disty'] = rtk_full_shirase['Filtered_Longitude'].diff().fillna(0.)
rtk_full_shirase['distz'] = rtk_full_shirase['Filtered_Height'].diff().fillna(0.)
rtk_full_shirase['horizontal_velocity'] = (np.sqrt(rtk_full_shirase['distx']**2 + rtk_full_shirase['disty']**2)/15)*60*60
rtk_full_shirase['vertical_velocity'] = (rtk_full_shirase['distz']/15)*60*60

rtkd_full_shirase['distx'] = rtkd_full_shirase['Filtered_Latitude'].diff().fillna(0.)
rtkd_full_shirase['disty'] = rtkd_full_shirase['Filtered_Longitude'].diff().fillna(0.)
rtkd_full_shirase['distz'] = rtkd_full_shirase['Filtered_Height'].diff().fillna(0.)
rtkd_full_shirase['horizontal_velocity'] = (np.sqrt(rtkd_full_shirase['distx']**2 + rtkd_full_shirase['disty']**2)/15)*60*60
rtkd_full_shirase['vertical_velocity'] = (rtkd_full_shirase['distz']/15)*60*60
#unit of velocity is meters per hour

cutoff_v = 0.002
magic_full_shirase['Filtered_vertical_velocity'] = butter_lowpass_filter(magic_full_shirase['vertical_velocity'], cutoff_v, fs, order)
csrs_full_shirase['Filtered_vertical_velocity'] = butter_lowpass_filter(csrs_full_shirase['vertical_velocity'], cutoff_v, fs, order)
rtk_full_shirase['Filtered_vertical_velocity'] = butter_lowpass_filter(rtk_full_shirase['vertical_velocity'], cutoff_v, fs, order)
rtkd_full_shirase['Filtered_vertical_velocity'] = butter_lowpass_filter(rtkd_full_shirase['vertical_velocity'], cutoff_v, fs, order)

magic_full_shirase['Filtered_horizontal_velocity'] = butter_lowpass_filter(magic_full_shirase['horizontal_velocity'], cutoff_v, fs, order)
csrs_full_shirase['Filtered_horizontal_velocity'] = butter_lowpass_filter(csrs_full_shirase['horizontal_velocity'], cutoff_v, fs, order)
rtk_full_shirase['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtk_full_shirase['horizontal_velocity'], cutoff_v, fs, order)
rtkd_full_shirase['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtkd_full_shirase['horizontal_velocity'], cutoff_v, fs, order)

fig12 = plt.figure(figsize=(18,14))
ax1 = fig12.add_subplot(111)
ax1.plot(magic_full_shirase['Filtered_vertical_velocity'], lw=2)
ax1.plot(csrs_full_shirase['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtk_full_shirase['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtkd_full_shirase['Filtered_vertical_velocity'], lw=2)
ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('Vertical velocity (mh^-1)', fontsize=12)
ax1.set_title('shirase (1105-14): Filtered Vertical Velocity', fontsize=18)
ax1.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])
#%%
"""Calculate the Raw Velocity to understand the real glacier movement"""
magic_full_shirase['distx'] = magic_full_shirase['latitude_M'].diff().fillna(0.)
magic_full_shirase['disty'] = magic_full_shirase['longitude_M'].diff().fillna(0.)
magic_full_shirase['distz'] = magic_full_shirase['h'].diff().fillna(0.)
magic_full_shirase['raw_horizontal_velocity'] = (np.sqrt(magic_full_shirase['distx']**2 + magic_full_shirase['disty']**2)/15)*60*60
magic_full_shirase['raw_vertical_velocity'] = (magic_full_shirase['distz']/15)*60*60

csrs_full_shirase['distx'] = csrs_full_shirase['latitude_M'].diff().fillna(0.)
csrs_full_shirase['disty'] = csrs_full_shirase['longitude_M'].diff().fillna(0.)
csrs_full_shirase['distz'] = csrs_full_shirase['HGT(m)'].diff().fillna(0.)
csrs_full_shirase['raw_horizontal_velocity'] = (np.sqrt(csrs_full_shirase['distx']**2 + csrs_full_shirase['disty']**2)/15)*60*60
csrs_full_shirase['raw_vertical_velocity'] = (csrs_full_shirase['distz']/15)*60*60

rtk_full_shirase['distx'] = rtk_full_shirase['latitude_M'].diff().fillna(0.)
rtk_full_shirase['disty'] = rtk_full_shirase['longitude_M'].diff().fillna(0.)
rtk_full_shirase['distz'] = rtk_full_shirase['height(m)'].diff().fillna(0.)
rtk_full_shirase['raw_horizontal_velocity'] = (np.sqrt(rtk_full_shirase['distx']**2 + rtk_full_shirase['disty']**2)/15)*60*60
rtk_full_shirase['raw_vertical_velocity'] = (rtk_full_shirase['distz']/15)*60*60

rtkd_full_shirase['distx'] = rtkd_full_shirase['latitude_M'].diff().fillna(0.)
rtkd_full_shirase['disty'] = rtkd_full_shirase['longitude_M'].diff().fillna(0.)
rtkd_full_shirase['distz'] = rtkd_full_shirase['height(m)'].diff().fillna(0.)
rtkd_full_shirase['raw_horizontal_velocity'] = (np.sqrt(rtkd_full_shirase['distx']**2 + rtkd_full_shirase['disty']**2)/15)*60*60
rtkd_full_shirase['raw_vertical_velocity'] = (rtkd_full_shirase['distz']/15)*60*60
#unit of velocity is meters per hour

magic_full_shirase = magic_full_shirase.drop(columns=['distx','disty','distz'])
csrs_full_shirase = csrs_full_shirase.drop(columns=['distx','disty','distz'])
rtk_full_shirase = rtk_full_shirase.drop(columns=['distx','disty','distz'])
rtkd_full_shirase = rtkd_full_shirase.drop(columns=['distx','disty','distz'])
#%%
"""Create dataframe for statistics"""
stat = {'CSRS' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']), 
      'RTK' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'RTKD' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'Magic' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar'])} 
shirase_stat = pd.DataFrame(stat)

shirase_stat['Magic']['xmean'] = round(magic_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['Magic']['xsd'] = round(magic_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['Magic']['xvar'] = round(magic_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTK']['xmean'] = round(rtk_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTK']['xsd'] = round(rtk_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTK']['xvar'] = round(rtk_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTKD']['xmean'] = round(rtkd_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTKD']['xsd'] = round(rtkd_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTKD']['xvar'] = round(rtkd_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['CSRS']['xmean'] = round(csrs_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['CSRS']['xsd'] = round(csrs_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['CSRS']['xvar'] = round(csrs_full_shirase['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['Magic']['ymean'] = round(magic_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['Magic']['ysd'] = round(magic_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['Magic']['yvar'] = round(magic_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTK']['ymean'] = round(rtk_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTK']['ysd'] = round(rtk_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTK']['yvar'] = round(rtk_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTKD']['ymean'] = round(rtkd_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTKD']['ysd'] = round(rtkd_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTKD']['yvar'] = round(rtkd_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['CSRS']['ymean'] = round(csrs_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['CSRS']['ysd'] = round(csrs_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['CSRS']['yvar'] = round(csrs_full_shirase['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['Magic']['zmean'] = round(magic_full_shirase['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['Magic']['zsd'] = round(magic_full_shirase['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['Magic']['zvar'] = round(magic_full_shirase['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTK']['zmean'] = round(rtk_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTK']['zsd'] = round(rtk_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTK']['zvar'] = round(rtk_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['RTKD']['zmean'] = round(rtkd_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['RTKD']['zsd'] = round(rtkd_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['RTKD']['zvar'] = round(rtkd_full_shirase['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

shirase_stat['CSRS']['zmean'] = round(csrs_full_shirase['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
shirase_stat['CSRS']['zsd'] = round(csrs_full_shirase['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
shirase_stat['CSRS']['zvar'] = round(csrs_full_shirase['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)
shirase_stat.to_csv('shirase_stat.csv', index=True)  

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Plotting statistics"""
"""Shirase"""

m1 = abs(magic_full_shirase['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m2 = abs(rtk_full_shirase['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m2d = abs(rtkd_full_shirase['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m3 = abs(csrs_full_shirase['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m11 = magic_full_shirase['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22 = rtk_full_shirase['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22d = rtkd_full_shirase['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m33 = csrs_full_shirase['latitude_M'].resample('D').std().to_numpy()[0:10]/1000

m4 = abs(magic_full_shirase['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m5 = abs(rtk_full_shirase['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m5d = abs(rtkd_full_shirase['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m6 = abs(csrs_full_shirase['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m44 = magic_full_shirase['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55 = rtk_full_shirase['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55d = rtkd_full_shirase['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m66 = csrs_full_shirase['longitude_M'].resample('D').std().to_numpy()[0:10]/1000

m7 = magic_full_shirase['h'].resample('D').mean().to_numpy()[0:10]
m8 = rtk_full_shirase['height(m)'].resample('D').mean().to_numpy()[0:10]
m8d = rtkd_full_shirase['height(m)'].resample('D').mean().to_numpy()[0:10]
m9 = csrs_full_shirase['HGT(m)'].resample('D').mean().to_numpy()[0:10]
m77 = magic_full_shirase['h'].resample('D').std().to_numpy()[0:10]/1000
m88 = rtk_full_shirase['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m88d = rtkd_full_shirase['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m99 = csrs_full_shirase['HGT(m)'].resample('D').std().to_numpy()[0:10]/1000
day = [309,310,311,312,313,314,315,316,317,318]

fig16, axes = plt.subplots(figsize=(18,18))
plt.suptitle("PPP Positioning Statistics (Shirase)",size=18,y=0.95)

ax01 = plt.subplot2grid((2,3),(0,0))
ax02 = plt.subplot2grid((2,3),(1,0))
ax03 = plt.subplot2grid((2,3),(0,1),colspan = 2)
ax04 = plt.subplot2grid((2,3),(1,1),colspan = 2)

df1 = pd.DataFrame({'magic': m1, 'rtk': m2, 'rtkd': m2d, 'csrs': m3}, index=day)
df1 = df1/1000
ax01 = df1[['magic','rtk','rtkd','csrs']].plot(ax=ax01,kind='bar',grid=True,width=1,ylim=[9.375,9.383],yerr=[m11,m22,m22d,m33],legend=False,rot=None)
ax01.set_title("(a) Polar stereographic x coordinates")
ax01.set(ylabel="km")

df2 = pd.DataFrame({'magic': m4, 'rtk': m5, 'rtkd': m5d, 'csrs': m6}, index=day)
df2 = df2/1000
ax02 = df2[['magic','rtk','rtkd','csrs']].plot(ax=ax02,kind='bar',grid=True,width=1,ylim=[15.526,15.534],yerr=[m44,m55,m55d,m66],legend=False,rot=None)
ax02.set_title("(b) Polar stereographic y coordinates")
ax02.set(ylabel="km")
ax02.set(xlabel="Day of the Year")

ax03 = magic_full_shirase['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = rtk_full_shirase['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = rtkd_full_shirase['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = csrs_full_shirase['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03.set_title("(c) Average Height")
ax03.set_ylabel('meter', fontsize=12)

ax04 = magic_full_shirase['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = rtk_full_shirase['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = rtkd_full_shirase['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = csrs_full_shirase['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04.set_title("(d) Vertical velocity")
ax04.set_ylabel('meter/hour', fontsize=12)

ax04.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.8, 1), loc='upper left')
ax02.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.6, 1), loc='upper left')

fig16.subplots_adjust(hspace=0.5)
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Station: Blake""" #repeat for Blake
"""Collect files"""
MAGICf = sorted(glob.glob('magicGNSS_blake*.txt')) #PPP
CSRSf = sorted(glob.glob('CSRS_blake*.pos')) #PPP
RTKf = sorted(glob.glob('PPP_RTK_blake*.pos')) #PPP
RTKDf = sorted(glob.glob('DGNSS_RTK_blake*.pos')) #DGNSS
magic_full_blake=pd.DataFrame()
csrs_full_blake=pd.DataFrame()
rtk_full_blake=pd.DataFrame()
rtkd_full_blake=pd.DataFrame()
dataperiod = len(RTKf)
headers = ["year", "month", "day", "hour", "min", "sec", "ddd_lat", "mm_lat", "ss_lat",\
           "ddd_lon", "mm_lon", "ss_lon", "h", "lat_sigma", "lon_sigma", "h_sigma"]   
parse_dates = ["year", "month", "day", "hour", "min", "sec"]

for i in range(dataperiod):
    """Input MagicGNSS data"""
    data_m = pd.read_csv(MAGICf[i], delim_whitespace=True, comment='#', usecols=\
                         [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18], header=None, names=headers,  \
                         parse_dates={'datetime': ['year', 'month', 'day', 'hour', 'min', 'sec']}, \
                         date_parser=parse_date)
    data_m = data_m.set_index(pd.DatetimeIndex(data_m['datetime']))
    #DMS converted to DD
    data_m["latitude_DD"] = data_m[['ddd_lat','mm_lat','ss_lat']].apply(dms2dd, axis=1)
    data_m["longitude_DD"] = data_m[['ddd_lon','mm_lon','ss_lon']].apply(dms2dd, axis=1)
    data_m.columns = ["datetime","ddd_lat", "mm_lat", "ss_lat", "ddd_lon", "mm_lon", "ss_lon"\
                      , "h", "lat_sigma", "lon_sigma", "h_sigma", "latitude_DD", "longitude_DD"]
    magic_full_blake = magic_full_blake.append(data_m, ignore_index=True)

    """Input CSRS data"""
    data_c = pd.read_csv(CSRSf[i],delim_whitespace=True, comment='#', \
                             header=6, usecols=[4,5,10,11,12,15,16,17,20,21,22,23,24,25,26],\
                             parse_dates=[['YEAR-MM-DD', 'HR:MN:SS.SS']])
    data_c.rename(columns={'YEAR-MM-DD_HR:MN:SS.SS':'datetime'}, inplace=True)
    data_c = data_c.set_index(pd.DatetimeIndex(data_c['datetime']))
    data_c["latitude_DD"] = data_c[['LATDD','LATMN','LATSS']].apply(dms2dd, axis=1)
    data_c["longitude_DD"] = data_c[['LONDD','LONMN','LONSS']].apply(dms2dd, axis=1)
    data_c.columns = ['datetime', 'DLAT(m)', 'DLON(m)', 'DHGT(m)', 'SDLAT(95%)', 'SDLON(95%)',\
                      'SDHGT(95%)', 'LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS', 'HGT(m)',\
                      'latitude_DD', 'longitude_DD']
    csrs_full_blake = csrs_full_blake.append(data_c, ignore_index=True)
    
    """Input RTKLIB PPP data"""
    data_r = pd.read_csv(RTKf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_r.rename(columns={'date_time':'datetime'}, inplace=True)
    data_r = data_r.set_index(pd.DatetimeIndex(data_r['datetime']))
    data_r.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtk_full_blake = rtk_full_blake.append(data_r, ignore_index=True)
    
    """Input RTKLIB DGNSS data"""
    data_rd = pd.read_csv(RTKDf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_rd.rename(columns={'date_time':'datetime'}, inplace=True)
    data_rd = data_rd.set_index(pd.DatetimeIndex(data_rd['datetime']))
    data_rd.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtkd_full_blake = rtkd_full_blake.append(data_rd, ignore_index=True)
    
#Drop excess columns
magic_full_blake = magic_full_blake.drop(['ddd_lat', 'mm_lat', 'ss_lat', 'ddd_lon', 'mm_lon', 'ss_lon'], axis=1)
csrs_full_blake = csrs_full_blake.drop(['LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS'], axis=1)

#Reset index
magic_full_blake = magic_full_blake.set_index(pd.DatetimeIndex(magic_full_blake['datetime']))
csrs_full_blake = csrs_full_blake.set_index(pd.DatetimeIndex(csrs_full_blake['datetime']))
rtk_full_blake = rtk_full_blake.set_index(pd.DatetimeIndex(rtk_full_blake['datetime']))
rtkd_full_blake = rtkd_full_blake.set_index(pd.DatetimeIndex(rtkd_full_blake['datetime']))

#1D interpolation
#RTK PPP
new_range_r = pd.date_range(rtk_full_blake.datetime[0], rtk_full_blake.datetime.values[-1], freq='S')
rtk_full_blake = rtk_full_blake[~rtk_full_blake.index.duplicated()]
rtk_full_blake.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#RTK DGNSS
new_range_rd = pd.date_range(rtkd_full_blake.datetime[0], rtkd_full_blake.datetime.values[-1], freq='S')
rtkd_full_blake = rtkd_full_blake[~rtkd_full_blake.index.duplicated()]
rtkd_full_blake.set_index('datetime').reindex(new_range_rd).interpolate(method='time').reset_index()
#CSRS
new_range_c = pd.date_range(csrs_full_blake.datetime[0], csrs_full_blake.datetime.values[-1], freq='S')
csrs_full_blake = csrs_full_blake[~csrs_full_blake.index.duplicated()]
csrs_full_blake.set_index('datetime').reindex(new_range_c).interpolate(method='time').reset_index()
#MagicGNSS
new_range = pd.date_range(magic_full_blake.datetime[0], magic_full_blake.datetime.values[-1], freq='S')
magic_full_blake = magic_full_blake[~magic_full_blake.index.duplicated()]
magic_full_blake.set_index('datetime').reindex(new_range).interpolate(method='time').reset_index()

#%%
"""polar stereographic coordinates"""
magic_full_blake[['latitude_M','longitude_M']] = magic_full_blake[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

csrs_full_blake[['latitude_M','longitude_M']] = csrs_full_blake[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_blake[['latitude_M','longitude_M']] = rtk_full_blake[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtkd_full_blake[['latitude_M','longitude_M']] = rtkd_full_blake[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_blake['latitude_M2'] = rtk_full_blake['latitude_M']

#%%
"""Spike Removal RTK PPP"""
rtk_full_blake["lat_diff"] = rtk_full_blake["latitude_M"].diff().abs()
rtk_lat_cutoff = rtk_full_blake["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtk_full_blake[(rtk_full_blake.index.hour == 23) | (rtk_full_blake.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtk_full_blake["latitude_M"].loc[daymargin].loc[(rtk_full_blake["lat_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_blake["lon_diff"] = rtk_full_blake["longitude_M"].diff().abs()
rtk_lon_cutoff = rtk_full_blake["lon_diff"].median()

for i in range(-5,5):
    rtk_full_blake["longitude_M"].loc[daymargin].loc[(rtk_full_blake["lon_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_blake["h_diff"] = rtk_full_blake["height(m)"].diff().abs()
rtk_h_cutoff = rtk_full_blake["h_diff"].median()

for i in range(-5,5):
    rtk_full_blake["height(m)"].loc[daymargin].loc[(rtk_full_blake["h_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtk_full_blake["latitude_M"].loc[(rtk_full_blake["lat_diff"].shift(i) > rtk_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_blake["longitude_M"].loc[(rtk_full_blake["lon_diff"].shift(i) > rtk_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_blake["height(m)"].loc[(rtk_full_blake["h_diff"].shift(i) > rtk_h_cutoff*10)] = np.nan

rtk_full_blake = rtk_full_blake.interpolate(method='time', axis=0).ffill().bfill()

"""Spike Removal RTK DGNSS"""
rtkd_full_blake["lat_diff"] = rtkd_full_blake["latitude_M"].diff().abs()
rtkd_lat_cutoff = rtkd_full_blake["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtkd_full_blake[(rtkd_full_blake.index.hour == 23) | (rtkd_full_blake.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtkd_full_blake["latitude_M"].loc[daymargin].loc[(rtkd_full_blake["lat_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan
rtkd_full_blake["lon_diff"] = rtkd_full_blake["longitude_M"].diff().abs()
rtkd_lon_cutoff = rtkd_full_blake["lon_diff"].median()

for i in range(-5,5):
    rtkd_full_blake["longitude_M"].loc[daymargin].loc[(rtkd_full_blake["lon_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan
rtkd_full_blake["h_diff"] = rtkd_full_blake["height(m)"].diff().abs()
rtkd_h_cutoff = rtkd_full_blake["h_diff"].median()

for i in range(-5,5):
    rtkd_full_blake["height(m)"].loc[daymargin].loc[(rtkd_full_blake["h_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtkd_full_blake["latitude_M"].loc[(rtkd_full_blake["lat_diff"].shift(i) > rtkd_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_blake["longitude_M"].loc[(rtkd_full_blake["lon_diff"].shift(i) > rtkd_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_blake["height(m)"].loc[(rtkd_full_blake["h_diff"].shift(i) > rtkd_h_cutoff*10)] = np.nan

rtkd_full_blake = rtkd_full_blake.interpolate(method='time', axis=0).ffill().bfill()


#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Demean and Detrend the positioning signals"""
magic_full_blake['Demean_detrend_Latitude'] = magic_full_blake['latitude_M']
magic_full_blake['Demean_detrend_Longitude'] = magic_full_blake['longitude_M']
magic_full_blake['Demean_detrend_Height'] = magic_full_blake['h']

csrs_full_blake['Demean_detrend_Latitude'] = csrs_full_blake['latitude_M']
csrs_full_blake['Demean_detrend_Longitude'] = csrs_full_blake['longitude_M']
csrs_full_blake['Demean_detrend_Height'] = csrs_full_blake['HGT(m)']

rtk_full_blake['Demean_detrend_Latitude'] = rtk_full_blake['latitude_M']
rtk_full_blake['Demean_detrend_Longitude'] = rtk_full_blake['longitude_M']
rtk_full_blake['Demean_detrend_Height'] = rtk_full_blake['height(m)']

rtkd_full_blake['Demean_detrend_Latitude'] = rtkd_full_blake['latitude_M']
rtkd_full_blake['Demean_detrend_Longitude'] = rtkd_full_blake['longitude_M']
rtkd_full_blake['Demean_detrend_Height'] = rtkd_full_blake['height(m)']

#MagicGNSS
magic_full_blake['Demean_detrend_Latitude'] = signal.detrend(magic_full_blake['Demean_detrend_Latitude'].sub(magic_full_blake['Demean_detrend_Latitude'].mean()))
magic_full_blake['Demean_detrend_Longitude'] = signal.detrend(magic_full_blake['Demean_detrend_Longitude'].sub(magic_full_blake['Demean_detrend_Longitude'].mean()))
magic_full_blake['Demean_detrend_Height'] = signal.detrend(magic_full_blake['Demean_detrend_Height'].sub(magic_full_blake['Demean_detrend_Height'].mean()))
#CSRS
csrs_full_blake['Demean_detrend_Latitude'] = signal.detrend(csrs_full_blake['Demean_detrend_Latitude'].sub(csrs_full_blake['Demean_detrend_Latitude'].mean()))
csrs_full_blake['Demean_detrend_Longitude'] = signal.detrend(csrs_full_blake['Demean_detrend_Longitude'].sub(csrs_full_blake['Demean_detrend_Longitude'].mean()))
csrs_full_blake['Demean_detrend_Height'] = signal.detrend(csrs_full_blake['Demean_detrend_Height'].sub(csrs_full_blake['Demean_detrend_Height'].mean()))
#RTK PPP
rtk_full_blake['Demean_detrend_Latitude'] = signal.detrend(rtk_full_blake['Demean_detrend_Latitude'].sub(rtk_full_blake['Demean_detrend_Latitude'].mean()))
rtk_full_blake['Demean_detrend_Longitude'] = signal.detrend(rtk_full_blake['Demean_detrend_Longitude'].sub(rtk_full_blake['Demean_detrend_Longitude'].mean()))
rtk_full_blake['Demean_detrend_Height'] = signal.detrend(rtk_full_blake['Demean_detrend_Height'].sub(rtk_full_blake['Demean_detrend_Height'].mean()))
#RTK DGNSS
rtkd_full_blake['Demean_detrend_Latitude'] = signal.detrend(rtkd_full_blake['Demean_detrend_Latitude'].sub(rtkd_full_blake['Demean_detrend_Latitude'].mean()))
rtkd_full_blake['Demean_detrend_Longitude'] = signal.detrend(rtkd_full_blake['Demean_detrend_Longitude'].sub(rtkd_full_blake['Demean_detrend_Longitude'].mean()))
rtkd_full_blake['Demean_detrend_Height'] = signal.detrend(rtkd_full_blake['Demean_detrend_Height'].sub(rtkd_full_blake['Demean_detrend_Height'].mean()))

#%%
"""Applying Butterworth Low Pass Filter"""
T = 60*60*24         # Sample Period
fs = 1      # sample rate, Hz
cutoff = 0.0025      # desired cutoff frequency
cutoff_rtk = 0.0008
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples
#Filtering Latitude
magic_full_blake['Filtered_Latitude'] = butter_lowpass_filter(magic_full_blake['Demean_detrend_Latitude'], cutoff, fs, order)
csrs_full_blake['Filtered_Latitude'] = butter_lowpass_filter(csrs_full_blake['Demean_detrend_Latitude'], cutoff, fs, order)
rtk_full_blake['Filtered_Latitude'] = butter_lowpass_filter(rtk_full_blake['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
rtkd_full_blake['Filtered_Latitude'] = butter_lowpass_filter(rtkd_full_blake['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
#Filtering Longitude
magic_full_blake['Filtered_Longitude'] = butter_lowpass_filter(magic_full_blake['Demean_detrend_Longitude'], cutoff, fs, order)
csrs_full_blake['Filtered_Longitude'] = butter_lowpass_filter(csrs_full_blake['Demean_detrend_Longitude'], cutoff, fs, order)
rtk_full_blake['Filtered_Longitude'] = butter_lowpass_filter(rtk_full_blake['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
rtkd_full_blake['Filtered_Longitude'] = butter_lowpass_filter(rtkd_full_blake['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
#Filtering Height
magic_full_blake['Filtered_Height'] = butter_lowpass_filter(magic_full_blake['Demean_detrend_Height'], cutoff, fs, order)
csrs_full_blake['Filtered_Height'] = butter_lowpass_filter(csrs_full_blake['Demean_detrend_Height'], cutoff, fs, order)
rtk_full_blake['Filtered_Height'] = butter_lowpass_filter(rtk_full_blake['Demean_detrend_Height'], cutoff_rtk, fs, order)
rtkd_full_blake['Filtered_Height'] = butter_lowpass_filter(rtkd_full_blake['Demean_detrend_Height'], cutoff_rtk, fs, order)

#%%
"""Plotting all filtered signals"""

fig9 = plt.figure(figsize=(18,14))
ax1 = fig9.add_subplot(311)
ax1.plot(magic_full_blake['Filtered_Latitude'], lw=2)
ax1.plot(csrs_full_blake['Filtered_Latitude'], lw=2)
ax1.plot(rtk_full_blake['Filtered_Latitude'], lw=2)
ax1.plot(rtkd_full_blake['Filtered_Latitude'], lw=2)
ax1.set_ylabel('Latitude (m)', fontsize=12)
ax1.set_title('blake (1105-14): Filtered Positioning', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.plot(magic_full_blake['Filtered_Longitude'], lw=2)
ax2.plot(csrs_full_blake['Filtered_Longitude'], lw=2)
ax2.plot(rtk_full_blake['Filtered_Longitude'], lw=2)
ax2.plot(rtkd_full_blake['Filtered_Longitude'], lw=2)
ax2.set_ylabel('Longitude (m)', fontsize=12)

ax3 = fig9.add_subplot(313)
ax3.plot(magic_full_blake['Filtered_Height'], lw=2)
ax3.plot(csrs_full_blake['Filtered_Height'], lw=2)
ax3.plot(rtk_full_blake['Filtered_Height'], lw=2)
ax3.plot(rtkd_full_blake['Filtered_Height'], lw=2)
ax3.set_ylabel('Datetime', fontsize=12)
ax3.set_ylabel('Height (m)', fontsize=12)
ax3.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])
#%%
"""Calculating Velocity by differencing filtered x,y,z distance"""

magic_full_blake['distx'] = magic_full_blake['Filtered_Latitude'].diff().fillna(0.)
magic_full_blake['disty'] = magic_full_blake['Filtered_Longitude'].diff().fillna(0.)
magic_full_blake['distz'] = magic_full_blake['Filtered_Height'].diff().fillna(0.)
magic_full_blake['horizontal_velocity'] = (np.sqrt(magic_full_blake['distx']**2 + magic_full_blake['disty']**2)/15)*60*60
magic_full_blake['vertical_velocity'] = (magic_full_blake['distz']/15)*60*60

csrs_full_blake['distx'] = csrs_full_blake['Filtered_Latitude'].diff().fillna(0.)
csrs_full_blake['disty'] = csrs_full_blake['Filtered_Longitude'].diff().fillna(0.)
csrs_full_blake['distz'] = csrs_full_blake['Filtered_Height'].diff().fillna(0.)
csrs_full_blake['horizontal_velocity'] = (np.sqrt(csrs_full_blake['distx']**2 + csrs_full_blake['disty']**2)/15)*60*60
csrs_full_blake['vertical_velocity'] = (csrs_full_blake['distz']/15)*60*60

rtk_full_blake['distx'] = rtk_full_blake['Filtered_Latitude'].diff().fillna(0.)
rtk_full_blake['disty'] = rtk_full_blake['Filtered_Longitude'].diff().fillna(0.)
rtk_full_blake['distz'] = rtk_full_blake['Filtered_Height'].diff().fillna(0.)
rtk_full_blake['horizontal_velocity'] = (np.sqrt(rtk_full_blake['distx']**2 + rtk_full_blake['disty']**2)/15)*60*60
rtk_full_blake['vertical_velocity'] = (rtk_full_blake['distz']/15)*60*60

rtkd_full_blake['distx'] = rtkd_full_blake['Filtered_Latitude'].diff().fillna(0.)
rtkd_full_blake['disty'] = rtkd_full_blake['Filtered_Longitude'].diff().fillna(0.)
rtkd_full_blake['distz'] = rtkd_full_blake['Filtered_Height'].diff().fillna(0.)
rtkd_full_blake['horizontal_velocity'] = (np.sqrt(rtkd_full_blake['distx']**2 + rtkd_full_blake['disty']**2)/15)*60*60
rtkd_full_blake['vertical_velocity'] = (rtkd_full_blake['distz']/15)*60*60
#unit of velocity is meters per hour

cutoff_v = 0.002
magic_full_blake['Filtered_vertical_velocity'] = butter_lowpass_filter(magic_full_blake['vertical_velocity'], cutoff_v, fs, order)
csrs_full_blake['Filtered_vertical_velocity'] = butter_lowpass_filter(csrs_full_blake['vertical_velocity'], cutoff_v, fs, order)
rtk_full_blake['Filtered_vertical_velocity'] = butter_lowpass_filter(rtk_full_blake['vertical_velocity'], cutoff_v, fs, order)
rtkd_full_blake['Filtered_vertical_velocity'] = butter_lowpass_filter(rtkd_full_blake['vertical_velocity'], cutoff_v, fs, order)

magic_full_blake['Filtered_horizontal_velocity'] = butter_lowpass_filter(magic_full_blake['horizontal_velocity'], cutoff_v, fs, order)
csrs_full_blake['Filtered_horizontal_velocity'] = butter_lowpass_filter(csrs_full_blake['horizontal_velocity'], cutoff_v, fs, order)
rtk_full_blake['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtk_full_blake['horizontal_velocity'], cutoff_v, fs, order)
rtkd_full_blake['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtkd_full_blake['horizontal_velocity'], cutoff_v, fs, order)

fig12 = plt.figure(figsize=(18,14))
ax1 = fig12.add_subplot(111)
ax1.plot(magic_full_blake['Filtered_vertical_velocity'], lw=2)
ax1.plot(csrs_full_blake['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtk_full_blake['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtkd_full_blake['Filtered_vertical_velocity'], lw=2)
ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('Vertical velocity (mh^-1)', fontsize=12)
ax1.set_title('blake (1105-14): Filtered Vertical Velocity', fontsize=18)
ax1.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])

#%%
"""Calculate the Raw Velocity to understand the real glacier movement"""

magic_full_blake['distx'] = magic_full_blake['latitude_M'].diff().fillna(0.)
magic_full_blake['disty'] = magic_full_blake['longitude_M'].diff().fillna(0.)
magic_full_blake['distz'] = magic_full_blake['h'].diff().fillna(0.)
magic_full_blake['raw_horizontal_velocity'] = (np.sqrt(magic_full_blake['distx']**2 + magic_full_blake['disty']**2)/15)*60*60
magic_full_blake['raw_vertical_velocity'] = (magic_full_blake['distz']/15)*60*60

csrs_full_blake['distx'] = csrs_full_blake['latitude_M'].diff().fillna(0.)
csrs_full_blake['disty'] = csrs_full_blake['longitude_M'].diff().fillna(0.)
csrs_full_blake['distz'] = csrs_full_blake['HGT(m)'].diff().fillna(0.)
csrs_full_blake['raw_horizontal_velocity'] = (np.sqrt(csrs_full_blake['distx']**2 + csrs_full_blake['disty']**2)/15)*60*60
csrs_full_blake['raw_vertical_velocity'] = (csrs_full_blake['distz']/15)*60*60

rtk_full_blake['distx'] = rtk_full_blake['latitude_M'].diff().fillna(0.)
rtk_full_blake['disty'] = rtk_full_blake['longitude_M'].diff().fillna(0.)
rtk_full_blake['distz'] = rtk_full_blake['height(m)'].diff().fillna(0.)
rtk_full_blake['raw_horizontal_velocity'] = (np.sqrt(rtk_full_blake['distx']**2 + rtk_full_blake['disty']**2)/15)*60*60
rtk_full_blake['raw_vertical_velocity'] = (rtk_full_blake['distz']/15)*60*60

rtkd_full_blake['distx'] = rtkd_full_blake['latitude_M'].diff().fillna(0.)
rtkd_full_blake['disty'] = rtkd_full_blake['longitude_M'].diff().fillna(0.)
rtkd_full_blake['distz'] = rtkd_full_blake['height(m)'].diff().fillna(0.)
rtkd_full_blake['raw_horizontal_velocity'] = (np.sqrt(rtkd_full_blake['distx']**2 + rtkd_full_blake['disty']**2)/15)*60*60
rtkd_full_blake['raw_vertical_velocity'] = (rtkd_full_blake['distz']/15)*60*60
#unit of velocity is meters per hour

magic_full_blake = magic_full_blake.drop(columns=['distx','disty','distz'])
csrs_full_blake = csrs_full_blake.drop(columns=['distx','disty','distz'])
rtk_full_blake = rtk_full_blake.drop(columns=['distx','disty','distz'])
rtkd_full_blake = rtkd_full_blake.drop(columns=['distx','disty','distz'])
#%%
"""Create dataframe for statistics"""
stat = {'CSRS' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']), 
      'RTK' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'RTKD' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'Magic' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar'])} 
blake_stat = pd.DataFrame(stat)

blake_stat['Magic']['xmean'] = round(magic_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['Magic']['xsd'] = round(magic_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['Magic']['xvar'] = round(magic_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTK']['xmean'] = round(rtk_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTK']['xsd'] = round(rtk_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTK']['xvar'] = round(rtk_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTKD']['xmean'] = round(rtkd_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTKD']['xsd'] = round(rtkd_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTKD']['xvar'] = round(rtkd_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['CSRS']['xmean'] = round(csrs_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['CSRS']['xsd'] = round(csrs_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['CSRS']['xvar'] = round(csrs_full_blake['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['Magic']['ymean'] = round(magic_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['Magic']['ysd'] = round(magic_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['Magic']['yvar'] = round(magic_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTK']['ymean'] = round(rtk_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTK']['ysd'] = round(rtk_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTK']['yvar'] = round(rtk_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTKD']['ymean'] = round(rtkd_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTKD']['ysd'] = round(rtkd_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTKD']['yvar'] = round(rtkd_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['CSRS']['ymean'] = round(csrs_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['CSRS']['ysd'] = round(csrs_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['CSRS']['yvar'] = round(csrs_full_blake['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['Magic']['zmean'] = round(magic_full_blake['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['Magic']['zsd'] = round(magic_full_blake['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['Magic']['zvar'] = round(magic_full_blake['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTK']['zmean'] = round(rtk_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTK']['zsd'] = round(rtk_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTK']['zvar'] = round(rtk_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['RTKD']['zmean'] = round(rtkd_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['RTKD']['zsd'] = round(rtkd_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['RTKD']['zvar'] = round(rtkd_full_blake['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

blake_stat['CSRS']['zmean'] = round(csrs_full_blake['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
blake_stat['CSRS']['zsd'] = round(csrs_full_blake['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
blake_stat['CSRS']['zvar'] = round(csrs_full_blake['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)
blake_stat.to_csv('blake_stat.csv', index=True)  

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Plotting statistics"""
"""blake"""

m1 = abs(magic_full_blake['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m2 = abs(rtk_full_blake['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m2d = abs(rtkd_full_blake['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m3 = abs(csrs_full_blake['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m11 = magic_full_blake['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22 = rtk_full_blake['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22d = rtkd_full_blake['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m33 = csrs_full_blake['latitude_M'].resample('D').std().to_numpy()[0:10]/1000

m4 = abs(magic_full_blake['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m5 = abs(rtk_full_blake['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m5d = abs(rtkd_full_blake['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m6 = abs(csrs_full_blake['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m44 = magic_full_blake['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55 = rtk_full_blake['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55d = rtkd_full_blake['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m66 = csrs_full_blake['longitude_M'].resample('D').std().to_numpy()[0:10]/1000

m7 = magic_full_blake['h'].resample('D').mean().to_numpy()[0:10]
m8 = rtk_full_blake['height(m)'].resample('D').mean().to_numpy()[0:10]
m8d = rtkd_full_blake['height(m)'].resample('D').mean().to_numpy()[0:10]
m9 = csrs_full_blake['HGT(m)'].resample('D').mean().to_numpy()[0:10]
m77 = magic_full_blake['h'].resample('D').std().to_numpy()[0:10]/1000
m88 = rtk_full_blake['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m88d = rtkd_full_blake['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m99 = csrs_full_blake['HGT(m)'].resample('D').std().to_numpy()[0:10]/1000
day = [309,310,311,312,313,314,315,316,317,318]

fig16, axes = plt.subplots(figsize=(18,18))
plt.suptitle("PPP Positioning Statistics (blake)",size=18,y=0.95)

ax01 = plt.subplot2grid((2,3),(0,0))
ax02 = plt.subplot2grid((2,3),(1,0))
ax03 = plt.subplot2grid((2,3),(0,1),colspan = 2)
ax04 = plt.subplot2grid((2,3),(1,1),colspan = 2)

df1 = pd.DataFrame({'magic': m1, 'rtk': m2, 'rtkd': m2d, 'csrs': m3}, index=day)
df1 = df1/1000
ax01 = df1[['magic','rtk','rtkd','csrs']].plot(ax=ax01,kind='bar',grid=True,width=1,ylim=[4.97,4.974],yerr=[m11,m22,m22d,m33],legend=False,rot=None)
ax01.set_title("(a) Polar stereographic x coordinates")
ax01.set(ylabel="km")

df2 = pd.DataFrame({'magic': m4, 'rtk': m5, 'rtkd': m5d, 'csrs': m6}, index=day)
df2 = df2/1000
ax02 = df2[['magic','rtk','rtkd','csrs']].plot(ax=ax02,kind='bar',grid=True,width=1,ylim=[2.843,2.848],yerr=[m44,m55,m55d,m66],legend=False,rot=None)
ax02.set_title("(b) Polar stereographic y coordinates")
ax02.set(ylabel="km")
ax02.set(xlabel="Day of the Year")

#only plotting 3 day period
ax03 = magic_full_blake['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = rtk_full_blake['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = rtkd_full_blake['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = csrs_full_blake['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03.set_title("(c) Average Height")
ax03.set_ylabel('meter', fontsize=12)

ax04 = magic_full_blake['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = rtk_full_blake['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = rtkd_full_blake['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = csrs_full_shirase['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04.set_title("(d) Vertical velocity")
ax04.set_ylabel('meter/hour', fontsize=12)

ax04.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.8, 1), loc='upper left')
ax02.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.6, 1), loc='upper left')

fig16.subplots_adjust(hspace=0.5)
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Station: Hillary""" #repeat for Hillary
"""Collect files"""
MAGICf = sorted(glob.glob('magicGNSS_hillary*.txt'))
CSRSf = sorted(glob.glob('CSRS_hillary*.pos'))
RTKf = sorted(glob.glob('PPP_RTK_hillary*.pos'))
RTKDf = sorted(glob.glob('DGNSS_RTK_hillary*.pos'))
magic_full_hillary=pd.DataFrame()
csrs_full_hillary=pd.DataFrame()
rtk_full_hillary=pd.DataFrame()
rtkd_full_hillary=pd.DataFrame()
dataperiod = len(RTKf)
headers = ["year", "month", "day", "hour", "min", "sec", "ddd_lat", "mm_lat", "ss_lat",
           "ddd_lon", "mm_lon", "ss_lon", "h", "lat_sigma", "lon_sigma", "h_sigma"]   
parse_dates = ["year", "month", "day", "hour", "min", "sec"]

for i in range(dataperiod):
    """Input MagicGNSS data"""
    data_m = pd.read_csv(MAGICf[i], delim_whitespace=True, comment='#', usecols=
                         [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18], header=None, names=headers,  
                         parse_dates={'datetime': ['year', 'month', 'day', 'hour', 'min', 'sec']}, 
                         date_parser=parse_date)
    data_m = data_m.set_index(pd.DatetimeIndex(data_m['datetime']))
    #DMS converted to DD
    data_m["latitude_DD"] = data_m[['ddd_lat','mm_lat','ss_lat']].apply(dms2dd, axis=1)
    data_m["longitude_DD"] = data_m[['ddd_lon','mm_lon','ss_lon']].apply(dms2dd, axis=1)
    data_m.columns = ["datetime","ddd_lat", "mm_lat", "ss_lat", "ddd_lon", "mm_lon", "ss_lon"
                      , "h", "lat_sigma", "lon_sigma", "h_sigma", "latitude_DD", "longitude_DD"]
    magic_full_hillary = magic_full_hillary.append(data_m, ignore_index=True)
    
    """Input CSRS data"""
    data_c = pd.read_csv(CSRSf[i],delim_whitespace=True, comment='#', 
                             header=6, usecols=[4,5,10,11,12,15,16,17,20,21,22,23,24,25,26],
                             parse_dates=[['YEAR-MM-DD', 'HR:MN:SS.SS']])
    data_c.rename(columns={'YEAR-MM-DD_HR:MN:SS.SS':'datetime'}, inplace=True)
    data_c = data_c.set_index(pd.DatetimeIndex(data_c['datetime']))
    data_c["latitude_DD"] = data_c[['LATDD','LATMN','LATSS']].apply(dms2dd, axis=1)
    data_c["longitude_DD"] = data_c[['LONDD','LONMN','LONSS']].apply(dms2dd, axis=1)
    data_c.columns = ['datetime', 'DLAT(m)', 'DLON(m)', 'DHGT(m)', 'SDLAT(95%)', 'SDLON(95%)',
                      'SDHGT(95%)', 'LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS', 'HGT(m)',
                      'latitude_DD', 'longitude_DD']
    csrs_full_hillary = csrs_full_hillary.append(data_c.iloc[:5760], ignore_index=True)
    """Input RTKLIB PPP data"""
    data_r = pd.read_csv(RTKf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], 
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],
                                             parse_dates=[['date', 'time']])
    data_r.rename(columns={'date_time':'datetime'}, inplace=True)
    data_r = data_r.set_index(pd.DatetimeIndex(data_r['datetime']))
    data_r.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtk_full_hillary = rtk_full_hillary.append(data_r, ignore_index=True)
    """Input RTKLIB DGNSS data"""
    data_rd = pd.read_csv(RTKDf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], 
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],
                                             parse_dates=[['date', 'time']])
    data_rd.rename(columns={'date_time':'datetime'}, inplace=True)
    data_rd = data_rd.set_index(pd.DatetimeIndex(data_rd['datetime']))
    data_rd.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtkd_full_hillary = rtkd_full_hillary.append(data_rd, ignore_index=True)
    
#Drop excess columns
magic_full_hillary = magic_full_hillary.drop(['ddd_lat', 'mm_lat', 'ss_lat', 'ddd_lon', 'mm_lon', 'ss_lon'], axis=1)
csrs_full_hillary = csrs_full_hillary.drop(['LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS'], axis=1)

#Reset index
magic_full_hillary = magic_full_hillary.set_index(pd.DatetimeIndex(magic_full_hillary['datetime']))
csrs_full_hillary = csrs_full_hillary.set_index(pd.DatetimeIndex(csrs_full_hillary['datetime']))
rtk_full_hillary = rtk_full_hillary.set_index(pd.DatetimeIndex(rtk_full_hillary['datetime']))
rtkd_full_hillary = rtkd_full_hillary.set_index(pd.DatetimeIndex(rtkd_full_hillary['datetime']))

#1D interpolation
#RTK PPP
new_range_r = pd.date_range(rtk_full_hillary.datetime[0], rtk_full_hillary.datetime.values[-1], freq='S')
rtk_full_hillary = rtk_full_hillary[~rtk_full_hillary.index.duplicated()]
rtk_full_hillary.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#RTK DGNSS
new_range_rd = pd.date_range(rtkd_full_hillary.datetime[0], rtkd_full_hillary.datetime.values[-1], freq='S')
rtkd_full_hillary = rtkd_full_hillary[~rtkd_full_hillary.index.duplicated()]
rtkd_full_hillary.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#CSRS
new_range_c = pd.date_range(csrs_full_hillary.datetime[0], csrs_full_hillary.datetime.values[-1], freq='S')
csrs_full_hillary = csrs_full_hillary[~csrs_full_hillary.index.duplicated()]
csrs_full_hillary.set_index('datetime').reindex(new_range_c).interpolate(method='time').reset_index()
#MagicGNSS
new_range = pd.date_range(magic_full_hillary.datetime[0], magic_full_hillary.datetime.values[-1], freq='S')
magic_full_hillary = magic_full_hillary[~magic_full_hillary.index.duplicated()]
magic_full_hillary.set_index('datetime').reindex(new_range).interpolate(method='time').reset_index()

#%%
"""Changing from degrees to meters"""
magic_full_hillary[['latitude_M','longitude_M']] = magic_full_hillary[['latitude_DD','longitude_DD']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

csrs_full_hillary[['latitude_M','longitude_M']] = csrs_full_hillary[['latitude_DD','longitude_DD']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_hillary[['latitude_M','longitude_M']] = rtk_full_hillary[['latitude(deg)','longitude(deg)']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtkd_full_hillary[['latitude_M','longitude_M']] = rtkd_full_hillary[['latitude(deg)','longitude(deg)']].apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

#%%
"""Spike Removal PPP"""
rtk_full_hillary["lat_diff"] = rtk_full_hillary["latitude_M"].diff().abs()
rtk_lat_cutoff = rtk_full_hillary["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtk_full_hillary[(rtk_full_hillary.index.hour == 23) | (rtk_full_hillary.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtk_full_hillary["latitude_M"].loc[daymargin].loc[(rtk_full_hillary["lat_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_hillary["lon_diff"] = rtk_full_hillary["longitude_M"].diff().abs()
rtk_lon_cutoff = rtk_full_hillary["lon_diff"].median()

for i in range(-5,5):
    rtk_full_hillary["longitude_M"].loc[daymargin].loc[(rtk_full_hillary["lon_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_hillary["h_diff"] = rtk_full_hillary["height(m)"].diff().abs()
rtk_h_cutoff = rtk_full_hillary["h_diff"].median()

for i in range(-5,5):
    rtk_full_hillary["height(m)"].loc[daymargin].loc[(rtk_full_hillary["h_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtk_full_hillary["latitude_M"].loc[(rtk_full_hillary["lat_diff"].shift(i) > rtk_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_hillary["longitude_M"].loc[(rtk_full_hillary["lon_diff"].shift(i) > rtk_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_hillary["height(m)"].loc[(rtk_full_hillary["h_diff"].shift(i) > rtk_h_cutoff*10)] = np.nan

rtk_full_hillary = rtk_full_hillary.interpolate(method='time', axis=0).ffill().bfill()

#%%
"""Spike Removal DGNSS"""
rtkd_full_hillary["lat_diff"] = rtkd_full_hillary["latitude_M"].diff().abs()
rtkd_lat_cutoff = rtkd_full_hillary["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtkd_full_hillary[(rtkd_full_hillary.index.hour == 23) | (rtkd_full_hillary.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtkd_full_hillary["latitude_M"].loc[daymargin].loc[(rtkd_full_hillary["lat_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_hillary["lon_diff"] = rtkd_full_hillary["longitude_M"].diff().abs()
rtkd_lon_cutoff = rtkd_full_hillary["lon_diff"].median()

for i in range(-5,5):
    rtkd_full_hillary["longitude_M"].loc[daymargin].loc[(rtkd_full_hillary["lon_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_hillary["h_diff"] = rtkd_full_hillary["height(m)"].diff().abs()
rtkd_h_cutoff = rtkd_full_hillary["h_diff"].median()

for i in range(-5,5):
    rtkd_full_hillary["height(m)"].loc[daymargin].loc[(rtkd_full_hillary["h_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtkd_full_hillary["latitude_M"].loc[(rtkd_full_hillary["lat_diff"].shift(i) > rtkd_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_hillary["longitude_M"].loc[(rtkd_full_hillary["lon_diff"].shift(i) > rtkd_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_hillary["height(m)"].loc[(rtkd_full_hillary["h_diff"].shift(i) > rtkd_h_cutoff*10)] = np.nan

rtkd_full_hillary = rtkd_full_hillary.interpolate(method='time', axis=0).ffill().bfill()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Demean and Detrend the positioning signals"""
magic_full_hillary['Demean_detrend_Latitude'] = magic_full_hillary['latitude_M']
magic_full_hillary['Demean_detrend_Longitude'] = magic_full_hillary['longitude_M']
magic_full_hillary['Demean_detrend_Height'] = magic_full_hillary['h']

csrs_full_hillary['Demean_detrend_Latitude'] = csrs_full_hillary['latitude_M']
csrs_full_hillary['Demean_detrend_Longitude'] = csrs_full_hillary['longitude_M']
csrs_full_hillary['Demean_detrend_Height'] = csrs_full_hillary['HGT(m)']

rtk_full_hillary['Demean_detrend_Latitude'] = rtk_full_hillary['latitude_M']
rtk_full_hillary['Demean_detrend_Longitude'] = rtk_full_hillary['longitude_M']
rtk_full_hillary['Demean_detrend_Height'] = rtk_full_hillary['height(m)']

rtkd_full_hillary['Demean_detrend_Latitude'] = rtkd_full_hillary['latitude_M']
rtkd_full_hillary['Demean_detrend_Longitude'] = rtkd_full_hillary['longitude_M']
rtkd_full_hillary['Demean_detrend_Height'] = rtkd_full_hillary['height(m)']

#MagicGNSS
magic_full_hillary['Demean_detrend_Latitude'] = signal.detrend(magic_full_hillary['Demean_detrend_Latitude'].sub(magic_full_hillary['Demean_detrend_Latitude'].mean()))
magic_full_hillary['Demean_detrend_Longitude'] = signal.detrend(magic_full_hillary['Demean_detrend_Longitude'].sub(magic_full_hillary['Demean_detrend_Longitude'].mean()))
magic_full_hillary['Demean_detrend_Height'] = signal.detrend(magic_full_hillary['Demean_detrend_Height'].sub(magic_full_hillary['Demean_detrend_Height'].mean()))
#CSRS
csrs_full_hillary['Demean_detrend_Latitude'] = signal.detrend(csrs_full_hillary['Demean_detrend_Latitude'].sub(csrs_full_hillary['Demean_detrend_Latitude'].mean()))
csrs_full_hillary['Demean_detrend_Longitude'] = signal.detrend(csrs_full_hillary['Demean_detrend_Longitude'].sub(csrs_full_hillary['Demean_detrend_Longitude'].mean()))
csrs_full_hillary['Demean_detrend_Height'] = signal.detrend(csrs_full_hillary['Demean_detrend_Height'].sub(csrs_full_hillary['Demean_detrend_Height'].mean()))
#RTK-PPP
rtk_full_hillary['Demean_detrend_Latitude'] = signal.detrend(rtk_full_hillary['Demean_detrend_Latitude'].sub(rtk_full_hillary['Demean_detrend_Latitude'].mean()))
rtk_full_hillary['Demean_detrend_Longitude'] = signal.detrend(rtk_full_hillary['Demean_detrend_Longitude'].sub(rtk_full_hillary['Demean_detrend_Longitude'].mean()))
rtk_full_hillary['Demean_detrend_Height'] = signal.detrend(rtk_full_hillary['Demean_detrend_Height'].sub(rtk_full_hillary['Demean_detrend_Height'].mean()))
#RTK-DGNSS
rtkd_full_hillary['Demean_detrend_Latitude'] = signal.detrend(rtkd_full_hillary['Demean_detrend_Latitude'].sub(rtkd_full_hillary['Demean_detrend_Latitude'].mean()))
rtkd_full_hillary['Demean_detrend_Longitude'] = signal.detrend(rtkd_full_hillary['Demean_detrend_Longitude'].sub(rtkd_full_hillary['Demean_detrend_Longitude'].mean()))
rtkd_full_hillary['Demean_detrend_Height'] = signal.detrend(rtkd_full_hillary['Demean_detrend_Height'].sub(rtkd_full_hillary['Demean_detrend_Height'].mean()))

"""Applying Butterworth Low Pass Filter"""
T = 60*60*24         # Sample Period
fs = 1      # sample rate, Hz
cutoff = 0.0025      # desired cutoff frequency
cutoff_rtk = 0.0008
cutoff_csrs = 0.0025
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#Filtering Latitude
magic_full_hillary['Filtered_Latitude'] = butter_lowpass_filter(magic_full_hillary['Demean_detrend_Latitude'], cutoff, fs, order)
csrs_full_hillary['Filtered_Latitude'] = butter_lowpass_filter(csrs_full_hillary['Demean_detrend_Latitude'], cutoff_csrs, fs, order)
rtk_full_hillary['Filtered_Latitude'] = butter_lowpass_filter(rtk_full_hillary['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
rtkd_full_hillary['Filtered_Latitude'] = butter_lowpass_filter(rtkd_full_hillary['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
#Filtering Longitude
magic_full_hillary['Filtered_Longitude'] = butter_lowpass_filter(magic_full_hillary['Demean_detrend_Longitude'], cutoff, fs, order)
csrs_full_hillary['Filtered_Longitude'] = butter_lowpass_filter(csrs_full_hillary['Demean_detrend_Longitude'], cutoff_csrs, fs, order)
rtk_full_hillary['Filtered_Longitude'] = butter_lowpass_filter(rtk_full_hillary['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
rtkd_full_hillary['Filtered_Longitude'] = butter_lowpass_filter(rtkd_full_hillary['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
#Filtering Height
magic_full_hillary['Filtered_Height'] = butter_lowpass_filter(magic_full_hillary['Demean_detrend_Height'], cutoff, fs, order)
csrs_full_hillary['Filtered_Height'] = butter_lowpass_filter(csrs_full_hillary['Demean_detrend_Height'], cutoff_csrs, fs, order)
rtk_full_hillary['Filtered_Height'] = butter_lowpass_filter(rtk_full_hillary['Demean_detrend_Height'], cutoff_rtk, fs, order)
rtkd_full_hillary['Filtered_Height'] = butter_lowpass_filter(rtkd_full_hillary['Demean_detrend_Height'], cutoff_rtk, fs, order)

#%%
"""Plotting filtered signals all together"""
fig9 = plt.figure(figsize=(18,14))
ax1 = fig9.add_subplot(311)
ax1.plot(magic_full_hillary['Filtered_Latitude'], lw=2)
ax1.plot(csrs_full_hillary['Filtered_Latitude'], lw=2)
ax1.plot(rtk_full_hillary['Filtered_Latitude'], lw=2)
ax1.plot(rtkd_full_hillary['Filtered_Latitude'], lw=2)
ax1.set_ylabel('Latitude (m)', fontsize=12)
ax1.set_title('hillary (1105-14): Filtered Positioning', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.plot(magic_full_hillary['Filtered_Longitude'], lw=2)
ax2.plot(csrs_full_hillary['Filtered_Longitude'], lw=2)
ax2.plot(rtk_full_hillary['Filtered_Longitude'], lw=2)
ax2.plot(rtkd_full_hillary['Filtered_Longitude'], lw=2)
ax2.set_ylabel('Longitude (m)', fontsize=12)

ax3 = fig9.add_subplot(313)
ax3.plot(magic_full_hillary['Filtered_Height'], lw=2)
ax3.plot(csrs_full_hillary['Filtered_Height'], lw=2)
ax3.plot(rtk_full_hillary['Filtered_Height'], lw=2)
ax3.plot(rtkd_full_hillary['Filtered_Height'], lw=2)
ax3.set_ylabel('Datetime', fontsize=12)
ax3.set_ylabel('Height (m)', fontsize=12)
ax3.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])

#%%
"""Calculating Velocity by differencing filtered x,y,z distance"""
magic_full_hillary['distx'] = magic_full_hillary['Filtered_Latitude'].diff().fillna(0.)
magic_full_hillary['disty'] = magic_full_hillary['Filtered_Longitude'].diff().fillna(0.)
magic_full_hillary['distz'] = magic_full_hillary['Filtered_Height'].diff().fillna(0.)
magic_full_hillary['horizontal_velocity'] = (np.sqrt(magic_full_hillary['distx']**2 + magic_full_hillary['disty']**2)/15)*60*60
magic_full_hillary['vertical_velocity'] = (magic_full_hillary['distz']/15)*60*60

csrs_full_hillary['distx'] = csrs_full_hillary['Filtered_Latitude'].diff().fillna(0.)
csrs_full_hillary['disty'] = csrs_full_hillary['Filtered_Longitude'].diff().fillna(0.)
csrs_full_hillary['distz'] = csrs_full_hillary['Filtered_Height'].diff().fillna(0.)
csrs_full_hillary['horizontal_velocity'] = (np.sqrt(csrs_full_hillary['distx']**2 + csrs_full_hillary['disty']**2)/15)*60*60
csrs_full_hillary['vertical_velocity'] = (csrs_full_hillary['distz']/15)*60*60

rtk_full_hillary['distx'] = rtk_full_hillary['Filtered_Latitude'].diff().fillna(0.)
rtk_full_hillary['disty'] = rtk_full_hillary['Filtered_Longitude'].diff().fillna(0.)
rtk_full_hillary['distz'] = rtk_full_hillary['Filtered_Height'].diff().fillna(0.)
rtk_full_hillary['horizontal_velocity'] = (np.sqrt(rtk_full_hillary['distx']**2 + rtk_full_hillary['disty']**2)/15)*60*60
rtk_full_hillary['vertical_velocity'] = (rtk_full_hillary['distz']/15)*60*60

rtkd_full_hillary['distx'] = rtkd_full_hillary['Filtered_Latitude'].diff().fillna(0.)
rtkd_full_hillary['disty'] = rtkd_full_hillary['Filtered_Longitude'].diff().fillna(0.)
rtkd_full_hillary['distz'] = rtkd_full_hillary['Filtered_Height'].diff().fillna(0.)
rtkd_full_hillary['horizontal_velocity'] = (np.sqrt(rtkd_full_hillary['distx']**2 + rtkd_full_hillary['disty']**2)/15)*60*60
rtkd_full_hillary['vertical_velocity'] = (rtkd_full_hillary['distz']/15)*60*60
#unit of velocity is meters per hour
#Pick the cutoff frequency and filter Vertical Velocity for the second time
#As noise can still be seen after transforming the filtered position to velocity, second low pass filter is applied once again
cutoff_v = 0.0018
magic_full_hillary['Filtered_vertical_velocity'] = butter_lowpass_filter(magic_full_hillary['vertical_velocity'], cutoff_v, fs, order)
csrs_full_hillary['Filtered_vertical_velocity'] = butter_lowpass_filter(csrs_full_hillary['vertical_velocity'], cutoff_v, fs, order)
rtk_full_hillary['Filtered_vertical_velocity'] = butter_lowpass_filter(rtk_full_hillary['vertical_velocity'], cutoff_v, fs, order)
rtkd_full_hillary['Filtered_vertical_velocity'] = butter_lowpass_filter(rtkd_full_hillary['vertical_velocity'], cutoff_v, fs, order)

magic_full_hillary['Filtered_horizontal_velocity'] = butter_lowpass_filter(magic_full_hillary['horizontal_velocity'], cutoff_v, fs, order)
csrs_full_hillary['Filtered_horizontal_velocity'] = butter_lowpass_filter(csrs_full_hillary['horizontal_velocity'], cutoff_v, fs, order)
rtk_full_hillary['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtk_full_hillary['horizontal_velocity'], cutoff_v, fs, order)
rtkd_full_hillary['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtkd_full_hillary['horizontal_velocity'], cutoff_v, fs, order)

fig12 = plt.figure(figsize=(18,14))
ax1 = fig12.add_subplot(111)
ax1.plot(magic_full_hillary['Filtered_vertical_velocity'], lw=2)
ax1.plot(csrs_full_hillary['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtk_full_hillary['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtkd_full_hillary['Filtered_vertical_velocity'], lw=2)
ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('Vertical velocity (mh^-1)', fontsize=12)
ax1.set_title('hillary (1105-14): Filtered Vertical Velocity', fontsize=18)
ax1.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])
#%%
"""Calculate the Raw Velocity to understand the real glacier movement"""
magic_full_hillary['distx'] = magic_full_hillary['latitude_M'].diff().fillna(0.)
magic_full_hillary['disty'] = magic_full_hillary['longitude_M'].diff().fillna(0.)
magic_full_hillary['distz'] = magic_full_hillary['h'].diff().fillna(0.)
magic_full_hillary['raw_horizontal_velocity'] = (np.sqrt(magic_full_hillary['distx']**2 + magic_full_hillary['disty']**2)/15)*60*60
magic_full_hillary['raw_vertical_velocity'] = (magic_full_hillary['distz']/15)*60*60

csrs_full_hillary['distx'] = csrs_full_hillary['latitude_M'].diff().fillna(0.)
csrs_full_hillary['disty'] = csrs_full_hillary['longitude_M'].diff().fillna(0.)
csrs_full_hillary['distz'] = csrs_full_hillary['HGT(m)'].diff().fillna(0.)
csrs_full_hillary['raw_horizontal_velocity'] = (np.sqrt(csrs_full_hillary['distx']**2 + csrs_full_hillary['disty']**2)/15)*60*60
csrs_full_hillary['raw_vertical_velocity'] = (csrs_full_hillary['distz']/15)*60*60

rtk_full_hillary['distx'] = rtk_full_hillary['latitude_M'].diff().fillna(0.)
rtk_full_hillary['disty'] = rtk_full_hillary['longitude_M'].diff().fillna(0.)
rtk_full_hillary['distz'] = rtk_full_hillary['height(m)'].diff().fillna(0.)
rtk_full_hillary['raw_horizontal_velocity'] = (np.sqrt(rtk_full_hillary['distx']**2 + rtk_full_hillary['disty']**2)/15)*60*60
rtk_full_hillary['raw_vertical_velocity'] = (rtk_full_hillary['distz']/15)*60*60

rtkd_full_hillary['distx'] = rtkd_full_hillary['latitude_M'].diff().fillna(0.)
rtkd_full_hillary['disty'] = rtkd_full_hillary['longitude_M'].diff().fillna(0.)
rtkd_full_hillary['distz'] = rtkd_full_hillary['height(m)'].diff().fillna(0.)
rtkd_full_hillary['raw_horizontal_velocity'] = (np.sqrt(rtkd_full_hillary['distx']**2 + rtkd_full_hillary['disty']**2)/15)*60*60
rtkd_full_hillary['raw_vertical_velocity'] = (rtkd_full_hillary['distz']/15)*60*60
#unit of velocity is meters per hour

magic_full_hillary = magic_full_hillary.drop(columns=['distx','disty','distz'])
csrs_full_hillary = csrs_full_hillary.drop(columns=['distx','disty','distz'])
rtk_full_hillary = rtk_full_hillary.drop(columns=['distx','disty','distz'])
rtkd_full_hillary = rtkd_full_hillary.drop(columns=['distx','disty','distz'])

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Statistical Table""" #export a statistical table.csv for the thesis paper

#Create dataframe
stat = {'CSRS' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']), 
      'RTK' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'RTKD' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'Magic' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar'])} 
hillary_stat = pd.DataFrame(stat)

hillary_stat['Magic']['xmean'] = round(magic_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['Magic']['xsd'] = round(magic_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['Magic']['xvar'] = round(magic_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTK']['xmean'] = round(rtk_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTK']['xsd'] = round(rtk_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTK']['xvar'] = round(rtk_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTKD']['xmean'] = round(rtkd_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTKD']['xsd'] = round(rtkd_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTKD']['xvar'] = round(rtkd_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['CSRS']['xmean'] = round(csrs_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['CSRS']['xsd'] = round(csrs_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['CSRS']['xvar'] = round(csrs_full_hillary['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['Magic']['ymean'] = round(magic_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['Magic']['ysd'] = round(magic_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['Magic']['yvar'] = round(magic_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTK']['ymean'] = round(rtk_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTK']['ysd'] = round(rtk_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTK']['yvar'] = round(rtk_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTKD']['ymean'] = round(rtkd_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTKD']['ysd'] = round(rtkd_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTKD']['yvar'] = round(rtkd_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['CSRS']['ymean'] = round(csrs_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['CSRS']['ysd'] = round(csrs_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['CSRS']['yvar'] = round(csrs_full_hillary['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['Magic']['zmean'] = round(magic_full_hillary['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['Magic']['zsd'] = round(magic_full_hillary['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['Magic']['zvar'] = round(magic_full_hillary['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTK']['zmean'] = round(rtk_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTK']['zsd'] = round(rtk_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTK']['zvar'] = round(rtk_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['RTKD']['zmean'] = round(rtkd_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['RTKD']['zsd'] = round(rtkd_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['RTKD']['zvar'] = round(rtkd_full_hillary['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

hillary_stat['CSRS']['zmean'] = round(csrs_full_hillary['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
hillary_stat['CSRS']['zsd'] = round(csrs_full_hillary['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
hillary_stat['CSRS']['zvar'] = round(csrs_full_hillary['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)
hillary_stat.to_csv('hillary_stat.csv', index=True)  

#%%
"""Plotting statistics"""
"""hillary"""

m1 = abs(magic_full_hillary['latitude_M'].resample('D').mean().to_numpy()[0:8] - magic_full_base['latitude_M'].mean())
m2 = abs(rtk_full_hillary['latitude_M'].resample('D').mean().to_numpy()[0:8] - rtk_full_base['latitude_M'].mean())
m2d = abs(rtkd_full_hillary['latitude_M'].resample('D').mean().to_numpy()[0:8] - rtk_full_base['latitude_M'].mean())
m3 = abs(csrs_full_hillary['latitude_M'].resample('D').mean().to_numpy()[0:8] - magic_full_base['latitude_M'].mean())
m11 = magic_full_hillary['latitude_M'].resample('D').std().to_numpy()[0:8]/1000
m22 = rtk_full_hillary['latitude_M'].resample('D').std().to_numpy()[0:8]/1000
m22d = rtkd_full_hillary['latitude_M'].resample('D').std().to_numpy()[0:8]/1000
m33 = csrs_full_hillary['latitude_M'].resample('D').std().to_numpy()[0:8]/1000

m4 = abs(magic_full_hillary['longitude_M'].resample('D').mean().to_numpy()[0:8] - magic_full_base['longitude_M'].mean())
m5 = abs(rtk_full_hillary['longitude_M'].resample('D').mean().to_numpy()[0:8] - rtk_full_base['longitude_M'].mean())
m5d = abs(rtkd_full_hillary['longitude_M'].resample('D').mean().to_numpy()[0:8] - rtk_full_base['longitude_M'].mean())
m6 = abs(csrs_full_hillary['longitude_M'].resample('D').mean().to_numpy()[0:8] - magic_full_base['longitude_M'].mean())
m44 = magic_full_hillary['longitude_M'].resample('D').std().to_numpy()[0:8]/1000
m55 = rtk_full_hillary['longitude_M'].resample('D').std().to_numpy()[0:8]/1000
m55d = rtkd_full_hillary['longitude_M'].resample('D').std().to_numpy()[0:8]/1000
m66 = csrs_full_hillary['longitude_M'].resample('D').std().to_numpy()[0:8]/1000

m7 = magic_full_hillary['h'].resample('D').mean().to_numpy()[0:8]
m8 = rtk_full_hillary['height(m)'].resample('D').mean().to_numpy()[0:8]
m8d = rtkd_full_hillary['height(m)'].resample('D').mean().to_numpy()[0:8]
m9 = csrs_full_hillary['HGT(m)'].resample('D').mean().to_numpy()[0:8]
m77 = magic_full_hillary['h'].resample('D').std().to_numpy()[0:8]/1000
m88 = rtk_full_hillary['height(m)'].resample('D').std().to_numpy()[0:8]/1000
m88d = rtkd_full_hillary['height(m)'].resample('D').std().to_numpy()[0:8]/1000
m99 = csrs_full_hillary['HGT(m)'].resample('D').std().to_numpy()[0:8]/1000

sd1 = [magic_full_hillary['latitude_M'].std(),rtk_full_hillary['latitude_M'].std(),rtkd_full_hillary['latitude_M'].std(),csrs_full_hillary['latitude_M'].std()]
sd2 = [magic_full_hillary['longitude_M'].std(),rtk_full_hillary['longitude_M'].std(),rtkd_full_hillary['longitude_M'].std(),csrs_full_hillary['longitude_M'].std()]
sd3 = [magic_full_hillary['h'].std(),rtk_full_hillary['height(m)'].std(),rtkd_full_hillary['height(m)'].std(),csrs_full_hillary['HGT(m)'].std()]
day = [309,310,311,312,313,314,315,316]

fig18, axes = plt.subplots(figsize=(18,18))
plt.suptitle("PPP Positioning Statistics (hillary)",size=18,y=0.95)

ax01 = plt.subplot2grid((2,3),(0,0))
ax02 = plt.subplot2grid((2,3),(1,0))
ax03 = plt.subplot2grid((2,3),(0,1),colspan = 2)
ax04 = plt.subplot2grid((2,3),(1,1),colspan = 2)

df1 = pd.DataFrame({'magic': m1, 'rtk': m2,'rtkd': m2d, 'csrs': m3}, index=day)
df1 = df1/1000
ax01 = df1[['magic','rtk','rtkd','csrs']].plot(ax=ax01,kind='bar',grid=True,ylim=[4.968,4.976],width=1, yerr=[m11,m22,m22d,m33],legend=False,rot=None)
ax01.set_title("(a) Polar stereographic x coordinates")
ax01.set(ylabel="km")

df2 = pd.DataFrame({'magic': m4, 'rtk': m5,'rtkd': m5d, 'csrs': m6}, index=day)
df2 = df2/1000
ax02 = df2[['magic','rtk','rtkd','csrs']].plot(ax=ax02,kind='bar',grid=True,ylim=[1.212,1.22],width=1, yerr=[m44,m55,m55d,m66],legend=False,rot=None)
ax02.set_title("(b) Polar stereographic y coordinates")
ax02.set(ylabel="km")
ax02.set(xlabel="Day of the Year")

ax03 = magic_full_hillary['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03 = rtk_full_hillary['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03 = rtkd_full_hillary['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03 = csrs_full_hillary['Filtered_Height'].loc["2018-11-11":"2018-11-12"].plot(ax=ax03,grid=True)
ax03.set_title("(c) Average Height")
ax03.set_ylabel('meter', fontsize=12)

ax04 = magic_full_hillary['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04 = rtk_full_hillary['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04 = rtkd_full_hillary['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04 = csrs_full_hillary['Filtered_vertical_velocity'].loc["2018-11-11":"2018-11-12"].plot(ax=ax04,grid=True)
ax04.set_title("(d) Vertical velocity")
ax04.set_ylabel('meter/hour', fontsize=12)

ax04.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.8, 1), loc='upper left')
ax02.legend(labels = ["magicGNSS","RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.6, 1), loc='upper left')

fig18.subplots_adjust(hspace=0.5)
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Station: Tuati""" #repeat for Tuati
"""Collect files"""
MAGICf = sorted(glob.glob('magicGNSS_tuati*.txt')) #PPP
CSRSf = sorted(glob.glob('CSRS_tuati*.pos')) #PPP
RTKf = sorted(glob.glob('PPP_RTK_tuati*.pos')) #PPP
RTKDf = sorted(glob.glob('DGNSS_RTK_tuati*.pos')) #DGNSS
magic_full_tuati=pd.DataFrame()
csrs_full_tuati=pd.DataFrame()
rtk_full_tuati=pd.DataFrame()
rtkd_full_tuati=pd.DataFrame()
dataperiod = len(RTKf)
headers = ["year", "month", "day", "hour", "min", "sec", "ddd_lat", "mm_lat", "ss_lat",\
           "ddd_lon", "mm_lon", "ss_lon", "h", "lat_sigma", "lon_sigma", "h_sigma"]   
parse_dates = ["year", "month", "day", "hour", "min", "sec"]

for i in range(dataperiod):
    """Input MagicGNSS data"""
    data_m = pd.read_csv(MAGICf[i], delim_whitespace=True, comment='#', usecols=\
                         [0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18], header=None, names=headers,  \
                         parse_dates={'datetime': ['year', 'month', 'day', 'hour', 'min', 'sec']}, \
                         date_parser=parse_date)
    data_m = data_m.set_index(pd.DatetimeIndex(data_m['datetime']))
    #DMS converted to DD
    data_m["latitude_DD"] = data_m[['ddd_lat','mm_lat','ss_lat']].apply(dms2dd, axis=1)
    data_m["longitude_DD"] = data_m[['ddd_lon','mm_lon','ss_lon']].apply(dms2dd, axis=1)
    data_m.columns = ["datetime","ddd_lat", "mm_lat", "ss_lat", "ddd_lon", "mm_lon", "ss_lon"\
                      , "h", "lat_sigma", "lon_sigma", "h_sigma", "latitude_DD", "longitude_DD"]
    magic_full_tuati = magic_full_tuati.append(data_m, ignore_index=True)

    """Input CSRS data"""
    data_c = pd.read_csv(CSRSf[i],delim_whitespace=True, comment='#', \
                             header=6, usecols=[4,5,10,11,12,15,16,17,20,21,22,23,24,25,26],\
                             parse_dates=[['YEAR-MM-DD', 'HR:MN:SS.SS']])
    data_c.rename(columns={'YEAR-MM-DD_HR:MN:SS.SS':'datetime'}, inplace=True)
    data_c = data_c.set_index(pd.DatetimeIndex(data_c['datetime']))
    data_c["latitude_DD"] = data_c[['LATDD','LATMN','LATSS']].apply(dms2dd, axis=1)
    data_c["longitude_DD"] = data_c[['LONDD','LONMN','LONSS']].apply(dms2dd, axis=1)
    data_c.columns = ['datetime', 'DLAT(m)', 'DLON(m)', 'DHGT(m)', 'SDLAT(95%)', 'SDLON(95%)',\
                      'SDHGT(95%)', 'LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS', 'HGT(m)',\
                      'latitude_DD', 'longitude_DD']
    csrs_full_tuati = csrs_full_tuati.append(data_c, ignore_index=True)
    
    """Input RTKLIB PPP data"""
    data_r = pd.read_csv(RTKf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_r.rename(columns={'date_time':'datetime'}, inplace=True)
    data_r = data_r.set_index(pd.DatetimeIndex(data_r['datetime']))
    data_r.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtk_full_tuati = rtk_full_tuati.append(data_r, ignore_index=True)
    
    """Input RTKLIB DGNSS data"""
    data_rd = pd.read_csv(RTKDf[i], delim_whitespace=True, comment='%', usecols=[0,1,2,3,4,7,8,9], \
                         header=None, names=['date','time','latitude(deg)','longitude(deg)','height(m)',\
                                             'latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'],\
                                             parse_dates=[['date', 'time']])
    data_rd.rename(columns={'date_time':'datetime'}, inplace=True)
    data_rd = data_rd.set_index(pd.DatetimeIndex(data_rd['datetime']))
    data_rd.columns = ["datetime","latitude(deg)","longitude(deg)","height(m)","latitude_sigma(m)"\
                      ,"longitude_sigma(m)","height_sigma(m)"]
    rtkd_full_tuati = rtkd_full_tuati.append(data_rd, ignore_index=True)
    
#Drop excess columns
magic_full_tuati = magic_full_tuati.drop(['ddd_lat', 'mm_lat', 'ss_lat', 'ddd_lon', 'mm_lon', 'ss_lon'], axis=1)
csrs_full_tuati = csrs_full_tuati.drop(['LATDD', 'LATMN', 'LATSS', 'LONDD', 'LONMN', 'LONSS'], axis=1)

#Reset index
magic_full_tuati = magic_full_tuati.set_index(pd.DatetimeIndex(magic_full_tuati['datetime']))
csrs_full_tuati = csrs_full_tuati.set_index(pd.DatetimeIndex(csrs_full_tuati['datetime']))
rtk_full_tuati = rtk_full_tuati.set_index(pd.DatetimeIndex(rtk_full_tuati['datetime']))
rtkd_full_tuati = rtkd_full_tuati.set_index(pd.DatetimeIndex(rtkd_full_tuati['datetime']))

#"""1D interpolation"""
#RTK PPP
new_range_r = pd.date_range(rtk_full_tuati.datetime[0], rtk_full_tuati.datetime.values[-1], freq='S')
rtk_full_tuati = rtk_full_tuati[~rtk_full_tuati.index.duplicated()]
rtk_full_tuati.set_index('datetime').reindex(new_range_r).interpolate(method='time').reset_index()
#RTK DGNSS
new_range_rd = pd.date_range(rtkd_full_tuati.datetime[0], rtkd_full_tuati.datetime.values[-1], freq='S')
rtkd_full_tuati = rtkd_full_tuati[~rtkd_full_tuati.index.duplicated()]
rtkd_full_tuati.set_index('datetime').reindex(new_range_rd).interpolate(method='time').reset_index()
#CSRS
new_range_c = pd.date_range(csrs_full_tuati.datetime[0], csrs_full_tuati.datetime.values[-1], freq='S')
csrs_full_tuati = csrs_full_tuati[~csrs_full_tuati.index.duplicated()]
csrs_full_tuati.set_index('datetime').reindex(new_range_c).interpolate(method='time').reset_index()
#MagicGNSS
new_range = pd.date_range(magic_full_tuati.datetime[0], magic_full_tuati.datetime.values[-1], freq='S')
magic_full_tuati = magic_full_tuati[~magic_full_tuati.index.duplicated()]
magic_full_tuati.set_index('datetime').reindex(new_range).interpolate(method='time').reset_index()
#%%
"""polar stereographic coordinates"""
magic_full_tuati[['latitude_M','longitude_M']] = magic_full_tuati[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

csrs_full_tuati[['latitude_M','longitude_M']] = csrs_full_tuati[['latitude_DD','longitude_DD']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_tuati[['latitude_M','longitude_M']] = rtk_full_tuati[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtkd_full_tuati[['latitude_M','longitude_M']] = rtkd_full_tuati[['latitude(deg)','longitude(deg)']].\
apply(polar_lonlat_to_xy, axis = 1, result_type="expand")

rtk_full_tuati['latitude_M2'] = rtk_full_tuati['latitude_M']

#%%
"""Spike Removal RTK PPP"""
rtk_full_tuati["lat_diff"] = rtk_full_tuati["latitude_M"].diff().abs()
rtk_lat_cutoff = rtk_full_tuati["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtk_full_tuati[(rtk_full_tuati.index.hour == 23) | (rtk_full_tuati.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtk_full_tuati["latitude_M"].loc[daymargin].loc[(rtk_full_tuati["lat_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_tuati["lon_diff"] = rtk_full_tuati["longitude_M"].diff().abs()
rtk_lon_cutoff = rtk_full_tuati["lon_diff"].median()

for i in range(-5,5):
    rtk_full_tuati["longitude_M"].loc[daymargin].loc[(rtk_full_tuati["lon_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

rtk_full_tuati["h_diff"] = rtk_full_tuati["height(m)"].diff().abs()
rtk_h_cutoff = rtk_full_tuati["h_diff"].median()

for i in range(-5,5):
    rtk_full_tuati["height(m)"].loc[daymargin].loc[(rtk_full_tuati["h_diff"].shift(i) > rtk_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtk_full_tuati["latitude_M"].loc[(rtk_full_tuati["lat_diff"].shift(i) > rtk_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_tuati["longitude_M"].loc[(rtk_full_tuati["lon_diff"].shift(i) > rtk_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtk_full_tuati["height(m)"].loc[(rtk_full_tuati["h_diff"].shift(i) > rtk_h_cutoff*10)] = np.nan

rtk_full_tuati = rtk_full_tuati.interpolate(method='time', axis=0).ffill().bfill()

"""Spike Removal RTK DGNSS"""
rtkd_full_tuati["lat_diff"] = rtkd_full_tuati["latitude_M"].diff().abs()
rtkd_lat_cutoff = rtkd_full_tuati["lat_diff"].median()

#Remove the spike at day margin
daymargin = rtkd_full_tuati[(rtkd_full_tuati.index.hour == 23) | (rtkd_full_tuati.index.hour == 0)].index.tolist() 
for i in range(-5,5):
    rtkd_full_tuati["latitude_M"].loc[daymargin].loc[(rtkd_full_tuati["lat_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_tuati["lon_diff"] = rtkd_full_tuati["longitude_M"].diff().abs()
rtkd_lon_cutoff = rtkd_full_tuati["lon_diff"].median()

for i in range(-5,5):
    rtkd_full_tuati["longitude_M"].loc[daymargin].loc[(rtkd_full_tuati["lon_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

rtkd_full_tuati["h_diff"] = rtkd_full_tuati["height(m)"].diff().abs()
rtkd_h_cutoff = rtkd_full_tuati["h_diff"].median()

for i in range(-5,5):
    rtkd_full_tuati["height(m)"].loc[daymargin].loc[(rtkd_full_tuati["h_diff"].shift(i) > rtkd_lat_cutoff)] = np.nan

#Remove the spike at other time of the day with a different margin value
for i in range(-15,15):
    rtkd_full_tuati["latitude_M"].loc[(rtkd_full_tuati["lat_diff"].shift(i) > rtkd_lat_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_tuati["longitude_M"].loc[(rtkd_full_tuati["lon_diff"].shift(i) > rtkd_lon_cutoff*10)] = np.nan
for i in range(-15,15):
    rtkd_full_tuati["height(m)"].loc[(rtkd_full_tuati["h_diff"].shift(i) > rtkd_h_cutoff*10)] = np.nan
rtkd_full_tuati = rtkd_full_tuati.interpolate(method='time', axis=0).ffill().bfill()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Demean and Detrend the positioning signals"""
magic_full_tuati['Demean_detrend_Latitude'] = magic_full_tuati['latitude_M']
magic_full_tuati['Demean_detrend_Longitude'] = magic_full_tuati['longitude_M']
magic_full_tuati['Demean_detrend_Height'] = magic_full_tuati['h']

csrs_full_tuati['Demean_detrend_Latitude'] = csrs_full_tuati['latitude_M']
csrs_full_tuati['Demean_detrend_Longitude'] = csrs_full_tuati['longitude_M']
csrs_full_tuati['Demean_detrend_Height'] = csrs_full_tuati['HGT(m)']

rtk_full_tuati['Demean_detrend_Latitude'] = rtk_full_tuati['latitude_M']
rtk_full_tuati['Demean_detrend_Longitude'] = rtk_full_tuati['longitude_M']
rtk_full_tuati['Demean_detrend_Height'] = rtk_full_tuati['height(m)']

rtkd_full_tuati['Demean_detrend_Latitude'] = rtkd_full_tuati['latitude_M']
rtkd_full_tuati['Demean_detrend_Longitude'] = rtkd_full_tuati['longitude_M']
rtkd_full_tuati['Demean_detrend_Height'] = rtkd_full_tuati['height(m)']

#MagicGNSS
magic_full_tuati['Demean_detrend_Latitude'] = signal.detrend(magic_full_tuati['Demean_detrend_Latitude'].sub(magic_full_tuati['Demean_detrend_Latitude'].mean()))
magic_full_tuati['Demean_detrend_Longitude'] = signal.detrend(magic_full_tuati['Demean_detrend_Longitude'].sub(magic_full_tuati['Demean_detrend_Longitude'].mean()))
magic_full_tuati['Demean_detrend_Height'] = signal.detrend(magic_full_tuati['Demean_detrend_Height'].sub(magic_full_tuati['Demean_detrend_Height'].mean()))
#CSRS
csrs_full_tuati['Demean_detrend_Latitude'] = signal.detrend(csrs_full_tuati['Demean_detrend_Latitude'].sub(csrs_full_tuati['Demean_detrend_Latitude'].mean()))
csrs_full_tuati['Demean_detrend_Longitude'] = signal.detrend(csrs_full_tuati['Demean_detrend_Longitude'].sub(csrs_full_tuati['Demean_detrend_Longitude'].mean()))
csrs_full_tuati['Demean_detrend_Height'] = signal.detrend(csrs_full_tuati['Demean_detrend_Height'].sub(csrs_full_tuati['Demean_detrend_Height'].mean()))
#RTK PPP
rtk_full_tuati['Demean_detrend_Latitude'] = signal.detrend(rtk_full_tuati['Demean_detrend_Latitude'].sub(rtk_full_tuati['Demean_detrend_Latitude'].mean()))
rtk_full_tuati['Demean_detrend_Longitude'] = signal.detrend(rtk_full_tuati['Demean_detrend_Longitude'].sub(rtk_full_tuati['Demean_detrend_Longitude'].mean()))
rtk_full_tuati['Demean_detrend_Height'] = signal.detrend(rtk_full_tuati['Demean_detrend_Height'].sub(rtk_full_tuati['Demean_detrend_Height'].mean()))
#RTK DGNSS
rtkd_full_tuati['Demean_detrend_Latitude'] = signal.detrend(rtkd_full_tuati['Demean_detrend_Latitude'].sub(rtkd_full_tuati['Demean_detrend_Latitude'].mean()))
rtkd_full_tuati['Demean_detrend_Longitude'] = signal.detrend(rtkd_full_tuati['Demean_detrend_Longitude'].sub(rtkd_full_tuati['Demean_detrend_Longitude'].mean()))
rtkd_full_tuati['Demean_detrend_Height'] = signal.detrend(rtkd_full_tuati['Demean_detrend_Height'].sub(rtkd_full_tuati['Demean_detrend_Height'].mean()))

"""Applying Butterworth Low Pass Filter"""
T = 60*60*24         # Sample Period
fs = 1      # sample rate, Hz
cutoff = 0.0025      # desired cutoff frequency
cutoff_rtk = 0.0008
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#Filtering Latitude
magic_full_tuati['Filtered_Latitude'] = butter_lowpass_filter(magic_full_tuati['Demean_detrend_Latitude'], cutoff, fs, order)
csrs_full_tuati['Filtered_Latitude'] = butter_lowpass_filter(csrs_full_tuati['Demean_detrend_Latitude'], cutoff, fs, order)
rtk_full_tuati['Filtered_Latitude'] = butter_lowpass_filter(rtk_full_tuati['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
rtkd_full_tuati['Filtered_Latitude'] = butter_lowpass_filter(rtkd_full_tuati['Demean_detrend_Latitude'], cutoff_rtk, fs, order)
#Filtering Longitude
magic_full_tuati['Filtered_Longitude'] = butter_lowpass_filter(magic_full_tuati['Demean_detrend_Longitude'], cutoff, fs, order)
csrs_full_tuati['Filtered_Longitude'] = butter_lowpass_filter(csrs_full_tuati['Demean_detrend_Longitude'], cutoff, fs, order)
rtk_full_tuati['Filtered_Longitude'] = butter_lowpass_filter(rtk_full_tuati['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
rtkd_full_tuati['Filtered_Longitude'] = butter_lowpass_filter(rtkd_full_tuati['Demean_detrend_Longitude'], cutoff_rtk, fs, order)
#Filtering Height
magic_full_tuati['Filtered_Height'] = butter_lowpass_filter(magic_full_tuati['Demean_detrend_Height'], cutoff, fs, order)
csrs_full_tuati['Filtered_Height'] = butter_lowpass_filter(csrs_full_tuati['Demean_detrend_Height'], cutoff, fs, order)
rtk_full_tuati['Filtered_Height'] = butter_lowpass_filter(rtk_full_tuati['Demean_detrend_Height'], cutoff_rtk, fs, order)
rtkd_full_tuati['Filtered_Height'] = butter_lowpass_filter(rtkd_full_tuati['Demean_detrend_Height'], cutoff_rtk, fs, order)

#%%
"""Plotting all filtered signals"""
fig9 = plt.figure(figsize=(18,14))
ax1 = fig9.add_subplot(311)
ax1.plot(magic_full_tuati['Filtered_Latitude'], lw=2)
ax1.plot(csrs_full_tuati['Filtered_Latitude'], lw=2)
ax1.plot(rtk_full_tuati['Filtered_Latitude'], lw=2)
ax1.plot(rtkd_full_tuati['Filtered_Latitude'], lw=2)
ax1.set_ylabel('Latitude (m)', fontsize=12)
ax1.set_title('tuati (1105-14): Filtered Positioning', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.plot(magic_full_tuati['Filtered_Longitude'], lw=2)
ax2.plot(csrs_full_tuati['Filtered_Longitude'], lw=2)
ax2.plot(rtk_full_tuati['Filtered_Longitude'], lw=2)
ax2.plot(rtkd_full_tuati['Filtered_Longitude'], lw=2)
ax2.set_ylabel('Longitude (m)', fontsize=12)

ax3 = fig9.add_subplot(313)
ax3.plot(magic_full_tuati['Filtered_Height'], lw=2)
ax3.plot(csrs_full_tuati['Filtered_Height'], lw=2)
ax3.plot(rtk_full_tuati['Filtered_Height'], lw=2)
ax3.plot(rtkd_full_tuati['Filtered_Height'], lw=2)
ax3.set_ylabel('Datetime', fontsize=12)
ax3.set_ylabel('Height (m)', fontsize=12)
ax3.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])

#%%
"""Calculating Velocity by differencing filtered x,y,z distance"""
magic_full_tuati['distx'] = magic_full_tuati['Filtered_Latitude'].diff().fillna(0.)
magic_full_tuati['disty'] = magic_full_tuati['Filtered_Longitude'].diff().fillna(0.)
magic_full_tuati['distz'] = magic_full_tuati['Filtered_Height'].diff().fillna(0.)
magic_full_tuati['horizontal_velocity'] = (np.sqrt(magic_full_tuati['distx']**2 + magic_full_tuati['disty']**2)/15)*60*60
magic_full_tuati['vertical_velocity'] = (magic_full_tuati['distz']/15)*60*60

csrs_full_tuati['distx'] = csrs_full_tuati['Filtered_Latitude'].diff().fillna(0.)
csrs_full_tuati['disty'] = csrs_full_tuati['Filtered_Longitude'].diff().fillna(0.)
csrs_full_tuati['distz'] = csrs_full_tuati['Filtered_Height'].diff().fillna(0.)
csrs_full_tuati['horizontal_velocity'] = (np.sqrt(csrs_full_tuati['distx']**2 + csrs_full_tuati['disty']**2)/15)*60*60
csrs_full_tuati['vertical_velocity'] = (csrs_full_tuati['distz']/15)*60*60

rtk_full_tuati['distx'] = rtk_full_tuati['Filtered_Latitude'].diff().fillna(0.)
rtk_full_tuati['disty'] = rtk_full_tuati['Filtered_Longitude'].diff().fillna(0.)
rtk_full_tuati['distz'] = rtk_full_tuati['Filtered_Height'].diff().fillna(0.)
rtk_full_tuati['horizontal_velocity'] = (np.sqrt(rtk_full_tuati['distx']**2 + rtk_full_tuati['disty']**2)/15)*60*60
rtk_full_tuati['vertical_velocity'] = (rtk_full_tuati['distz']/15)*60*60

rtkd_full_tuati['distx'] = rtkd_full_tuati['Filtered_Latitude'].diff().fillna(0.)
rtkd_full_tuati['disty'] = rtkd_full_tuati['Filtered_Longitude'].diff().fillna(0.)
rtkd_full_tuati['distz'] = rtkd_full_tuati['Filtered_Height'].diff().fillna(0.)
rtkd_full_tuati['horizontal_velocity'] = (np.sqrt(rtkd_full_tuati['distx']**2 + rtkd_full_tuati['disty']**2)/15)*60*60
rtkd_full_tuati['vertical_velocity'] = (rtkd_full_tuati['distz']/15)*60*60
#unit of velocity is meters per hour

cutoff_v = 0.002
magic_full_tuati['Filtered_vertical_velocity'] = butter_lowpass_filter(magic_full_tuati['vertical_velocity'], cutoff_v, fs, order)
csrs_full_tuati['Filtered_vertical_velocity'] = butter_lowpass_filter(csrs_full_tuati['vertical_velocity'], cutoff_v, fs, order)
rtk_full_tuati['Filtered_vertical_velocity'] = butter_lowpass_filter(rtk_full_tuati['vertical_velocity'], cutoff_v, fs, order)
rtkd_full_tuati['Filtered_vertical_velocity'] = butter_lowpass_filter(rtkd_full_tuati['vertical_velocity'], cutoff_v, fs, order)

magic_full_tuati['Filtered_horizontal_velocity'] = butter_lowpass_filter(magic_full_tuati['horizontal_velocity'], cutoff_v, fs, order)
csrs_full_tuati['Filtered_horizontal_velocity'] = butter_lowpass_filter(csrs_full_tuati['horizontal_velocity'], cutoff_v, fs, order)
rtk_full_tuati['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtk_full_tuati['horizontal_velocity'], cutoff_v, fs, order)
rtkd_full_tuati['Filtered_horizontal_velocity'] = butter_lowpass_filter(rtkd_full_tuati['horizontal_velocity'], cutoff_v, fs, order)

fig12 = plt.figure(figsize=(18,14))
ax1 = fig12.add_subplot(111)
ax1.plot(magic_full_tuati['Filtered_vertical_velocity'], lw=2)
ax1.plot(csrs_full_tuati['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtk_full_tuati['Filtered_vertical_velocity'], lw=2)
ax1.plot(rtkd_full_tuati['Filtered_vertical_velocity'], lw=2)
ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('Vertical velocity (mh^-1)', fontsize=12)
ax1.set_title('tuati (1105-14): Filtered Vertical Velocity', fontsize=18)
ax1.legend(labels = ["magicGNSS","CSRS","RTKLIB-PPP","RTKLIB-DGNSS"])

#%%
"""Calculate the Raw Velocity to understand the real glacier movement"""
magic_full_tuati['distx'] = magic_full_tuati['latitude_M'].diff().fillna(0.)
magic_full_tuati['disty'] = magic_full_tuati['longitude_M'].diff().fillna(0.)
magic_full_tuati['distz'] = magic_full_tuati['h'].diff().fillna(0.)
magic_full_tuati['raw_horizontal_velocity'] = (np.sqrt(magic_full_tuati['distx']**2 + magic_full_tuati['disty']**2)/15)*60*60
magic_full_tuati['raw_vertical_velocity'] = (magic_full_tuati['distz']/15)*60*60

csrs_full_tuati['distx'] = csrs_full_tuati['latitude_M'].diff().fillna(0.)
csrs_full_tuati['disty'] = csrs_full_tuati['longitude_M'].diff().fillna(0.)
csrs_full_tuati['distz'] = csrs_full_tuati['HGT(m)'].diff().fillna(0.)
csrs_full_tuati['raw_horizontal_velocity'] = (np.sqrt(csrs_full_tuati['distx']**2 + csrs_full_tuati['disty']**2)/15)*60*60
csrs_full_tuati['raw_vertical_velocity'] = (csrs_full_tuati['distz']/15)*60*60

rtk_full_tuati['distx'] = rtk_full_tuati['latitude_M'].diff().fillna(0.)
rtk_full_tuati['disty'] = rtk_full_tuati['longitude_M'].diff().fillna(0.)
rtk_full_tuati['distz'] = rtk_full_tuati['height(m)'].diff().fillna(0.)
rtk_full_tuati['raw_horizontal_velocity'] = (np.sqrt(rtk_full_tuati['distx']**2 + rtk_full_tuati['disty']**2)/15)*60*60
rtk_full_tuati['raw_vertical_velocity'] = (rtk_full_tuati['distz']/15)*60*60

rtkd_full_tuati['distx'] = rtkd_full_tuati['latitude_M'].diff().fillna(0.)
rtkd_full_tuati['disty'] = rtkd_full_tuati['longitude_M'].diff().fillna(0.)
rtkd_full_tuati['distz'] = rtkd_full_tuati['height(m)'].diff().fillna(0.)
rtkd_full_tuati['raw_horizontal_velocity'] = (np.sqrt(rtkd_full_tuati['distx']**2 + rtkd_full_tuati['disty']**2)/15)*60*60
rtkd_full_tuati['raw_vertical_velocity'] = (rtkd_full_tuati['distz']/15)*60*60
#unit of velocity is meters per hour

magic_full_tuati = magic_full_tuati.drop(columns=['distx','disty','distz'])
csrs_full_tuati = csrs_full_tuati.drop(columns=['distx','disty','distz'])
rtk_full_tuati = rtk_full_tuati.drop(columns=['distx','disty','distz'])
rtkd_full_tuati = rtkd_full_tuati.drop(columns=['distx','disty','distz'])
#%%
"""Create dataframe for statistics"""
stat = {'CSRS' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']), 
      'RTK' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'RTKD' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar']),
      'Magic' : pd.Series([99., 99, 99, 99, 99, 99, 99, 99, 99], index = ['xmean','xsd','xvar','ymean','ysd','yvar','zmean','zsd','zvar'])} 
tuati_stat = pd.DataFrame(stat)

tuati_stat['Magic']['xmean'] = round(magic_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['Magic']['xsd'] = round(magic_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['Magic']['xvar'] = round(magic_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTK']['xmean'] = round(rtk_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTK']['xsd'] = round(rtk_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTK']['xvar'] = round(rtk_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTKD']['xmean'] = round(rtkd_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTKD']['xsd'] = round(rtkd_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTKD']['xvar'] = round(rtkd_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['CSRS']['xmean'] = round(csrs_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['CSRS']['xsd'] = round(csrs_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['CSRS']['xvar'] = round(csrs_full_tuati['latitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['Magic']['ymean'] = round(magic_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['Magic']['ysd'] = round(magic_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['Magic']['yvar'] = round(magic_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTK']['ymean'] = round(rtk_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTK']['ysd'] = round(rtk_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTK']['yvar'] = round(rtk_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTKD']['ymean'] = round(rtkd_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTKD']['ysd'] = round(rtkd_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTKD']['yvar'] = round(rtkd_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['CSRS']['ymean'] = round(csrs_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['CSRS']['ysd'] = round(csrs_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['CSRS']['yvar'] = round(csrs_full_tuati['longitude_M'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['Magic']['zmean'] = round(magic_full_tuati['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['Magic']['zsd'] = round(magic_full_tuati['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['Magic']['zvar'] = round(magic_full_tuati['h'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTK']['zmean'] = round(rtk_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTK']['zsd'] = round(rtk_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTK']['zvar'] = round(rtk_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['RTKD']['zmean'] = round(rtkd_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['RTKD']['zsd'] = round(rtkd_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['RTKD']['zvar'] = round(rtkd_full_tuati['height(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)

tuati_stat['CSRS']['zmean'] = round(csrs_full_tuati['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].mean(),3)
tuati_stat['CSRS']['zsd'] = round(csrs_full_tuati['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].std(),3)
tuati_stat['CSRS']['zvar'] = round(csrs_full_tuati['HGT(m)'].loc['2018-11-05 01:00:00':'2018-11-14 00:00:00'].var(),3)
tuati_stat.to_csv('tuati_stat.csv', index=True)  

#%%
"""Plotting statistics"""
"""tuati"""

m1 = abs(magic_full_tuati['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m2 = abs(rtk_full_tuati['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m2d = abs(rtkd_full_tuati['latitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['latitude_M'].mean())
m3 = abs(csrs_full_tuati['latitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['latitude_M'].mean())
m11 = magic_full_tuati['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22 = rtk_full_tuati['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m22d = rtkd_full_tuati['latitude_M'].resample('D').std().to_numpy()[0:10]/1000
m33 = csrs_full_tuati['latitude_M'].resample('D').std().to_numpy()[0:10]/1000

m4 = abs(magic_full_tuati['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m5 = abs(rtk_full_tuati['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m5d = abs(rtkd_full_tuati['longitude_M'].resample('D').mean().to_numpy()[0:10] - rtk_full_base['longitude_M'].mean())
m6 = abs(csrs_full_tuati['longitude_M'].resample('D').mean().to_numpy()[0:10] - magic_full_base['longitude_M'].mean())
m44 = magic_full_tuati['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55 = rtk_full_tuati['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m55d = rtkd_full_tuati['longitude_M'].resample('D').std().to_numpy()[0:10]/1000
m66 = csrs_full_tuati['longitude_M'].resample('D').std().to_numpy()[0:10]/1000

m7 = magic_full_tuati['h'].resample('D').mean().to_numpy()[0:10]
m8 = rtk_full_tuati['height(m)'].resample('D').mean().to_numpy()[0:10]
m8d = rtkd_full_tuati['height(m)'].resample('D').mean().to_numpy()[0:10]
m9 = csrs_full_tuati['HGT(m)'].resample('D').mean().to_numpy()[0:10]
m77 = magic_full_tuati['h'].resample('D').std().to_numpy()[0:10]/1000
m88 = rtk_full_tuati['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m88d = rtkd_full_tuati['height(m)'].resample('D').std().to_numpy()[0:10]/1000
m99 = csrs_full_tuati['HGT(m)'].resample('D').std().to_numpy()[0:10]/1000
day = [309,310,311,312,313,314,315,316,317,318]

fig16, axes = plt.subplots(figsize=(18,18))
plt.suptitle("PPP Positioning Statistics (tuati)",size=18,y=0.95)

ax01 = plt.subplot2grid((2,3),(0,0))
ax02 = plt.subplot2grid((2,3),(1,0))
ax03 = plt.subplot2grid((2,3),(0,1),colspan = 2)
ax04 = plt.subplot2grid((2,3),(1,1),colspan = 2)

#X
df1 = pd.DataFrame({'magic': m1, 'rtk': m2, 'rtkd': m2d, 'csrs': m3}, index=day)
df1 = df1/1000
ax01 = df1[['rtk','rtkd','csrs']].plot(ax=ax01,kind='bar',grid=True,width=1,ylim=[5.002,5.007],yerr=[m22,m22d,m33],legend=False,rot=None)
ax01.set_title("(a) Polar stereographic x coordinates")
ax01.set(ylabel="km")

#Y
df2 = pd.DataFrame({'magic': m4, 'rtk': m5, 'rtkd': m5d, 'csrs': m6}, index=day)
df2 = df2/1000
ax02 = df2[['rtk','rtkd','csrs']].plot(ax=ax02,kind='bar',grid=True,width=1,ylim=[4.995,5.002],yerr=[m55,m55d,m66],legend=False,rot=None)
ax02.set_title("(b) Polar stereographic y coordinates")
ax02.set(ylabel="km")
ax02.set(xlabel="Day of the Year")

#Z
ax03 = rtk_full_tuati['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = rtkd_full_tuati['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03 = csrs_full_tuati['Filtered_Height'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax03,grid=True)
ax03.set_title("(c) Average Height")
ax03.set_ylabel('meter', fontsize=12)

#Velocity
ax04 = rtk_full_tuati['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = rtkd_full_tuati['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04 = csrs_full_tuati['Filtered_vertical_velocity'].loc["2018-11-10 06:00:00":"2018-11-13 06:00:00"].plot(ax=ax04,grid=True)
ax04.set_title("(d) Vertical velocity")
ax04.set_ylabel('meter/hour', fontsize=12)
ax04.legend(labels = ["RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.8, 1), loc='upper left')
ax02.legend(labels = ["RTKLIB-PPP","RTKLIB-DGNSS","CSRS"],bbox_to_anchor=(0.6, 1), loc='upper left')

fig16.subplots_adjust(hspace=0.5)
plt.show()

#%%
"""Height: Comparing different stations"""
fig16 = plt.figure(figsize=(18,14))
ax1 = fig16.add_subplot(111)
ax1.plot(csrs_full_tuati['Filtered_Height'], lw=2)
ax1.plot(csrs_full_blake['Filtered_Height'], lw=2)
ax1.plot(csrs_full_shirase['Filtered_Height'], lw=2)
ax1.plot(csrs_full_hillary['Filtered_Height'], lw=2)

ax1.set_xlabel('Datetime', fontsize=12)
ax1.set_ylabel('H', fontsize=12)
ax1.set_title('Height', fontsize=18)
ax1.legend(labels = ["tuati","blake","shirase","hillary"])
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Dropping all unused columns"""
magic_full_base = magic_full_base.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',\
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','lat_sigma','lon_sigma','h_sigma'])
csrs_full_base = csrs_full_base.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',\
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','SDLAT(95%)','SDLON(95%)','SDHGT(95%)','DLAT(m)','DLON(m)','DHGT(m)'])
rtk_full_base = rtk_full_base.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',\
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
    
magic_full_blake = magic_full_blake.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','lat_sigma','lon_sigma','h_sigma'])
csrs_full_blake = csrs_full_blake.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','SDLAT(95%)','SDLON(95%)','SDHGT(95%)','DLAT(m)','DLON(m)','DHGT(m)'])
rtk_full_blake = rtk_full_blake.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
rtkd_full_blake = rtkd_full_blake.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])

magic_full_hillary = magic_full_hillary.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','lat_sigma','lon_sigma','h_sigma'])
csrs_full_hillary = csrs_full_hillary.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','SDLAT(95%)','SDLON(95%)','SDHGT(95%)','DLAT(m)','DLON(m)','DHGT(m)'])
rtk_full_hillary = rtk_full_hillary.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
rtkd_full_hillary = rtkd_full_hillary.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])

magic_full_tuati = magic_full_tuati.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','lat_sigma','lon_sigma','h_sigma'])
csrs_full_tuati = csrs_full_tuati.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','SDLAT(95%)','SDLON(95%)','SDHGT(95%)','DLAT(m)','DLON(m)','DHGT(m)'])
rtk_full_tuati = rtk_full_tuati.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
rtkd_full_tuati = rtkd_full_tuati.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
    
magic_full_shirase = magic_full_shirase.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','lat_sigma','lon_sigma','h_sigma'])
csrs_full_shirase = csrs_full_shirase.drop(columns=['latitude_DD','longitude_DD','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','SDLAT(95%)','SDLON(95%)','SDHGT(95%)','DLAT(m)','DLON(m)','DHGT(m)'])
rtk_full_shirase = rtk_full_shirase.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])
rtkd_full_shirase = rtkd_full_shirase.drop(columns=['latitude(deg)','longitude(deg)','Demean_detrend_Latitude','Demean_detrend_Longitude',
                              'Demean_detrend_Height','horizontal_velocity','vertical_velocity','latitude_sigma(m)','longitude_sigma(m)','height_sigma(m)'])

#%%
"""Interpolating Tuati from 30s to 15s"""
csrs_full_tuati_15s = pd.DataFrame()
rtk_full_tuati_15s = pd.DataFrame()
rtkd_full_tuati_15s = pd.DataFrame()
magic_full_tuati_15s = pd.DataFrame()
csrs_full_tuati_15s = csrs_full_tuati.resample('15S').interpolate("cubic")
rtk_full_tuati_15s = rtk_full_tuati.resample('15S').interpolate("cubic")
rtkd_full_tuati_15s = rtkd_full_tuati.resample('15S').interpolate("cubic")
magic_full_tuati_15s = magic_full_tuati.resample('15S').interpolate("cubic")

#%%
"""Dynamic time Wrapping (DTW)"""
#DTW is calculated as minimum distance for all stations except base as it is only the reference
#Z score is first calculated for normalization as only relative distance (variations) is considered.

"""CSRS"""
#Calculate z score
csrs_full_shirase['z_Height'] = stats.zscore(csrs_full_shirase['Filtered_Height'])
csrs_full_blake['z_Height'] = stats.zscore(csrs_full_blake['Filtered_Height'])
csrs_full_hillary['z_Height'] = stats.zscore(csrs_full_hillary['Filtered_Height'])
csrs_full_tuati_15s['z_Height'] = stats.zscore(csrs_full_tuati_15s['Filtered_Height'])

#Create dataframe
station_csrs = pd.DataFrame()
station_csrs['shirase'] = csrs_full_shirase['z_Height']
station_csrs['tuati'] = csrs_full_tuati_15s['z_Height']
station_csrs['blake'] = csrs_full_blake['z_Height']
station_csrs['hillary'] = csrs_full_hillary['z_Height']
station_csrs.dropna(inplace=True)
dist_matrix_c = np.zeros(shape=(4,4))

"""MagicGNSS"""
#Calculate z score
magic_full_shirase['z_Height'] = stats.zscore(magic_full_shirase['Filtered_Height'])
magic_full_blake['z_Height'] = stats.zscore(magic_full_blake['Filtered_Height'])
magic_full_hillary['z_Height'] = stats.zscore(magic_full_hillary['Filtered_Height'])
magic_full_tuati_15s['z_Height'] = stats.zscore(magic_full_tuati_15s['Filtered_Height'])

station_magic = pd.DataFrame()
magic_full_tuati_15s['z_Height'] = np.nan
station_magic['shirase'] = magic_full_shirase['z_Height']
station_magic['tuati'] = 0 #nan
station_magic['blake'] = magic_full_blake['z_Height']
station_magic['hillary'] = magic_full_hillary['z_Height']
station_magic.dropna(axis=0, inplace=True)

"""RTKLIB-PPP"""
#Calculate z score
rtk_full_shirase['z_Height'] = stats.zscore(rtk_full_shirase['Filtered_Height'])
rtk_full_blake['z_Height'] = stats.zscore(rtk_full_blake['Filtered_Height'])
rtk_full_hillary['z_Height'] = stats.zscore(rtk_full_hillary['Filtered_Height'])
rtk_full_tuati_15s['z_Height'] = stats.zscore(rtk_full_tuati_15s['Filtered_Height'])

station_rtk = pd.DataFrame()
station_rtk['shirase'] = rtk_full_shirase['z_Height']
station_rtk['tuati'] = rtk_full_tuati_15s['z_Height']
station_rtk['blake'] = rtk_full_blake['z_Height']
station_rtk['hillary'] = rtk_full_hillary['z_Height']
station_rtk.dropna(inplace=True)

"""RTKLIB-DGNSS"""
#Calculate z score
rtkd_full_shirase['z_Height'] = stats.zscore(rtkd_full_shirase['Filtered_Height'])
rtkd_full_blake['z_Height'] = stats.zscore(rtkd_full_blake['Filtered_Height'])
rtkd_full_hillary['z_Height'] = stats.zscore(rtkd_full_hillary['Filtered_Height'])
rtkd_full_tuati_15s['z_Height'] = stats.zscore(rtkd_full_tuati_15s['Filtered_Height'])

station_rtkd = pd.DataFrame()
station_rtkd['shirase'] = rtkd_full_shirase['z_Height']
station_rtkd['tuati'] = rtkd_full_tuati_15s['z_Height']
station_rtkd['blake'] = rtkd_full_blake['z_Height']
station_rtkd['hillary'] = rtkd_full_hillary['z_Height']
station_rtkd.dropna(inplace=True)

#%%
"""Calculating Matrix (Dynamic Time Warping)"""

## Matrix for CSRS
for i in range(len(dist_matrix_c)):
    for j in range(len(dist_matrix_c[0])):
        dist_matrix_c[i][j],_ = fastdtw(station_csrs.iloc[:, i].to_numpy(),station_csrs.iloc[:, j].to_numpy(), dist=euclidean)
alpha = ['shirase', 'tuati', 'blake', 'hillary']

## Matrix for MagicGNSS
dist_matrix_m = np.zeros(shape=(4,4))
for i in range(len(dist_matrix_m)):
    for j in range(len(dist_matrix_m[0])):
        dist_matrix_m[i][j],_ = fastdtw(station_magic.iloc[:, i].to_numpy(),station_magic.iloc[:, j].to_numpy(), dist=euclidean)

## Matrix for RTKLIB PPP
dist_matrix_r = np.zeros(shape=(4,4))
for i in range(len(dist_matrix_r)):
    for j in range(len(dist_matrix_r[0])):
        dist_matrix_r[i][j],_ = fastdtw(station_rtk.iloc[:, i].to_numpy(),station_rtk.iloc[:, j].to_numpy(), dist=euclidean)
        
## Matrix for RTKLIB DGNSS
dist_matrix_rd = np.zeros(shape=(4,4))
for i in range(len(dist_matrix_rd)):
    for j in range(len(dist_matrix_rd[0])):
        dist_matrix_rd[i][j],_ = fastdtw(station_rtkd.iloc[:, i].to_numpy(),station_rtkd.iloc[:, j].to_numpy(), dist=euclidean)

dist_matrix_m[1,:] = np.nan
dist_matrix_m[:,1] = np.nan

#%%
## Plotting the DTW matrix with imshow
fig_DTW, axs = plt.subplots(1, 4, sharex=True, sharey=True,figsize=(36,24))
cmap = plt.cm.viridis
cmap.set_bad(color='navy') #color map

imc = axs[0].imshow(dist_matrix_c,vmin=0,vmax=20000,cmap=cmap)
axs[0].set_xticklabels(alpha)
axs[0].set_yticklabels(alpha)
axs[0].set_title('CSRS',fontweight="bold",fontsize=14)
axs[0].set_xticks(np.arange(0, 4, 1))
axs[0].set_yticks(np.arange(0, 4, 1))
axs[0].tick_params(labelsize=13)
axs[1].tick_params(labelsize=13)
axs[2].tick_params(labelsize=13)

imm = axs[1].imshow(dist_matrix_m,vmin=0,vmax=20000,cmap=cmap)
axs[1].set_xticklabels(alpha)
axs[1].set_yticklabels(alpha)
axs[1].set_title('MagicGNSS',fontweight="bold",fontsize=14)
axs[1].set_xticks(np.arange(0, 4, 1))
axs[1].set_yticks(np.arange(0, 4, 1))

imr = axs[2].imshow(dist_matrix_r,vmin=0,vmax=20000,cmap=cmap)
axs[2].set_xticklabels(alpha)
axs[2].set_yticklabels(alpha)
axs[2].set_title('RTKLIB-PPP',fontweight="bold",fontsize=14)
axs[2].set_xticks(np.arange(0, 4, 1))
axs[2].set_yticks(np.arange(0, 4, 1))

imr = axs[3].imshow(dist_matrix_rd,vmin=0,vmax=20000,cmap=cmap)
axs[3].set_xticklabels(alpha)
axs[3].set_yticklabels(alpha)
axs[3].set_title('RTKLIB-DGNSS',fontweight="bold",fontsize=14)
axs[3].set_xticks(np.arange(0, 4, 1))
axs[3].set_yticks(np.arange(0, 4, 1))

#layout settings
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(imr,cax=cax,label="Minimum Distance (m)")

plt.subplots_adjust(left=0.1, right=0.11, wspace=0.1)
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
#Plotting height for all platforms and stations
fig30 = plt.figure(figsize=(30,24))
axa = fig30.add_subplot(411)
axa.plot(station_rtkd.loc["2018-11-06 00:00:00":"2018-11-11 00:00:00"], lw=2, alpha = 0.8)

axb = fig30.add_subplot(412,sharex = axa)
axb.plot(station_csrs.loc["2018-11-06 00:00:00":"2018-11-11 00:00:00"], lw=2, alpha = 0.8)

axc = fig30.add_subplot(413,sharex = axa)
axc.plot(station_magic.loc["2018-11-06 00:00:00":"2018-11-11 00:00:00"], lw=2, alpha = 0.8)

axd = fig30.add_subplot(414,sharex = axa)
axd.plot(station_rtk.loc["2018-11-06 00:00:00":"2018-11-11 00:00:00"], lw=2, alpha = 0.8)

axd.set_xlabel('Datetime', fontsize=12)
axa.set_ylabel('RTKLIB/DGNSS', fontsize=12)
axb.set_ylabel('CSRS/PPP', fontsize=12)
axc.set_ylabel('magicGNSS/PPP', fontsize=12)
axd.set_ylabel('RTKLIB/PPP', fontsize=12)
fig30.text(0.06, 0.5, 'Height (m)', va='center', rotation='vertical', fontsize=12)

axa.legend(labels = ["shirase","tuati","blake","hillary"],loc='upper right')
axb.legend(labels = ["shirase","tuati","blake","hillary"],loc='upper right')
axc.legend(labels = ["shirase","tuati","blake","hillary"],loc='upper right')
axd.legend(labels = ["shirase","tuati","blake","hillary"],loc='upper right')
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Boxplot"""
#Create a new dataframe stat_df for all positioning data need to be plotted
stat_df = pd.DataFrame()

#Calculate mean position
avg_base_lat = (csrs_full_base['latitude_M'].mean() + magic_full_base['latitude_M'].mean() + rtk_full_base['latitude_M'].mean())/3
avg_base_lon = (csrs_full_base['longitude_M'].mean() + magic_full_base['longitude_M'].mean() + rtk_full_base['longitude_M'].mean())/3
avg_base_h = (csrs_full_base["HGT(m)"].mean() + magic_full_base['h'].mean() + rtk_full_base["height(m)"].mean())/3

avg_tuati_lat = (csrs_full_tuati['latitude_M'].mean() + magic_full_tuati['latitude_M'].mean() + rtk_full_tuati['latitude_M'].mean() + rtkd_full_tuati['latitude_M'].mean())/4
avg_tuati_lon = (csrs_full_tuati['longitude_M'].mean() + magic_full_tuati['longitude_M'].mean() + rtk_full_tuati['longitude_M'].mean() + rtkd_full_tuati['longitude_M'].mean())/4
avg_tuati_h = (csrs_full_tuati["HGT(m)"].mean() + magic_full_tuati['h'].mean() + rtk_full_tuati["height(m)"].mean() + rtkd_full_tuati["height(m)"].mean())/4

avg_hillary_lat = (csrs_full_hillary['latitude_M'].mean() + magic_full_hillary['latitude_M'].mean() + rtk_full_hillary['latitude_M'].mean() + rtkd_full_hillary['latitude_M'].mean())/4
avg_hillary_lon = (csrs_full_hillary['longitude_M'].mean() + magic_full_hillary['longitude_M'].mean() + rtk_full_hillary['longitude_M'].mean() + rtkd_full_hillary['longitude_M'].mean())/4
avg_hillary_h = (csrs_full_hillary["HGT(m)"].mean() + magic_full_hillary['h'].mean() + rtk_full_hillary["height(m)"].mean() + rtkd_full_hillary["height(m)"].mean())/4

avg_shirase_lat = (csrs_full_shirase['latitude_M'].mean() + magic_full_shirase['latitude_M'].mean() + rtk_full_shirase['latitude_M'].mean() + rtkd_full_shirase['latitude_M'].mean())/4
avg_shirase_lon = (csrs_full_shirase['longitude_M'].mean() + magic_full_shirase['longitude_M'].mean() + rtk_full_shirase['longitude_M'].mean() + rtkd_full_shirase['longitude_M'].mean())/4
avg_shirase_h = (csrs_full_shirase["HGT(m)"].mean() + magic_full_shirase['h'].mean() + rtk_full_shirase["height(m)"].mean() + rtkd_full_shirase["height(m)"].mean())/4

avg_blake_lat = (csrs_full_blake['latitude_M'].mean() + magic_full_blake['latitude_M'].mean() + rtk_full_blake['latitude_M'].mean() + rtkd_full_blake['latitude_M'].mean())/4
avg_blake_lon = (csrs_full_blake['longitude_M'].mean() + magic_full_blake['longitude_M'].mean() + rtk_full_blake['longitude_M'].mean() + rtkd_full_blake['longitude_M'].mean())/4
avg_blake_h = (csrs_full_blake["HGT(m)"].mean() + magic_full_blake['h'].mean() + rtk_full_blake["height(m)"].mean() + rtkd_full_blake["height(m)"].mean())/4

"""CSRS""" #calculating relative position 
stat_df['tuati_lat_csrs'] = csrs_full_tuati["latitude_M"] - avg_tuati_lat
stat_df['tuati_lon_csrs'] = csrs_full_tuati["longitude_M"] - avg_tuati_lon
stat_df['tuati_h_csrs'] = csrs_full_tuati["HGT(m)"] - avg_tuati_h
stat_df['base_lat_csrs'] = csrs_full_base["latitude_M"] - avg_base_lat
stat_df['base_lon_csrs'] = csrs_full_base["longitude_M"] - avg_base_lon
stat_df['base_h_csrs'] = csrs_full_base["HGT(m)"] - avg_base_h
stat_df['blake_lat_csrs'] = csrs_full_blake["latitude_M"] - avg_blake_lat
stat_df['blake_lon_csrs'] = csrs_full_blake["longitude_M"] - avg_blake_lon
stat_df['blake_h_csrs'] = csrs_full_blake["HGT(m)"] - avg_blake_h
stat_df['hillary_lat_csrs'] = csrs_full_hillary["latitude_M"] - avg_hillary_lat
stat_df['hillary_lon_csrs'] = csrs_full_hillary["longitude_M"] - avg_hillary_lon
stat_df['hillary_h_csrs'] = csrs_full_hillary["HGT(m)"] - avg_hillary_h
stat_df['shirase_lat_csrs'] = csrs_full_shirase["latitude_M"] - avg_shirase_lat
stat_df['shirase_lon_csrs'] = csrs_full_shirase["longitude_M"] - avg_shirase_lon
stat_df['shirase_h_csrs'] = csrs_full_shirase["HGT(m)"] - avg_shirase_h

"""MagicGNSS"""
stat_df['tuati_lat_magic'] = np.nan
stat_df['tuati_lon_magic'] = np.nan
stat_df['tuati_h_magic'] = np.nan
stat_df['base_lat_magic'] = magic_full_base["latitude_M"] - avg_base_lat
stat_df['base_lon_magic'] = magic_full_base["longitude_M"] - avg_base_lon
stat_df['base_h_magic'] = magic_full_base["h"] - avg_base_h
stat_df['blake_lat_magic'] = magic_full_blake["latitude_M"] - avg_blake_lat
stat_df['blake_lon_magic'] = magic_full_blake["longitude_M"] - avg_blake_lon
stat_df['blake_h_magic'] = magic_full_blake["h"] - avg_blake_h
stat_df['hillary_lat_magic'] = magic_full_hillary["latitude_M"] - avg_hillary_lat
stat_df['hillary_lon_magic'] = magic_full_hillary["longitude_M"] - avg_hillary_lon
stat_df['hillary_h_magic'] = magic_full_hillary["h"] - avg_hillary_h
stat_df['shirase_lat_magic'] = magic_full_shirase["latitude_M"] - avg_shirase_lat
stat_df['shirase_lon_magic'] = magic_full_shirase["longitude_M"] - avg_shirase_lon
stat_df['shirase_h_magic'] = magic_full_shirase["h"] - avg_shirase_h

"""RTKLIB PPP"""
stat_df['tuati_lat_rtk'] = rtk_full_tuati["latitude_M"] - avg_tuati_lat
stat_df['tuati_lon_rtk'] = rtk_full_tuati["longitude_M"] - avg_tuati_lon
stat_df['tuati_h_rtk'] = rtk_full_tuati["height(m)"] - avg_tuati_h
stat_df['base_lat_rtk'] = rtk_full_base["latitude_M"] - avg_base_lat
stat_df['base_lon_rtk'] = rtk_full_base["longitude_M"] - avg_base_lon
stat_df['base_h_rtk'] = rtk_full_base["height(m)"] - avg_base_h
stat_df['blake_lat_rtk'] = rtk_full_blake["latitude_M"] - avg_blake_lat
stat_df['blake_lon_rtk'] = rtk_full_blake["longitude_M"] - avg_blake_lon
stat_df['blake_h_rtk'] = rtk_full_blake["height(m)"] - avg_blake_h
stat_df['hillary_lat_rtk'] = rtk_full_hillary["latitude_M"] - avg_hillary_lat
stat_df['hillary_lon_rtk'] = rtk_full_hillary["longitude_M"] - avg_hillary_lon
stat_df['hillary_h_rtk'] = rtk_full_hillary["height(m)"] - avg_hillary_h
stat_df['shirase_lat_rtk'] = rtk_full_shirase["latitude_M"] - avg_shirase_lat
stat_df['shirase_lon_rtk'] = rtk_full_shirase["longitude_M"] - avg_shirase_lon
stat_df['shirase_h_rtk'] = rtk_full_shirase["height(m)"] - avg_shirase_h

"""RTKLIB DGNSS"""
stat_df['tuati_lat_rtkd'] = rtkd_full_tuati["latitude_M"] - avg_tuati_lat
stat_df['tuati_lon_rtkd'] = rtkd_full_tuati["longitude_M"] - avg_tuati_lon
stat_df['tuati_h_rtkd'] = rtkd_full_tuati["height(m)"] - avg_tuati_h
stat_df['base_lat_rtkd'] = np.nan
stat_df['base_lon_rtkd'] = np.nan
stat_df['base_h_rtkd'] = np.nan
stat_df['blake_lat_rtkd'] = rtkd_full_blake["latitude_M"] - avg_blake_lat
stat_df['blake_lon_rtkd'] = rtkd_full_blake["longitude_M"] - avg_blake_lon
stat_df['blake_h_rtkd'] = rtkd_full_blake["height(m)"] - avg_blake_h
stat_df['hillary_lat_rtkd'] = rtkd_full_hillary["latitude_M"] - avg_hillary_lat
stat_df['hillary_lon_rtkd'] = rtkd_full_hillary["longitude_M"] - avg_hillary_lon
stat_df['hillary_h_rtkd'] = rtkd_full_hillary["height(m)"] - avg_hillary_h
stat_df['shirase_lat_rtkd'] = rtkd_full_shirase["latitude_M"] - avg_shirase_lat
stat_df['shirase_lon_rtkd'] = rtkd_full_shirase["longitude_M"] - avg_shirase_lon
stat_df['shirase_h_rtkd'] = rtkd_full_shirase["height(m)"] - avg_shirase_h

"""Multi-index Dataframe"""
#Defining dataframe hierarchy with three levels: dimension (lat,lon,h), locations and software packages
index = pd.MultiIndex.from_arrays([("lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h",\
 "lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h",\
     "lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h",\
         "lat","lon","h","lat","lon","h","lat","lon","h","lat","lon","h"),\
("Tuati","Tuati","Tuati","Base","Base","Base",\
"Blake","Blake","Blake","Hillary","Hillary",\
"Hillary","Shirase","Shirase","Shirase","Tuati"\
,"Tuati","Tuati","Base","Base","Base","Blake"\
,"Blake","Blake","Hillary","Hillary","Hillary",\
"Shirase","Shirase","Shirase","Tuati"\
,"Tuati","Tuati","Base","Base","Base","Blake","Blake","Blake","Hillary",\
"Hillary","Hillary","Shirase","Shirase","Shirase","Tuati"\
,"Tuati","Tuati","Base","Base","Base","Blake"\
,"Blake","Blake","Hillary","Hillary","Hillary",\
"Shirase","Shirase","Shirase"),\
    ("csrs","csrs","csrs","csrs","csrs","csrs","csrs","csrs","csrs","csrs",\
     "csrs","csrs","csrs","csrs","csrs","magic","magic","magic","magic","magic",\
         "magic","magic","magic","magic","magic","magic","magic","magic","magic","magic",\
             "rtk","rtk","rtk","rtk","rtk",\
         "rtk","rtk","rtk","rtk","rtk","rtk","rtk","rtk","rtk","rtk","rtkd","rtkd","rtkd","rtkd","rtkd"\
             ,"rtkd","rtkd","rtkd","rtkd","rtkd","rtkd","rtkd","rtkd","rtkd","rtkd")],names=['dimension','location','package'])
stat_df.columns = index
stat_df.dropna(how='all', inplace=True)

#%%
"""Plotting statistics in subplots with Pandas groupby function"""
props = dict(whiskers="DarkOrange", medians="DarkBlue", caps="Gray") #define properties for the plot layout
fig_box, axes_box = plt.subplots(3, 5, gridspec_kw=dict(hspace=0.2,wspace = 0.2),sharex=True) #set up the subplots
fig_box.suptitle('Comparison of positioning results', fontsize=18,fontweight="bold")

bp_dict_lat = stat_df['lat'].groupby(axis=1,level=0).boxplot(notch=True, sym='c+', positions=np.linspace(1,2,num=4),rot=45,\
layout=(1,5), sharey=False, grid=True, color=props, patch_artist=True,return_type='both', widths=0.2,ax=axes_box[0,[0,1,2,3,4]]) #boxplot

colors = ["dodgerblue","forestgreen","orangered","aqua"] #set colours

n = 0
for row_key, (ax,row) in bp_dict_lat.iteritems(): #Filling grouped colour, xtick and ylabels for the boxplot
    ##removing shared axes:
    grouper = ax.get_shared_x_axes()
    shared_xs = [a for a in grouper]
    for ax_list in shared_xs:
        for ax2 in ax_list:
            grouper.remove(ax2)        
    if n == 0:
        ax.set_ylabel('X (m)', fontsize=14,fontweight="bold")
        ax.set_ylim([-2,2])
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP"," "])
    elif n == 3:
        ax.set_xticklabels(["CSRS"," ","RTKLIB-PPP","RTKLIB-DGNSS"])
    else:
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP","RTKLIB-DGNSS"])
    for i,box1 in enumerate(row['boxes']):
        box1.set_facecolor(colors[i])
    n += 1

#Plotting longitude
bp_dict_lon = stat_df['lon'].groupby(axis=1,level=0).boxplot(notch=True, sym='c+', positions=np.linspace(1,2,num=4),rot=45,\
layout=(1,5), sharey=False, grid=True, color=props, patch_artist=True,return_type='both', widths=0.2,ax=axes_box[1,[0,1,2,3,4]])

n = 0
for row_key, (ax,row) in bp_dict_lon.iteritems(): #Properties
    if n == 0:
        ax.set_ylabel('Y (m)', fontsize=14,fontweight="bold")
        ax.set_ylim([-2,2])
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP"," "])
    else:
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP","RTKLIB-DGNSS"])
    for i,box2 in enumerate(row['boxes']):
        box2.set_facecolor(colors[i])
    n += 1

#Plotting height
bp_dict_h = stat_df['h'].groupby(axis=1,level=0).boxplot(notch=True, sym='c+', positions=np.linspace(1,2,num=4),rot=45,\
layout=(1,5), sharey=False, grid=True, color=props, patch_artist=True,return_type='both', widths=0.2,ax=axes_box[2,[0,1,2,3,4]])

n = 0
for row_key, (ax,row) in bp_dict_h.iteritems(): #Properties
    if n == 2:
        ax.set_xlabel('GNSS Packages', fontsize=14,fontweight="bold")
    if n == 0:
        ax.set_ylabel('Height (m)', fontsize=14,fontweight="bold")
        ax.set_ylim([-2,2])
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP"," "])
    else:
        ax.set_xticklabels(["CSRS","MagicGNSS","RTKLIB-PPP","RTKLIB-DGNSS"])
    for i,box3 in enumerate(row['boxes']):
        box3.set_facecolor(colors[i])
    n += 1
     
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
"""Base station Kinematic Position comparison"""
fig21 = plt.figure(figsize=(18,14))
base_x = magic_full_base['longitude_M'].mean()
base_y = magic_full_base['latitude_M'].mean()
plt.scatter(magic_full_base['longitude_M'].resample('1D').mean() - base_x, magic_full_base['latitude_M'].resample('1D').mean() - base_y, color = 'r')
plt.scatter(rtk_full_base['longitude_M'].resample('1D').mean() - base_x, rtk_full_base['latitude_M'].resample('1D').mean() - base_y, color = 'g')
plt.scatter(csrs_full_base['longitude_M'].resample('1D').mean() - base_x, csrs_full_base['latitude_M'].resample('1D').mean() - base_y, color = 'b')
plt.legend(['MagicGNSS','RTKLIB','CSRS'])
plt.show()

#%%
"""Singular Spectrum Analysis (SSA)"""
#SSA is implemented to discompose periodic signals which might be correlated to 
#the tidal patterns from the GNSS signals. Signals are first discompose into 10 
#components and groupped together based on the correlation of elementary components 
#in the w-correlation matrix. Bar chart is plotted to examine the contributions of
#components to the varinace of the signals. The function is provided by pymssa library
#from GitHub.
#reference: https://www.kaggle.com/edumotya/ssa-dimensionality-reduction-lanl-toy-example
#package information: https://github.com/kieferk/pymssa

N_COMPONENTS = 10 #number of components
weeksize = 60*24*7 #window size
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #set up the parameters

#Shirase
    #Magic
magic_resam_shirase = magic_full_shirase['Filtered_Height'] #use the filtered height (detrend and demean) as input
magic_resam_shirase = magic_resam_shirase.resample('60S').mean().interpolate(method='time') #take the mean every minutes to reduce the size of dataset
magic_resam_shirase_arr = magic_resam_shirase.values #from pandas dataframe to np.array
mssa.fit(magic_resam_shirase_arr) #fit the model
shirase_magic_signal = magic_resam_shirase_arr[:weeksize] #take the first week for analysis
cumulative_recon = np.zeros_like(shirase_magic_signal) #built up the empty frame for reconstruction

#Plot the components
for comp in range(N_COMPONENTS):  #loop through components
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp] #current component
    cumulative_recon = cumulative_recon + current_component #add up
    ax.plot(magic_resam_shirase.index[:weeksize], shirase_magic_signal, lw=3, alpha=0.2, c='k', label='MagicGNSS Shirase: first day zoom')
    ax.plot(magic_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    #plotting current component, add up total and true signals for comparison
    ax.legend()
    plt.show()

#Plotting the bar chart to explain the variance contributions of components to the true signals
x = np.arange(10) #number of elementary components
y = mssa.explained_variance_ratio_[0:10].ravel() #explained variance
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("Shirase MagicGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :] #total components from SSA
print(total_comps.shape)
total_wcorr = mssa.w_correlation(total_comps) #built up correlation
total_wcorr_abs = np.abs(total_wcorr) #abs value
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax) #visualize the correlation in heatmap
ax.set_title('{} component w-correlations'.format(pd.DataFrame(magic_resam_shirase).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
"""Replotting"""
#From the w-corr matrix, some components are groupped together as they are highly correlated
#the reconstruction are plotted once again

#Group assignment
ts0_groups = [
    [0,1],
    [2,3,4],
    [5,6,7,8],
    [9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups) #set customed assignment
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped) #w-corr matrix
fig, ax = plt.subplots(figsize=(12,9)) #plotting
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax) #visualize it
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(magic_resam_shirase).columns[0]), fontsize = 16)
plt.show()

#Plotting final reconstruction
fig21, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(shirase_magic_signal)

for comp in range(4): #loop through elements for plotting them in subplots
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 411 + comp #plotting in subplots
    ax = plt.subplot(temp)
    ax.plot(magic_resam_shirase.index[:weeksize], shirase_magic_signal, lw=3, alpha=0.2, c='k', label='True Signal') #element 1
    ax.plot(magic_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp)) #element 2
    ax.plot(magic_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp)) #element 3
    ax.legend()
    
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical') #y-label
plt.tight_layout()
plt.show()

#%%
    #CSRS
   #repeat the code for CSRS 
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False)

csrs_resam_shirase = csrs_full_shirase['Filtered_Height']
csrs_resam_shirase = csrs_resam_shirase.resample('60S').mean().interpolate(method='time')
csrs_resam_shirase_arr = csrs_resam_shirase.values
mssa.fit(csrs_resam_shirase_arr)
shirase_csrs_signal = csrs_resam_shirase_arr[:weeksize]
cumulative_recon = np.zeros_like(shirase_csrs_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(csrs_resam_shirase.index[:weeksize], shirase_csrs_signal, lw=3, alpha=0.2, c='k', label='CSRS GNSS Shirase')
    ax.plot(csrs_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
#%%
#Plotting Bar Chart: Explained Variance
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("Shirase csrsGNSS: Explained Variance for each component", fontsize = 12)
plt.show()
total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(csrs_resam_shirase).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Groupings
ts0_groups = [
    [0,1],
    [2,3,4],
    [5,6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(csrs_resam_shirase).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(shirase_csrs_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(csrs_resam_shirase.index[:weeksize], shirase_csrs_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(csrs_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
    #RTKLIB
    #repeat the code for RTKLIB
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtk_resam_shirase = rtk_full_shirase['Filtered_Height']
rtk_resam_shirase = rtk_resam_shirase.resample('60S').mean().interpolate(method='time')
rtk_resam_shirase_arr = rtk_resam_shirase.values
mssa.fit(rtk_resam_shirase_arr)
shirase_rtk_signal = rtk_resam_shirase_arr[:weeksize]
cumulative_recon = np.zeros_like(shirase_rtk_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtk_resam_shirase.index[:weeksize], shirase_rtk_signal, lw=3, alpha=0.2, c='k', label='rtk GNSS Shirase')
    ax.plot(rtk_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
#%%
#Bar Chart: Explained variance
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("Shirase rtkGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtk_resam_shirase).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting after group assignment

ts0_groups = [
    [0,1],
    [2,3,4,5,6],
    [7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtk_resam_shirase).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(shirase_rtk_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtk_resam_shirase.index[:weeksize], shirase_rtk_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtk_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #RTKLIB DGNSS
    #repeat the code for RTKLIB
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtkd_resam_shirase = rtkd_full_shirase['Filtered_Height']
rtkd_resam_shirase = rtkd_resam_shirase.resample('60S').mean().interpolate(method='time')
rtkd_resam_shirase_arr = rtkd_resam_shirase.values
mssa.fit(rtkd_resam_shirase_arr)
shirase_rtkd_signal = rtkd_resam_shirase_arr[:weeksize]
cumulative_recon = np.zeros_like(shirase_rtkd_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtkd_resam_shirase.index[:weeksize], shirase_rtkd_signal, lw=3, alpha=0.2, c='k', label='rtkd GNSS Shirase')
    ax.plot(rtkd_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("Shirase rtkdGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtkd_resam_shirase).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1,2,3],
    [4,5],
    [6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtkd_resam_shirase).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(shirase_rtkd_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtkd_resam_shirase.index[:weeksize], shirase_rtkd_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtkd_resam_shirase.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_shirase.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
#Blake (Repeat SSA for Blake)
    #Magic
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

magic_resam_blake = magic_full_blake['Filtered_Height']
magic_resam_blake = magic_resam_blake.resample('60S').mean().interpolate(method='time')
magic_resam_blake_arr = magic_resam_blake.values
mssa.fit(magic_resam_blake_arr)
blake_magic_signal = magic_resam_blake_arr[:weeksize]
cumulative_recon = np.zeros_like(blake_magic_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(magic_resam_blake.index[:weeksize], blake_magic_signal, lw=3, alpha=0.2, c='k', label='MagicGNSS blake: first day zoom')
    ax.plot(magic_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
#%%
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("blake MagicGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(magic_resam_blake).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,4,5,6],
    [7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(magic_resam_blake).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(blake_magic_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(magic_resam_blake.index[:weeksize], blake_magic_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(magic_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #CSRS    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

csrs_resam_blake = csrs_full_blake['Filtered_Height']
csrs_resam_blake = csrs_resam_blake.resample('60S').mean().interpolate(method='time')
csrs_resam_blake_arr = csrs_resam_blake.values
mssa.fit(csrs_resam_blake_arr)
blake_csrs_signal = csrs_resam_blake_arr[:weeksize]
cumulative_recon = np.zeros_like(blake_csrs_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(csrs_resam_blake.index[:weeksize], blake_csrs_signal, lw=3, alpha=0.2, c='k', label='CSRS GNSS blake')
    ax.plot(csrs_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
#%%
#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("blake csrsGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(csrs_resam_blake).columns[0]))
plt.title('Total component w-correlations')
plt.show()
#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,6,7,8,9],
    [4,5],
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(csrs_resam_blake).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(blake_csrs_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(csrs_resam_blake.index[:weeksize], blake_csrs_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(csrs_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #RTKLIB PPP
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtk_resam_blake = rtk_full_blake['Filtered_Height']
rtk_resam_blake = rtk_resam_blake.resample('60S').mean().interpolate(method='time')
rtk_resam_blake_arr = rtk_resam_blake.values
mssa.fit(rtk_resam_blake_arr)
blake_rtk_signal = rtk_resam_blake_arr[:weeksize]
cumulative_recon = np.zeros_like(blake_rtk_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtk_resam_blake.index[:weeksize], blake_rtk_signal, lw=3, alpha=0.2, c='k', label='rtk GNSS blake')
    ax.plot(rtk_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("blake rtkGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtk_resam_blake).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3],
    [6,7,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtk_resam_blake).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(blake_rtk_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtk_resam_blake.index[:weeksize], blake_rtk_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtk_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%% Repeat SSA
    #RTKLIB DGNSS
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtkd_resam_blake = rtkd_full_blake['Filtered_Height']
rtkd_resam_blake = rtkd_resam_blake.resample('60S').mean().interpolate(method='time')
rtkd_resam_blake_arr = rtkd_resam_blake.values
mssa.fit(rtkd_resam_blake_arr)
blake_rtkd_signal = rtkd_resam_blake_arr[:weeksize]
cumulative_recon = np.zeros_like(blake_rtkd_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtkd_resam_blake.index[:weeksize], blake_rtkd_signal, lw=3, alpha=0.2, c='k', label='rtkd GNSS blake')
    ax.plot(rtkd_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
    
#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("blake rtkdGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtkd_resam_blake).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
#Replotting
ts0_groups = [
    [0,1,2,3],
    [4,5],
    [6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtkd_resam_blake).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(blake_rtkd_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtkd_resam_blake.index[:weeksize], blake_rtkd_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtkd_resam_blake.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_blake.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
#Tuati (Repeat the code for Tuati station)
    #Magic
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

magic_resam_tuati = magic_full_tuati['Filtered_Height']
magic_resam_tuati = magic_resam_tuati.resample('60S').mean().interpolate(method='time')
magic_resam_tuati_arr = magic_resam_tuati.values
mssa.fit(magic_resam_tuati_arr)
tuati_magic_signal = magic_resam_tuati_arr[:weeksize]
cumulative_recon = np.zeros_like(tuati_magic_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(magic_resam_tuati.index[:weeksize], tuati_magic_signal, lw=3, alpha=0.2, c='k', label='MagicGNSS tuati: first day zoom')
    ax.plot(magic_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()
#%%
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("tuati MagicGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

#%%
total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(magic_resam_tuati).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,4,5,6],
    [7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(magic_resam_tuati).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(tuati_magic_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(magic_resam_tuati.index[:weeksize], tuati_magic_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(magic_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
    #CSRS
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False)

csrs_resam_tuati = csrs_full_tuati['Filtered_Height']
csrs_resam_tuati = csrs_resam_tuati.resample('60S').mean().interpolate(method='time')
csrs_resam_tuati_arr = csrs_resam_tuati.values
mssa.fit(csrs_resam_tuati_arr)
tuati_csrs_signal = csrs_resam_tuati_arr[:weeksize]
cumulative_recon = np.zeros_like(tuati_csrs_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(csrs_resam_tuati.index[:weeksize], tuati_csrs_signal, lw=3, alpha=0.2, c='k', label='CSRS GNSS tuati')
    ax.plot(csrs_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("tuati csrsGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(csrs_resam_tuati).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3],
    [5,6,7,8],
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(csrs_resam_tuati).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(tuati_csrs_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(csrs_resam_tuati.index[:weeksize], tuati_csrs_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(csrs_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #RTKLIB PPP
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False)

rtk_resam_tuati = rtk_full_tuati['Filtered_Height']
rtk_resam_tuati = rtk_resam_tuati.resample('60S').mean().interpolate(method='time')
rtk_resam_tuati_arr = rtk_resam_tuati.values
mssa.fit(rtk_resam_tuati_arr)
tuati_rtk_signal = rtk_resam_tuati_arr[:weeksize]
cumulative_recon = np.zeros_like(tuati_rtk_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtk_resam_tuati.index[:weeksize], tuati_rtk_signal, lw=3, alpha=0.2, c='k', label='rtk GNSS tuati')
    ax.plot(rtk_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("tuati rtkGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtk_resam_tuati).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3],
    [4,5,6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtk_resam_tuati).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(tuati_rtk_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtk_resam_tuati.index[:weeksize], tuati_rtk_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtk_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')
#%%
    #RTKLIB DGNSS
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False)

rtkd_resam_tuati = rtkd_full_tuati['Filtered_Height']
rtkd_resam_tuati = rtkd_resam_tuati.resample('60S').mean().interpolate(method='time')
rtkd_resam_tuati_arr = rtkd_resam_tuati.values
mssa.fit(rtkd_resam_tuati_arr)
tuati_rtkd_signal = rtkd_resam_tuati_arr[:weeksize]
cumulative_recon = np.zeros_like(tuati_rtkd_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtkd_resam_tuati.index[:weeksize], tuati_rtkd_signal, lw=3, alpha=0.2, c='k', label='rtkd GNSS tuati')
    ax.plot(rtkd_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("tuati rtkdGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtkd_resam_tuati).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,4,5],
    [6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtkd_resam_tuati).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(tuati_rtkd_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtkd_resam_tuati.index[:weeksize], tuati_rtkd_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtkd_resam_tuati.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_tuati.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
#Hillary (Repeat SSA for Hillary station)
    #Magic
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

magic_resam_hillary = magic_full_hillary['Filtered_Height']
magic_resam_hillary = magic_resam_hillary.resample('60S').mean().interpolate(method='time')
magic_resam_hillary_arr = magic_resam_hillary.values
mssa.fit(magic_resam_hillary_arr)
hillary_magic_signal = magic_resam_hillary_arr[:weeksize]
cumulative_recon = np.zeros_like(hillary_magic_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(magic_resam_hillary.index[:weeksize], hillary_magic_signal, lw=3, alpha=0.2, c='k', label='MagicGNSS hillary: first day zoom')
    ax.plot(magic_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("hillary MagicGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(magic_resam_hillary).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,4,5,6],
    [7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(magic_resam_hillary).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(hillary_magic_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(magic_resam_hillary.index[:weeksize], hillary_magic_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(magic_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(magic_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()

ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #CSRS
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False)

csrs_resam_hillary = csrs_full_hillary['Filtered_Height']
csrs_resam_hillary = csrs_resam_hillary.resample('60S').mean().interpolate(method='time')
csrs_resam_hillary_arr = csrs_resam_hillary.values
mssa.fit(csrs_resam_hillary_arr)
hillary_csrs_signal = csrs_resam_hillary_arr[:weeksize]
cumulative_recon = np.zeros_like(hillary_csrs_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(csrs_resam_hillary.index[:weeksize], hillary_csrs_signal, lw=3, alpha=0.2, c='k', label='CSRS GNSS hillary')
    ax.plot(csrs_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp)) 
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("hillary csrsGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(csrs_resam_hillary).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3,4,5,6,7,8],
    [9],
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(csrs_resam_hillary).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(hillary_csrs_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(csrs_resam_hillary.index[:weeksize], hillary_csrs_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(csrs_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(csrs_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
    #RTKLIB PPP
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtk_resam_hillary = rtk_full_hillary['Filtered_Height']
rtk_resam_hillary = rtk_resam_hillary.resample('60S').mean().interpolate(method='time')
rtk_resam_hillary_arr = rtk_resam_hillary.values
mssa.fit(rtk_resam_hillary_arr)
hillary_rtk_signal = rtk_resam_hillary_arr[:weeksize]
cumulative_recon = np.zeros_like(hillary_rtk_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtk_resam_hillary.index[:weeksize], hillary_rtk_signal, lw=3, alpha=0.2, c='k', label='rtk GNSS hillary')
    ax.plot(rtk_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("hillary rtkGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtk_resam_hillary).columns[0]))
plt.title('Total component w-correlations')
plt.show()
#%%
#Replotting
ts0_groups = [
    [0,1],
    [2,3],
    [5,6],
    [8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtk_resam_hillary).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(hillary_rtk_signal)
for comp in range(4):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 411 + comp
    ax = plt.subplot(temp)
    ax.plot(rtk_resam_hillary.index[:weeksize], hillary_rtk_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtk_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtk_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
    #RTKLIB DGNSS
    
N_COMPONENTS = 10
weeksize = 60*24*7
mssa = MSSA(n_components=N_COMPONENTS, window_size=60*24, verbose=False) #window size = 1 day

rtkd_resam_hillary = rtkd_full_hillary['Filtered_Height']
rtkd_resam_hillary = rtkd_resam_hillary.resample('60S').mean().interpolate(method='time')
rtkd_resam_hillary_arr = rtkd_resam_hillary.values
mssa.fit(rtkd_resam_hillary_arr)
hillary_rtkd_signal = rtkd_resam_hillary_arr[:weeksize]
cumulative_recon = np.zeros_like(hillary_rtkd_signal)

for comp in range(N_COMPONENTS):  
    fig20, ax = plt.subplots(figsize=(18, 7))
    current_component = mssa.components_[0, :weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    ax.plot(rtkd_resam_hillary.index[:weeksize], hillary_rtkd_signal, lw=3, alpha=0.2, c='k', label='rtkd GNSS hillary')
    ax.plot(rtkd_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    
    ax.legend()
    plt.show()

#Bar Chart
x = np.arange(10)
y = mssa.explained_variance_ratio_[0:10].ravel()
fig21, ax = plt.subplots()
ax.bar(x,y)
ax.set_xticks(x)
plt.title("hillary rtkdGNSS: Explained Variance for each component", fontsize = 12)
plt.show()

total_comps = mssa.components_[0, :, :]
print(total_comps.shape)

#W-Correlation Matrix
total_wcorr = mssa.w_correlation(total_comps)
total_wcorr_abs = np.abs(total_wcorr)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
ax.set_title('{} component w-correlations'.format(pd.DataFrame(rtkd_resam_hillary).columns[0]))
plt.title('Total component w-correlations')
plt.show()

#%%
#Replotting
ts0_groups = [
    [0,1,2,3],
    [4,5],
    [6,7,8,9]
]

mssa = mssa.set_ts_component_groups(0, ts0_groups)
ts0_grouped = mssa.grouped_components_[0]
print(ts0_grouped.shape)

ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
ax.set_title('{} grouped component w-correlations'.format(pd.DataFrame(rtkd_resam_hillary).columns[0]), fontsize = 16)
plt.show()

fig21, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(18, 7))
cumulative_recon = np.zeros_like(hillary_rtkd_signal)
for comp in range(3):
    current_component = mssa.grouped_components_[0][:weeksize, comp]
    cumulative_recon = cumulative_recon + current_component
    temp = 311 + comp
    ax = plt.subplot(temp)
    ax.plot(rtkd_resam_hillary.index[:weeksize], hillary_rtkd_signal, lw=3, alpha=0.2, c='k', label='True Signal')
    ax.plot(rtkd_resam_hillary.index[:weeksize], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
    ax.plot(rtkd_resam_hillary.index[:weeksize], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
    ax.legend()
ax.set_xlabel('Date')
fig21.text(0.01, 0.5, 'Height (m)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.clf()
plt.cla()
plt.close('all')

#%%
#Plotting vertical displacement
vertdisp = [csrs_full_hillary['Filtered_Height'].abs(),csrs_full_blake['Filtered_Height'].abs(),\
            csrs_full_tuati['Filtered_Height'].abs(),csrs_full_shirase['Filtered_Height'].abs()]
ybar = [csrs_full_hillary['Filtered_Height'].abs().mean(),csrs_full_blake['Filtered_Height'].abs().mean(),\
            csrs_full_tuati['Filtered_Height'].abs().mean(),csrs_full_shirase['Filtered_Height'].abs().mean()]
x = ["Hillary","Blake","Tuati","Shirase"]
vertdisp_pd = pd.DataFrame({'Station': x, 'Vertical Displacement (m)': vertdisp}) 
vertdisp_pd = vertdisp_pd.explode('Station')
sns.boxplot(x = 'Station', y = 'Vertical Displacement (m)', data = vertdisp_pd, width=.3)
plt.show()

#%%
"""Calculating Horizontal Velocity using derivative"""
#Tuati
tuati_v_lon_comp = csrs_full_tuati['Filtered_Longitude'].rolling(720*2).mean().diff() + 10
tuati_v_lat_comp = csrs_full_tuati['Filtered_Latitude'].rolling(720*2).mean().diff() + 10
csrs_full_tuati['horizontal_derivative'] = np.sqrt((tuati_v_lat_comp)**2 + (tuati_v_lat_comp)**2)*120*1000 #mm/day
csrs_full_tuati['horizontal_derivative'] = csrs_full_tuati['horizontal_derivative'].sub(csrs_full_tuati['horizontal_derivative'].mean())
csrs_full_tuati['horizontal_derivative'].plot()

#Shirase
shirase_v_lon_comp = csrs_full_shirase['Filtered_Longitude'].rolling(720*4).mean().diff() + 10
shirase_v_lat_comp = csrs_full_shirase['Filtered_Latitude'].rolling(720*4).mean().diff() + 10
csrs_full_shirase['horizontal_derivative'] = np.sqrt((shirase_v_lat_comp)**2 + (shirase_v_lat_comp)**2)*120*1000 #mm/day
csrs_full_shirase['horizontal_derivative'] = csrs_full_shirase['horizontal_derivative'].sub(csrs_full_shirase['horizontal_derivative'].mean())                                                                                      
csrs_full_shirase['horizontal_derivative'].plot()

#Blake
blake_v_lon_comp = csrs_full_blake['Filtered_Longitude'].rolling(720*4).mean().diff() + 10
blake_v_lat_comp = csrs_full_blake['Filtered_Latitude'].rolling(720*4).mean().diff() + 10
csrs_full_blake['horizontal_derivative'] = np.sqrt((blake_v_lat_comp)**2 + (blake_v_lat_comp)**2)*120*1000 #mm/day
csrs_full_blake['horizontal_derivative'] = csrs_full_blake['horizontal_derivative'].sub(csrs_full_blake['horizontal_derivative'].mean())                                                                                      
csrs_full_blake['horizontal_derivative'].plot()

#Hillary
hillary_v_lon_comp = csrs_full_hillary['Filtered_Longitude'].rolling(720*4).mean().diff() + 10
hillary_v_lat_comp = csrs_full_hillary['Filtered_Latitude'].rolling(720*4).mean().diff() + 10
csrs_full_hillary['horizontal_derivative'] = np.sqrt((hillary_v_lat_comp)**2 + (hillary_v_lat_comp)**2)*120*1000 #mm/day
csrs_full_hillary['horizontal_derivative'] = csrs_full_hillary['horizontal_derivative'].sub(csrs_full_hillary['horizontal_derivative'].mean())                                                                                       
csrs_full_hillary['horizontal_derivative'].plot()
plt.show()

#%%
"""Calculating distance between stations"""
distance_shirase_tuati = np.sqrt((csrs_full_shirase["latitude_M"].mean()/1000 - csrs_full_tuati["latitude_M"].mean()/1000)**2 + \
    (csrs_full_shirase["longitude_M"].mean()/1000 - csrs_full_tuati["longitude_M"].mean()/1000)**2)

distance_tuati_blake = np.sqrt((csrs_full_tuati["latitude_M"].mean()/1000 - csrs_full_blake["latitude_M"].mean()/1000)**2 + \
    (csrs_full_tuati["longitude_M"].mean()/1000 - csrs_full_blake["longitude_M"].mean()/1000)**2)

distance_blake_hillary = np.sqrt((csrs_full_blake["latitude_M"].mean()/1000 - csrs_full_hillary["latitude_M"].mean()/1000)**2 + \
    (csrs_full_blake["longitude_M"].mean()/1000 - csrs_full_hillary["longitude_M"].mean()/1000)**2)

distance_hillary_base = np.sqrt((csrs_full_hillary["latitude_M"].mean()/1000 - csrs_full_base["latitude_M"].mean()/1000)**2 + \
    (csrs_full_hillary["longitude_M"].mean()/1000 - csrs_full_base["longitude_M"].mean()/1000)**2)
    
print("Distance between Shirase and Tuati is: (m)", distance_shirase_tuati)
print("Distance between Tuati and Blake is: (m)", distance_tuati_blake)
print("Distance between Blake and Hillary is: (m)", distance_blake_hillary)
print("Distance between Hillary and Base is: (m)", distance_hillary_base)

#%%
"""Tides"""
#Reading tide data
tides_doy = pd.read_csv('Tides_Doy_Amplitude.txt',delim_whitespace=True,header = None,\
    names=["DOY","tides_amplitude(m)"],parse_dates=\
        {'datetime': ['DOY']},date_parser=parse_doy2date)

##reset datetime index
tides_doy.set_index(pd.DatetimeIndex(tides_doy['datetime']),inplace=True)
tides_doy = tides_doy.drop(columns=['datetime'])
##slide the right time frame with 15s interval
tides_doy = tides_doy.loc["2018-11-05 00:00:00":"2018-11-15 00:00:00"]
tides_doy = tides_doy.iloc[:-1]
tides_doy = tides_doy.resample('s').interpolate('cubic').resample('15s').asfreq().dropna()

#Plotting rolling horizontal velocity with tides (CSRS data)
tidal_height = plt.figure(figsize=(18,14))
ax011 = tidal_height.add_subplot(211)
tides_doy.loc["2018-11-08 00:00:00":"2018-11-12 00:00:00"].plot(ax=ax011,color='gray',linewidth=2.0)
ax012 = tidal_height.add_subplot(212, sharex = ax011)

csrs_full_shirase['horizontal_derivative'].loc["2018-11-08 00:00:00":"2018-11-12 12:00:00"].rolling(window=720*2).mean().plot(ax=ax012)
csrs_full_tuati['horizontal_derivative'].loc["2018-11-08 00:00:00":"2018-11-12 12:00:00"].rolling(window=720).mean().plot(ax=ax012)
csrs_full_blake['horizontal_derivative'].loc["2018-11-08 00:00:00":"2018-11-12 12:00:00"].rolling(window=720*2).mean().plot(ax=ax012)
csrs_full_hillary['horizontal_derivative'].loc["2018-11-08 00:00:00":"2018-11-12 12:00:00"].rolling(window=720*2).mean().plot(ax=ax012)

ax011.legend(["Amplitude of tides (m)"])
ax012.legend(["Shirase","Tuati","Blake","Hillary"])
plt.xlabel("Datetime")
plt.ylabel("Horizontal Velocity (mm/day)")
plt.tight_layout()
plt.show()

#%%
"""Plotting Bootscrapping convergence"""
magic_sample_h = magic_full_base['Filtered_Height'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
magic_h_boot_means = []
rtk_sample_h = rtk_full_base['Filtered_Height'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
rtk_h_boot_means = []
csrs_sample_h = csrs_full_base['Filtered_Height'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
csrs_h_boot_means = []
bootscrap_mean_magic = []
bootscrap_mean_rtk = []
bootscrap_mean_csrs = []

#Setting constants (sample size)
n = 5000
n1 = 5000
n2 = 5000

#Magic
x = magic_sample_h
reps = 1000
conv, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30,24), sharex=True, sharey=True)
xb = np.random.choice(x, (n, reps), replace=True)
yb = 1/np.arange(1, n+1)[:, None] * np.cumsum(xb, axis=0)
upper, lower = np.percentile(yb, [2.5, 97.5], axis=1)
ax1.plot(np.arange(1, n+1)[:, None], yb, c='grey', alpha=0.04)
ax1.plot(np.arange(1, n+1), yb[:, 0], c='red', linewidth=2,label="magicGNSS/PPP")
ax1.plot(np.arange(1, n+1), upper, 'b', np.arange(1, n+1), lower, 'b')
ax1.legend()

#RTK
x1 = rtk_sample_h
xb1 = np.random.choice(x1, (n1, reps), replace=True)
yb1 = 1/np.arange(1, n1+1)[:, None] * np.cumsum(xb1, axis=0)
upper1, lower1 = np.percentile(yb1, [2.5, 97.5], axis=1)
ax2.plot(np.arange(1, n1+1)[:, None], yb1, c='grey', alpha=0.04)
ax2.plot(np.arange(1, n1+1), yb1[:, 0], c='green', linewidth=2,label="RTKLIB/PPP")
ax2.plot(np.arange(1, n1+1), upper1, 'b', np.arange(1, n1+1), lower1, 'b')
ax2.legend()

#CSRS
x2 = csrs_sample_h
xb2 = np.random.choice(x2, (n2, reps), replace=True)
yb2 = 1/np.arange(1, n2+1)[:, None] * np.cumsum(xb2, axis=0)
upper2, lower2 = np.percentile(yb2, [2.5, 97.5], axis=1)
ax3.plot(np.arange(1, n2+1)[:, None], yb2, c='grey', alpha=0.04)
ax3.plot(np.arange(1, n2+1), yb2[:, 0], c='aqua', linewidth=2,label='CSRS/PPP')
ax3.plot(np.arange(1, n2+1), upper2, 'b', np.arange(1, n2+1), lower2, 'b')

#Layout
conv.text(0.07,0.5, "z value (m)", ha="center", va="center", rotation=90, fontsize=20)
plt.xlabel("Number of Data Sample", fontsize=20)
plt.legend()
plt.show()

#%%
""" X, Y, Z Best Estimate """
magic_full_base['Demean_detrend_Latitude'] = magic_full_base['latitude_M']
magic_full_base['Demean_detrend_Longitude'] = magic_full_base['longitude_M']
magic_full_base['Demean_detrend_Height'] = magic_full_base['h']

csrs_full_base['Demean_detrend_Latitude'] = csrs_full_base['latitude_M']
csrs_full_base['Demean_detrend_Longitude'] = csrs_full_base['longitude_M']
csrs_full_base['Demean_detrend_Height'] = csrs_full_base['HGT(m)']

rtk_full_base['Demean_detrend_Latitude'] = rtk_full_base['latitude_M']
rtk_full_base['Demean_detrend_Longitude'] = rtk_full_base['longitude_M']
rtk_full_base['Demean_detrend_Height'] = rtk_full_base['height(m)']

#Excluding the margin of time period for calculation
magic_sample_h = magic_full_base['h'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
rtk_sample_h = rtk_full_base['height(m)'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
csrs_sample_h = csrs_full_base['HGT(m)'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()

magic_sample_x = magic_full_base['latitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
rtk_sample_x = rtk_full_base['latitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
csrs_sample_x = csrs_full_base['latitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()

magic_sample_y = magic_full_base['longitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
rtk_sample_y = rtk_full_base['longitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()
csrs_sample_y = csrs_full_base['longitude_M'].loc["2018-11-06 00:00:00":"2018-11-12 00:00:00"].to_numpy()

#Cutting out inconsistency of data length
magic_sample = [magic_sample_x[:33000],magic_sample_y[:33000],magic_sample_h[:33000]]
csrs_sample = [csrs_sample_x[:33000],csrs_sample_y[:33000],csrs_sample_h[:33000]]
rtk_sample = [rtk_sample_x[:33000],rtk_sample_y[:33000],rtk_sample_h[:33000]]

#Constants
n = 1000 #sample size
reps = 1000 #repetitions

#Fill in table
magic_x_pool = magic_sample[0]
magic_y_pool = magic_sample[1]
magic_h_pool = magic_sample[2]
csrs_x_pool = csrs_sample[0]
csrs_y_pool = csrs_sample[1]
csrs_h_pool = csrs_sample[2]
rtk_x_pool = rtk_sample[0]
rtk_y_pool = rtk_sample[1]
rtk_h_pool = rtk_sample[2]

#Empty list for mean
sample_i = np.random.choice(np.arange(0,33000), (reps, n), replace=True)
magic_mean_x = []
magic_mean_y = []
magic_mean_h = []

csrs_mean_x = []
csrs_mean_y = []
csrs_mean_h = []

rtk_mean_x = []
rtk_mean_y = []
rtk_mean_h = []

#Looping the sample mean
for i in range(len(sample_i)):
    magic_mean_x.append(np.mean(magic_x_pool[sample_i[i]]))
    magic_mean_y.append(np.mean(magic_y_pool[sample_i[i]]))
    magic_mean_h.append(np.mean(magic_h_pool[sample_i[i]]))
    
    rtk_mean_x.append(np.mean(rtk_x_pool[sample_i[i]]))
    rtk_mean_y.append(np.mean(rtk_y_pool[sample_i[i]]))
    rtk_mean_h.append(np.mean(rtk_h_pool[sample_i[i]]))
    
    csrs_mean_x.append(np.mean(csrs_x_pool[sample_i[i]]))
    csrs_mean_y.append(np.mean(csrs_y_pool[sample_i[i]]))
    csrs_mean_h.append(np.mean(csrs_h_pool[sample_i[i]]))

#Results of best estimates
x_bar = (np.mean(magic_mean_x) + np.mean(csrs_mean_x) + np.mean(rtk_mean_x))/3
y_bar = (np.mean(magic_mean_y) + np.mean(csrs_mean_y) + np.mean(rtk_mean_y))/3
h_bar = (np.mean(magic_mean_h) + np.mean(csrs_mean_h) + np.mean(rtk_mean_h))/3

#%%
#Calculating RMSE
RMSE_magic = [np.nan, np.nan, np.nan]
RMSE_csrs = [np.nan, np.nan, np.nan]
RMSE_rtk = [np.nan, np.nan, np.nan]

#x
RMSE_magic[0] = np.sum(((magic_x_pool - x_bar)**2)/len(magic_x_pool))
RMSE_csrs[0] = np.sum(((csrs_x_pool - x_bar)**2)/len(csrs_x_pool))
RMSE_rtk[0] = np.sum(((rtk_x_pool - x_bar)**2)/len(rtk_x_pool))
#y
RMSE_magic[1] = np.sum(((magic_y_pool - y_bar)**2)/len(magic_y_pool))
RMSE_csrs[1] = np.sum(((csrs_y_pool - y_bar)**2)/len(csrs_y_pool))
RMSE_rtk[1] = np.sum(((rtk_y_pool - y_bar)**2)/len(rtk_y_pool))
#h
RMSE_magic[2] = np.sum(((magic_h_pool - h_bar)**2)/len(magic_h_pool))
RMSE_csrs[2] = np.sum(((csrs_h_pool - h_bar)**2)/len(csrs_h_pool))
RMSE_rtk[2] = np.sum(((rtk_h_pool - h_bar)**2)/len(rtk_h_pool))

### END OF CODE ###















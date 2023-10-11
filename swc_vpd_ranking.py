import numpy as np
import pandas as pd
from datetime import datetime

from utility_modules import load_swc_ne3, load_swc_ne2, load_swc_bo1
from run_case import flux_file_suffix_dict, aflux_file_suffix_dict
from noah_energy.phm.flux_subroutines import ESAT


# this script plots the binned scatter response of PHM and Beta models
# plot time: maize 7/1-9/1, soy 7/15 - 9/1

def main():
    # set up multiple sites and years
    sites = ['US-Bo1']
    # for maize
    site_years = {'US-Ne2': [2005, 2007, 2009, 2010, 2011, 2012], 'US-Ne3': [2005, 2007, 2009, 2011],
                  'US-Bo1': [2001, 2003, 2005, 2007]}
    # for soy
    # site_years = {'US-Ne2': [2004, 2006, 2008], 'US-Ne3': [2004, 2006, 2008, 2010, 2012]}
    obs_hourly = 0

    # loading data
    ET_flux_l = []
    swc_l = []
    vpd_l = []
    mean_ET_flux_l = []
    mean_swc_l = []
    mean_vpd_l = []
    for site in sites:
        for year in site_years[site]:
            print('processing site: {}, year: {}'.format(site, year))
            start = datetime(year, 7, 1)
            end = datetime(year, 8, 15)

            # file path
            flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
            flux_file_suffix = flux_file_suffix_dict[site]
            aflux_file_suffix = aflux_file_suffix_dict[site]
            flux_file = flux_path + 'fluxnet/' + flux_file_suffix
            aflux_file = flux_path + 'ameriflux/' + aflux_file_suffix

            # load flux
            # print('reading obs. data...')
            if len(flux_file_suffix) > 0:
                df = pd.read_csv(flux_file)
                ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE_F_MDS'].values, -9999)
                temp_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA_F'].values, -9999)
                rh_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH'].values, -9999)
            else:
                df = pd.read_csv(aflux_file, skiprows=2)
                ET_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LE'].values, -9999)
                temp_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA'].values, -9999)
                rh_flux = np.ma.masked_values(df[(df.TIMESTAMP_START >= int(
                    '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
                    '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH'].values, -9999)

            # load SWC
            swc = 'holder'
            if site == 'US-Ne3':
                swc_h, swc_v = load_swc_ne3(aflux_file, start, end)
                swc = swc_v[1]
            elif site == 'US-Ne2':
                swc_h, swc_v = load_swc_ne2(aflux_file, start, end)
                swc = swc_v[1]
            elif site == 'US-Bo1':
                swc_h, swc_v = load_swc_bo1(aflux_file, start, end)
                swc = swc_v[1]

            # convert all to hourly
            if not obs_hourly:
                ET_flux = np.ma.mean(np.ma.reshape(ET_flux, (-1, 2)), 1)
                swc = np.ma.mean(np.ma.reshape(swc, (-1, 2)), 1)
                temp_flux = np.ma.mean(np.ma.reshape(temp_flux, (-1, 2)), 1)
                rh_flux = np.ma.mean(np.ma.reshape(rh_flux, (-1, 2)), 1)

            # get vpd
            ESW, ESI, DESW, DESI = ESAT(temp_flux)
            vpd_flux = (100 - rh_flux) / 100 * ESW

            # calculate and record mean
            mean_length = 24
            ET_flux_md = ET_flux[mean_length // 2::mean_length]
            swc_md = swc[mean_length // 2::mean_length]
            vpd_md = vpd_flux[mean_length // 2::mean_length]
            mean_ET_flux_l.append(np.mean(ET_flux_md))
            mean_swc_l.append(np.mean(swc_md))
            mean_vpd_l.append(np.mean(vpd_md))

            # record data
            ET_flux_l.append(ET_flux)
            swc_l.append(swc)
            vpd_l.append(vpd_flux)

    # print
    site_years_l = [[(site, year) for year in site_years[site]] for site in sites]
    print(site_years_l)
    print('mean_ET_flux_l:', mean_ET_flux_l)
    print('mean_swc_l:', mean_swc_l)
    print('mean_vpd_l:', mean_vpd_l)


if __name__ == '__main__':
    main()

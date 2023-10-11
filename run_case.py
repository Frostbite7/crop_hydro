from datetime import datetime
from noah_energy.crop_hydro.snap_run_model import run_model


# case info
# Sim period: Ne1-3 7/1 - 9/15


def main():
    # sim config
    site = 'US-Ne2'
    crop = 'soy'
    year = 2004
    mode = 'beta'
    start = datetime(year, 7, 1)
    end = datetime(year, 9, 15)
    test_suffix = '_kmax4.18'
    mode_suffix = ''
    CAP_beta = 1
    if CAP_beta == 1 and mode == 'beta':
        mode_suffix = '_capped'

    # site info
    coordinates = coord_dict[site]
    force_start_year = site_year_dict[site][0]
    force_end_year = site_year_dict[site][1]
    utc_offset = -6
    if site == 'US-Ne3' or site == 'US-Ne2' or site == 'US-Ne1':
        DT = 3600
    else:
        DT = 3600

    # file paths
    parm_path = '/Users/yangyicge/Desktop/watercon/script/noah_energy/parameters/{}/energy_parameter_{}_{}.ini'.format(site, crop, mode)
    flux_file_suffix = flux_file_suffix_dict[site]
    aflux_file_suffix = aflux_file_suffix_dict[site]
    results_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/case_run/{}/{}_{}_{}/'.format(site, site, start.year, crop)
    results_suffix = '_' + mode + mode_suffix + test_suffix

    # run case
    run_model(site, coordinates, utc_offset, DT, parm_path, start, end, results_path, results_suffix, flux_file_suffix,
              aflux_file_suffix, force_start_year, force_end_year, mode, DYNAMIC_beta=0, CAP_beta=CAP_beta)

    return


coord_dict = {'US-Ne1': [41.165, -96.476], 'US-Ne2': [41.165, -96.476], 'US-Ne3': [41.165, -96.476], 'US-Bo1': [40.0062, -88.2904]}
site_year_dict = {'US-Ne1': [2003, 2012], 'US-Ne2': [2003, 2012], 'US-Ne3': [2003, 2012], 'US-Bo1': [2000, 2008]}
flux_file_suffix_dict = {'US-Ne3': 'FLX_US-Ne3_FLUXNET2015_SUBSET_2001-2013_1-4/FLX_US-Ne3_FLUXNET2015_SUBSET_HR_2001-2013_1-4.csv',
                         'US-Ne2': 'FLX_US-Ne2_FLUXNET2015_SUBSET_2001-2013_1-3/FLX_US-Ne2_FLUXNET2015_SUBSET_HR_2001-2013_1-3.csv',
                         'US-Ne1': 'FLX_US-Ne1_FLUXNET2015_SUBSET_2001-2013_1-3/FLX_US-Ne1_FLUXNET2015_SUBSET_HR_2001-2013_1-3.csv',
                         'US-Bo1': '',
                         }
aflux_file_suffix_dict = {'US-Ne3': 'AMF_US-Ne3_BASE-BADM_9-5/AMF_US-Ne3_BASE_HR_9-5.csv',
                          'US-Ne2': 'AMF_US-Ne2_BASE-BADM_12-5/AMF_US-Ne2_BASE_HR_12-5.csv',
                          'US-Ne1': 'AMF_US-Ne1_BASE-BADM_12-5/AMF_US-Ne1_BASE_HR_12-5.csv',
                          'US-Bo1': 'AMF_US-Bo1_BASE-BADM_2-1/AMF_US-Bo1_BASE_HH_2-1.csv',
                          }

if __name__ == '__main__':
    main()

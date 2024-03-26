import configparser
import os
from datetime import datetime
from datetime import timedelta
# from datetime import timezone

import numpy as np
import pandas as pd

from utility_modules import load_swc_ne3, load_swc_ne2, load_swc_ne1, load_swc_bo1, load_swc_br1, load_swc_br3
from noah_energy.no_stress.energy_driver import ENERGY_DRIVER_NOSTRESS
from noah_energy.phm.energy_driver import ENERGY_DRIVER_PHM
from noah_energy.btran.energy_driver import ENERGY_DRIVER_BTRAN


def run_model(site, coordinates, utc_offset, DT, parm_path, start, end, results_path, results_suffix, flux_file_suffix,
              aflux_file_suffix, force_start_year, force_end_year, mode, DYNAMIC_beta, CAP_beta):
    # site info
    LAT = coordinates[0]
    LON = coordinates[1]

    # flux data
    flux_path = '/Users/yangyicge/Desktop/watercon/flux/'
    flux_file = flux_path + 'fluxnet/' + flux_file_suffix
    aflux_file = flux_path + 'ameriflux/' + aflux_file_suffix

    # read full model force data
    print('reading data...')
    full_model_force_path = '/Users/yangyicge/Desktop/watercon/forcing/gapfilled_crop/'
    force_start = datetime(force_start_year, 1, 1)
    lines_to_skip = 71
    slice_start = (start - force_start).days * 24 + lines_to_skip
    slice_end = (end - force_start).days * 24 + lines_to_skip
    full_model_force = open(full_model_force_path + '{}_{}-{}/force.dat'.format(site[3:], force_start_year, force_end_year), 'r')

    lai_force = []
    ws_force = []
    temp_force = []
    rh_force = []
    prs_force = []
    soldn_force = []
    lwdn_force = []
    prec_force = []
    gh_force = []
    i = 0
    for line in full_model_force.readlines():
        if slice_start <= i < slice_end:
            lai_force.append(float(line[157:169]))
            ws_force.append(float(line[20:33]))
            temp_force.append(float(line[53:67]))
            rh_force.append(float(line[70:84]))
            prs_force.append(float(line[86:101]))
            soldn_force.append(float(line[103:118]))
            lwdn_force.append(float(line[121:135]))
            prec_force.append(float(line[140:152]))
            gh_force.append(0)
        i = i + 1
    print('lai_force: ', lai_force)

    # read flux forcing data
    time_flux = pd.date_range(start=start, end=end, freq='{}S'.format(DT), tz='UTC')[0:-1]
    time_flux = time_flux.to_pydatetime()
    if len(flux_file_suffix) > 0:
        print('get forcing from fluxnet...')
        df = pd.read_csv(flux_file)
        # time_flux = df[(df.TIMESTAMP_START >= int(
        #     '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
        #     '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TIMESTAMP_START']
        ws_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'WS_F']
        temp_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'TA_F']
        rh_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'RH']
        prs_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'PA_F']
        soldn_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'SW_IN_F']
        lwdn_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'LW_IN_F']
        prec_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'P_F']
        gh_flux = df[(df.TIMESTAMP_START >= int(
            '{:4d}{:02d}{:02d}0000'.format(start.year, start.month, start.day))) & (df.TIMESTAMP_START < int(
            '{:4d}{:02d}{:02d}0000'.format(end.year, end.month, end.day)))].loc[:, 'G_F_MDS']
    else:
        print('get forcing from full model...')
        ws_flux = pd.Series(ws_force)
        temp_flux = pd.Series(temp_force) - 273.15
        rh_flux = pd.Series(rh_force)
        prs_flux = pd.Series(prs_force) / 10
        soldn_flux = pd.Series(soldn_force)
        lwdn_flux = pd.Series(lwdn_force)
        prec_flux = pd.Series(prec_force) * DT
        gh_flux = pd.Series(gh_force)

    # inspect gaps
    print('Forcing Gaps! ws, temp, rh, prs, soldn, lwdn, prec, gh:\n', np.sum(ws_flux == -9999), np.sum(temp_flux == -9999),
          np.sum(rh_flux == -9999), np.sum(prs_flux == -9999), np.sum(soldn_flux == -9999), np.sum(lwdn_flux == -9999),
          np.sum(prec_flux == -9999), np.sum(gh_flux == -9999))

    # gap filling
    ws_flux = ws_flux.replace(-9999, np.nan).interpolate()
    temp_flux = temp_flux.replace(-9999, np.nan).interpolate()
    rh_flux = rh_flux.replace(-9999, np.nan).interpolate()
    prs_flux = prs_flux.replace(-9999, np.nan).interpolate()
    soldn_flux = soldn_flux.replace(-9999, np.nan).interpolate()
    lwdn_flux = lwdn_flux.replace(-9999, np.nan).interpolate()
    prec_flux = prec_flux.replace(-9999, np.nan).interpolate()
    gh_flux = gh_flux.replace(-9999, np.nan).interpolate()

    # load SWC
    print('reading obs. data...')
    swc_v = np.random.rand(1, 1)
    if site == 'US-Ne3':
        swc_h, swc_v = load_swc_ne3(aflux_file, start, end)
    elif site == 'US-Ne2':
        swc_h, swc_v = load_swc_ne2(aflux_file, start, end)
    elif site == 'US-Ne1':
        swc_h, swc_v = load_swc_ne1(aflux_file, start, end)
    elif site == 'US-Bo1':
        swc_h, swc_v = load_swc_bo1(aflux_file, start, end)
    elif site == 'US-Br1':
        swc_h, swc_v = load_swc_br1(aflux_file, start, end)
    elif site == 'US-Br3':
        swc_h, swc_v = load_swc_br3(aflux_file, start, end)
    swc_v = pd.DataFrame(swc_v).interpolate(axis=1).values

    # load parameters
    config = configparser.ConfigParser()
    config.read(parm_path)

    # model config
    DT_unused, NSOIL, ZSOIL, DZSNSO = SOIL_LAYER_CONFIG(config)
    if DT == 1800:
        lai_force = np.repeat(lai_force, 2)

    # initialize results
    times = []
    transpiration = []
    soil_evaporation = []
    canopy_evaporation = []
    rssun = []
    rssha = []
    psn = []
    fsun = []
    fsh = []
    sav = []
    sag = []
    fsa = []
    fsr = []
    fira = []
    apar = []
    parsun = []
    parsha = []
    lai = []
    tr_p = []
    fstomata = []

    # run
    print('\nbegin running...')
    n_records = time_flux.shape[0]
    for i in range(n_records):
        # time = str(time_flux.iloc[i])
        # TIME = datetime(int(time[:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12]),
        #                 tzinfo=timezone.utc) + timedelta(hours=-utc_offset)
        TIME = time_flux[i] + timedelta(hours=-utc_offset)
        # print(TIME)
        if i % 240 == 0:
            print('\n', TIME, TIME.tzinfo)

        # forcing variables from flux data
        WS = ws_flux.iloc[i]
        SFCTMP = temp_flux.iloc[i] + 273.15
        RH = rh_flux.iloc[i]
        SFCPRS = prs_flux.iloc[i] * 1000
        SOLDN = soldn_flux.iloc[i]
        LWDN = lwdn_flux.iloc[i]
        PRECP = prec_flux.iloc[i] / DT
        GH = gh_flux.iloc[i]

        # variables from flux data
        SH2O = 'holder'
        if site == 'US-Ne3':
            SH2O = set_swc_ne3(swc_v, i)
        elif site == 'US-Ne2':
            SH2O = set_swc_ne2(swc_v, i)
        elif site == 'US-Ne1':
            SH2O = set_swc_ne1(swc_v, i)
        elif site == 'US-Bo1':
            SH2O = set_swc_bo1(swc_v, i)
        elif site == 'US-Br1':
            SH2O = set_swc_br1(swc_v, i)
        elif site == 'US-Br3':
            SH2O = set_swc_br3(swc_v, i)
        SMC = SH2O

        # variables from full model
        LAI = lai_force[i]

        # pre defined variables
        # LAI = 2.1
        # SH2O = [0.3, 0.3, 0.3, 0.3]
        # SMC = [0.3, 0.3, 0.3, 0.3]
        CANLIQ = 0
        FWET = 0

        # print focring for diagnostics
        # print('forcing:')
        # print('WS:', WS, 'SFCTMP:', SFCTMP, 'RH:', RH, 'SFCPRS:', SFCPRS, 'SOLDN:', SOLDN, 'LWDN:', LWDN, 'PRECP:', PRECP, 'GH:', GH,
        #       'LAI:', LAI, 'SH2O:', SH2O)

        if mode == 'nostress':
            SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
                TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
                BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
                QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
                PRECP_out, FSUN, PARSUN, PARSHA = ENERGY_DRIVER_NOSTRESS(config, LAT, LON, TIME, DT, NSOIL, ZSOIL, DZSNSO, WS, SFCTMP,
                                                                         RH, SFCPRS, SOLDN, LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ,
                                                                         FWET)
            TR_P = FCTR
            FSTOMATA_l = [-9999.1]

        elif mode == 'phm':
            soil_hydro = 0
            SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
                TRAD, T2M, PSN, APAR, SSOIL, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
                BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
                QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
                PRECP_out, FSUN, PARSUN, PARSHA, \
                TV_l, FSTOMATA_l, SH2O_out, SMC_out, L12, L23, ROOTU, TR_P = ENERGY_DRIVER_PHM(config, soil_hydro, LAT, LON, TIME, DT,
                                                                                               NSOIL,
                                                                                               ZSOIL, DZSNSO, WS, SFCTMP, RH, SFCPRS,
                                                                                               SOLDN,
                                                                                               LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ,
                                                                                               FWET)

        else:
            assert mode == 'beta'
            SAV, SAG, FSA, FSR, TAUX, TAUY, FIRA, FSH, FCEV, FGEV, FCTR, \
                TRAD, T2M, PSN, APAR, SSOIL, BTRANI, BTRAN, LATHEA, FSRV, FSRG, RSSUN, RSSHA, CHSTAR, TSTAR, T2MV, T2MB, \
                BGAP, WGAP, GAP, EMISSI, Q2V, Q2B, Q2E, TGV, CHV, TGB, CHB, \
                QSFC, TS, TV, TG, EAH, TAH, CM, CH, Q1, \
                PRECP_out, FSUN, PARSUN, PARSHA, TR_P = ENERGY_DRIVER_BTRAN(config, LAT, LON, TIME, DT, NSOIL, ZSOIL, DZSNSO, WS, SFCTMP,
                                                                            RH,
                                                                            SFCPRS, SOLDN, LWDN, PRECP, LAI, SH2O, SMC, GH, CANLIQ, FWET,
                                                                            DYNAMIC_beta, CAP_beta)
            FSTOMATA_l = [-9999.1]

        # print output for diagnostics
        # print('WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP:', WS, SFCTMP, RH, SFCPRS, SOLDN, LWDN, PRECP)
        # print('SAV, SAG, FSA, FSR:', SAV, SAG, FSA, FSR)
        # print('TS, TV, TG, EAH, TAH:', TS, TV, TG, EAH, TAH)

        times.append(TIME)
        transpiration.append(FCTR)
        soil_evaporation.append(FGEV)
        canopy_evaporation.append(FCEV)
        rssun.append(RSSUN)
        rssha.append(RSSHA)
        psn.append(PSN)
        fsun.append(FSUN)
        fsh.append(FSH)
        sav.append(SAV)
        sag.append(SAG)
        fsa.append(FSA)
        fsr.append(FSR)
        fira.append(FIRA)
        apar.append(APAR)
        parsun.append(PARSUN)
        parsha.append(PARSHA)
        lai.append(LAI)
        tr_p.append(TR_P)
        fstomata.append(FSTOMATA_l[-1])

    # save results
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results = np.array(
        [times, transpiration, soil_evaporation, canopy_evaporation, rssun, rssha, psn, fsun, fsh, sav, sag, fsa, fsr, fira, apar,
         parsun, parsha, lai, tr_p])
    np.save(results_path + 'standalone_flux' + results_suffix, results)

    return


def SOIL_LAYER_CONFIG(config):
    DT = float(config['configuration']['DT'])
    NSOIL = int(config['configuration']['NSOIL'])
    ZSOIL = np.array([float(idx) for idx in config['configuration']['ZSOIL'].split(',')])
    DZSNSO = np.array([float(idx) for idx in config['configuration']['DZSNSO'].split(',')])

    return DT, NSOIL, ZSOIL, DZSNSO


def set_swc_ne3(swc_v, i):
    # -0.1, -0.25, -0.5, -1.0
    SH2O = [swc_v[0, i], swc_v[1, i], swc_v[2, i], swc_v[3, i]]
    return SH2O


def set_swc_ne2(swc_v, i):
    # -0.1, -0.25, -0.5, -1.0
    SH2O = [swc_v[0, i], swc_v[1, i], swc_v[2, i], swc_v[3, i]]
    return SH2O


def set_swc_ne1(swc_v, i):
    # -0.1, -0.25, -0.5, -1.0
    SH2O = [swc_v[0, i], swc_v[1, i], swc_v[2, i], swc_v[3, i]]
    return SH2O


def set_swc_bo1(swc_v, i):
    # -0.1, -0.2
    SH2O = [swc_v[0, i], swc_v[1, i], swc_v[1, i], swc_v[1, i]]
    return SH2O


def set_swc_br1(swc_v, i):
    # -0.05
    SH2O = [swc_v[0, i], swc_v[0, i], swc_v[0, i], swc_v[0, i]]
    return SH2O


def set_swc_br3(swc_v, i):
    # -0.05
    SH2O = [swc_v[0, i], swc_v[0, i], swc_v[0, i], swc_v[0, i]]
    return SH2O

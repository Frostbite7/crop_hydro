import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os

# set up constants
HVAP = 2.51e6


def main():
    # set up paths
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/numerical_test/'
    fig_name = 'DHLF_vs_PHM_var_psil_rev.pdf'
    save_fig = 0

    # set up fixed parameters
    SPAI = 15.8E-4
    VGA1 = 1
    # VGSP50 = -150
    # VGA2 = 3

    # ref values for variable parameters
    # VEGH = 1
    # VGKSAT = 1.5e-6
    # VGTLP = -100
    # VGA3 = 4

    # kmax_adjusted = 6.2
    # krmax = DKSAT * np.sqrt(RAI) / DZSNSO / np.pi * HVAP * 1000
    # kmax = VGKSAT * SPAI / VGA1 / VEGH * HVAP * 1000
    # kmax_adjusted = kmax * ((-VGTLP) / 100) ** (1.2 + (VEGH - 9) / 45) * (1 - (VEGH - 9) / 80)

    # set up variable parameters
    vars_l = [{'VEGH': 1, 'VGKSAT': 1.0e-6, 'VGTLP': -100, 'VGA3': 4, 'VGSP50': -150, 'VGA2': 3},
              {'VEGH': 1, 'VGKSAT': 1.5e-6, 'VGTLP': -100, 'VGA3': 4, 'VGSP50': -150, 'VGA2': 3},
              {'VEGH': 1, 'VGKSAT': 1.5e-6, 'VGTLP': -100, 'VGA3': 4, 'VGSP50': -150, 'VGA2': 3},
              {'VEGH': 1, 'VGKSAT': 1.5e-6, 'VGTLP': -100, 'VGA3': 4, 'VGSP50': -150, 'VGA2': 3}]
    kmax_phm_l = [var['VGKSAT'] * SPAI / VGA1 / var['VEGH'] * HVAP * 1000 for var in vars_l]
    kmax_adjusted_ref_l = [
        var['VGKSAT'] * SPAI / VGA1 / var['VEGH'] * HVAP * 1000 * ((-var['VGTLP']) / 100) ** (1.2 + (var['VEGH'] - 9) / 45) * (
                1 - (var['VEGH'] - 9) / 80) for var in vars_l]
    kmax_adjusted_range_l = [kmax_adjusted_ref * np.arange(0.8, 1.2, 0.04) for kmax_adjusted_ref in kmax_adjusted_ref_l]
    n_var_parms = len(vars_l)

    # set up variables
    TR_P_l = np.arange(0, 1000, 1)

    # begin calculation
    TR_PHM_l_l = []
    TR_beta_capped_l_l = []
    PSIL_l_l = []
    for i in range(n_var_parms):
        var = vars_l[i]
        VGKSAT = var['VGKSAT']
        VGTLP = var['VGTLP']
        VGA3 = var['VGA3']
        VEGH = var['VEGH']
        VGSP50 = var['VGSP50']
        VGA2 = var['VGA2']
        kmax_adjusted_range = kmax_adjusted_range_l[i]
        print('running for VGKSAT = {}, VGTLP = {}, VGA3 = {}, VEGH = {}'.format(VGKSAT, VGTLP, VGA3, VEGH))

        TR_beta_capped_l_range = []
        TR_PHM_l = []
        PSIL_l = []
        j = 0
        for kmax_adjusted in kmax_adjusted_range:
            TR_beta_capped_l = []
            for TR_P in TR_P_l:
                if j == 0:
                    TR_PHM, PSIL = PHM_ww_solve(TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3)
                    TR_PHM_l.append(TR_PHM)
                    PSIL_l.append(PSIL)

                TR_beta_capped = beta_capped_solve(kmax_adjusted, TR_P, 9, -100, 6)
                TR_beta_capped_l.append(TR_beta_capped)
            j += 1
            TR_beta_capped_l_range.append(np.array(TR_beta_capped_l))
        TR_PHM_l_l.append(np.array(TR_PHM_l))
        TR_beta_capped_l_l.append(np.array(TR_beta_capped_l_range))
        PSIL_l_l.append(np.array(PSIL_l))

    # find the best matching kmax_adjusted
    best_kmax_adjusted_idx_l = []
    best_kmax_adjusted_l = []
    for i in range(n_var_parms):
        kmax_adjusted_range = kmax_adjusted_range_l[i]

        TR_PHM_l = TR_PHM_l_l[i]
        TR_beta_capped_l_range = TR_beta_capped_l_l[i]

        TR_diff_l = np.abs(TR_PHM_l - TR_beta_capped_l_range)
        TR_diff_l_sum = np.sum(TR_diff_l, axis=1)
        idx = np.argmin(TR_diff_l_sum)
        kmax_adjusted_best = kmax_adjusted_range[idx]
        print('var:', vars_l[i], 'kmax_phm:', kmax_phm_l[i], 'best kmax_adjusted:', kmax_adjusted_best, 'idx:',
              kmax_adjusted_best / kmax_adjusted_ref_l[i])
        best_kmax_adjusted_idx_l.append(idx)
        best_kmax_adjusted_l.append(kmax_adjusted_best)

    # plot
    TR_range = [0, 1000]
    TR_diff_range = [-100, 100]
    PSIL_range = [-300, 0]

    fig, ax = plt.subplots(3, 4, figsize=(18, 12))
    for i in range(n_var_parms):
        var = vars_l[i]
        VGKSAT = var['VGKSAT']
        kmax_phm = kmax_phm_l[i]
        VGTLP = var['VGTLP']
        VGA3 = var['VGA3']
        VEGH = var['VEGH']
        VGSP50 = var['VGSP50']
        VGA2 = var['VGA2']
        best_kmax_adjusted = best_kmax_adjusted_l[i]

        TR_PHM_l = TR_PHM_l_l[i]
        TR_beta_capped_l = TR_beta_capped_l_l[i][best_kmax_adjusted_idx_l[i]]
        PSIL_l = PSIL_l_l[i]

        ax[0, i].plot(TR_P_l, TR_PHM_l, label='PHM', color='k', lw=2)
        ax[0, i].plot(TR_P_l, TR_beta_capped_l, label='DHLF', color='red', lw=2, ls='--')
        ax[0, i].set_xlabel('$T_{NHL}$ W/m2', fontsize=14)
        ax[0, i].set_ylabel('T W/m2', fontsize=14)
        ax[0, i].set_xlim(TR_range)
        ax[0, i].set_ylim(TR_range)
        ax[0, 1].tick_params(axis='both', which='major', labelsize=14)
        ax[0, i].text(0.03, 0.92, r'PHM', transform=ax[0, i].transAxes, fontsize=14)
        ax[0, i].text(0.03, 0.84, r'$K_x$ = ' + '{:.2f}'.format(kmax_phm), transform=ax[0, i].transAxes, fontsize=14)
        ax[0, i].text(0.03, 0.76, r'$\psi_{x,50}$ = ' + '{}'.format(VGSP50), transform=ax[0, i].transAxes, fontsize=14, c='k',
                      # bbox=dict(facecolor='grey', alpha=0.5)
                      )
        ax[0, i].text(0.03, 0.68, r'$a_1$ = ' + '{}'.format(VGA2), transform=ax[0, i].transAxes, fontsize=14, c='k',
                      # bbox=dict(facecolor='grey', alpha=0.5)
                      )
        ax[0, i].text(0.03, 0.60, r'$h$ = ' + '{}'.format(VEGH), transform=ax[0, i].transAxes, fontsize=14, c='k',
                      # bbox=dict(facecolor='grey', alpha=0.5)
                      )
        ax[0, i].text(0.03, 0.52, r'$\psi_{l,50}$ = ' + '{}'.format(VGTLP), transform=ax[0, i].transAxes, fontsize=14, c='k',
                      bbox=dict(facecolor='grey', alpha=0.5)
                      )
        ax[0, i].text(0.03, 0.44, r'$a_2$ = ' + '{}'.format(VGA3), transform=ax[0, i].transAxes, fontsize=14, c='k',
                      # bbox=dict(facecolor='grey', alpha=0.5)
                      )
        ax[0, i].text(0.64, 0.92, r'DHLF', transform=ax[0, i].transAxes, fontsize=14, c='r')
        ax[0, i].text(0.64, 0.84, r'$K_x$ = ' + '{:.2f}'.format(best_kmax_adjusted), transform=ax[0, i].transAxes, fontsize=14, c='r',
                      bbox=dict(facecolor='grey', alpha=0.5))
        if i == 0:
            ax[0, i].legend(fontsize=14, loc='lower right')

        ax[1, i].plot(TR_P_l, TR_beta_capped_l - TR_PHM_l)
        ax[1, i].set_xlabel('$T_{NHL}$ w/m2')
        ax[1, i].set_ylabel('T DHLF-PHM w/m2')
        ax[1, i].set_xlim(TR_range)
        ax[1, i].set_ylim(TR_diff_range)

        ax[2, i].plot(TR_P_l, PSIL_l)
        ax[2, i].set_xlabel('$T_{NHL}$ w/m2')
        ax[2, i].set_ylabel('$\\psi_l$ m')
        ax[2, i].set_xlim(TR_range)
        ax[2, i].set_ylim(PSIL_range)

    fig.tight_layout()

    if save_fig:
        if os.path.exists(fig_path) is False:
            os.makedirs(fig_path)
        fig.savefig(fig_path + fig_name)

    plt.show()

    return


def PHM_ww_solve(TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3):
    res = minimize_scalar(PHM_ww_r_l, args=(TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3))
    VGPSIL = res.x

    FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
    TR_solution = FSTOMATA * TR_P

    return TR_solution, VGPSIL


def PHM_ww_r_l(VGPSIL, TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3):
    VGPSIS = 0
    # KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)
    # KA = VGKSAT / (1 + ((VGPSIS + VGPSIL) / 2 / VGSP50) ** VGA2)
    KA = VGKSAT / (1 + ((VGPSIS + max(VGPSIL, 2 * VGTLP)) / 2 / VGSP50) ** VGA2)
    # KA = VGKSAT / (1 + ((VGPSIS + max(VGPSIL, -200)) / 2 / VGSP50) ** VGA2)
    KAC = max(KA * SPAI, 1e-20) / VEGH / VGA1
    RHS = KAC * (VGPSIS - VGPSIL - VEGH) * 1000 * HVAP

    FSTOMATA = 1. / (1. + (VGPSIL / VGTLP) ** VGA3)
    LHS = FSTOMATA * TR_P
    R_L = RHS - LHS

    result = R_L ** 2

    return result


def beta_capped_solve(kmax, TR_P, VEGH, VGTLP, VGA3):
    # optimize and result
    res = minimize_scalar(psil_cap_opt, args=(TR_P, VGTLP, VGA3, kmax, VEGH))
    VGPSIL = res.x
    TR_solution = 1. / (1. + (VGPSIL / VGTLP) ** VGA3) * TR_P

    return TR_solution


def psil_cap_opt(VGPSIL, TR_P, VGTLP, VGA3, kmax, VEGH):
    VGPSIS = 0
    td = 1. / (1. + (VGPSIL / VGTLP) ** VGA3) * TR_P
    ts = kmax * (VGPSIS - VGPSIL - VEGH)
    result = (td - ts) ** 2

    return result


if __name__ == '__main__':
    main()

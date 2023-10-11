import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os

# set up constants
HVAP = 2.51e6


def main():
    # set up paths
    fig_path = '/Users/yangyicge/Desktop/watercon/crop_hydro_case/fig/numerical_test/'
    fig_name = 'beta_capped_vs_PHM_var_kx.pdf'
    save_fig = 0

    # set up fixed parameters
    SPAI = 15.8E-4
    VEGH = 1
    # VGKSAT = 1.5e-6
    VGSP50 = -150
    VGA2 = 3
    VGA1 = 1
    VGTLP = -100
    VGA3 = 4

    # kmax_adjusted = 6.2
    # krmax = DKSAT * np.sqrt(RAI) / DZSNSO / np.pi * HVAP * 1000
    # kmax = VGKSAT * SPAI / VGA1 / VEGH * HVAP * 1000
    # kmax_adjusted = kmax * ((-VGTLP) / 100) ** (1.2 + (VEGH - 9) / 45) * (1 - (VEGH - 9) / 80)

    # set up variable parameters
    n_var_parms = 3
    # VGKSAT_l = [VGKSAT] * n_var_parms
    VGKSAT_l = [1.5e-6, 5e-6, 5e-7]
    VGTLP_l = [VGTLP] * n_var_parms
    # VGTLP_l = [-50, -100, -200]

    kmax_adjusted_l = [VGKSAT * SPAI / VGA1 / VEGH * HVAP * 1000 * ((-VGTLP) / 100) ** (1.2 + (VEGH - 9) / 45) * (
            1 - (VEGH - 9) / 80) for VGKSAT in VGKSAT_l]
    # kmax_adjusted_l = [VGKSAT * SPAI / VGA1 / VEGH * HVAP * 1000 * ((-VGTLP) / 100) ** (1.2 + (VEGH - 9) / 45) * (
    #         1 - (VEGH - 9) / 80) for VGTLP in VGTLP_l]

    # set up variables
    TR_P_l = np.arange(0, 800, 1)

    # begin calculation
    TR_PHM_l_l = []
    TR_beta_capped_l_l = []
    for i in range(n_var_parms):
        VGKSAT = VGKSAT_l[i]
        VGTLP = VGTLP_l[i]
        kmax_adjusted = kmax_adjusted_l[i]
        TR_PHM_l = []
        TR_beta_capped_l = []
        for TR_P in TR_P_l:
            TR_PHM = PHM_ww_solve(TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3)
            TR_PHM_l.append(TR_PHM)

            TR_beta_capped = beta_capped_solve(kmax_adjusted, TR_P, 9, -100, 6)
            TR_beta_capped_l.append(TR_beta_capped)
        TR_PHM_l_l.append(np.array(TR_PHM_l))
        TR_beta_capped_l_l.append(np.array(TR_beta_capped_l))

    # plot
    TR_range = [0, 800]
    TR_diff_range = [-100, 100]

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for i in range(n_var_parms):
        VGKSAT = VGKSAT_l[i]
        VGTLP = VGTLP_l[i]
        TR_PHM_l = TR_PHM_l_l[i]
        TR_beta_capped_l = TR_beta_capped_l_l[i]
        ax[0, i].plot(TR_P_l, TR_PHM_l, label='PHM')
        ax[0, i].plot(TR_P_l, TR_beta_capped_l, label='beta_capped')
        ax[0, i].set_xlabel('Tr demand w/m2')
        ax[0, i].set_ylabel('Tr w/m2')
        ax[0, i].set_xlim(TR_range)
        ax[0, i].set_ylim(TR_range)
        ax[0, i].legend()
        ax[0, i].set_title(r'$K_x$:{}, '.format(VGKSAT) + '$\psi_{l50}$:' + '{}, shape:{}'.format(VGTLP, VGA3))

        ax[1, i].plot(TR_P_l, TR_beta_capped_l - TR_PHM_l)
        ax[1, i].set_xlabel('Tr demand w/m2')
        ax[1, i].set_ylabel('Tr beta_cap-PHM w/m2')
        ax[1, i].set_xlim(TR_range)
        ax[1, i].set_ylim(TR_diff_range)
        # ax[0, 1].legend()

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

    return TR_solution


def PHM_ww_r_l(VGPSIL, TR_P, SPAI, VEGH, VGKSAT, VGSP50, VGA2, VGA1, VGTLP, VGA3):
    VGPSIS = 0
    KA = VGKSAT / (1 + (VGPSIS / VGSP50) ** VGA2)
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

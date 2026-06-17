import torch

from .constants import ev

_SQPI = 1.77245385090552
_FPI = 12.5663706143592
_A3 = 0.333333333333333
_A4 = 0.666666666666667
_ALIM = 0.317
_ABLIM = 0.1
_BIGEXP = 50.0
_TOL = 12 * 2.302585093


_FCTRL = (1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0)
_DFCTRL = (1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0)
_DFCTRL_FJ = (1.0, 3.0, 15.0, 105.0)


def _horner_piecewise_table(x, table, h):
    if x.numel() == 0:
        return x.clone()

    shape = x.shape
    xf = x.reshape(-1)

    interval = (xf / h).to(torch.long)
    interval.clamp_(0, table.shape[0] - 1)

    coeff = table.index_select(0, interval)

    val = coeff[:, 0]
    for k in range(1, table.shape[1]):
        val = coeff[:, k] + xf * val

    return val.reshape(shape)


def _stack2(d, n0, n1, z):
    out = z.new_zeros((*z.shape, n0, n1))
    for (i, j), v in d.items():
        if 0 <= i < n0 and 0 <= j < n1:
            out[..., i, j] = v
    return out


def _stack3(d, n0, n1, n2, z):
    out = z.new_zeros((*z.shape, n0, n1, n2))
    for (i, j, k), v in d.items():
        if 0 <= i < n0 and 0 <= j < n1 and 0 <= k < n2:
            out[..., i, j, k] = v
    return out


def _dawf_vec(y, ppecp):
    x = y.abs()

    if x.is_cuda:
        # GPU path: avoid mask.any() host syncs.
        xs = x.clamp_max(10.0)
        small_val = _horner_piecewise_table(xs, ppecp["dawf_table"], ppecp["dawf_h"])

        xl = x.clamp_min(10.0)
        txt = 0.5 / (xl * xl)
        large_val = (txt * xl) * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + 105.0 * txt))))

        out = torch.where(x < 10.0, small_val, large_val)
        return torch.where(y < 0.0, -out, out)

    # CPU path: avoid computing both branches.
    out = torch.empty_like(x)

    small = x < 10.0
    if small.any():
        out[small] = _horner_piecewise_table(x[small], ppecp["dawf_table"], ppecp["dawf_h"])

    large = ~small
    if large.any():
        xl = x[large]
        txt = 0.5 / (xl * xl)
        out[large] = (txt * xl) * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + 105.0 * txt))))

    return torch.where(y < 0.0, -out, out)


def _dawerf_vec(y, ppecp):
    x = y.abs()

    if x.is_cuda:
        # GPU path: avoid mask.any() host syncs.
        xs = x.clamp_max(10.0)
        small_val = _horner_piecewise_table(xs, ppecp["dawerf_table"], ppecp["dawerf_h"])

        xl = x.clamp_min(10.0)
        txt = 0.5 / (xl * xl)
        large_val = (txt * xl) * (
            1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + txt * (105.0 + 945.0 * txt))))
        )

        return torch.where(x < 10.0, small_val, large_val)

    # CPU path: avoid computing both branches.
    out = torch.empty_like(x)

    small = x < 10.0
    if small.any():
        out[small] = _horner_piecewise_table(x[small], ppecp["dawerf_table"], ppecp["dawerf_h"])

    large = ~small
    if large.any():
        xl = x[large]
        txt = 0.5 / (xl * xl)
        out[large] = (txt * xl) * (
            1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + txt * (105.0 + 945.0 * txt))))
        )

    return out


def _fsips_vec(n, l, alfa, xp0):
    nl = n + l

    if nl % 2 == 0:
        lam = nl // 2
        x = alfa * alfa
        l2 = 2 * l + 1

        term = torch.full_like(alfa, _DFCTRL[lam] / _DFCTRL[l + 1])
        total = term.clone()

        for k in range(1, 11):
            term = (term * x * (2 * k + nl - 1)) / (k * (2 * k + l2))
            total = total + term

        return total * _SQPI * xp0 * (2.0**lam) * (alfa**l)

    lam = (nl - 1) // 2
    x = 2.0 * alfa * alfa
    l2 = 2 * l + 1

    term = torch.full_like(alfa, _FCTRL[lam] / _DFCTRL[l + 1])
    total = term.clone()

    for k in range(1, 11):
        term = (term * x * (k + lam)) / (k * (2 * k + l2))
        total = total + term

    return total * xp0 * (2.0**nl) * (alfa**l)


def _fjps_vec(n, lalf, lbet, alf, bet, xi, si):
    out = torch.empty_like(alf)
    scale = (0.5 * xi) ** (n + 1)

    m = alf > bet
    if m.any():
        b = bet[m]
        si_m = si[m]

        x = b * b
        term = (b**lbet) / _DFCTRL_FJ[lbet]
        l1 = lalf + 1
        l2 = lbet + n + 1
        l3 = 2 * lbet + 1

        total = term * si_m[..., l2, l1]
        for k in range(1, 11):
            term = term * x / (2.0 * k * (2.0 * k + l3))
            total = total + term * si_m[..., 2 * k + l2, l1]

        out[m] = total * scale[m]

    m = ~m
    if m.any():
        a = alf[m]
        si_m = si[m]

        x = a * a
        term = (a**lalf) / _DFCTRL_FJ[lalf]
        l1 = lbet + 1
        l2 = lalf + n + 1
        l3 = 2 * lalf + 1

        total = term * si_m[..., l2, l1]
        for k in range(1, 11):
            term = term * x / (2.0 * k * (2.0 * k + l3))
            total = total + term * si_m[..., 2 * k + l2, l1]

        out[m] = total * scale[m]

    return out


def _fsi0_vec(n, alfa, xp0, xp1, ppecp):
    out = torch.empty_like(alfa)

    small = alfa <= _ALIM
    if small.any():
        out[small] = _fsips_vec(n, 0, alfa[small], xp0[small])

    large = ~small
    if large.any():
        a = alfa[large]
        if n != 1:
            dawfs = _dawf_vec(a, ppecp)
            out[large] = _SQPI * dawfs * xp1[large] / a
        else:
            errfs = _SQPI * torch.erf(a) * xp1[large]
            out[large] = errfs / a

    return out


def _fsi1_vec(n, alfa, xp0, xp1, ppecp):
    out = torch.empty_like(alfa)

    small = alfa <= _ALIM
    if small.any():
        out[small] = _fsips_vec(n, 1, alfa[small], xp0[small])

    large = ~small
    if large.any():
        a = alfa[large]

        if n <= 0:
            errfs = _SQPI * torch.erf(a) * xp1[large]
            out[large] = (0.5 * errfs / a - xp0[large]) / a
        elif n == 1:
            dawfs = _dawf_vec(a, ppecp)
            out[large] = _SQPI * (a - dawfs) * xp1[large] / (a * a)
        else:
            a2 = a * a
            errfs = _SQPI * torch.erf(a) * xp1[large]
            out[large] = (2.0 * a * xp0[large] + (2.0 * a2 - 1.0) * errfs) / a2

    return out


def _fsi2_vec(n, alfa, xp0, xp1, ppecp):
    out = torch.empty_like(alfa)

    small = alfa <= _ALIM
    if small.any():
        out[small] = _fsips_vec(n, 2, alfa[small], xp0[small])

    large = ~small
    if large.any():
        a = alfa[large]
        a2 = a * a
        dawfs = _dawf_vec(a, ppecp)
        errfs = _SQPI * torch.erf(a) * xp1[large]

        if n == 0:
            out[large] = 0.25 * _SQPI * (3.0 * a - (2.0 * a2 + 3.0) * dawfs) * xp1[large] / (a2 * a)
        elif n == 1:
            out[large] = (0.5 * (2.0 * a2 - 3.0) * errfs + 3.0 * a * xp0[large]) / (a2 * a)
        elif n == 2:
            out[large] = _SQPI * (a * (2.0 * a2 - 3.0) + 3.0 * dawfs) * xp1[large] / (a2 * a)
        elif n == 3:
            out[large] = ((a2 * (4.0 * a2 - 4.0) + 3.0) * errfs + 2.0 * a * (2.0 * a2 - 3.0) * xp0[large]) / (
                a2 * a
            )
        else:
            raise NotImplementedError("FSI2 current OM1 PPECP scope only needs N<=3")

    return out


def _get2(d, i, j, z):
    return d.get((i, j), z)


def _get3(d, i, j, k, z):
    return d.get((i, j, k), z)


def _recur_si_cache(si, nmin, nmax, lmax, x, z):
    tx = 2.0 * x
    for n in range(nmin, nmax + 1, 2):
        tfnm2 = 2.0 * n - 4.0
        si[(n + 1, 1)] = (tfnm2 + 2.0) * _get2(si, n - 1, 1, z) + tx * _get2(si, n, 2, z)
        si[(n + 2, 2)] = tfnm2 * _get2(si, n, 2, z) + tx * si[(n + 1, 1)]
        if lmax >= 2:
            si[(n + 3, 3)] = tfnm2 * _get2(si, n + 1, 3, z) + tx * si[(n + 2, 2)]


def _sitabl_vec(lemax, lomax, alfj, betj, xpls, xmns, xp, ppecp):
    alfj, betj, xpls, xmns, xp = torch.broadcast_tensors(alfj, betj, xpls, xmns, xp)

    m = alfj <= betj
    alfa = torch.where(m, betj, alfj)
    xp1 = torch.where(m, xmns, xpls)

    x = alfa
    xp0 = xp
    z = torch.zeros_like(alfa)
    si = {}

    if lemax >= 0:
        si[(1, 1)] = _fsi0_vec(0, alfa, xp0, xp1, ppecp)
        si[(2, 2)] = _fsi1_vec(1, alfa, xp0, xp1, ppecp)
        if lemax >= 2:
            si[(3, 3)] = _fsi2_vec(2, alfa, xp0, xp1, ppecp)
        _recur_si_cache(si, 2, 22, lemax, x, z)

    if lomax >= 0:
        si[(1, 2)] = _fsi1_vec(0, alfa, xp0, xp1, ppecp)
        si[(2, 1)] = _fsi0_vec(1, alfa, xp0, xp1, ppecp)
        si[(3, 2)] = _fsi1_vec(2, alfa, xp0, xp1, ppecp)
        _recur_si_cache(si, 3, 21, lomax, x, z)

    return _stack2(si, 27, 5, z)


def _exp_neg_cutoff(x, cutoff=_BIGEXP):
    m = x < cutoff

    if x.is_cuda:
        # Avoid GPU sync from `if m.any()`.
        safe_x = torch.where(m, x, torch.zeros_like(x))
        return torch.where(m, torch.exp(-safe_x), torch.zeros_like(x))

    # CPU path: avoid computing exp on inactive values.
    out = torch.zeros_like(x)
    if m.any():
        out[m] = torch.exp(-x[m])
    return out


def _fiprep_vec(nlpk, nij, zetc, zetb, r, zlp, clp, ppecp):
    zetc, zetb, r, zlp, clp = torch.broadcast_tensors(zetc, zetb, r, zlp, clp)
    lead_shape = zetc.shape

    zetc_f = zetc.reshape(-1)
    zetb_f = zetb.reshape(-1)
    r_f = r.reshape(-1)
    zlp_f = zlp.reshape(-1)
    clp_f = clp.reshape(-1)

    nitem = zetc_f.numel()
    full_zero = zetc_f.new_zeros((nitem, nij + 1, nij + 1))

    zetcb = zetc_f + zetb_f
    alfa_all = zetcb * r_f
    xalfa = alfa_all * r_f

    xp0_all = _exp_neg_cutoff(xalfa)

    xi_all = 1.0 / torch.sqrt(zetcb + zlp_f)
    alf_all = alfa_all * xi_all
    dum = xalfa - alf_all * alf_all

    active = dum <= _TOL
    if not active.any():
        return full_zero.reshape((*lead_shape, nij + 1, nij + 1))

    idx = active.nonzero(as_tuple=False).squeeze(1)

    alfa = alfa_all[idx]
    alfi = 0.5 / alfa
    xp0 = xp0_all[idx]
    xi = xi_all[idx]
    alf = alf_all[idx]
    xp1 = torch.exp(-dum[idx])
    clp_a = clp_f[idx]

    z = torch.zeros_like(alfa)
    fit = {}

    nmx = nij - 1 + nlpk
    yi = 0.5 * xi
    a1 = xi * alf
    a2 = xi * yi
    x = alf * alf

    if nlpk % 2 == 0:
        f11 = torch.empty_like(alf)
        f22 = torch.empty_like(alf)

        small = alf <= _ALIM
        if small.any():
            alf_s = alf[small]
            x_s = x[small]
            xp0_s = xp0[small]

            t11 = torch.ones_like(alf_s)
            t22 = torch.full_like(alf_s, 1.0 / 3.0)
            s11 = t11.clone()
            s22 = t22.clone()

            for k in range(1, 11):
                t11 = (t11 * x_s * (2 * k - 1)) / (k * (2 * k + 1))
                t22 = (t22 * x_s * (2 * k + 1)) / (k * (2 * k + 3))
                s11 = s11 + t11
                s22 = s22 + t22

            f11[small] = s11 * _SQPI * xp0_s
            f22[small] = s22 * _SQPI * xp0_s * 2.0 * alf_s

        large = ~small
        if large.any():
            alf_l = alf[large]
            dawfs = _dawf_vec(alf_l, ppecp)
            f11[large] = _SQPI * dawfs * xp1[large] / alf_l
            f22[large] = _SQPI * (alf_l - dawfs) * xp1[large] / x[large]

        fit[(1, 1)] = f11 * yi

        if nmx >= 1:
            fit[(2, 2)] = f22 * yi * yi
            for n in range(2, nmx + 1, 2):
                fit[(n + 1, 1)] = a1 * _get2(fit, n, 2, z) + (n - 1) * a2 * _get2(fit, n - 1, 1, z)
                fit[(n + 2, 2)] = a1 * fit[(n + 1, 1)] + (n - 2) * a2 * _get2(fit, n, 2, z)

    else:
        f12 = torch.empty_like(alf)
        f21 = torch.empty_like(alf)

        small = alf <= _ALIM
        if small.any():
            alf_s = alf[small]
            xx = 2.0 * x[small]
            xp0_s = xp0[small]

            t12 = torch.full_like(alf_s, 1.0 / 3.0)
            t21 = torch.ones_like(alf_s)
            s12 = t12.clone()
            s21 = t21.clone()

            for k in range(1, 11):
                t12 = t12 * xx / (2 * k + 3)
                t21 = t21 * xx / (2 * k + 1)
                s12 = s12 + t12
                s21 = s21 + t21

            f12[small] = s12 * xp0_s * 2.0 * alf_s
            f21[small] = s21 * xp0_s * 2.0

        large = ~small
        if large.any():
            alf_l = alf[large]
            errfs = _SQPI * torch.erf(alf_l) * xp1[large]
            f21[large] = errfs / alf_l
            f12[large] = (0.5 * errfs / alf_l - xp0[large]) / alf_l

        fit[(1, 2)] = f12 * yi
        fit[(2, 1)] = f21 * yi * yi

        for n in range(2, nmx + 1, 2):
            fit[(n + 1, 2)] = a1 * _get2(fit, n, 1, z) + (n - 3) * a2 * _get2(fit, n - 1, 2, z)
            fit[(n + 2, 1)] = a1 * fit[(n + 1, 2)] + n * a2 * _get2(fit, n, 1, z)

    rr = {}

    for n in range(1, nij + 1, 2):
        rr[(n, 1)] = _get2(fit, n + nlpk, 1, z) * clp_a

    if nij > 1:
        for n in range(2, nij + 1, 2):
            rr[(n, 2)] = _get2(fit, n + nlpk, 2, z) * clp_a

    if nij > 2:
        for lp in range(3, nij + 1):
            lp1 = lp - 1
            lp2 = lp - 2
            fp = (lp1 + lp2) * alfi
            for n in range(lp, nij + 1, 2):
                rr[(n, lp)] = _get2(rr, n, lp2, z) - fp * _get2(rr, n - 1, lp1, z)

    rr_a = _stack2(rr, nij + 1, nij + 1, z)
    rr_full = full_zero.index_copy(0, idx, rr_a)
    return rr_full.reshape((*lead_shape, nij + 1, nij + 1))


def _fjprep_vec(nlpk, nij, zetc, zetb, r, zlp, clp, ppecp):
    zetc, zetb, r, zlp, clp = torch.broadcast_tensors(zetc, zetb, r, zlp, clp)
    lead_shape = zetc.shape

    shell_type = 1 if nij == 3 else 0
    lijc = shell_type + 1
    lijb = shell_type + 1

    zetc_f = zetc.reshape(-1)
    zetb_f = zetb.reshape(-1)
    r_f = r.reshape(-1)
    zlp_f = zlp.reshape(-1)
    clp_f = clp.reshape(-1)

    nitem = zetc_f.numel()
    full_zero = zetc_f.new_zeros((nitem, nij + 1, lijc + 1, lijb + 1))

    alfa_all = zetc_f * r_f
    beta_all = zetb_f * r_f
    zetcb = zetc_f + zetb_f

    alfbet_all = (alfa_all + beta_all) * r_f

    xpb_all = _exp_neg_cutoff(alfbet_all)

    zeta = zetcb + zlp_f
    dumtol = zlp_f * (((alfa_all + beta_all) ** 2) / zetcb) / zeta
    active1 = dumtol <= _TOL

    if not active1.any():
        return full_zero.reshape((*lead_shape, nij + 1, lijc + 1, lijb + 1))

    idx1 = active1.nonzero(as_tuple=False).squeeze(1)

    alfa1 = alfa_all[idx1]
    beta1 = beta_all[idx1]
    zetcb1 = zetcb[idx1]
    zlp1 = zlp_f[idx1]
    clp1 = clp_f[idx1]
    alfbet1 = alfbet_all[idx1]
    xpb1 = xpb_all[idx1]

    xi1 = torch.rsqrt(zetcb1 + zlp1)
    alef1 = alfa1 * xi1
    beit1 = beta1 * xi1
    ab1 = alef1 * beit1

    dum1 = torch.empty_like(ab1)
    dum2 = torch.empty_like(ab1)

    large_ab = ab1 > _ABLIM

    ap = alef1 + beit1
    am = alef1 - beit1

    dum1_large = alfbet1 - ap * ap
    dum2_large = alfbet1 - am * am

    dum1_small = alfbet1 - alef1 * alef1
    dum2_small = alfbet1 - beit1 * beit1

    dum1 = torch.where(large_ab, dum1_large, dum1_small)
    dum2 = torch.where(large_ab, dum2_large, dum2_small)

    active2 = (dum1 < _BIGEXP) | (dum2 < _BIGEXP)
    if not active2.any():
        return full_zero.reshape((*lead_shape, nij + 1, lijc + 1, lijb + 1))

    idx = idx1[active2]

    clp_a = clp1[active2]
    xi = xi1[active2]
    alef = alef1[active2]
    beit = beit1[active2]
    xpb = xpb1[active2]
    dum1 = dum1[active2]
    dum2 = dum2[active2]

    xpls = _exp_neg_cutoff(dum1)
    xmns = _exp_neg_cutoff(dum2)

    nmx = nij - 1 + nlpk
    xa = xi * alef
    xb = xi * beit
    xx = 0.5 * xi * xi

    z = torch.zeros_like(alef)
    f = {}

    if nlpk % 2 == 0:
        f111, f122, f221, f212 = _fj_base_even_vec(nmx, alef, beit, xi, xpls, xmns, xpb, ppecp)

        f[(1, 1, 1)] = f111
        f[(1, 2, 2)] = f122

        if nmx >= 1:
            f[(2, 2, 1)] = f221
            f[(2, 1, 2)] = f212

            for n in range(2, nmx + 1, 2):
                np1 = n + 1

                f[(np1, 1, 1)] = (
                    xa * _get3(f, n, 2, 1, z)
                    + xb * _get3(f, n, 1, 2, z)
                    + (n - 1) * xx * _get3(f, n - 1, 1, 1, z)
                )

                f[(np1, 2, 2)] = (
                    xa * _get3(f, n, 1, 2, z)
                    + xb * _get3(f, n, 2, 1, z)
                    + (n - 5) * xx * _get3(f, n - 1, 2, 2, z)
                )

                f[(n + 2, 2, 1)] = (
                    xa * f[(np1, 1, 1)] + xb * f[(np1, 2, 2)] + (n - 2) * xx * _get3(f, n, 2, 1, z)
                )

                f[(n + 2, 1, 2)] = (
                    xb * f[(np1, 1, 1)] + xa * f[(np1, 2, 2)] + (n - 2) * xx * _get3(f, n, 1, 2, z)
                )

    else:
        f121, f112, f211, f222 = _fj_base_odd_vec(nmx, alef, beit, xi, xpls, xmns, xpb, ppecp)

        f[(1, 2, 1)] = f121
        f[(1, 1, 2)] = f112
        f[(2, 1, 1)] = f211
        f[(2, 2, 2)] = f222

        for n in range(2, nmx + 1, 2):
            np1 = n + 1

            f[(np1, 2, 1)] = (
                xa * _get3(f, n, 1, 1, z)
                + xb * _get3(f, n, 2, 2, z)
                + (n - 3) * xx * _get3(f, n - 1, 2, 1, z)
            )

            f[(np1, 1, 2)] = (
                xb * _get3(f, n, 1, 1, z)
                + xa * _get3(f, n, 2, 2, z)
                + (n - 3) * xx * _get3(f, n - 1, 1, 2, z)
            )

            f[(n + 2, 1, 1)] = xa * f[(np1, 2, 1)] + xb * f[(np1, 1, 2)] + n * xx * _get3(f, n, 1, 1, z)

            f[(n + 2, 2, 2)] = xa * f[(np1, 1, 2)] + xb * f[(np1, 2, 1)] + (n - 4) * xx * _get3(f, n, 2, 2, z)

    rr = {}

    for n in range(1, nij + 1, 2):
        rr[(n, 1, 1)] = _get3(f, n + nlpk, 1, 1, z) * clp_a

    if lijc > 1 and lijb > 1:
        for n in range(1, nij + 1, 2):
            rr[(n, 2, 2)] = _get3(f, n + nlpk, 2, 2, z) * clp_a

    if lijb > 1:
        for n in range(2, nij + 1, 2):
            rr[(n, 1, 2)] = _get3(f, n + nlpk, 1, 2, z) * clp_a

    if lijc > 1:
        for n in range(2, nij + 1, 2):
            rr[(n, 2, 1)] = _get3(f, n + nlpk, 2, 1, z) * clp_a

    rr_a = _stack3(rr, nij + 1, lijc + 1, lijb + 1, z)
    rr_full = full_zero.index_copy(0, idx, rr_a)
    return rr_full.reshape((*lead_shape, nij + 1, lijc + 1, lijb + 1))


def _fj_base_even_vec(nmx, a, b, xi, xpls, xmns, xp, ppecp):
    """
    Computes the even-nlpk base FJ values in one fused pass.

    Returns:
        f111 = fj00(n=0)
        f122 = fj11(n=0)
        f221 = fj10(n=1), or None if nmx < 1
        f212 = fj01(n=1), or None if nmx < 1
    """
    need_extra = nmx >= 1

    f111 = a.new_zeros(a.shape)
    f122 = a.new_zeros(a.shape)
    f221 = a.new_zeros(a.shape) if need_extra else None
    f212 = a.new_zeros(a.shape) if need_extra else None

    small = a * b <= _ABLIM

    if small.any():
        idx = small.nonzero(as_tuple=False).squeeze(1)

        aa = a[idx]
        bb = b[idx]
        xx = xi[idx]
        xp_p = xpls[idx]
        xp_m = xmns[idx]
        xp0 = xp[idx]

        si = _sitabl_vec(2, -1, aa, bb, xp_p, xp_m, xp0, ppecp)

        f111 = f111.index_copy(0, idx, _fjps_vec(0, 0, 0, aa, bb, xx, si))
        f122 = f122.index_copy(0, idx, _fjps_vec(0, 1, 1, aa, bb, xx, si))

        if need_extra:
            f221 = f221.index_copy(0, idx, _fjps_vec(1, 1, 0, aa, bb, xx, si))
            f212 = f212.index_copy(0, idx, _fjps_vec(1, 0, 1, aa, bb, xx, si))

    large = ~small

    if large.any():
        idx = large.nonzero(as_tuple=False).squeeze(1)

        aa = a[idx]
        bb = b[idx]
        xx = xi[idx]
        xp_p = xpls[idx]
        xp_m = xmns[idx]

        ap = aa + bb
        am = aa - bb

        tab = 2.0 * aa * bb
        est = 0.5 * (xp_p - xp_m)
        ect = 0.5 * (xp_p + xp_m)

        fm0 = est / tab
        fm1 = (-fm0 + ect) / tab

        dawfp = _dawf_vec(ap, ppecp)
        dawfm = _dawf_vec(am, ppecp)

        tp = xp_p * dawfp
        tm = xp_m * dawfm

        dp = tp + tm
        dm = tp - tm

        aa2 = aa * aa
        bb2 = bb * bb

        v111 = _SQPI * xx * (aa * dm + bb * dp - tab * fm0) / (2.0 * tab)

        v122 = (
            _SQPI * xx * (2.0 * (aa2 + bb2) * fm0 - tab * fm1 - bb2 * dp / aa - aa2 * dm / bb) / (6.0 * tab)
        )

        f111 = f111.index_copy(0, idx, v111)
        f122 = f122.index_copy(0, idx, v122)

        if need_extra:
            xx2 = xx * xx

            v221 = _SQPI * xx2 * (2.0 * fm0 - dp / aa) / (8.0 * aa)
            v212 = _SQPI * xx2 * (2.0 * fm0 - dm / bb) / (8.0 * bb)

            f221 = f221.index_copy(0, idx, v221)
            f212 = f212.index_copy(0, idx, v212)

    return f111, f122, f221, f212


def _fj_base_odd_vec(nmx, a, b, xi, xpls, xmns, xp, ppecp):
    """
    Computes the odd-nlpk base FJ values in one fused pass.

    Returns:
        f121 = fj10(n=0)
        f112 = fj01(n=0)
        f211 = fj00(n=1)
        f222 = fj11(n=1)
    """
    f121 = a.new_zeros(a.shape)
    f112 = a.new_zeros(a.shape)
    f211 = a.new_zeros(a.shape)
    f222 = a.new_zeros(a.shape)

    small = a * b <= _ABLIM

    if small.any():
        idx = small.nonzero(as_tuple=False).squeeze(1)

        aa = a[idx]
        bb = b[idx]
        xx = xi[idx]
        xp_p = xpls[idx]
        xp_m = xmns[idx]
        xp0 = xp[idx]

        si = _sitabl_vec(-1, 2, aa, bb, xp_p, xp_m, xp0, ppecp)

        f121 = f121.index_copy(0, idx, _fjps_vec(0, 1, 0, aa, bb, xx, si))
        f112 = f112.index_copy(0, idx, _fjps_vec(0, 0, 1, aa, bb, xx, si))
        f211 = f211.index_copy(0, idx, _fjps_vec(1, 0, 0, aa, bb, xx, si))
        f222 = f222.index_copy(0, idx, _fjps_vec(1, 1, 1, aa, bb, xx, si))

    large = ~small

    if large.any():
        idx = large.nonzero(as_tuple=False).squeeze(1)

        aa = a[idx]
        bb = b[idx]
        xx = xi[idx]
        xp_p = xpls[idx]
        xp_m = xmns[idx]
        xp0 = xp[idx]

        ap = aa + bb
        am = aa - bb
        apam = ap * am

        tab = 2.0 * aa * bb

        dawfp = _dawf_vec(ap, ppecp)
        dawfm = _dawf_vec(am, ppecp)

        tp_daw = xp_p * dawfp
        tm_daw = xp_m * dawfm
        dm_daw = tp_daw - tm_daw

        tp_erf = xp_p * torch.erf(ap)
        tm_erf = xp_m * torch.erf(am)

        ep = tp_erf + tm_erf
        em = tp_erf - tm_erf

        hm = xp_p * _dawerf_vec(ap, ppecp) - xp_m * _dawerf_vec(am, ppecp)

        aa2 = aa * aa
        bb2 = bb * bb
        xx2 = xx * xx

        v121 = xx * (
            _SQPI * ((1.0 + 2.0 * apam) * hm + bb * ep - aa * em) / (8.0 * aa * tab) - xp0 / (4.0 * aa)
        )

        v112 = xx * (
            _SQPI * ((1.0 - 2.0 * apam) * hm - bb * ep + aa * em) / (8.0 * bb * tab) - xp0 / (4.0 * bb)
        )

        v211 = _SQPI * xx2 * dm_daw / (4.0 * tab)

        v222 = xx2 * (
            _SQPI * (aa * em + bb * ep - (1.0 + 2.0 * (aa2 + bb2)) * hm) / (8.0 * tab) - xp0 / (4.0 * tab)
        )

        f121 = f121.index_copy(0, idx, v121)
        f112 = f112.index_copy(0, idx, v112)
        f211 = f211.index_copy(0, idx, v211)
        f222 = f222.index_copy(0, idx, v222)

    return f121, f112, f211, f222


def _primitive_g1_vec(kd, acz, zfn, zetc, zetb, r, zlp, clp, ppecp):
    """
    Batched primitive_g1.

    zetc, zetb: [M, 6]
    acz, zfn, r: [M, 1] or broadcastable
    zlp, clp: [M, 3], columns are local, corr1, corr2

    Returns:
        g1 [M, 6, 6]
    """
    nij = 1 if kd == 1 else 3

    acz, zfn, zetc, zetb, r = torch.broadcast_tensors(acz, zfn, zetc, zetb, r)

    local_zlp = zlp[:, 0:1]
    corr1_zlp = zlp[:, 1:2]
    corr2_zlp = zlp[:, 2:3]

    local_clp = clp[:, 0:1]
    corr1_clp = clp[:, 1:2]
    corr2_clp = clp[:, 2:3]

    r1 = _fiprep_vec(1, nij, zetc, zetb, r, local_zlp, local_clp, ppecp)

    r2_corr1 = _fjprep_vec(0, nij, zetc, zetb, r, corr1_zlp, corr1_clp, ppecp)
    r2_corr2 = _fjprep_vec(2, nij, zetc, zetb, r, corr2_zlp, corr2_clp, ppecp)
    r2 = r2_corr1 + r2_corr2

    g1 = torch.zeros((*zetc.shape, 6), dtype=zetc.dtype, device=zetc.device)

    g1[..., 1] = (r1[..., 1, 1] + r2[..., 1, 1, 1]) * _FPI

    if kd > 1:
        g1[..., 2] = (
            acz * r1[..., 1, 1] + zfn * r1[..., 2, 2] + acz * r2[..., 1, 1, 1] + zfn * r2[..., 2, 1, 2]
        ) * _FPI

        g1[..., 3] = (
            acz * r1[..., 1, 1] + zfn * r1[..., 2, 2] + acz * r2[..., 1, 1, 1] + zfn * r2[..., 2, 2, 1]
        ) * _FPI

        acb1 = acz * acz
        acb5 = 2.0 * acz * zfn
        aczz = acz * zfn

        g1[..., 4] = (
            acb1 * r1[..., 1, 1]
            + acb5 * r1[..., 2, 2]
            + _A3 * r1[..., 3, 1]
            + _A4 * r1[..., 3, 3]
            + acb1 * r2[..., 1, 1, 1]
            + aczz * (r2[..., 2, 2, 1] + r2[..., 2, 1, 2])
            + r2[..., 3, 2, 2]
        ) * _FPI

        g1[..., 5] = (_A3 * (r1[..., 3, 1] - r1[..., 3, 3])) * _FPI

    return g1


def om1_ppecp_local_vectorized(ni, nj, rij, basis_tables, basis_i, basis_j):
    """
    Vectorized OM1 PPECP local block.

    Inputs:
        ni, nj:    [npair] atomic numbers
        rij:       [npair] pair distance in OM1 units
        basis:     basis table from build_omx_basis_tables, with basis["ppecp"]

    Returns:
        corpp [npair, 4, 2] in eV
    """
    device = rij.device
    dtype = rij.dtype

    ppecp = basis_tables["ppecp"]

    npair = ni.numel()
    if npair == 0:
        return torch.zeros((0, 4, 2), dtype=dtype, device=device)

    corpp = torch.zeros((npair, 4, 2), dtype=dtype, device=device)

    # Pair-column representation.
    # column 0: shell center i, ECP center j, acz=+rij, zfn=-1
    # column 1: shell center j, ECP center i, acz=-rij, zfn=+1
    shell_z = torch.stack((ni, nj), dim=1)
    ecp_z = torch.stack((nj, ni), dim=1)
    acz = torch.stack((rij, -rij), dim=1)
    zfn = rij.new_empty((npair, 2))
    zfn[:, 0] = -1.0
    zfn[:, 1] = 1.0
    shell_z_f = shell_z.reshape(-1)
    ecp_z_f = ecp_z.reshape(-1)
    acz_f = acz.reshape(-1)
    zfn_f = zfn.reshape(-1)
    r_f = rij.repeat_interleave(2)

    # H has no ECP. Padding should also be inactive if Z<=0.
    active = (shell_z_f > 0) & (ecp_z_f > 2)

    if not active.any():
        return corpp

    ecp_z_a = ecp_z_f[active]
    acz_a = acz_f[active]
    zfn_a = zfn_f[active]
    r_a = r_f[active]
    active_flat_col_idx = active.nonzero(as_tuple=False).squeeze(1)

    supported = ppecp["ecp_supported"]
    ecp_zlp_table = ppecp["ecp_zlp"]
    ecp_clp_table = ppecp["ecp_clp"]

    max_ecp_z = supported.numel() - 1

    ok_range = ecp_z_a <= max_ecp_z
    ok = torch.zeros_like(ok_range)
    if ok_range.any():
        ok[ok_range] = supported[ecp_z_a[ok_range]]

    if not ok.all():
        bad = torch.unique(ecp_z_a[~ok]).detach().cpu().tolist()
        raise NotImplementedError(
            f"OM1 PPECP is currently restricted to first-row ECP atoms C/N/O/F; got ECP center Z={bad}"
        )

    zlp_a = ecp_zlp_table[ecp_z_a]
    clp_a = ecp_clp_table[ecp_z_a]

    shell_type = torch.stack((basis_i["shell_type"], basis_j["shell_type"]), dim=1).reshape(-1)[active]
    exponents = torch.stack((basis_i["exponents"], basis_j["exponents"]), dim=1).reshape(-1, 3)[active]
    coeff_s = torch.stack((basis_i["coeff_s"], basis_j["coeff_s"]), dim=1).reshape(-1, 3)[active]
    coeff_p = torch.stack((basis_i["coeff_p"], basis_j["coeff_p"]), dim=1).reshape(-1, 3)[active]

    # shell_type = shell_type.to(torch.long)
    # exponents = exponents.to(dtype)
    # coeff_s = coeff_s.to(dtype)
    # coeff_p = coeff_p.to(dtype)

    tri_i = ppecp["tri_i"]
    tri_j = ppecp["tri_j"]
    sym = ppecp["tri_sym"]
    offdiag = ppecp["tri_offdiag"]

    zetc_all = exponents[:, tri_i]
    zetb_all = exponents[:, tri_j]

    cs_i_all = coeff_s[:, tri_i]
    cs_j_all = coeff_s[:, tri_j]
    cp_i_all = coeff_p[:, tri_i]
    cp_j_all = coeff_p[:, tri_j]

    corpp_active = torch.zeros((shell_type.numel(), 4), dtype=dtype, device=device)

    def _run_group(mask, kd):
        idxg = mask.nonzero(as_tuple=False).squeeze(1)
        if idxg.numel() == 0:
            return

        zetc = zetc_all.index_select(0, idxg)
        zetb = zetb_all.index_select(0, idxg)

        g1 = _primitive_g1_vec(
            kd=kd,
            acz=acz_a.index_select(0, idxg).unsqueeze(1),
            zfn=zfn_a.index_select(0, idxg).unsqueeze(1),
            zetc=zetc,
            zetb=zetb,
            r=r_a.index_select(0, idxg).unsqueeze(1),
            zlp=zlp_a.index_select(0, idxg),
            clp=clp_a.index_select(0, idxg),
            ppecp=ppecp,
        )

        cs_i = cs_i_all.index_select(0, idxg)
        cs_j = cs_j_all.index_select(0, idxg)
        cp_i = cp_i_all.index_select(0, idxg)
        cp_j = cp_j_all.index_select(0, idxg)

        out = torch.zeros((idxg.numel(), 4), dtype=dtype, device=device)

        out[:, 0] = (sym * cs_i * cs_j * g1[..., 1]).sum(dim=1)

        if kd > 1:
            out[:, 1] = (cs_i * cp_j * g1[..., 2] + offdiag * cs_j * cp_i * g1[..., 3]).sum(dim=1)

            pp = sym * cp_i * cp_j
            out[:, 2] = (pp * g1[..., 4]).sum(dim=1)
            out[:, 3] = (pp * g1[..., 5]).sum(dim=1)

        corpp_active.index_copy_(0, idxg, out)

    # H shell: s only, kd=1.
    _run_group(shell_type == 0, kd=1)

    # C/N/O/F shell: sp shell, kd=4.
    _run_group(shell_type == 1, kd=4)

    # bad_shell = (shell_type != 0) & (shell_type != 1)
    # if bad_shell.any():
    #     bad = torch.unique(shell_z_a[bad_shell]).detach().cpu().tolist()
    #     raise NotImplementedError(f"Unsupported shell types for atoms {bad}")

    # Scatter active pair-columns back to [npair, 4, 2].
    pair_idx = active_flat_col_idx // 2
    col_idx = active_flat_col_idx % 2

    corpp[pair_idx, :, col_idx] = corpp_active

    return corpp * ev


def om1_ppecp_local(ni, nj, rij, basis_tables, basis_i, basis_j):
    """
    Public wrapper matching the old interface.
    """
    return om1_ppecp_local_vectorized(ni, nj, rij, basis_tables, basis_i, basis_j)

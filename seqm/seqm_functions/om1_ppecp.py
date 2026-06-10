import torch

from .constants import ev
from .omx_basis import gather_om1_basis

_SQPI = 1.77245385090552
_FPI = 12.5663706143592
_A3 = 0.333333333333333
_A4 = 0.666666666666667
_ALIM = 0.317
_ABLIM = 0.1
_BIGEXP = 50.0
_TOL = 12 * 2.302585093


def _horner_piecewise_vec(x, coeffs_t, ifirst_t, ilast_t, h):
    """
    Vectorized table/Horner evaluator.

    coeffs_t, ifirst_t, ilast_t are already tensors on the correct device.
    """
    if x.numel() == 0:
        return x.clone()

    nx = (x / h).to(torch.long) + 1
    out = torch.empty_like(x)

    # Loop only over unique table intervals, not over molecules/pairs/primitives.
    for nx_val in torch.unique(nx.detach()).cpu().tolist():
        m = nx == int(nx_val)
        if not m.any():
            continue

        start = int(ifirst_t[nx_val - 1].item()) - 1
        end = int(ilast_t[nx_val - 1].item()) - 1

        xm = x[m]
        val = coeffs_t[end].expand_as(xm)
        for idx in range(end - 1, start - 1, -1):
            val = coeffs_t[idx] + xm * val

        out[m] = val

    return out


def _dawf_vec(y, ppecp):
    """
    Vectorized Dawson-like helper matching scalar _dawf.
    """
    x = y.abs()
    out = torch.empty_like(x)

    small = x < 10.0
    if small.any():
        out[small] = _horner_piecewise_vec(
            x[small], ppecp["dawf_c"], ppecp["dawf_ifirst"], ppecp["dawf_ilast"], ppecp["dawf_h"]
        )

    large = ~small
    if large.any():
        xl = x[large]
        txt = 0.5 / (xl * xl)
        tx = txt * xl
        out[large] = tx * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + 105.0 * txt))))

    neg = y < 0.0
    if neg.any():
        out[neg] = -out[neg]

    return out


def _dawerf_vec(y, ppecp):
    """
    Vectorized helper matching scalar _dawerf.
    """
    x = y.abs()
    out = torch.empty_like(x)

    small = x < 10.0
    if small.any():
        out[small] = _horner_piecewise_vec(
            x[small], ppecp["dawerf_c"], ppecp["dawerf_ifirst"], ppecp["dawerf_ilast"], ppecp["dawerf_h"]
        )

    large = ~small
    if large.any():
        xl = x[large]
        txt = 0.5 / (xl * xl)
        tx = txt * xl
        out[large] = tx * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + txt * (105.0 + 945.0 * txt)))))

    return out


def _fsips_vec(n, l, alfa, xp0):
    fctrl = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0]
    dfctrl = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]

    nl = n + l

    if nl % 2 == 0:
        lam = nl // 2
        x = alfa * alfa
        l2 = 2 * l + 1
        term = torch.full_like(alfa, dfctrl[lam] / dfctrl[l + 1])
        total = term.clone()
        for k in range(1, 11):
            term = (term * x * (2 * k + nl - 1)) / (k * (2 * k + l2))
            total = total + term
        return total * _SQPI * xp0 * (2.0**lam) * (alfa**l)

    lam = (nl - 1) // 2
    x = 2.0 * alfa * alfa
    l2 = 2 * l + 1
    term = torch.full_like(alfa, fctrl[lam] / dfctrl[l + 1])
    total = term.clone()
    for k in range(1, 11):
        term = (term * x * (k + lam)) / (k * (2 * k + l2))
        total = total + term
    return total * xp0 * (2.0**nl) * (alfa**l)


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


def _stack2(d, n0, n1, z):
    return torch.stack(
        [torch.stack([d.get((i, j), z) for j in range(n1)], dim=-1) for i in range(n0)], dim=-2
    )


def _stack3(d, n0, n1, n2, z):
    return torch.stack(
        [
            torch.stack(
                [torch.stack([d.get((i, j, k), z) for k in range(n2)], dim=-1) for j in range(n1)], dim=-2
            )
            for i in range(n0)
        ],
        dim=-3,
    )


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

    alfa = alfj.clone()
    xp1 = xpls.clone()

    m = alfj <= betj
    if m.any():
        alfa[m] = betj[m]
        xp1[m] = xmns[m]

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


def _fm_vec(l, a, b, xpls, xmns):
    t = 2.0 * a * b
    est = 0.5 * (xpls - xmns)
    ect = 0.5 * (xpls + xmns)

    if l == -1:
        return t
    if l == 0:
        return est / t
    if l == 1:
        return (-est / t + ect) / t

    raise NotImplementedError("OM1 PPECP current scope only needs FM(-1:1)")


def _fjps_vec(n, lalf, lbet, alf, bet, xi, si):
    dfctrl = [1.0, 3.0, 15.0, 105.0]

    out = torch.empty_like(alf)

    m = alf > bet
    if m.any():
        b = bet[m]
        si_m = si[m]

        x = b * b
        term = (b**lbet) / dfctrl[lbet]
        l1 = lalf + 1
        l2 = lbet + n + 1
        l3 = 2 * lbet + 1

        total = term * si_m[..., l2, l1]
        for k in range(1, 11):
            term = term * x / (2.0 * k * (2.0 * k + l3))
            total = total + term * si_m[..., 2 * k + l2, l1]

        out[m] = total * (0.5 * xi[m]) ** (n + 1)

    m = alf <= bet
    if m.any():
        a = alf[m]
        si_m = si[m]

        x = a * a
        term = (a**lalf) / dfctrl[lalf]
        l1 = lbet + 1
        l2 = lalf + n + 1
        l3 = 2 * lalf + 1

        total = term * si_m[..., l2, l1]
        for k in range(1, 11):
            term = term * x / (2.0 * k * (2.0 * k + l3))
            total = total + term * si_m[..., 2 * k + l2, l1]

        out[m] = total * (0.5 * xi[m]) ** (n + 1)

    return out


def _fj00_vec(n, a, b, xi, xpls, xmns, xp, si, ppecp):
    out = torch.empty_like(a)

    small = a * b <= _ABLIM
    if small.any():
        if si is None:
            raise RuntimeError("_fj00_vec needs si for small a*b branch")
        out[small] = _fjps_vec(n, 0, 0, a[small], b[small], xi[small], si[small])

    large = ~small
    if large.any():
        aa = a[large]
        bb = b[large]
        xpls_l = xpls[large]
        xmns_l = xmns[large]
        xi_l = xi[large]

        tab = _fm_vec(-1, aa, bb, xpls_l, xmns_l)

        if n == 1:
            tp = xpls_l * _dawf_vec(aa + bb, ppecp)
            tm = xmns_l * _dawf_vec(aa - bb, ppecp)
            hm = tp - tm
            out[large] = _SQPI * xi_l * xi_l * hm / (4.0 * tab)
        else:
            tp = xpls_l * _dawf_vec(aa + bb, ppecp)
            tm = xmns_l * _dawf_vec(aa - bb, ppecp)
            dp = tp + tm
            dm = tp - tm
            out[large] = (
                _SQPI * xi_l * (aa * dm + bb * dp - tab * _fm_vec(0, aa, bb, xpls_l, xmns_l)) / (2.0 * tab)
            )

    return out


def _fj10_vec(n, a, b, xi, xpls, xmns, xp, si, ppecp):
    out = torch.empty_like(a)

    small = a * b <= _ABLIM
    if small.any():
        if si is None:
            raise RuntimeError("_fj10_vec needs si for small a*b branch")
        out[small] = _fjps_vec(n, 1, 0, a[small], b[small], xi[small], si[small])

    large = ~small
    if large.any():
        aa = a[large]
        bb = b[large]
        xpls_l = xpls[large]
        xmns_l = xmns[large]
        xi_l = xi[large]
        xp_l = xp[large]

        tp = xpls_l * torch.erf(aa + bb)
        tm = xmns_l * torch.erf(aa - bb)
        ep = tp + tm
        em = tp - tm
        hm = xpls_l * _dawerf_vec(aa + bb, ppecp) - xmns_l * _dawerf_vec(aa - bb, ppecp)
        tab = _fm_vec(-1, aa, bb, xpls_l, xmns_l)

        if n == 1:
            dp = xpls_l * _dawf_vec(aa + bb, ppecp) + xmns_l * _dawf_vec(aa - bb, ppecp)
            out[large] = (
                _SQPI * xi_l * xi_l * (2.0 * _fm_vec(0, aa, bb, xpls_l, xmns_l) - dp / aa) / (8.0 * aa)
            )
        else:
            out[large] = xi_l * (
                _SQPI * ((1.0 + 2.0 * (aa + bb) * (aa - bb)) * hm + bb * ep - aa * em) / (8.0 * aa * tab)
                - xp_l / (4.0 * aa)
            )

    return out


def _fj01_vec(n, a, b, xi, xpls, xmns, xp, si, ppecp):
    out = torch.empty_like(a)

    small = a * b <= _ABLIM
    if small.any():
        if si is None:
            raise RuntimeError("_fj01_vec needs si for small a*b branch")
        out[small] = _fjps_vec(n, 0, 1, a[small], b[small], xi[small], si[small])

    large = ~small
    if large.any():
        aa = a[large]
        bb = b[large]
        xpls_l = xpls[large]
        xmns_l = xmns[large]
        xi_l = xi[large]
        xp_l = xp[large]

        tp = xpls_l * torch.erf(aa + bb)
        tm = xmns_l * torch.erf(aa - bb)
        ep = tp + tm
        em = tp - tm
        hm = xpls_l * _dawerf_vec(aa + bb, ppecp) - xmns_l * _dawerf_vec(aa - bb, ppecp)
        tab = _fm_vec(-1, aa, bb, xpls_l, xmns_l)

        if n == 1:
            dm = xpls_l * _dawf_vec(aa + bb, ppecp) - xmns_l * _dawf_vec(aa - bb, ppecp)
            out[large] = (
                _SQPI * xi_l * xi_l * (2.0 * _fm_vec(0, aa, bb, xpls_l, xmns_l) - dm / bb) / (8.0 * bb)
            )
        else:
            out[large] = xi_l * (
                _SQPI * ((1.0 - 2.0 * (aa + bb) * (aa - bb)) * hm - bb * ep + aa * em) / (8.0 * bb * tab)
                - xp_l / (4.0 * bb)
            )

    return out


def _fj11_vec(n, a, b, xi, xpls, xmns, xp, si, ppecp):
    out = torch.empty_like(a)

    small = a * b <= _ABLIM
    if small.any():
        if si is None:
            raise RuntimeError("_fj11_vec needs si for small a*b branch")
        out[small] = _fjps_vec(n, 1, 1, a[small], b[small], xi[small], si[small])

    large = ~small
    if large.any():
        aa = a[large]
        bb = b[large]
        xpls_l = xpls[large]
        xmns_l = xmns[large]
        xi_l = xi[large]
        xp_l = xp[large]

        tp = xpls_l * _dawf_vec(aa + bb, ppecp)
        tm = xmns_l * _dawf_vec(aa - bb, ppecp)
        dp = tp + tm
        dm = tp - tm
        tab = _fm_vec(-1, aa, bb, xpls_l, xmns_l)

        if n == 1:
            tp = xpls_l * torch.erf(aa + bb)
            tm = xmns_l * torch.erf(aa - bb)
            ep = tp + tm
            em = tp - tm
            hm = xpls_l * _dawerf_vec(aa + bb, ppecp) - xmns_l * _dawerf_vec(aa - bb, ppecp)

            out[large] = (
                xi_l
                * xi_l
                * (
                    _SQPI * (aa * em + bb * ep - (1.0 + 2.0 * (aa * aa + bb * bb)) * hm) / (8.0 * tab)
                    - xp_l / (4.0 * tab)
                )
            )
        else:
            a2 = aa * aa
            b2 = bb * bb
            out[large] = (
                _SQPI
                * xi_l
                * (
                    2.0 * (a2 + b2) * _fm_vec(0, aa, bb, xpls_l, xmns_l)
                    - tab * _fm_vec(1, aa, bb, xpls_l, xmns_l)
                    - b2 * dp / aa
                    - a2 * dm / bb
                )
                / (6.0 * tab)
            )

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

    xp0_all = torch.zeros_like(xalfa)
    m_exp = xalfa < _BIGEXP
    if m_exp.any():
        xp0_all[m_exp] = torch.exp(-xalfa[m_exp])

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

    alfbet_all = alfa_all * r_f + beta_all * r_f
    xpb_all = torch.zeros_like(alfbet_all)
    m_xpb = alfbet_all < _BIGEXP
    if m_xpb.any():
        xpb_all[m_xpb] = torch.exp(-alfbet_all[m_xpb])

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

    xi1 = 1.0 / torch.sqrt(zetcb1 + zlp1)
    alef1 = alfa1 * xi1
    beit1 = beta1 * xi1
    ab1 = alef1 * beit1

    dum1 = torch.empty_like(ab1)
    dum2 = torch.empty_like(ab1)

    large_ab = ab1 > _ABLIM
    if large_ab.any():
        dum1[large_ab] = alfbet1[large_ab] - (alef1[large_ab] + beit1[large_ab]) ** 2
        dum2[large_ab] = alfbet1[large_ab] - (alef1[large_ab] - beit1[large_ab]) ** 2

    small_ab = ~large_ab
    if small_ab.any():
        dum1[small_ab] = alfbet1[small_ab] - alef1[small_ab] * alef1[small_ab]
        dum2[small_ab] = alfbet1[small_ab] - beit1[small_ab] * beit1[small_ab]

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

    xpls = torch.zeros_like(dum1)
    xmns = torch.zeros_like(dum2)

    m = dum1 < _BIGEXP
    if m.any():
        xpls[m] = torch.exp(-dum1[m])

    m = dum2 < _BIGEXP
    if m.any():
        xmns[m] = torch.exp(-dum2[m])

    nmx = nij - 1 + nlpk
    xa = xi * alef
    xb = xi * beit
    xx = xi * xi * 0.5

    z = torch.zeros_like(alef)
    f = {}

    si = None
    if (alef * beit <= _ABLIM).any():
        if nlpk % 2 == 0:
            si = _sitabl_vec(2, -1, alef, beit, xpls, xmns, xpb, ppecp)
        else:
            si = _sitabl_vec(-1, 2, alef, beit, xpls, xmns, xpb, ppecp)

    if nlpk % 2 == 0:
        f[(1, 1, 1)] = _fj00_vec(0, alef, beit, xi, xpls, xmns, xpb, si, ppecp)
        f[(1, 2, 2)] = _fj11_vec(0, alef, beit, xi, xpls, xmns, xpb, si, ppecp)

        if nmx >= 1:
            f[(2, 2, 1)] = _fj10_vec(1, alef, beit, xi, xpls, xmns, xpb, si, ppecp)
            f[(2, 1, 2)] = _fj01_vec(1, alef, beit, xi, xpls, xmns, xpb, si, ppecp)

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
        f[(1, 2, 1)] = _fj10_vec(0, alef, beit, xi, xpls, xmns, xpb, si, ppecp)
        f[(1, 1, 2)] = _fj01_vec(0, alef, beit, xi, xpls, xmns, xpb, si, ppecp)
        f[(2, 1, 1)] = _fj00_vec(1, alef, beit, xi, xpls, xmns, xpb, si, ppecp)
        f[(2, 2, 2)] = _fj11_vec(1, alef, beit, xi, xpls, xmns, xpb, si, ppecp)

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
        rr[(n, 1, 1)] = _get3(rr, n, 1, 1, z) + _get3(f, n + nlpk, 1, 1, z) * clp_a

    if lijc > 1 and lijb > 1:
        for n in range(1, nij + 1, 2):
            rr[(n, 2, 2)] = _get3(rr, n, 2, 2, z) + _get3(f, n + nlpk, 2, 2, z) * clp_a

    if lijb > 1:
        for n in range(2, nij + 1, 2):
            rr[(n, 1, 2)] = _get3(rr, n, 1, 2, z) + _get3(f, n + nlpk, 1, 2, z) * clp_a

    if lijc > 1:
        for n in range(2, nij + 1, 2):
            rr[(n, 2, 1)] = _get3(rr, n, 2, 1, z) + _get3(f, n + nlpk, 2, 1, z) * clp_a

    rr_a = _stack3(rr, nij + 1, lijc + 1, lijb + 1, z)
    rr_full = full_zero.index_copy(0, idx, rr_a)
    return rr_full.reshape((*lead_shape, nij + 1, lijc + 1, lijb + 1))


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


def om1_ppecp_local_vectorized(ni, nj, rij, zeta_i, zeta_j, basis):
    """
    Vectorized OM1 PPECP local block.

    Inputs:
        ni, nj:    [npair] atomic numbers
        rij:       [npair] pair distance in OM1 units
        zeta_i/j:  [npair] shell zeta values
        basis:     basis table from build_omx_basis_tables, with basis["ppecp"]

    Returns:
        corpp [npair, 4, 2] in eV
    """
    device = rij.device
    dtype = rij.dtype

    ppecp = basis["ppecp"]

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
    shell_zeta = torch.stack((zeta_i, zeta_j), dim=1)

    shell_z_f = shell_z.reshape(-1)
    ecp_z_f = ecp_z.reshape(-1)
    acz_f = acz.reshape(-1)
    zfn_f = zfn.reshape(-1)
    shell_zeta_f = shell_zeta.reshape(-1)
    r_f = rij.repeat_interleave(2)

    # H has no ECP. Padding should also be inactive if Z<=0.
    active = (shell_z_f > 0) & (ecp_z_f > 2)

    if not active.any():
        return corpp

    shell_z_a = shell_z_f[active]
    ecp_z_a = ecp_z_f[active]
    acz_a = acz_f[active]
    zfn_a = zfn_f[active]
    r_a = r_f[active]
    shell_zeta_a = shell_zeta_f[active]
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

    shell_type, exponents, coeff_s, coeff_p = gather_om1_basis(shell_z_a, shell_zeta_a, basis)

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

    corpp_active = torch.zeros((shell_z_a.numel(), 4), dtype=dtype, device=device)

    def _run_group(mask, kd):
        if not mask.any():
            return

        zetc = zetc_all[mask]
        zetb = zetb_all[mask]

        g1 = _primitive_g1_vec(
            kd=kd,
            acz=acz_a[mask].unsqueeze(1),
            zfn=zfn_a[mask].unsqueeze(1),
            zetc=zetc,
            zetb=zetb,
            r=r_a[mask].unsqueeze(1),
            zlp=zlp_a[mask],
            clp=clp_a[mask],
            ppecp=ppecp,
        )

        cs_i = cs_i_all[mask]
        cs_j = cs_j_all[mask]
        cp_i = cp_i_all[mask]
        cp_j = cp_j_all[mask]

        ngrp = cs_i.shape[0]
        out = torch.zeros((ngrp, 4), dtype=dtype, device=device)

        out[:, 0] = (sym * cs_i * cs_j * g1[..., 1]).sum(dim=1)

        if kd > 1:
            out[:, 1] = (cs_i * cp_j * g1[..., 2] + offdiag * cs_j * cp_i * g1[..., 3]).sum(dim=1)

            out[:, 2] = (sym * cp_i * cp_j * g1[..., 4]).sum(dim=1)
            out[:, 3] = (sym * cp_i * cp_j * g1[..., 5]).sum(dim=1)

        corpp_active[mask] = out

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


def om1_ppecp_local(ni, nj, rij, zeta_i, zeta_j, basis):
    """
    Public wrapper matching the old interface.
    """
    return om1_ppecp_local_vectorized(ni, nj, rij, zeta_i, zeta_j, basis)

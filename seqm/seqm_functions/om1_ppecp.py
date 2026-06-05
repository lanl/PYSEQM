import math

import torch

from .constants import ev
from .om1_overlap import _lookup_basis
from .om1_ppecp_tables import (
    DAWERF_C,
    DAWERF_H,
    DAWERF_IFIRST,
    DAWERF_ILAST,
    DAWF_C,
    DAWF_H,
    DAWF_IFIRST,
    DAWF_ILAST,
    OM1_FIRST_ROW_ECP,
)

_SQPI = 1.77245385090552
_FPI = 12.5663706143592
_A3 = 0.333333333333333
_A4 = 0.666666666666667
_ALIM = 0.317
_ABLIM = 0.1
_BIGEXP = 50.0
_TOL = 12 * 2.302585093


def _horner_piecewise(x, coeffs, ifirst, ilast, h):
    nx = int(x / h) + 1
    start = ifirst[nx - 1] - 1
    end = ilast[nx - 1] - 1
    value = coeffs[end]
    for idx in range(end - 1, start - 1, -1):
        value = coeffs[idx] + x * value
    return value


def _dawf(y):
    x = abs(y)
    if x < 10.0:
        value = _horner_piecewise(x, DAWF_C, DAWF_IFIRST, DAWF_ILAST, DAWF_H)
    else:
        txt = 0.5 / (x * x)
        tx = txt * x
        value = tx * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + 105.0 * txt))))
    return -value if y < 0.0 else value


def _dawerf(y):
    x = abs(y)
    if x < 10.0:
        return _horner_piecewise(x, DAWERF_C, DAWERF_IFIRST, DAWERF_ILAST, DAWERF_H)
    txt = 0.5 / (x * x)
    tx = txt * x
    return tx * (1.0 + txt * (1.0 + txt * (3.0 + txt * (15.0 + txt * (105.0 + 945.0 * txt)))))


def _basis_primitives(atomic_number, zeta):
    shell_type, exponents, coeff_s, coeff_p = _lookup_basis(
        torch.tensor([atomic_number], dtype=torch.int64), torch.tensor([zeta], dtype=torch.float64)
    )
    shell_type = int(shell_type.item())
    return shell_type, exponents[0].tolist(), coeff_s[0].tolist(), coeff_p[0].tolist()


def _fsips(n, l, alfa, xp0):
    fctrl = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0]
    dfctrl = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
    nl = n + l
    if nl % 2 == 0:
        lam = nl // 2
        x = alfa * alfa
        l2 = 2 * l + 1
        term = dfctrl[lam] / dfctrl[l + 1]
        total = term
        for k in range(1, 11):
            term = (term * x * (2 * k + nl - 1)) / (k * (2 * k + l2))
            total += term
        return total * _SQPI * xp0 * (2.0**lam) * (alfa**l)
    lam = (nl - 1) // 2
    x = 2.0 * alfa * alfa
    l2 = 2 * l + 1
    term = fctrl[lam] / dfctrl[l + 1]
    total = term
    for k in range(1, 11):
        term = (term * x * (k + lam)) / (k * (2 * k + l2))
        total += term
    return total * xp0 * (2.0**nl) * (alfa**l)


def _fsi0(n, alfa, xp0, xp1):
    if alfa <= _ALIM:
        return _fsips(n, 0, alfa, xp0)
    if n != 1:
        dawfs = _dawf(alfa)
        return _SQPI * dawfs * xp1 / alfa
    errfs = _SQPI * math.erf(alfa) * xp1
    return errfs / alfa


def _fsi1(n, alfa, xp0, xp1):
    if alfa <= _ALIM:
        return _fsips(n, 1, alfa, xp0)
    if n <= 0:
        errfs = _SQPI * math.erf(alfa) * xp1
        return (0.5 * errfs / alfa - xp0) / alfa
    if n == 1:
        dawfs = _dawf(alfa)
        return _SQPI * (alfa - dawfs) * xp1 / (alfa * alfa)
    alfa2 = alfa * alfa
    errfs = _SQPI * math.erf(alfa) * xp1
    return (2.0 * alfa * xp0 + (2.0 * alfa2 - 1.0) * errfs) / alfa2


def _fsi2(n, alfa, xp0, xp1):
    if alfa <= _ALIM:
        return _fsips(n, 2, alfa, xp0)
    alfa2 = alfa * alfa
    dawfs = _dawf(alfa)
    errfs = _SQPI * math.erf(alfa) * xp1
    if n == 0:
        return 0.25 * _SQPI * (3.0 * alfa - (2.0 * alfa2 + 3.0) * dawfs) * xp1 / (alfa2 * alfa)
    if n == 1:
        return (0.5 * (2.0 * alfa2 - 3.0) * errfs + 3.0 * alfa * xp0) / (alfa2 * alfa)
    if n == 2:
        return _SQPI * (alfa * (2.0 * alfa2 - 3.0) + 3.0 * dawfs) * xp1 / (alfa2 * alfa)
    if n == 3:
        return ((alfa2 * (4.0 * alfa2 - 4.0) + 3.0) * errfs + 2.0 * alfa * (2.0 * alfa2 - 3.0) * xp0) / (
            alfa2 * alfa
        )
    raise NotImplementedError("FSI2 current OM1 PPECP scope only needs N<=3")


def _recur(si, nmin, nmax, lmax, x):
    tx = 2.0 * x
    for n in range(nmin, nmax + 1, 2):
        tfnm2 = 2.0 * n - 4.0
        si[n + 1][1] = (tfnm2 + 2.0) * si[n - 1][1] + tx * si[n][2]
        si[n + 2][2] = tfnm2 * si[n][2] + tx * si[n + 1][1]
        if lmax >= 2:
            si[n + 3][3] = tfnm2 * si[n + 1][3] + tx * si[n + 2][2]


def _sitabl(lemax, lomax, alfj, betj, xj, xpls, xmns, xp):
    si = [[0.0] * 5 for _ in range(27)]
    if alfj <= betj:
        alfa = betj
        x = betj
        xp0 = xp
        xp1 = xmns
    else:
        alfa = alfj
        x = alfj
        xp0 = xp
        xp1 = xpls

    if lemax >= 0:
        si[1][1] = _fsi0(0, alfa, xp0, xp1)
        si[2][2] = _fsi1(1, alfa, xp0, xp1)
        if lemax >= 2:
            si[3][3] = _fsi2(2, alfa, xp0, xp1)
        _recur(si, 2, 22, lemax, x)

    if lomax >= 0:
        si[1][2] = _fsi1(0, alfa, xp0, xp1)
        si[2][1] = _fsi0(1, alfa, xp0, xp1)
        si[3][2] = _fsi1(2, alfa, xp0, xp1)
        _recur(si, 3, 21, lomax, x)
    return si


def _fm(l, a, b, xpls, xmns):
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


def _fjps(n, lalf, lbet, alf, bet, xi, si):
    dfctrl = [1.0, 3.0, 15.0, 105.0]
    if alf > bet:
        x = bet * bet
        term = bet**lbet / dfctrl[lbet]
        l1 = lalf + 1
        l2 = lbet + n + 1
        l3 = 2 * lbet + 1
    else:
        x = alf * alf
        term = alf**lalf / dfctrl[lalf]
        l1 = lbet + 1
        l2 = lalf + n + 1
        l3 = 2 * lalf + 1
    total = term * si[l2][l1]
    for k in range(1, 11):
        term = term * x / (2.0 * k * (2.0 * k + l3))
        total += term * si[2 * k + l2][l1]
    return total * (0.5 * xi) ** (n + 1)


def _fj00(n, a, b, xi, xpls, xmns, xp, si):
    if a * b <= _ABLIM:
        return _fjps(n, 0, 0, a, b, xi, si)
    if n == 1:
        tab = _fm(-1, a, b, xpls, xmns)
        tp = xpls * _dawf(a + b)
        tm = xmns * _dawf(a - b)
        hm = tp - tm
        return _SQPI * xi * xi * hm / (4.0 * tab)
    tp = xpls * _dawf(a + b)
    tm = xmns * _dawf(a - b)
    dp = tp + tm
    dm = tp - tm
    tab = _fm(-1, a, b, xpls, xmns)
    return _SQPI * xi * (a * dm + b * dp - tab * _fm(0, a, b, xpls, xmns)) / (2.0 * tab)


def _fj10(n, a, b, xi, xpls, xmns, xp, si):
    if a * b <= _ABLIM:
        return _fjps(n, 1, 0, a, b, xi, si)
    tp = xpls * math.erf(a + b)
    tm = xmns * math.erf(a - b)
    ep = tp + tm
    em = tp - tm
    hm = xpls * _dawerf(a + b) - xmns * _dawerf(a - b)
    tab = _fm(-1, a, b, xpls, xmns)
    if n == 1:
        dp = xpls * _dawf(a + b) + xmns * _dawf(a - b)
        return _SQPI * xi * xi * (2.0 * _fm(0, a, b, xpls, xmns) - dp / a) / (8.0 * a)
    return xi * (
        _SQPI * ((1.0 + 2.0 * (a + b) * (a - b)) * hm + b * ep - a * em) / (8.0 * a * tab) - xp / (4.0 * a)
    )


def _fj01(n, a, b, xi, xpls, xmns, xp, si):
    if a * b <= _ABLIM:
        return _fjps(n, 0, 1, a, b, xi, si)
    tp = xpls * math.erf(a + b)
    tm = xmns * math.erf(a - b)
    ep = tp + tm
    em = tp - tm
    hm = xpls * _dawerf(a + b) - xmns * _dawerf(a - b)
    tab = _fm(-1, a, b, xpls, xmns)
    if n == 1:
        dm = xpls * _dawf(a + b) - xmns * _dawf(a - b)
        return _SQPI * xi * xi * (2.0 * _fm(0, a, b, xpls, xmns) - dm / b) / (8.0 * b)
    return xi * (
        _SQPI * ((1.0 - 2.0 * (a + b) * (a - b)) * hm - b * ep + a * em) / (8.0 * b * tab) - xp / (4.0 * b)
    )


def _fj11(n, a, b, xi, xpls, xmns, xp, si):
    if a * b <= _ABLIM:
        return _fjps(n, 1, 1, a, b, xi, si)
    tp = xpls * _dawf(a + b)
    tm = xmns * _dawf(a - b)
    dp = tp + tm
    dm = tp - tm
    tab = _fm(-1, a, b, xpls, xmns)
    if n == 1:
        tp = xpls * math.erf(a + b)
        tm = xmns * math.erf(a - b)
        ep = tp + tm
        em = tp - tm
        hm = xpls * _dawerf(a + b) - xmns * _dawerf(a - b)
        return (
            xi
            * xi
            * (
                _SQPI * (a * em + b * ep - (1.0 + 2.0 * (a * a + b * b)) * hm) / (8.0 * tab)
                - xp / (4.0 * tab)
            )
        )
    a2 = a * a
    b2 = b * b
    return (
        _SQPI
        * xi
        * (
            2.0 * (a2 + b2) * _fm(0, a, b, xpls, xmns)
            - tab * _fm(1, a, b, xpls, xmns)
            - b2 * dp / a
            - a2 * dm / b
        )
        / (6.0 * tab)
    )


def _fiprep(nlpk, nij, zetc, zetb, r, zlp, clp):
    rr = [[0.0] * (nij + 1) for _ in range(nij + 1)]
    zetcb = zetc + zetb
    alfa = zetcb * r
    xalfa = alfa * r
    alfi = 0.5 / alfa
    xp0 = math.exp(-xalfa) if xalfa < _BIGEXP else 0.0
    xi = 1.0 / math.sqrt(zetcb + zlp)
    alf = alfa * xi
    dum = xalfa - alf * alf
    if dum <= _TOL:
        xp1 = math.exp(-dum)
        nmx = nij - 1 + nlpk
        yi = 0.5 * xi
        a1 = xi * alf
        a2 = xi * yi
        fit = [[0.0, 0.0, 0.0] for _ in range(11)]
        x = alf * alf
        if nlpk % 2 == 0:
            if alf <= _ALIM:
                t11 = 1.0
                t22 = 1.0 / 3.0
                s11 = t11
                s22 = t22
                for k in range(1, 11):
                    t11 = (t11 * x * (2 * k - 1)) / (k * (2 * k + 1))
                    t22 = (t22 * x * (2 * k + 1)) / (k * (2 * k + 3))
                    s11 += t11
                    s22 += t22
                f11 = s11 * _SQPI * xp0
                f22 = s22 * _SQPI * xp0 * 2.0 * alf
            else:
                dawfs = _dawf(alf)
                f11 = _SQPI * dawfs * xp1 / alf
                f22 = _SQPI * (alf - dawfs) * xp1 / x
            fit[1][1] = f11 * yi
            if nmx >= 1:
                fit[2][2] = f22 * yi * yi
                for n in range(2, nmx + 1, 2):
                    fit[n + 1][1] = a1 * fit[n][2] + (n - 1) * a2 * fit[n - 1][1]
                    fit[n + 2][2] = a1 * fit[n + 1][1] + (n - 2) * a2 * fit[n][2]
        else:
            if alf <= _ALIM:
                xx = 2.0 * x
                t12 = 1.0 / 3.0
                t21 = 1.0
                s12 = t12
                s21 = t21
                for k in range(1, 11):
                    t12 = t12 * xx / (2 * k + 3)
                    t21 = t21 * xx / (2 * k + 1)
                    s12 += t12
                    s21 += t21
                f12 = s12 * xp0 * 2.0 * alf
                f21 = s21 * xp0 * 2.0
            else:
                errfs = _SQPI * math.erf(alf) * xp1
                f21 = errfs / alf
                f12 = (0.5 * errfs / alf - xp0) / alf
            fit[1][2] = f12 * yi
            fit[2][1] = f21 * yi * yi
            for n in range(2, nmx + 1, 2):
                fit[n + 1][2] = a1 * fit[n][1] + (n - 3) * a2 * fit[n - 1][2]
                fit[n + 2][1] = a1 * fit[n + 1][2] + n * a2 * fit[n][1]
        for n in range(1, nij + 1, 2):
            rr[n][1] += fit[n + nlpk][1] * clp
        if nij > 1:
            for n in range(2, nij + 1, 2):
                rr[n][2] += fit[n + nlpk][2] * clp
        if nij > 2:
            for lp in range(3, nij + 1):
                lp1 = lp - 1
                lp2 = lp - 2
                fp = (lp1 + lp2) * alfi
                for n in range(lp, nij + 1, 2):
                    rr[n][lp] = rr[n][lp2] - fp * rr[n - 1][lp1]
    return rr


def _fjprep(nlpk, nij, zetc, zetb, r, zlp, clp):
    shell_type = 1 if nij == 3 else 0
    lijc = shell_type + 1
    lijb = shell_type + 1
    rr = [[[0.0] * (lijb + 1) for _ in range(lijc + 1)] for _ in range(nij + 1)]

    alfa = zetc * r
    beta = zetb * r
    zetcb = zetc + zetb
    alfi = 0.5 / alfa
    beti = 0.5 / beta
    alfbet = alfa * r + beta * r
    xpb = math.exp(-alfbet) if alfbet < _BIGEXP else 0.0
    zeta = zetcb + zlp
    dumtol = zlp * (((alfa + beta) ** 2) / zetcb) / zeta
    if dumtol > _TOL:
        return rr

    xi = 1.0 / math.sqrt(zeta)
    alef = alfa * xi
    beit = beta * xi
    xpls = 0.0
    xmns = 0.0
    if alef * beit > _ABLIM:
        dum1 = alfbet - (alef + beit) ** 2
        dum2 = alfbet - (alef - beit) ** 2
    else:
        dum1 = alfbet - alef * alef
        dum2 = alfbet - beit * beit
    if dum1 >= _BIGEXP and dum2 >= _BIGEXP:
        return rr
    if dum1 < _BIGEXP:
        xpls = math.exp(-dum1)
    if dum2 < _BIGEXP:
        xmns = math.exp(-dum2)

    nmx = nij - 1 + nlpk
    f = [[[0.0] * 3 for _ in range(3)] for _ in range(11)]
    xa = xi * alef
    xb = xi * beit
    xx = xi * xi * 0.5
    si = None
    if nlpk % 2 == 0:
        if alef * beit <= _ABLIM:
            si = _sitabl(2, -1, alef, beit, xi, xpls, xmns, xpb)
        f[1][1][1] = _fj00(0, alef, beit, xi, xpls, xmns, xpb, si)
        f[1][2][2] = _fj11(0, alef, beit, xi, xpls, xmns, xpb, si)
        if nmx >= 1:
            f[2][2][1] = _fj10(1, alef, beit, xi, xpls, xmns, xpb, si)
            f[2][1][2] = _fj01(1, alef, beit, xi, xpls, xmns, xpb, si)
            for n in range(2, nmx + 1, 2):
                np1 = n + 1
                f[np1][1][1] = xa * f[n][2][1] + xb * f[n][1][2] + (n - 1) * xx * f[n - 1][1][1]
                f[np1][2][2] = xa * f[n][1][2] + xb * f[n][2][1] + (n - 5) * xx * f[n - 1][2][2]
                f[n + 2][2][1] = xa * f[np1][1][1] + xb * f[np1][2][2] + (n - 2) * xx * f[n][2][1]
                f[n + 2][1][2] = xb * f[np1][1][1] + xa * f[np1][2][2] + (n - 2) * xx * f[n][1][2]
    else:
        if alef * beit <= _ABLIM:
            si = _sitabl(-1, 2, alef, beit, xi, xpls, xmns, xpb)
        f[1][2][1] = _fj10(0, alef, beit, xi, xpls, xmns, xpb, si)
        f[1][1][2] = _fj01(0, alef, beit, xi, xpls, xmns, xpb, si)
        f[2][1][1] = _fj00(1, alef, beit, xi, xpls, xmns, xpb, si)
        f[2][2][2] = _fj11(1, alef, beit, xi, xpls, xmns, xpb, si)
        for n in range(2, nmx + 1, 2):
            np1 = n + 1
            f[np1][2][1] = xa * f[n][1][1] + xb * f[n][2][2] + (n - 3) * xx * f[n - 1][2][1]
            f[np1][1][2] = xb * f[n][1][1] + xa * f[n][2][2] + (n - 3) * xx * f[n - 1][1][2]
            f[n + 2][1][1] = xa * f[np1][2][1] + xb * f[np1][1][2] + n * xx * f[n][1][1]
            f[n + 2][2][2] = xa * f[np1][1][2] + xb * f[np1][2][1] + (n - 4) * xx * f[n][2][2]

    for n in range(1, nij + 1, 2):
        rr[n][1][1] += f[n + nlpk][1][1] * clp
    if lijc > 1 and lijb > 1:
        for n in range(1, nij + 1, 2):
            rr[n][2][2] += f[n + nlpk][2][2] * clp
    if lijb > 1:
        for n in range(2, nij + 1, 2):
            rr[n][1][2] += f[n + nlpk][1][2] * clp
    if lijc > 1:
        for n in range(2, nij + 1, 2):
            rr[n][2][1] += f[n + nlpk][2][1] * clp

    if nij >= 2 and (lijc + lijb) >= 4:
        if lijc > 2:
            fc = 3.0 * alfi
            for n in range(3, nij + 1, 2):
                rr[n][3][1] = rr[n][1][1] - fc * rr[n - 1][2][1]
        if lijb > 2:
            fb = 3.0 * beti
            for n in range(3, nij + 1, 2):
                rr[n][1][3] = rr[n][1][1] - fb * rr[n - 1][1][2]
    return rr


def _primitive_g1(kd, acz, zfn, zetc, zetb, r, ecp):
    nij = 1 if kd == 1 else 3
    r1 = [[0.0] * (nij + 1) for _ in range(nij + 1)]
    r2 = [[[0.0] * (2 + (kd > 1)) for _ in range(2 + (kd > 1))] for _ in range(nij + 1)]

    local_nlp, corr1_nlp, corr2_nlp = ecp["nlp"]
    local_zlp, corr1_zlp, corr2_zlp = ecp["zlp"]
    local_clp, corr1_clp, corr2_clp = ecp["clp"]

    r1_local = _fiprep(local_nlp, nij, zetc, zetb, r, local_zlp, local_clp)
    for n in range(1, nij + 1):
        for l in range(1, nij + 1):
            r1[n][l] += r1_local[n][l]

    for nlp, zlp, clp in ((corr1_nlp, corr1_zlp, corr1_clp), (corr2_nlp, corr2_zlp, corr2_clp)):
        r2_term = _fjprep(nlp, nij, zetc, zetb, r, zlp, clp)
        for n in range(1, nij + 1):
            for lc in range(1, len(r2[n])):
                for lb in range(1, len(r2[n][lc])):
                    r2[n][lc][lb] += r2_term[n][lc][lb]

    g1 = [0.0] * 6
    g1[1] = r1[1][1] * _FPI + r2[1][1][1] * _FPI
    if kd > 1:
        g1[2] = (acz * r1[1][1] + zfn * r1[2][2]) * _FPI + (acz * r2[1][1][1] + zfn * r2[2][1][2]) * _FPI
        g1[3] = (acz * r1[1][1] + zfn * r1[2][2]) * _FPI + (acz * r2[1][1][1] + zfn * r2[2][2][1]) * _FPI
        acb1 = acz * acz
        acb5 = 2.0 * acz * zfn
        aczz = acz * zfn
        g1[4] = (acb1 * r1[1][1] + acb5 * r1[2][2] + _A3 * r1[3][1] + _A4 * r1[3][3]) * _FPI + (
            acb1 * r2[1][1][1] + aczz * (r2[2][2][1] + r2[2][1][2]) + r2[3][2][2]
        ) * _FPI
        g1[5] = (_A3 * (r1[3][1] - r1[3][3])) * _FPI
    return g1


def om1_ppecp_local(ni, nj, rij, zeta_s):
    """
    Native OM1 PPECP core-valence pseudopotential for the current H/C/N/O/F
    ``s/sp`` scope. Returns the local ``CORPP(4,2)`` block in eV before the
    outer COROM ``FKO`` scaling.
    """
    corpp = torch.zeros((4, 2), dtype=torch.float64)
    for column, shell_z, ecp_z, acz, zfn in ((0, ni, nj, rij, -1.0), (1, nj, ni, -rij, 1.0)):
        if ecp_z <= 2:
            continue
        if ecp_z not in OM1_FIRST_ROW_ECP:
            raise NotImplementedError(
                f"OM1 PPECP is currently restricted to H/C/N/O/F; got ECP center Z={ecp_z}"
            )
        shell_type, exponents, coeff_s, coeff_p = _basis_primitives(shell_z, float(zeta_s[int(shell_z)]))
        kd = 1 if shell_type == 0 else 4
        ecp = OM1_FIRST_ROW_ECP[ecp_z]

        for ig, zetc in enumerate(exponents):
            for jg in range(ig + 1):
                zetb = exponents[jg]
                g1 = _primitive_g1(kd, acz, zfn, zetc, zetb, rij, ecp)
                corpp[0, column] += coeff_s[ig] * coeff_s[jg] * g1[1]
                if kd > 1:
                    corpp[1, column] += coeff_s[ig] * coeff_p[jg] * g1[2]
                    corpp[2, column] += coeff_p[ig] * coeff_p[jg] * g1[4]
                    corpp[3, column] += coeff_p[ig] * coeff_p[jg] * g1[5]
                if ig != jg:
                    corpp[0, column] += coeff_s[jg] * coeff_s[ig] * g1[1]
                    if kd > 1:
                        corpp[1, column] += coeff_s[jg] * coeff_p[ig] * g1[3]
                        corpp[2, column] += coeff_p[jg] * coeff_p[ig] * g1[4]
                        corpp[3, column] += coeff_p[jg] * coeff_p[ig] * g1[5]
    return corpp * ev

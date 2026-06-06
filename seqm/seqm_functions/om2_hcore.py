import torch

from .om1_overlap import diatom_overlap_matrix_OM1, diatom_resonance_matrix_OM1, omx_betom_terms
from .om1_pair_backend import omx_pair_hcore_terms


def _atom_orbs(z):
    z = int(z)
    if z <= 0:
        return 0
    return 1 if z == 1 else 4


def _atom_block_start(atom_index):
    return int(atom_index) * 4


def _reshape_blocks_to_square(blocks, nmol, molsize, orb_dim=4):
    return (
        blocks.reshape(nmol, molsize, molsize, orb_dim, orb_dim)
        .transpose(2, 3)
        .reshape(nmol, orb_dim * molsize, orb_dim * molsize)
    )


def _reshape_square_to_blocks(square, nmol, molsize, orb_dim=4):
    return (
        square.reshape(nmol, molsize, orb_dim, molsize, orb_dim)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, orb_dim, orb_dim)
    )


def _sym_get(mat, r, c):
    """
    Read a symmetric AO matrix stored in only one triangle.

    Your pair construction stores idxi < idxj blocks only. Therefore, when
    reading the reverse direction, use mat[c, r].
    """
    return mat[r, c] if r <= c else mat[c, r]


def _extract_atom_pair_vec(mat, atom_a, atom_k, aorbs, korbs, orb_dim=4):
    """
    Return Fortran-style 16-vector for pair A-K.

    Layout matches BETORT:
        index = mu * 4 + nu

    where:
        mu = orbital on atom A
        nu = orbital on atom K

    Thus:
        0  = sA-sK
        1  = sA-pxK
        2  = sA-pyK
        3  = sA-pzK
        4  = pxA-sK
        8  = pyA-sK
        12 = pzA-sK
    """
    out = torch.zeros(16, dtype=mat.dtype, device=mat.device)

    a0 = atom_a * orb_dim
    k0 = atom_k * orb_dim

    for mu in range(aorbs):
        for nu in range(korbs):
            out[mu * 4 + nu] = _sym_get(mat, a0 + mu, k0 + nu)

    return out


def _build_cor_table(molecule, pair_core_semi, tables):
    """
    Build OM2 COR table used by BETORT.

    Fortran source:

        COR(IA,J) = USS(NI)+CORE(1,1)
        COR(JA,I) = USS(NJ)+CORE(1,2)

        TEMP = UPP(NI)+(CORE(3,1)+2*CORE(4,1))/3
        COR(IA+1,J) = TEMP
        COR(IA+2,J) = TEMP
        COR(IA+3,J) = TEMP

    Important:
        core_semi must be CORE before COROM adds PEN/CORPP/VALPP.
    """
    nmol, molsize = molecule.species.shape
    dtype = molecule.coordinates.dtype
    device = molecule.coordinates.device

    cor = torch.zeros((nmol, molsize * 4, molsize), dtype=dtype, device=device)

    for p in range(molecule.ni.shape[0]):
        mol = int(molecule.pair_molid[p].item())

        atom_i = int(molecule.idxi[p].item())
        atom_j = int(molecule.idxj[p].item())

        ni = int(molecule.ni[p].item())
        nj = int(molecule.nj[p].item())

        core_semi = pair_core_semi[p]

        row_i = _atom_block_start(atom_i)
        row_j = _atom_block_start(atom_j)

        cor[mol, row_i + 0, atom_j] = tables["U_ss"][ni] + core_semi[0, 0]
        cor[mol, row_j + 0, atom_i] = tables["U_ss"][nj] + core_semi[0, 1]

        if ni > 1:
            temp_i = tables["U_pp"][ni] + (core_semi[2, 0] + 2.0 * core_semi[3, 0]) / 3.0
            cor[mol, row_i + 1, atom_j] = temp_i
            cor[mol, row_i + 2, atom_j] = temp_i
            cor[mol, row_i + 3, atom_j] = temp_i

        if nj > 1:
            temp_j = tables["U_pp"][nj] + (core_semi[2, 1] + 2.0 * core_semi[3, 1]) / 3.0
            cor[mol, row_j + 1, atom_i] = temp_j
            cor[mol, row_j + 2, atom_i] = temp_j
            cor[mol, row_j + 3, atom_i] = temp_j

    return cor


def betort_torch(H, S, B, COR, species, gval1, gval2):
    """
    Literal but simplified PyTorch port of Fortran BETORT.

    H is accepted for API compatibility but not used.
    S, B are padded AO square matrices with 4 slots per atom.
    COR has shape:
        (nmol, molsize * 4, molsize)

    Returns:
        HO, same shape as H/S/B.

    The Fortran result is computed for I > J lower-triangle elements, but this
    routine stores the transposed upper-triangle block convention used by the
    rest of the PyTorch Hcore builder.
    """
    nmol, _, _ = S.shape
    dtype = S.dtype
    device = S.device

    HO = torch.zeros_like(S)

    for mol in range(nmol):
        natoms = int((species[mol] > 0).sum().item())

        for ia in range(1, natoms):
            ni = int(species[mol, ia].item())
            iorbs = _atom_orbs(ni)
            ia0 = ia * 4

            for ja in range(ia):
                nj = int(species[mol, ja].item())
                jorbs = _atom_orbs(nj)
                ja0 = ja * 4

                case1 = iorbs == 1 and jorbs == 1
                case2 = iorbs == 1 and jorbs >= 4
                case3 = iorbs >= 4 and jorbs == 1

                ts1 = torch.zeros(16, dtype=dtype, device=device)
                ts2 = torch.zeros(16, dtype=dtype, device=device)

                # Fortran:
                #   FT1 = PT25  *(FTS1(NI)+FTS1(NJ))
                #   FT2 = PT625 *(FTS2(NI)+FTS2(NJ))
                #   PT625 = 0.0625
                ft1 = 0.25 * (gval1[ni] + gval1[nj])
                ft2 = 0.0625 * (gval2[ni] + gval2[nj])

                for k in range(natoms):
                    if k == ia or k == ja:
                        continue

                    nk = int(species[mol, k].item())
                    korbs = _atom_orbs(nk)
                    if korbs == 0:
                        continue

                    ka0 = k * 4

                    sik = _extract_atom_pair_vec(S[mol], ia, k, iorbs, korbs)
                    bik = _extract_atom_pair_vec(B[mol], ia, k, iorbs, korbs)

                    sjk = _extract_atom_pair_vec(S[mol], ja, k, jorbs, korbs)
                    bjk = _extract_atom_pair_vec(B[mol], ja, k, jorbs, korbs)

                    hi = torch.zeros(4, dtype=dtype, device=device)
                    hj = torch.zeros(4, dtype=dtype, device=device)
                    hk = torch.zeros(4, dtype=dtype, device=device)

                    hi[0] = COR[mol, ia0 + 0, k]
                    hj[0] = COR[mol, ja0 + 0, k]
                    hk[0] = COR[mol, ka0 + 0, ia] + COR[mol, ka0 + 0, ja]

                    if iorbs >= 4:
                        hi[1] = COR[mol, ia0 + 1, k]
                        hi[2] = hi[1]
                        hi[3] = hi[1]

                    if jorbs >= 4:
                        hj[1] = COR[mol, ja0 + 1, k]
                        hj[2] = hj[1]
                        hj[3] = hj[1]

                    if korbs >= 4:
                        hk[1] = COR[mol, ka0 + 1, ia] + COR[mol, ka0 + 1, ja]
                        hk[2] = hk[1]
                        hk[3] = hk[1]

                    # S(I)-S(J)
                    for l in range(korbs):
                        ts1[0] = ts1[0] + sik[l] * bjk[l] + bik[l] * sjk[l]
                        ts2[0] = ts2[0] + sik[l] * sjk[l] * (hi[0] + hj[0] - hk[l])

                    if case1:
                        continue

                    # S(I)-P(J)
                    if case2:
                        for l in range(korbs):
                            ts1[1] = ts1[1] + sik[l] * bjk[l + 4] + bik[l] * sjk[l + 4]
                            ts1[2] = ts1[2] + sik[l] * bjk[l + 8] + bik[l] * sjk[l + 8]
                            ts1[3] = ts1[3] + sik[l] * bjk[l + 12] + bik[l] * sjk[l + 12]

                            ts2[1] = ts2[1] + sik[l] * sjk[l + 4] * (hi[0] + hj[1] - hk[l])
                            ts2[2] = ts2[2] + sik[l] * sjk[l + 8] * (hi[0] + hj[2] - hk[l])
                            ts2[3] = ts2[3] + sik[l] * sjk[l + 12] * (hi[0] + hj[3] - hk[l])

                    # P(I)-S(J)
                    elif case3:
                        for l in range(korbs):
                            ts1[4] = ts1[4] + sik[l + 4] * bjk[l] + bik[l + 4] * sjk[l]
                            ts1[8] = ts1[8] + sik[l + 8] * bjk[l] + bik[l + 8] * sjk[l]
                            ts1[12] = ts1[12] + sik[l + 12] * bjk[l] + bik[l + 12] * sjk[l]

                            ts2[4] = ts2[4] + sik[l + 4] * sjk[l] * (hi[1] + hj[0] - hk[l])
                            ts2[8] = ts2[8] + sik[l + 8] * sjk[l] * (hi[2] + hj[0] - hk[l])
                            ts2[12] = ts2[12] + sik[l + 12] * sjk[l] * (hi[3] + hj[0] - hk[l])

                    # S/P(I)-S/P(J)
                    else:
                        for l in range(korbs):
                            # TS1
                            ts1[1] = ts1[1] + sik[l] * bjk[l + 4] + bik[l] * sjk[l + 4]
                            ts1[2] = ts1[2] + sik[l] * bjk[l + 8] + bik[l] * sjk[l + 8]
                            ts1[3] = ts1[3] + sik[l] * bjk[l + 12] + bik[l] * sjk[l + 12]

                            ts1[4] = ts1[4] + sik[l + 4] * bjk[l] + bik[l + 4] * sjk[l]
                            ts1[5] = ts1[5] + sik[l + 4] * bjk[l + 4] + bik[l + 4] * sjk[l + 4]
                            ts1[6] = ts1[6] + sik[l + 4] * bjk[l + 8] + bik[l + 4] * sjk[l + 8]
                            ts1[7] = ts1[7] + sik[l + 4] * bjk[l + 12] + bik[l + 4] * sjk[l + 12]

                            ts1[8] = ts1[8] + sik[l + 8] * bjk[l] + bik[l + 8] * sjk[l]
                            ts1[9] = ts1[9] + sik[l + 8] * bjk[l + 4] + bik[l + 8] * sjk[l + 4]
                            ts1[10] = ts1[10] + sik[l + 8] * bjk[l + 8] + bik[l + 8] * sjk[l + 8]
                            ts1[11] = ts1[11] + sik[l + 8] * bjk[l + 12] + bik[l + 8] * sjk[l + 12]

                            ts1[12] = ts1[12] + sik[l + 12] * bjk[l] + bik[l + 12] * sjk[l]
                            ts1[13] = ts1[13] + sik[l + 12] * bjk[l + 4] + bik[l + 12] * sjk[l + 4]
                            ts1[14] = ts1[14] + sik[l + 12] * bjk[l + 8] + bik[l + 12] * sjk[l + 8]
                            ts1[15] = ts1[15] + sik[l + 12] * bjk[l + 12] + bik[l + 12] * sjk[l + 12]

                            # TS2
                            ts2[1] = ts2[1] + sik[l] * sjk[l + 4] * (hi[0] + hj[1] - hk[l])
                            ts2[2] = ts2[2] + sik[l] * sjk[l + 8] * (hi[0] + hj[2] - hk[l])
                            ts2[3] = ts2[3] + sik[l] * sjk[l + 12] * (hi[0] + hj[3] - hk[l])

                            ts2[4] = ts2[4] + sik[l + 4] * sjk[l] * (hi[1] + hj[0] - hk[l])
                            ts2[5] = ts2[5] + sik[l + 4] * sjk[l + 4] * (hi[1] + hj[1] - hk[l])
                            ts2[6] = ts2[6] + sik[l + 4] * sjk[l + 8] * (hi[1] + hj[2] - hk[l])
                            ts2[7] = ts2[7] + sik[l + 4] * sjk[l + 12] * (hi[1] + hj[3] - hk[l])

                            ts2[8] = ts2[8] + sik[l + 8] * sjk[l] * (hi[2] + hj[0] - hk[l])
                            ts2[9] = ts2[9] + sik[l + 8] * sjk[l + 4] * (hi[2] + hj[1] - hk[l])
                            ts2[10] = ts2[10] + sik[l + 8] * sjk[l + 8] * (hi[2] + hj[2] - hk[l])
                            ts2[11] = ts2[11] + sik[l + 8] * sjk[l + 12] * (hi[2] + hj[3] - hk[l])

                            ts2[12] = ts2[12] + sik[l + 12] * sjk[l] * (hi[3] + hj[0] - hk[l])
                            ts2[13] = ts2[13] + sik[l + 12] * sjk[l + 4] * (hi[3] + hj[1] - hk[l])
                            ts2[14] = ts2[14] + sik[l + 12] * sjk[l + 8] * (hi[3] + hj[2] - hk[l])
                            ts2[15] = ts2[15] + sik[l + 12] * sjk[l + 12] * (hi[3] + hj[3] - hk[l])

                def _put(mu, nu, q):
                    # Fortran computes H(I_mu, J_nu), with I > J.
                    # Your storage keeps the upper triangle, so store H(J_nu, I_mu).
                    HO[mol, ja0 + nu, ia0 + mu] = -ft1 * ts1[q] + ft2 * ts2[q]

                # Write HO block for atom pair ia-ja.
                _put(0, 0, 0)

                if case1:
                    continue

                if case2:
                    # I has only s, J has s,p.
                    # Fortran: H(I_s, J_px/py/pz)
                    _put(0, 1, 1)
                    _put(0, 2, 2)
                    _put(0, 3, 3)

                elif case3:
                    # I has s,p, J has only s.
                    # Fortran: H(I_px/py/pz, J_s)
                    _put(1, 0, 4)
                    _put(2, 0, 8)
                    _put(3, 0, 12)

                else:
                    for mu in range(4):
                        for nu in range(4):
                            _put(mu, nu, mu * 4 + nu)

    return HO


def _build_om_hcore(molecule, doTETCI=True, method="OM2", use_cor=True):
    nmol, molsize = molecule.species.shape
    orb_dim = 4
    device = molecule.coordinates.device
    dtype = molecule.coordinates.dtype

    npairs = molecule.xij.size(0)
    nblocks = nmol * molsize * molsize

    H_blocks = torch.zeros((nblocks, orb_dim, orb_dim), dtype=dtype, device=device)
    S_blocks = torch.zeros_like(H_blocks)
    B_blocks = torch.zeros_like(H_blocks)
    w = torch.zeros((npairs, 10, 10), dtype=dtype, device=device)

    def _table(name):
        return molecule.packpar.p[:, molecule.packpar.required_list.index(name)]

    tables = {name: _table(name) for name in molecule.packpar.required_list}

    # One-center diagonal terms.
    for orb, key in enumerate(["U_ss", "U_pp", "U_pp", "U_pp"]):
        H_blocks[molecule.maskd, orb, orb] = molecule.parameters[key].to(dtype=dtype, device=device)

    pair_core_semi = torch.zeros((npairs, 4, 2), dtype=dtype, device=device) if use_cor else None

    resonance_tables = {
        name: tables[name]
        for name in [
            "beta_s",
            "beta_p",
            "beta_pi",
            "beta_sh",
            "beta_ph",
            "alpha_s",
            "alpha_p",
            "alpha_pi",
            "alpha_s_h",
            "alpha_p_h",
        ]
    }

    for p in range(npairs):
        ni = int(molecule.ni[p].item())
        nj = int(molecule.nj[p].item())

        idxi = molecule.idxi[p : p + 1]
        idxj = molecule.idxj[p : p + 1]

        rij = molecule.rij[p : p + 1]
        xij = molecule.xij[p : p + 1]

        zeta_i_s = molecule.parameters["zeta_s"][idxi]
        zeta_j_s = molecule.parameters["zeta_s"][idxj]

        # Local BETOM terms for COROM/VALPOT.
        s_local, t_local = omx_betom_terms(
            molecule.ni[p : p + 1], molecule.nj[p : p + 1], rij, tables, zeta_i_s, zeta_j_s
        )

        pair = omx_pair_hcore_terms(
            method,
            ni,
            nj,
            xij[0],
            float(rij.item()),
            tables["zeta_s"],
            tables["g_ss"],
            molecule.const.tore,
            s_local[0],
            t_local[0],
            tables["U_ss"],
            tables["U_pp"],
            tables["fval1"],
            tables["fval2"],
            om2_tables=tables,
        )

        w[p] = pair["w"]

        if use_cor:
            pair_core_semi[p] = pair["core_semi"]

        # Core-electron attraction contributions to atom i and atom j blocks.
        H_blocks.index_add_(0, molecule.maskd[idxi], pair["e1b"].unsqueeze(0))
        H_blocks.index_add_(0, molecule.maskd[idxj], pair["e2a"].unsqueeze(0))

        # Rotated overlap S and resonance B for BETORT.
        #
        # This follows your existing code path. The important point is that these
        # must match Fortran:
        #
        #   CALL ROTBET(..., SIJ, ..., S)
        #   CALL ROTBET(..., T,   ..., B)
        #
        zeta_pair_i = torch.stack(
            [molecule.parameters["zeta_s"][idxi], molecule.parameters["zeta_p"][idxi]], dim=1
        )
        zeta_pair_j = torch.stack(
            [molecule.parameters["zeta_s"][idxj], molecule.parameters["zeta_p"][idxj]], dim=1
        )

        s_pair = diatom_overlap_matrix_OM1(
            molecule.ni[p : p + 1], molecule.nj[p : p + 1], xij, rij, zeta_pair_i, zeta_pair_j
        )[0]

        b_pair = diatom_resonance_matrix_OM1(
            molecule.ni[p : p + 1], molecule.nj[p : p + 1], xij, rij, resonance_tables
        )[0]

        S_blocks[molecule.mask[p : p + 1]] = s_pair
        B_blocks[molecule.mask[p : p + 1]] = b_pair

    H_sq = _reshape_blocks_to_square(H_blocks, nmol, molsize, orb_dim)
    S_sq = _reshape_blocks_to_square(S_blocks, nmol, molsize, orb_dim)
    B_sq = _reshape_blocks_to_square(B_blocks, nmol, molsize, orb_dim)

    has_three_or_more_atoms = bool(((molecule.species > 0).sum(dim=1) > 2).any().item())

    if doTETCI and has_three_or_more_atoms:
        if use_cor:
            COR = _build_cor_table(molecule, pair_core_semi, tables)
            HO_sq = betort_torch(H_sq, S_sq, B_sq, COR, molecule.species, tables["gval1"], tables["gval2"])
            H_sq = H_sq + B_sq + HO_sq
        else:
            HO_sq = betor3_torch(S_sq, B_sq, molecule.species, tables["gval1"])
            H_sq = H_sq + B_sq + HO_sq
    else:
        H_sq = H_sq + B_sq

    H_blocks = _reshape_square_to_blocks(H_sq, nmol, molsize, orb_dim)

    return H_blocks, w, None, None, None, None


def build_om2_hcore(molecule, doTETCI=True):
    return _build_om_hcore(molecule, doTETCI=doTETCI, method="OM2", use_cor=True)


def build_om3_hcore(molecule, doTETCI=True):
    return _build_om_hcore(molecule, doTETCI=doTETCI, method="OM3", use_cor=False)


def betor3_torch(S, B, species, gval1, cuts=1.0e-12):
    """
    PyTorch port of Fortran BETOR3 for OM3.

    Fortran:
        CALL BETOR3(HO, S, B, B, LM2, LM4, IZERO)

    Energy-evaluation behavior:
        H_ij = -FT1 * TS1_ij

    where:
        FT1 = 0.25 * (FTS1(NI) + FTS1(NJ))

    This is the first-perturbation-sum part of BETORT only.
    There is no COR table, no FTS2/gval2, and no TS2 contribution.

    Parameters
    ----------
    S, B : torch.Tensor
        AO square matrices with shape (nmol, nao, nao), using 4 AO slots per atom.
    species : torch.Tensor
        Atomic numbers, shape (nmol, molsize). Padding atoms should be <= 0.
    gval1 : torch.Tensor
        OM3 FTS1/gval1 table indexed by atomic number.
    cuts : float or None
        Fortran-style ICUTS overlap-product cutoff.
        If None, no cutoff is applied.
        Default 1.0e-12 matches SMALLS when ICUTS <= 0.

    Returns
    -------
    HO : torch.Tensor
        Orthogonalization correction matrix, same shape as S/B.

    Notes
    -----
    Fortran computes lower-triangle H(I_mu, J_nu) for I > J.
    This routine follows your existing PyTorch convention and stores the
    transposed upper-triangle block: HO[J_nu, I_mu].
    """
    nmol, _, _ = S.shape
    dtype = S.dtype
    device = S.device

    HO = torch.zeros_like(S)

    for mol in range(nmol):
        natoms = int((species[mol] > 0).sum().item())

        for ia in range(1, natoms):
            ni = int(species[mol, ia].item())
            iorbs = _atom_orbs(ni)
            ia0 = ia * 4

            for ja in range(ia):
                nj = int(species[mol, ja].item())
                jorbs = _atom_orbs(nj)
                ja0 = ja * 4

                case1 = iorbs == 1 and jorbs == 1
                case2 = iorbs == 1 and jorbs >= 4
                case3 = iorbs >= 4 and jorbs == 1

                ts1 = torch.zeros(16, dtype=dtype, device=device)

                # Fortran:
                #   FT1 = PT25 * (FTS1(NI) + FTS1(NJ))
                ft1 = 0.25 * (gval1[ni] + gval1[nj])

                for k in range(natoms):
                    if k == ia or k == ja:
                        continue

                    nk = int(species[mol, k].item())
                    korbs = _atom_orbs(nk)
                    if korbs == 0:
                        continue

                    # Fortran cutoff:
                    #
                    #   IF(ICUTS.GE.0) THEN
                    #      ...
                    #      IF((S(IKSS)*S(JKSS)).LT.CUTS) GO TO 90
                    #   ENDIF
                    #
                    # IKSS and JKSS are the s-s overlap elements for I-K and J-K.
                    # The Fortran does not use ABS here.
                    if cuts is not None:
                        sik_ss = _extract_atom_pair_vec(S[mol], ia, k, iorbs, korbs)[0]
                        sjk_ss = _extract_atom_pair_vec(S[mol], ja, k, jorbs, korbs)[0]

                        if (sik_ss * sjk_ss).detach().item() < cuts:
                            continue

                    sik = _extract_atom_pair_vec(S[mol], ia, k, iorbs, korbs)
                    bik = _extract_atom_pair_vec(B[mol], ia, k, iorbs, korbs)

                    sjk = _extract_atom_pair_vec(S[mol], ja, k, jorbs, korbs)
                    bjk = _extract_atom_pair_vec(B[mol], ja, k, jorbs, korbs)

                    # S(I)-S(J)
                    for l in range(korbs):
                        ts1[0] = ts1[0] + sik[l] * bjk[l] + bik[l] * sjk[l]

                    if case1:
                        continue

                    if case2:
                        # S(I)-P(J)
                        for l in range(korbs):
                            ts1[1] = ts1[1] + sik[l] * bjk[l + 4] + bik[l] * sjk[l + 4]
                            ts1[2] = ts1[2] + sik[l] * bjk[l + 8] + bik[l] * sjk[l + 8]
                            ts1[3] = ts1[3] + sik[l] * bjk[l + 12] + bik[l] * sjk[l + 12]

                    elif case3:
                        # P(I)-S(J)
                        for l in range(korbs):
                            ts1[4] = ts1[4] + sik[l + 4] * bjk[l] + bik[l + 4] * sjk[l]
                            ts1[8] = ts1[8] + sik[l + 8] * bjk[l] + bik[l + 8] * sjk[l]
                            ts1[12] = ts1[12] + sik[l + 12] * bjk[l] + bik[l + 12] * sjk[l]

                    else:
                        # S/P(I)-S/P(J)
                        for l in range(korbs):
                            # S(I)-P(J)
                            ts1[1] = ts1[1] + sik[l] * bjk[l + 4] + bik[l] * sjk[l + 4]
                            ts1[2] = ts1[2] + sik[l] * bjk[l + 8] + bik[l] * sjk[l + 8]
                            ts1[3] = ts1[3] + sik[l] * bjk[l + 12] + bik[l] * sjk[l + 12]

                            # P(I)-S(J), P(I)-P(J)
                            ts1[4] = ts1[4] + sik[l + 4] * bjk[l] + bik[l + 4] * sjk[l]
                            ts1[5] = ts1[5] + sik[l + 4] * bjk[l + 4] + bik[l + 4] * sjk[l + 4]
                            ts1[6] = ts1[6] + sik[l + 4] * bjk[l + 8] + bik[l + 4] * sjk[l + 8]
                            ts1[7] = ts1[7] + sik[l + 4] * bjk[l + 12] + bik[l + 4] * sjk[l + 12]

                            ts1[8] = ts1[8] + sik[l + 8] * bjk[l] + bik[l + 8] * sjk[l]
                            ts1[9] = ts1[9] + sik[l + 8] * bjk[l + 4] + bik[l + 8] * sjk[l + 4]
                            ts1[10] = ts1[10] + sik[l + 8] * bjk[l + 8] + bik[l + 8] * sjk[l + 8]
                            ts1[11] = ts1[11] + sik[l + 8] * bjk[l + 12] + bik[l + 8] * sjk[l + 12]

                            ts1[12] = ts1[12] + sik[l + 12] * bjk[l] + bik[l + 12] * sjk[l]
                            ts1[13] = ts1[13] + sik[l + 12] * bjk[l + 4] + bik[l + 12] * sjk[l + 4]
                            ts1[14] = ts1[14] + sik[l + 12] * bjk[l + 8] + bik[l + 12] * sjk[l + 8]
                            ts1[15] = ts1[15] + sik[l + 12] * bjk[l + 12] + bik[l + 12] * sjk[l + 12]

                def _put(mu, nu, q):
                    # Fortran computes H(I_mu, J_nu), with I > J.
                    # Your storage keeps the upper triangle, so store H(J_nu, I_mu).
                    HO[mol, ja0 + nu, ia0 + mu] = -ft1 * ts1[q]

                # S(I)-S(J)
                _put(0, 0, 0)

                if case1:
                    continue

                if case2:
                    # I has only s, J has s,p.
                    # Fortran: H(I_s, J_px/py/pz)
                    _put(0, 1, 1)
                    _put(0, 2, 2)
                    _put(0, 3, 3)

                elif case3:
                    # I has s,p, J has only s.
                    # Fortran: H(I_px/py/pz, J_s)
                    _put(1, 0, 4)
                    _put(2, 0, 8)
                    _put(3, 0, 12)

                else:
                    # Both atoms have s,p.
                    for mu in range(4):
                        for nu in range(4):
                            _put(mu, nu, mu * 4 + nu)

    return HO

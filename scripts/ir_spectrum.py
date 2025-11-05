#!/usr/bin/env python3

# Calculate IR spectrum from dipole time series.
# The IR spectrum can be calculated from dipole autocorrelation function.
# A quantum correction factor requires multiplying intensity with square of frequency 
# See: https://doi.org/10.1039/B924048A , Eq. 2
#      https://doi.org/10.1039/C1FD00004G , Eq. 4
#      J. Chem. Phys. 121, 3973–3983 (2004) https://doi.org/10.1063/1.1774986, Eqs. 27, 28, 31
#            quantum-corrected line shape I_QC(w) = Q_QC(w))I_cl(w) satisfies the principle of detailed balance. 


import argparse
from collections import defaultdict

import numpy as np

# ---------- constants (SI and AU helpers) ----------
a0_to_ang = 0.529177210903
au_to_fs = 2.4188843265857e-2
ang_per_fs_to_au = au_to_fs / a0_to_ang
ea0 = 8.478353552e-30          # C·m per 1 a.u. dipole
debye_per_au = 2.541746
c_cm_fs = 2.99792458e-5        # cm/fs

# ---------- defaults ----------
MASSES_AMU = defaultdict(lambda: 12.0107, **{
    "H":1.00784,"C":12.0107,"N":14.0067,"O":15.999,"F":18.998,
    "P":30.9738,"S":32.065,"Cl":35.453,"Na":22.9898,"K":39.0983,
    "Li":6.941,"Si":28.0855,"Ca":40.078
})

# ---------- utilities ----------
def window_normalization(win): return (win**2).mean()

def unit_vel_to_au(units):
    u = units.lower()
    if u == "bohr/au": return 1.0
    if u == "a/fs":    return ang_per_fs_to_au
    if u == "a/ps":    return ang_per_fs_to_au * 1e-3
    if u == "m/s":     return ang_per_fs_to_au * 1e-5
    raise ValueError(f"Unknown --vel-units: {units}")

def unit_dip_to_au(units):
    u = units.lower()
    if u in ("au","a.u.","atomic"): return 1.0
    if u in ("debye","d"):          return 1.0 / debye_per_au
    if u in ("c*m","c·m","c m"):    return 1.0 / ea0
    raise ValueError(f"Unknown --dip-units: {units}")

def dt_to_au(dt_value, dt_units):
    u = dt_units.lower()
    if u == "au": return float(dt_value)
    if u == "fs": return float(dt_value) / au_to_fs
    raise ValueError(f"Unknown --dt-units: {dt_units}")

def load_dipoles(path, has_time="auto"):
    data = np.loadtxt(path)
    if data.ndim == 1: data = data[None, :]
    if has_time == "auto": has_time = (data.shape[1] == 4)
    return data[:,1:4] if has_time else data[:,0:3]  # [T,3]

def detrend_linear(X, dt):
    X = X - X.mean(axis=0, keepdims=True)
    t = np.arange(X.shape[0]) * dt
    A = np.vstack([t, np.ones_like(t)]).T
    Xc = X.copy()
    for k in range(X.shape[1]):
        a, b = np.linalg.lstsq(A, X[:,k], rcond=None)[0]
        Xc[:,k] -= a*t + b
    return Xc

def exponential_window(n, ts_au, width_cm=5.0):
    # w(t)=exp(-t/τ) with τ = 1/(π * width_cm * c * au2fs)
    tau = 1.0 / (np.pi * width_cm * c_cm_fs * au_to_fs)
    t = (np.arange(n, dtype=float)) * ts_au
    return np.exp(-t / tau)

def spectrum_from_time_series(X, dt, window, exp_width_cm, zp_factor):
    T = X.shape[0]
    if window == "hann":
        win = np.hanning(T)
    elif window == "exp":
        win = exponential_window(T, dt, exp_width_cm)
    else:
        win = np.ones(T)

    ENBW = window_normalization(win)
    Xw = X * win[:,None]
    Tpad = int(2**np.ceil(np.log2(T * max(1, zp_factor))))
    if Tpad > T:
        Xw = np.pad(Xw, ((0, Tpad-T),(0,0)))
    F = np.fft.rfft(Xw, axis=0)
    S = np.sum(np.abs(F)**2, axis=1).real
    S = S / ENBW * (dt / T)             # PSD-like, relative shape
    freq = np.fft.rfftfreq(Xw.shape[0], d=dt)  # au^-1
    return freq, S

def ir_from_dipoles(M, dt, use_derivative, window, exp_width_cm, zp_factor, max_wn):
    M = detrend_linear(M, dt)
    if use_derivative:
        Mdot = np.gradient(M, dt, axis=0)
        freq, S = spectrum_from_time_series(Mdot, dt, window, exp_width_cm, zp_factor)
    else:
        freq, S = spectrum_from_time_series(M, dt, window, exp_width_cm, zp_factor)
        S *= (2*np.pi*freq)**2

    wn = freq / (au_to_fs * c_cm_fs)
    mask = (wn >= 0) & (wn <= max_wn)
    I = S[mask]; I /= I.max() if I.max() > 0 else 1.0
    return wn[mask], I

def parse_xyz_with_velocities(path, vel_cols):
    elems = None; frames = []
    v1, v2, v3 = vel_cols
    with open(path, "r") as f:
        while True:
            head = f.readline()
            if not head: break
            try:
                n = int(head.strip())
            except ValueError:
                break
            _comment = f.readline()
            v_list = []; frame_elems = []
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) < 7:
                    raise ValueError("Expect: Sym x y z vx vy vz per atom line.")
                frame_elems.append(parts[0])
                v_list.append([float(parts[v1]), float(parts[v2]), float(parts[v3])])
            if elems is None:
                elems = frame_elems
            elif frame_elems != elems:
                raise ValueError("Atom ordering changed between frames.")
            frames.append(np.array(v_list, float))
    return elems, np.stack(frames, axis=0)  # [T,N,3]

def masses_from_elems(elems):
    return np.array([MASSES_AMU[e] for e in elems])

def remove_com(V, masses):
    msum = masses.sum()
    com = (V * masses[None,:,None]).sum(axis=1) / msum
    return V - com[:,None,:]

def vdos_from_vel(V, dt, masses, window, exp_width_cm, zp_factor, max_wn, mass_weight, remove_com_flag):
    if remove_com_flag:
        V = remove_com(V, masses)
    T, N, _ = V.shape
    win = np.hanning(T) if window=="hann" else (exponential_window(T, dt, exp_width_cm) if window=="exp" else np.ones(T))
    ENBW = window_normalization(win)
    wV = V * win[:,None,None]
    if mass_weight:
        wV = wV * np.sqrt(masses)[None,:,None]

    Tpad = int(2**np.ceil(np.log2(T * max(1, zp_factor))))
    if Tpad > T:
        wV = np.pad(wV, ((0, Tpad-T),(0,0),(0,0)))
    F = np.fft.rfft(wV, axis=0)
    S = np.sum(np.abs(F)**2, axis=(1,2)).real
    S = S / ENBW * (dt / T)
    freq = np.fft.rfftfreq(wV.shape[0], d=dt)
    wn = freq / (au_to_fs * c_cm_fs)
    mask = (wn >= 0) & (wn <= max_wn)
    I = S[mask]; I /= I.max() if I.max() > 0 else 1.0
    return wn[mask], I

def main():
    ap = argparse.ArgumentParser(description="Compute IR (from dipoles) and/or VDOS (from velocities) and optionally plot.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap = argparse.ArgumentParser(
    description=(
        "Compute IR (dipole) and/or VDOS (velocity) spectra from time series.\n"
        "It reads either a dipole file (3–4 columns) or an extended XYZ with velocities."
    ),
    epilog=(
        "TIPS & NOTES:\n"
        "  • For smoother spectra, increase --zp (zero-padding factor, e.g. 5 or 10).\n"
        "  • The exponential window width (--exp-width, cm⁻¹) controls damping applied to exponential window function:\n"
        "      smaller values → slower decay, higher resolution.\n"
        "  • Use --use-deriv to get IR spectra from d (dipole)/dt (current autocorr) instead of raw dipoles.\n"
        "  • If the spectrum looks noisy, try window='hann'\n"
        "  • Units:\n"
        "      Dipoles: au (default), Debye, or C*m.\n"
        "      Velocities: bohr/au (default), A/fs, A/ps, m/s.\n"
        "  • COM drift removal (--no-remove-com) may not be needed if velocities already centered.\n"
        "  • Save spectra via --out-prefix; use --plot or --plot out.png to visualize results.\n"
        "  • All output files have two columns: wavenumber[cm⁻¹] and normalized intensity."
        "  • Use --skip N to discard initial frames (e.g. equilibration) before analysis."

    ),
    formatter_class=argparse.RawTextHelpFormatter
)
    # Common sampling
    ap.add_argument("--dt", type=float, required=True, help="Time step value")
    ap.add_argument("--dt-units", choices=["fs","au"], default="au", help="Units for --dt")
    ap.add_argument("--max-wn", type=float, default=800.0, help="Max wavenumber (cm^-1)")
    ap.add_argument("--window", choices=["exp","hann","rect"], default="exp", help="Window type")
    ap.add_argument("--exp-width", type=float, default=5.0, help="Exponential window width (cm^-1)")
    ap.add_argument("--zp", type=float, default=4.0, help="Zero-padding factor (>=1)")
    ap.add_argument("--skip", type=int, default=0,
    help="Number of initial frames (time steps) to skip before computing the spectrum")


    # Dipoles
    ap.add_argument("--dip", metavar="FILE", help="Dipole time series file (3 or 4 columns)")
    ap.add_argument("--dip-units", default="au", help="Dipole units: au / Debye / C*m")
    ap.add_argument("--dip-has-time", choices=["auto","yes","no"], default="auto", help="Whether dipole file has a time column")
    ap.add_argument("--use-deriv", action="store_true", help="Use time-derivative Ṁ path (recommended)")
    ap.add_argument("--no-use-deriv", dest="use_deriv", action="store_false")
    ap.set_defaults(use_deriv=False)

    # Velocities
    ap.add_argument("--xyz", metavar="FILE", help="Extended XYZ with velocities")
    ap.add_argument("--vel-units", default="bohr/au", help="Velocity units: bohr/au, A/fs, A/ps, m/s")
    ap.add_argument("--vel-cols", default="-3,-2,-1", help="Indices of vx,vy,vz in each atom line (comma-separated, Python-style)")
    ap.add_argument("--no-remove-com", action="store_true", help="Disable COM drift removal")
    ap.add_argument("--mass-weight", action="store_true", help="Mass-weight velocities for VDOS")

    # Output & plotting
    ap.add_argument("--out-prefix", default="", help="Prefix for output files")
    ap.add_argument("--plot", nargs="?", const="show", help='Plot spectrum(s): "show" (default) or filename.png/.pdf')

    args = ap.parse_args()

    dt_au = dt_to_au(args.dt, args.dt_units)
    spectra = []

    # Dipole-based IR
    if args.dip:
        has_time = {"auto":"auto","yes":True,"no":False}[args.dip_has_time]
        M = load_dipoles(args.dip, has_time)
        if args.skip:
            M = M[args.skip:]
        M *= unit_dip_to_au(args.dip_units)
        wn, I = ir_from_dipoles(M, dt_au, args.use_deriv, args.window, args.exp_width, args.zp, args.max_wn)
        np.savetxt(f"{args.out_prefix}ir_from_dipole_cm-1.dat", np.c_[wn, I])
        spectra.append(("IR (dipole)", wn, I))

    # Velocity-based VDOS
    if args.xyz:
        vcols = tuple(int(x) for x in args.vel_cols.split(","))
        elems, V = parse_xyz_with_velocities(args.xyz, vcols)
        if args.skip:
            V = V[args.skip:]
        V *= unit_vel_to_au(args.vel_units)
        masses = masses_from_elems(elems)
        wn_v, I_v = vdos_from_vel(V, dt_au, masses, args.window, args.exp_width, args.zp,
                                  args.max_wn, args.mass_weight, not args.no_remove_com)
        np.savetxt(f"{args.out_prefix}vdos_cm-1.dat", np.c_[wn_v, I_v])
        spectra.append(("VDOS (vel)", wn_v, I_v))

    if not spectra:
        raise SystemExit("Nothing to do: provide --dip and/or --xyz.")

    # Optional plotting
    if args.plot:
        import matplotlib.pyplot as plt
        for label, wn, I in spectra:
            plt.figure()
            plt.plot(wn, I, lw=1.2)
            plt.xlabel("Wavenumber (cm$^{-1}$)")
            plt.ylabel("Intensity (arb. units)")
            plt.title(label)
            plt.xlim(0, max(wn) if len(wn) else args.max_wn)
            plt.tight_layout()
            if args.plot == "show":
                plt.show()
            else:
                plt.savefig(args.plot, dpi=200)
                plt.close()

if __name__ == "__main__":
    main()

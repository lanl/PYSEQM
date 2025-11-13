#!/usr/bin/env python3
"""
Gaussian broadening of up to two stick spectra (.dat: energy[eV] intensity):
absorption and emission on the same plot.

Examples
--------
python gaussian_broaden_dual.py --abs abs.dat --em em.dat --fwhm 0.10 --axis eV --show-sticks
python gaussian_broaden_dual.py --abs abs.dat --em em.dat --fwhm-abs 0.08 --fwhm-em 0.12 --axis nm
python gaussian_broaden_dual.py --abs abs.dat --fwhm 0.10 --axis nm --out-csv both.csv
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib font sizes
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size':       18,
    'axes.titlesize':  18,
    'axes.labelsize':  18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 16,
    'lines.markersize': 10.0
})

HC_EV_NM = 1239.8419843320026  # h*c in eV*nm

def smart_loadtxt(path, xcol=0, ycol=1):
    """
    Loads a text file with at least two numeric columns.
    Ignores comment lines starting with # or ; or !
    Tries common delimiters automatically.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", ";", "!", "//"))]
    if not data_lines:
        raise ValueError("No data lines found in file.")

    probe = data_lines[0]
    if "\t" in probe:
        delimiter = "\t"
    elif "," in probe:
        delimiter = ","
    else:
        delimiter = None  # any whitespace

    arr = np.genfromtxt(data_lines, dtype=float, comments=None, delimiter=delimiter, invalid_raise=False)

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("Could not parse two numeric columns.")
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError("Parsed array is not 2D; check the input formatting.")

    if xcol < 0 or ycol < 0 or max(xcol, ycol) >= arr.shape[1]:
        raise IndexError(f"Requested columns xcol={xcol}, ycol={ycol} but file has {arr.shape[1]} columns.")

    x = np.asarray(arr[:, xcol], dtype=float)
    y = np.asarray(arr[:, ycol], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    idx = np.argsort(x)
    return x[idx], y[idx]

def broaden(x_sticks, y_sticks, x_grid, sigma):
    """Sum of Gaussians (unit-area kernels scaled by y_sticks) over a grid."""
    X = x_grid[:, None]
    mu = x_sticks[None, :]
    G = np.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return (G * y_sticks[None, :]).sum(axis=1)

def _sigma_from_args(g_sigma, g_fwhm, local_sigma, local_fwhm, label):
    """Resolve sigma (eV) for a curve using local overrides then global."""
    if local_sigma is not None and local_sigma > 0:
        return float(local_sigma), f"σ = {local_sigma:g} eV"
    if local_fwhm is not None and local_fwhm > 0:
        s = local_fwhm / (2 * np.sqrt(2 * np.log(2)))
        return s, f"FWHM = {local_fwhm:g} eV"
    if g_sigma is not None and g_sigma > 0:
        return float(g_sigma), f"σ = {g_sigma:g} eV"
    if g_fwhm is not None and g_fwhm > 0:
        s = g_fwhm / (2 * np.sqrt(2 * np.log(2)))
        return s, f"FWHM = {g_fwhm:g} eV"
    raise ValueError(f"No width provided for {label}. Provide --sigma/--fwhm or a per-curve override.")

def main():
    p = argparse.ArgumentParser(description="Plot Gaussian-broadened absorption/emission spectra on one figure.")
    io = p.add_argument_group("Inputs")
    io.add_argument("--abs", dest="absfile", help="Absorption input .dat (two columns: energy[eV], intensity)")
    io.add_argument("--em",  dest="emfile",  help="Emission   input .dat (two columns: energy[eV], intensity)")
    io.add_argument("--xcol", type=int, default=0, help="0-based column index for energy (default: 0)")
    io.add_argument("--ycol", type=int, default=1, help="0-based column index for intensity (default: 1)")

    widths = p.add_argument_group("Line broadening (eV)")
    widths.add_argument("--fwhm", type=float, help="Global FWHM for both curves (overridden by per-curve flags)")
    widths.add_argument("--sigma", type=float, help="Global σ for both curves (overridden by per-curve flags)")
    widths.add_argument("--fwhm-abs", type=float, help="FWHM for absorption only")
    widths.add_argument("--sigma-abs", type=float, help="σ for absorption only")
    widths.add_argument("--fwhm-em", type=float, help="FWHM for emission only")
    widths.add_argument("--sigma-em", type=float, help="σ for emission only")

    grid = p.add_argument_group("Grid / plotting")
    grid.add_argument("--xmin", type=float, help="Min energy for grid in eV (default: min(data)-1.5*max(FWHM))")
    grid.add_argument("--xmax", type=float, help="Max energy for grid in eV (default: max(data)+1.5*max(FWHM))")
    grid.add_argument("--dx", type=float, default=None, help="Energy grid spacing in eV (default: auto)")
    grid.add_argument("--points", type=int, default=None, help="Alternative to dx: number of grid points")
    grid.add_argument("--axis", choices=("eV","nm"), default="eV",
                      help="Plot x-axis in energy (eV) or wavelength (nm). Input x must be in eV.")
    grid.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                      help="Normalize each broadened curve to max = 1 (sticks scaled consistently).")
    grid.add_argument("--no-normalize", dest="normalize", action="store_false",
                      help="Disable normalization.")
    grid.add_argument("--show-sticks", action="store_true", help="Overlay the original stick spectra")
    grid.add_argument("--title", default=None, help="Custom plot title")

    out = p.add_argument_group("Output")
    out.add_argument("--out-png", default=None, help="Save plot to PNG path")
    out.add_argument("--out-csv", default=None, help="Save broadened data to CSV (x, y_abs, y_em)")

    args = p.parse_args()

    if not args.absfile and not args.emfile:
        print("Provide at least one input: --abs and/or --em.", file=sys.stderr)
        sys.exit(1)

    # Load data
    xs_abs = ys_abs = None
    xs_em  = ys_em  = None
    try:
        if args.absfile:
            xs_abs, ys_abs = smart_loadtxt(args.absfile, xcol=args.xcol, ycol=args.ycol)
        if args.emfile:
            xs_em, ys_em = smart_loadtxt(args.emfile,  xcol=args.xcol, ycol=args.ycol)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve sigmas (eV) per curve
    try:
        sigma_abs = width_lbl_abs = None
        sigma_em  = width_lbl_em  = None
        if args.absfile:
            sigma_abs, width_lbl_abs = _sigma_from_args(args.sigma, args.fwhm, args.sigma_abs, args.fwhm_abs, "absorption")
        if args.emfile:
            sigma_em,  width_lbl_em  = _sigma_from_args(args.sigma, args.fwhm, args.sigma_em,  args.fwhm_em,  "emission")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Build a common energy grid (in eV) that covers both datasets
    widths_for_pad = []
    xmins = []
    xmaxs = []
    if args.absfile:
        fwhm_abs = 2.354820045 * sigma_abs
        widths_for_pad.append(fwhm_abs)
        xmins.append(xs_abs.min())
        xmaxs.append(xs_abs.max())
    if args.emfile:
        fwhm_em = 2.354820045 * sigma_em
        widths_for_pad.append(fwhm_em)
        xmins.append(xs_em.min())
        xmaxs.append(xs_em.max())

    pad = 1.5 * (max(widths_for_pad) if widths_for_pad else 0.1)
    xmin_eV = args.xmin if args.xmin is not None else (min(xmins) - pad)
    xmax_eV = args.xmax if args.xmax is not None else (max(xmaxs) + pad)

    if args.axis == "nm":
        xmin_eV = max(xmin_eV, 1e-6)  # forbid nonpositive energies for nm conversion

    if not (np.isfinite(xmin_eV) and np.isfinite(xmax_eV)) or xmin_eV >= xmax_eV:
        print("Invalid energy range: ensure xmin < xmax and both finite.", file=sys.stderr)
        sys.exit(1)

    # Choose grid spacing
    if args.points is not None:
        Eg = np.linspace(xmin_eV, xmax_eV, args.points)
    else:
        if args.dx is None:
            # Estimate from native spacings and widths
            native_dx_candidates = []
            if args.absfile and xs_abs.size > 1:
                diffs = np.diff(np.unique(xs_abs))
                if diffs.size:
                    native_dx_candidates.append(np.median(diffs))
            if args.emfile and xs_em.size > 1:
                diffs = np.diff(np.unique(xs_em))
                if diffs.size:
                    native_dx_candidates.append(np.median(diffs))
            native_dx = min(native_dx_candidates) if native_dx_candidates else (max(widths_for_pad)/50.0 if widths_for_pad else 0.002)
            width_dx = min([(w/20.0) for w in widths_for_pad]) if widths_for_pad else native_dx
            dx = min(width_dx, native_dx/5.0)
        else:
            dx = args.dx
        if dx <= 0:
            print("dx must be positive.", file=sys.stderr)
            sys.exit(1)
        npts = int(np.floor((xmax_eV - xmin_eV) / dx)) + 1
        Eg = np.linspace(xmin_eV, xmax_eV, npts)

    # Broaden each curve on the shared energy grid (intensity per eV)
    Yg_abs_E = Yg_em_E = None
    if args.absfile:
        Yg_abs_E = broaden(xs_abs, ys_abs, Eg, sigma_abs)
    if args.emfile:
        Yg_em_E  = broaden(xs_em,  ys_em,  Eg, sigma_em)

    # Prepare axis conversion and plotting arrays
    if args.axis == "eV":
        Xplot = Eg
        Y_abs_plot = Yg_abs_E.copy() if Yg_abs_E is not None else None
        Y_em_plot  = Yg_em_E.copy()  if Yg_em_E  is not None else None
        xlabel = "Energy (eV)"
        ylabel_units = "(per eV)"
        # Sticks
        if args.absfile:
            sticks_abs_x = xs_abs
            sticks_abs_y = ys_abs.copy()
        if args.emfile:
            sticks_em_x = xs_em
            sticks_em_y = ys_em.copy()
    else:
        # Wavelength axis (nm) with Jacobian |dE/dλ| = HC/λ^2
        lam = HC_EV_NM / Eg
        jac = HC_EV_NM / (lam ** 2)
        order = np.argsort(lam)
        Xplot = lam[order]
        Y_abs_plot = (Yg_abs_E * jac)[order] if Yg_abs_E is not None else None
        Y_em_plot  = (Yg_em_E  * jac)[order] if Yg_em_E  is not None else None
        xlabel = "Wavelength (nm)"
        ylabel_units = "(per nm)"
        # Sticks converted to nm with consistent scaling
        if args.absfile:
            sticks_abs_x = HC_EV_NM / xs_abs
            sticks_abs_y = ys_abs * ((xs_abs**2) / HC_EV_NM)
        if args.emfile:
            sticks_em_x = HC_EV_NM / xs_em
            sticks_em_y = ys_em * ((xs_em**2) / HC_EV_NM)
        # Sort sticks by wavelength for nicer vlines drawing
        if args.absfile:
            o = np.argsort(sticks_abs_x); sticks_abs_x, sticks_abs_y = sticks_abs_x[o], sticks_abs_y[o]
        if args.emfile:
            o = np.argsort(sticks_em_x); sticks_em_x, sticks_em_y = sticks_em_x[o], sticks_em_y[o]

    # Normalize (per-curve)
    if args.normalize:
        if Y_abs_plot is not None and Y_abs_plot.size:
            m = np.max(Y_abs_plot)
            if m > 0: Y_abs_plot = Y_abs_plot / m
        if Y_em_plot is not None and Y_em_plot.size:
            m = np.max(Y_em_plot)
            if m > 0: Y_em_plot = Y_em_plot / m
        # Scale sticks per curve for visualization consistency
        if args.show_sticks:
            if args.absfile and sticks_abs_y.size:
                s = np.max(sticks_abs_y); 
                if s > 0: sticks_abs_y = sticks_abs_y / s
            if args.emfile and sticks_em_y.size:
                s = np.max(sticks_em_y);
                if s > 0: sticks_em_y = sticks_em_y / s

    # Save CSV if requested
    if args.out_csv:
        cols = [Xplot]
        hdrs = ["energy_eV" if args.axis=="eV" else "wavelength_nm"]
        if Y_abs_plot is not None:
            cols.append(Y_abs_plot); hdrs.append("broadened_abs")
        if Y_em_plot is not None:
            cols.append(Y_em_plot);  hdrs.append("broadened_em")
        out = np.column_stack(cols)
        np.savetxt(args.out_csv, out, delimiter=",", header=",".join(hdrs), comments="")
        print(f"Saved broadened data to {args.out_csv}")

    # Plot
    fig = plt.figure()
    if args.show_sticks:
        if args.absfile:
            for x_i, y_i in zip(sticks_abs_x, sticks_abs_y):
                plt.vlines(x_i, 0.0, y_i, linewidth=1, alpha=0.45, label=None)
        if args.emfile:
            for x_i, y_i in zip(sticks_em_x, sticks_em_y):
                plt.vlines(x_i, 0.0, y_i, linewidth=1, alpha=0.45, label=None)

    # Draw broadened curves with labels
    if Y_abs_plot is not None:
        plt.plot(Xplot, Y_abs_plot, linewidth=1.8, ls='-', label=f"Absorption")# ({width_lbl_abs})")
    if Y_em_plot is not None:
        plt.plot(Xplot, Y_em_plot,  linewidth=2.0,ls='-.', label=f"Emission")# ({width_lbl_em})")

    plt.xlabel(xlabel)
    plt.xlim([Xplot[0], Xplot[-1]])
    plt.ylabel("Intensity " + "(normalized)" )
    if args.absfile and args.emfile:
        default_title = "Absorption and Emission (Gaussian broadened)"
    elif args.absfile:
        default_title = "Absorption (Gaussian broadened)"
    else:
        default_title = "Emission (Gaussian broadened)"
    plt.title(args.title if args.title is not None else default_title)
    plt.legend()
    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=600, bbox_inches="tight")
        print(f"Saved plot to {args.out_png}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()


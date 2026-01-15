#!/usr/bin/env python3
"""
Gaussian broadening of stick spectra; supports multiple input files and eV↔nm axis.

Input
-----
Each input file must have at least two numeric columns:
  x: energy in eV   (col --xcol)
  y: intensity      (col --ycol)
Comments starting with #, ;, !, // are ignored. Delimiters auto-detected.

Examples
--------
# Two spectra overlaid, 0.10 eV FWHM, energy axis
python gaussian_broaden_multi.py a.dat b.dat --fwhm 0.10 --axis eV --show-sticks

# Four spectra, wavelength axis with proper Jacobian, save PNG & CSVs
python gaussian_broaden_multi.py a.dat b.dat c.dat d.dat --sigma 0.06 --axis nm \
  --out-png overlay.png --out-csv broadened.csv

# Custom labels (same count/order as files), custom grid, per-curve normalization
python gaussian_broaden_multi.py a.dat b.dat --fwhm 0.08 --labels "A (298K)" "B (320K)" \
  --xmin 1.6 --xmax 3.8 --dx 0.001 --normalize

Notes
-----
- Broadening is done in energy (eV), then optionally converted to wavelength (nm).
- When plotting in nm, intensities are multiplied by |dE/dλ| = (hc)/λ^2 so they are "per nm".
- If --out-csv is provided and multiple inputs are given, files are saved as:
    <template_stem>_<input_stem>.csv
  For example: broadened.csv + a.dat -> broadened_a.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Matplotlib font sizes
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "lines.markersize": 10.0,
    }
)

HC_EV_NM = 1239.8419843320026  # Planck*c in eV*nm

# --------------------------- IO & math helpers ---------------------------


def smart_loadtxt(path, xcol=0, ycol=1):
    """Load two numeric columns from a text file with auto delimiter detection."""
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
        delimiter = None  # whitespace

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


def broaden_energy(x_sticks_eV, y_sticks, E_grid, sigma_eV):
    """Gaussian broadening on an energy grid (unit-area kernels scaled by y)."""
    X = E_grid[:, None]  # (Ng, 1)
    MU = x_sticks_eV[None, :]  # (1, Ns)
    G = np.exp(-0.5 * ((X - MU) / sigma_eV) ** 2) / (sigma_eV * np.sqrt(2 * np.pi))
    return (G * y_sticks[None, :]).sum(axis=1)  # intensity per eV


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def csv_path(template, infile):
    """Insert input stem before extension of template; handle single/multiple inputs."""
    base, ext = os.path.splitext(template)
    return f"{base}_{stem(infile)}{ext or '.csv'}"


# --------------------------- main ---------------------------


def main():
    p = argparse.ArgumentParser(description="Overlay Gaussian-broadened spectra from multiple files.")
    p.add_argument("infiles", nargs="+", help="Input files (columns: energy[eV], intensity)")
    p.add_argument("--xcol", type=int, default=0, help="0-based column index for energy (default: 0)")
    p.add_argument("--ycol", type=int, default=1, help="0-based column index for intensity (default: 1)")

    width = p.add_mutually_exclusive_group(required=True)
    width.add_argument("--fwhm", type=float, help="Full Width at Half Maximum (eV)")
    width.add_argument("--sigma", type=float, help="Standard deviation (eV)")

    p.add_argument("--xmin", type=float, help="Min energy for grid in eV (default: min(all)-3*FWHM)")
    p.add_argument("--xmax", type=float, help="Max energy for grid in eV (default: max(all)+3*FWHM)")
    p.add_argument("--dx", type=float, default=None, help="Energy grid spacing in eV (default: auto)")
    p.add_argument("--points", type=int, default=None, help="Alternative to dx: number of grid points")

    p.add_argument(
        "--axis",
        choices=("eV", "nm"),
        default="eV",
        help="Plot x-axis in energy (eV) or wavelength (nm). Input x must be in eV.",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize each broadened curve to its own max (per-curve).",
    )
    p.add_argument(
        "--show-sticks", action="store_true", help="Overlay original stick spectra (unit-consistent)."
    )
    p.add_argument(
        "--labels", nargs="+", default=None, help="Legend labels (must match number/order of input files)."
    )
    p.add_argument("--title", default=None, help="Custom plot title")
    p.add_argument("--out-png", default=None, help="Save overlay plot to PNG path")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Save broadened data for each file to CSV using this template path; "
        "output becomes <template_stem>_<input_stem>.csv",
    )
    args = p.parse_args()

    # FWHM/sigma in eV
    if args.fwhm is not None:
        sigma_eV = args.fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2*sqrt(2 ln 2)*sigma
        width_for_bounds = args.fwhm
        width_label = f"FWHM = {args.fwhm:g} eV"
    else:
        sigma_eV = args.sigma
        width_for_bounds = 2.354820045 * sigma_eV
        width_label = f"σ = {sigma_eV:g} eV"

    if sigma_eV is None or sigma_eV <= 0:
        print("Width must be positive (in eV).", file=sys.stderr)
        sys.exit(1)

    # Load all inputs
    datasets = []
    for f in args.infiles:
        try:
            x_eV, y = smart_loadtxt(f, xcol=args.xcol, ycol=args.ycol)
        except Exception as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)
            sys.exit(1)
        datasets.append((f, x_eV, y))

    # Build a shared ENERGY grid for overlay
    all_min = min(x.min() for _, x, _ in datasets)
    all_max = max(x.max() for _, x, _ in datasets)
    pad = 1.5 * width_for_bounds
    xmin_eV = args.xmin if args.xmin is not None else (all_min - pad)
    xmax_eV = args.xmax if args.xmax is not None else (all_max + pad)

    if args.axis == "nm":
        xmin_eV = max(xmin_eV, 1e-6)  # forbid nonpositive energies for λ conversion

    if not (np.isfinite(xmin_eV) and np.isfinite(xmax_eV)) or xmin_eV >= xmax_eV:
        print("Invalid energy range: ensure xmin < xmax and both finite.", file=sys.stderr)
        sys.exit(1)

    if args.points is not None:
        Eg = np.linspace(xmin_eV, xmax_eV, args.points)
    else:
        if args.dx is None:
            # heuristic dx: min(width/20, median native spacing / 5 across datasets)
            native_dxs = []
            for _, x, _ in datasets:
                diffs = np.diff(np.unique(x))
                if diffs.size:
                    native_dxs.append(np.median(diffs))
            native_dx = np.median(native_dxs) if native_dxs else width_for_bounds / 50.0
            dx = min(width_for_bounds / 20.0, native_dx / 5.0)
        else:
            dx = args.dx
        if dx <= 0:
            print("dx must be positive.", file=sys.stderr)
            sys.exit(1)
        npts = int(np.floor((xmax_eV - xmin_eV) / dx)) + 1
        Eg = np.linspace(xmin_eV, xmax_eV, npts)

    # Convert to wavelength axis if requested (for x-grid and Jacobian)
    if args.axis == "nm":
        lam = HC_EV_NM / Eg
        order_grid = np.argsort(lam)
        Xplot_grid = lam[order_grid]
        xlabel = "Wavelength (nm)"
    else:
        Xplot_grid = Eg
        order_grid = slice(None)  # identity
        xlabel = "Energy (eV)"

    # Labels
    if args.labels:
        if len(args.labels) != len(datasets):
            print("Number of --labels must match number of input files.", file=sys.stderr)
            sys.exit(1)
        labels = args.labels
    else:
        labels = [stem(f) for f, _, _ in datasets]

    # Prepare figure
    plt.figure()

    # Process each dataset
    ticktype = "-"
    for (f, xs_eV, ys), lab in zip(datasets, labels):
        # Broaden in energy
        Yg_E = broaden_energy(xs_eV, ys, Eg, sigma_eV)  # per eV

        # Convert for plotting/output
        if args.axis == "eV":
            Xplot = Xplot_grid
            Yplot = Yg_E
            sticks_x = xs_eV
            sticks_y = ys
        else:
            lam_vals = HC_EV_NM / Eg
            jac = HC_EV_NM / (lam_vals**2)  # |dE/dλ|
            Y_lambda = Yg_E * jac

            # reorder increasing wavelength
            Xplot = lam_vals[order_grid]
            Yplot = Y_lambda[order_grid]

            # sticks transformed consistently
            sticks_x = HC_EV_NM / xs_eV
            sticks_y = ys * (xs_eV**2) / HC_EV_NM  # equivalent to HC/λ^2 at stick

        # Normalize per curve if requested
        if args.normalize:
            m = np.max(Yplot) if Yplot.size else 1.0
            if m > 0:
                Yplot = Yplot / m
            if args.show_sticks and np.size(sticks_y):
                sm = np.max(sticks_y)
                if sm > 0:
                    sticks_y = sticks_y / sm

        # CSV (per input)
        if args.out_csv:
            out_xy = np.column_stack([Xplot, Yplot])
            xhdr = "energy_eV" if args.axis == "eV" else "wavelength_nm"
            out_path = csv_path(args.out_csv, f)
            np.savetxt(out_path, out_xy, delimiter=",", header=f"{xhdr},broadened_intensity", comments="")
            print(f"Saved broadened data to {out_path}")

        # Plot curve
        plt.plot(Xplot, Yplot, ticktype, linewidth=2.0, label=lab)
        ticktype = ":" if ticktype == "-" else "-"

        # Optional sticks
        if args.show_sticks:
            # use a faint overlay for sticks
            for x_i, y_i in zip(np.atleast_1d(sticks_x), np.atleast_1d(sticks_y)):
                plt.vlines(float(x_i), 0.0, float(y_i), linewidth=1, alpha=0.35)

    # Final plot cosmetics
    plt.xlabel(xlabel)
    plt.xlim([Xplot[0], Xplot[-1]])
    plt.ylabel("Intensity " + ("(normalized)" if args.normalize else "(arb. units)"))
    ttl = args.title if args.title else f"Gaussian-broadened spectrum ({width_label})"
    ttl = args.title if args.title else ""
    plt.title(ttl)
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

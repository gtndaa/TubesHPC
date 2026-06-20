#!/usr/bin/env python3
"""
analyze_results.py - Analisis hasil simulasi N-Body
Membaca file timing dan trajectory dari ../results/, lalu membuat:
- Plot perbandingan performa (serial vs paralel)
- Plot error energi
- Visualisasi 3D trajektori (opsional)

Usage:
    python analyze_results.py                    # jalankan semua analisis
    python analyze_results.py --plot-time       # hanya plot waktu
    python analyze_results.py --plot-energy     # hanya plot energi
    python analyze_results.py --vis <csv_file>  # visualisasi trajektori dari CSV
    python analyze_results.py --help            # tampilkan bantuan
"""

import os
import sys
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- Konfigurasi path ---
RESULTS_DIR = "../results"
TIMING_PATTERN = "*_timing.txt"
TRAJECTORY_PATTERN = "*_trajectory.csv"

# --- Fungsi parsing timing file ---
def parse_timing_file(filepath):
    """Baca file timing dan kembalikan dictionary data."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=')
                data[key] = val
    # Tambahkan informasi dari nama file
    basename = os.path.basename(filepath)
    # Coba ekstrak N, procs, threads dari nama file jika ada
    # Format contoh: serial_timing.txt, parallel_timing_N256_P4_OMP2.txt
    # Kita gunakan regex
    match = re.search(r'N(\d+)', basename)
    if match:
        data['N'] = int(match.group(1))
    else:
        data['N'] = int(data.get('N', 0))
    match = re.search(r'P(\d+)', basename)
    if match:
        data['procs'] = int(match.group(1))
    else:
        data['procs'] = 1
    match = re.search(r'OMP(\d+)', basename)
    if match:
        data['threads'] = int(match.group(1))
    else:
        data['threads'] = 1
    # Mode dari nama file
    if 'serial' in basename.lower():
        data['mode'] = 'serial'
    else:
        data['mode'] = 'parallel'
    return data

# --- Fungsi kumpulkan semua timing ---
def collect_timing_data():
    """Scan RESULTS_DIR, baca semua timing file, kembalikan DataFrame."""
    timing_files = glob.glob(os.path.join(RESULTS_DIR, TIMING_PATTERN))
    records = []
    for f in timing_files:
        data = parse_timing_file(f)
        # Pastikan field penting ada
        required = ['total_time', 'energy_error_pct', 'N']
        if not all(k in data for k in required):
            print(f"Warning: {f} missing required fields, skip")
            continue
        records.append({
            'file': os.path.basename(f),
            'mode': data.get('mode', 'unknown'),
            'N': data.get('N', 0),
            'procs': data.get('procs', 1),
            'threads': data.get('threads', 1),
            'total_time': float(data.get('total_time', 0)),
            'force_time': float(data.get('force_time', 0)),
            'energy_error': float(data.get('energy_error_pct', 0))
        })
    if not records:
        print(f"Tidak ada file timing ditemukan di {RESULTS_DIR}")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Label konfigurasi
    df['config'] = df.apply(
        lambda r: f"serial" if r['mode']=='serial' else f"P{r['procs']}xOMP{r['threads']}",
        axis=1
    )
    return df

# --- Fungsi plot waktu ---
def plot_time_comparison(df, output_file='time_comparison.png'):
    if df.empty:
        print("Tidak ada data untuk plot waktu.")
        return
    fig, ax = plt.subplots(figsize=(10,6))
    for config, group in df.groupby('config'):
        # Urutkan berdasarkan N
        group_sorted = group.sort_values('N')
        ax.plot(group_sorted['N'], group_sorted['total_time'], 'o-', label=config)
    ax.set_xlabel('Jumlah Partikel (N)')
    ax.set_ylabel('Waktu Total (detik)')
    ax.set_title('Perbandingan Performa N-Body')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, output_file))
    print(f"Plot waktu disimpan ke {os.path.join(RESULTS_DIR, output_file)}")
    plt.close()

# --- Fungsi plot energi error ---
def plot_energy_error(df, output_file='energy_error.png'):
    if df.empty:
        print("Tidak ada data untuk plot energi.")
        return
    # Ambil data terakhir untuk setiap konfigurasi (asumsikan N terbesar)
    # Atau kita bisa tampilkan semua, tapi lebih baik ambil rata-rata atau N tertentu
    # Kita ambil data dengan N maksimum untuk setiap konfigurasi
    idx = df.groupby('config')['N'].idxmax()
    df_max = df.loc[idx]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(df_max['config'], df_max['energy_error'], color=['blue' if m=='serial' else 'orange' for m in df_max['mode']])
    ax.set_ylabel('Error Energi (%)')
    ax.set_title('Konservasi Energi (N terbesar)')
    for bar, err in zip(bars, df_max['energy_error']):
        ax.text(bar.get_x() + bar.get_width()/2, err + 0.2, f"{err:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, output_file))
    print(f"Plot energi disimpan ke {os.path.join(RESULTS_DIR, output_file)}")
    plt.close()

# --- Fungsi visualisasi trajektori ---
def visualize_trajectory(csv_path, output=None, steps=None, interval=100):
    """Visualisasi 3D trajektori dari CSV."""
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan.")
        return
    df = pd.read_csv(csv_path)
    if steps:
        df = df[df['step'] <= steps]
    steps_all = df['step'].unique()
    steps_all.sort()
    particle_ids = df['particle_id'].unique()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set limits
    margin = 0.2
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()
    zmin, zmax = df['z'].min(), df['z'].max()
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_zlim(zmin - margin, zmax + margin)
    # Scatter objects
    colors = plt.cm.jet(np.linspace(0,1,len(particle_ids)))
    scatters = []
    for pid, col in zip(particle_ids, colors):
        scat = ax.scatter([], [], [], c=[col], s=30, label=f'P{pid}')
        scatters.append(scat)
    def update(step_idx):
        step = steps_all[step_idx]
        data_step = df[df['step'] == step]
        for i, pid in enumerate(particle_ids):
            row = data_step[data_step['particle_id'] == pid]
            if len(row) > 0:
                x = row['x'].values[0]
                y = row['y'].values[0]
                z = row['z'].values[0]
                scatters[i]._offsets3d = ([x], [y], [z])
            else:
                scatters[i]._offsets3d = ([], [], [])
        ax.set_title(f'Step {step}')
        return scatters
    ani = animation.FuncAnimation(fig, update, frames=len(steps_all), interval=interval, blit=False)
    if output:
        if output.endswith('.gif'):
            ani.save(output, writer='pillow', fps=1000/interval)
        else:
            ani.save(output, fps=1000/interval)
        print(f"Animasi disimpan ke {output}")
    else:
        plt.show()

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Analisis hasil simulasi N-Body")
    parser.add_argument('--plot-time', action='store_true', help='Buat plot perbandingan waktu')
    parser.add_argument('--plot-energy', action='store_true', help='Buat plot error energi')
    parser.add_argument('--vis', metavar='CSV_FILE', help='Visualisasi trajektori dari CSV (opsional)')
    parser.add_argument('--output', help='Nama file output untuk animasi (jika --vis digunakan)')
    parser.add_argument('--steps', type=int, help='Jumlah step untuk visualisasi')
    parser.add_argument('--all', action='store_true', help='Jalankan semua analisis (default)')
    args = parser.parse_args()

    # Jika tidak ada argumen spesifik, jalankan semua
    if not any([args.plot_time, args.plot_energy, args.vis]):
        args.all = True

    if args.all or args.plot_time or args.plot_energy:
        df = collect_timing_data()
        if not df.empty:
            if args.all or args.plot_time:
                plot_time_comparison(df)
            if args.all or args.plot_energy:
                plot_energy_error(df)
        else:
            print("Tidak ada data timing. Jalankan simulasi terlebih dahulu.")

    if args.vis:
        visualize_trajectory(args.vis, args.output, args.steps)

if __name__ == "__main__":
    main()
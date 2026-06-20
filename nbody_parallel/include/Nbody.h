#ifndef NBODY_H
#define NBODY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ============================================================
   Konstanta fisika dan parameter simulasi
   ============================================================ */
#define G_CONST       6.674e-11   /* Konstanta gravitasi Newton (m^3 kg^-1 s^-2) */
#define SOFTENING     1.0e-5      /* Softening parameter untuk hindari singularitas */
#define DIM           3           /* Dimensi ruang (3D) */

/* Parameter default (bisa di-override via argv) */
#define DEFAULT_N     512         /* Jumlah partikel */
#define DEFAULT_T     100         /* Jumlah timestep */
#define DEFAULT_DT    0.01        /* Ukuran timestep (s) */
#define DEFAULT_SEED  42          /* Random seed untuk reproduksibilitas */

/* ============================================================
   Struktur data partikel
   ============================================================ */
typedef struct {
    double x, y, z;       /* Posisi (m) */
    double vx, vy, vz;    /* Kecepatan (m/s) */
    double fx, fy, fz;    /* Gaya/akselerasi (N) */
    double mass;           /* Massa (kg) */
} Particle;

/* ============================================================
   Struktur untuk hasil validasi energi
   ============================================================ */
typedef struct {
    double kinetic;        /* Energi kinetik total */
    double potential;      /* Energi potensial total */
    double total;          /* Energi mekanik total */
} Energy;

/* ============================================================
   Struktur parameter simulasi
   ============================================================ */
typedef struct {
    int    n_particles;    /* Jumlah partikel */
    int    n_steps;        /* Jumlah timestep */
    double dt;             /* Ukuran timestep */
    int    seed;           /* Random seed */
    int    output_freq;    /* Frekuensi output (setiap N step) */
    char   output_file[256]; /* Nama file output */
} SimParams;

/* ============================================================
   Deklarasi fungsi utilitas
   ============================================================ */

/* Inisialisasi partikel secara acak */
void init_particles(Particle *particles, int n, int seed);

/* Hitung gaya gravitasi antara dua partikel */
void compute_force_pair(const Particle *pi, const Particle *pj,
                        double *fx, double *fy, double *fz);

/* Hitung semua gaya (serial O(N^2)) */
void compute_forces_serial(Particle *particles, int n);

/* Integrasi Velocity Verlet step */
void velocity_verlet_step(Particle *particles, int n, double dt);

/* Hitung energi sistem */
Energy compute_energy(const Particle *particles, int n);

/* Validasi konservasi energi */
int validate_energy_conservation(double E0, double E1, double tolerance);

/* Print statistik */
void print_energy(int step, Energy e, double elapsed);

/* Simpan state ke file */
void save_state(FILE *fp, const Particle *particles, int n, int step, double time);

/* Print parameter simulasi */
void print_params(const SimParams *p, const char *mode);

/* Inisialisasi SimParams dengan default */
void default_params(SimParams *p);

/* Parse argumen baris perintah */
void parse_args(int argc, char *argv[], SimParams *p);

#endif /* NBODY_H */
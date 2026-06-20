/*
 * nbody_parallel_fixed.c
 * ============================================================
 * N-Body Hybrid MPI + OpenMP — VERSI FIXED
 *
 * PERBAIKAN dari versi sebelumnya:
 *   1. Unit adimensional (G=1, M_total=1, R=1)
 *      → tidak ada lagi overflow/instabilitas skala
 *   2. Leapfrog (Kick-Drift-Kick) yang benar
 *      → konservasi energi jauh lebih baik dari Velocity Verlet naive
 *   3. Softening otomatis menyesuaikan skala sistem
 *   4. dt default = 0.001 (aman untuk unit adimensional)
 *
 * UNIT ADIMENSIONAL:
 *   G = 1
 *   M_total = N  (tiap partikel massa = 1)
 *   R_virial = 1  (partikel tersebar di radius ~1)
 *   → crossing time ≈ 1, simulasikan T=2 sudah cukup
 *
 * Kompilasi:
 *   mpicc -O2 -fopenmp -o nbody_parallel nbody_parallel.c -lm
 *
 * Jalankan:
 *   mpirun --oversubscribe --allow-run-as-root \
 *          --mca plm_rsh_agent "" -np 4 \
 *          ./nbody_parallel 512 200 4
 *   (arg: N_partikel  N_steps  N_threads)
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/* ---- Konstanta unit adimensional ---- */
#define G_CONST   1.0      /* gravitasi adimensional                  */
#define SOFTENING 0.05     /* ~5% dari radius sistem, cegah singulari */
#define DT_DEF    0.001    /* timestep aman untuk unit adimensional   */

/* ---- Struct partikel ---- */
typedef struct {
    double x,  y,  z;
    double vx, vy, vz;
    double ax, ay, az;   /* akselerasi (butuh untuk Leapfrog KDK)   */
    double mass;
} Particle;

/* ================================================================
   inisialisasi_plummer
   Distribusi Plummer: model galaksi sederhana yang stabil.
   Lebih realistis dari posisi acak seragam.
   r_scale = 1 (unit adimensional).
   ================================================================ */
void inisialisasi_plummer(Particle *p, int N, unsigned int seed) {
    srand(seed);
    double inv_N = 1.0 / N;

    for (int i = 0; i < N; i++) {
        /* --- posisi dari distribusi Plummer ---
         * Inverse CDF: r = 1/sqrt(u^(-2/3) - 1), u ~ Uniform(0,1)
         */
        double u   = 0.001 + 0.998 * ((double)rand() / RAND_MAX);
        double r   = 1.0 / sqrt(pow(u, -2.0/3.0) - 1.0);
        double cos_theta = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        double phi       = 2.0 * M_PI * ((double)rand() / RAND_MAX);

        p[i].x = r * sin_theta * cos(phi);
        p[i].y = r * sin_theta * sin(phi);
        p[i].z = r * cos_theta;

        /* --- kecepatan: setengah dari kecepatan escape lokal ---
         * v_esc = sqrt(2 * |phi(r)|), phi(r) = -G*M/sqrt(r²+1)
         * Gunakan 50% untuk sistem terikat (virial theorem)
         */
        double phi_r = -G_CONST * N / sqrt(r*r + 1.0);
        double v_max = sqrt(2.0 * fabs(phi_r)) * 0.5;

        /* rejection sampling untuk distribusi kecepatan */
        double v, g;
        do {
            v = ((double)rand() / RAND_MAX) * v_max;
            g = v*v * pow(1.0 - v*v/(v_max*v_max + 1e-10), 3.5);
        } while (((double)rand() / RAND_MAX) > g / (v_max*v_max * 0.1 + 1e-10));

        double cos_tv = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double sin_tv = sqrt(1.0 - cos_tv*cos_tv);
        double phi_v  = 2.0 * M_PI * ((double)rand() / RAND_MAX);

        p[i].vx = v * sin_tv * cos(phi_v);
        p[i].vy = v * sin_tv * sin(phi_v);
        p[i].vz = v * cos_tv;

        p[i].ax   = 0.0;
        p[i].ay   = 0.0;
        p[i].az   = 0.0;
        p[i].mass = 1.0;   /* massa = 1 per partikel (unit adimensional) */
    }

    /* Koreksi pusat massa ke (0,0,0) dan kecepatan pusat ke nol */
    double cx=0,cy=0,cz=0, cvx=0,cvy=0,cvz=0;
    for (int i=0;i<N;i++){cx+=p[i].x;cy+=p[i].y;cz+=p[i].z;
                          cvx+=p[i].vx;cvy+=p[i].vy;cvz+=p[i].vz;}
    cx*=inv_N;cy*=inv_N;cz*=inv_N;
    cvx*=inv_N;cvy*=inv_N;cvz*=inv_N;
    for (int i=0;i<N;i++){
        p[i].x-=cx; p[i].y-=cy; p[i].z-=cz;
        p[i].vx-=cvx;p[i].vy-=cvy;p[i].vz-=cvz;
    }
}

/* ================================================================
   hitung_akselerasi_omp
   Hitung akselerasi partikel lokal akibat SEMUA partikel global.
   OpenMP mem-parallelize loop i (partikel lokal).
   Tidak ada race condition: setiap i menulis ke p_lokal[i].ax/ay/az.
   ================================================================ */
void hitung_akselerasi_omp(Particle *lokal, int n_lokal,
                            const Particle *global, int N)
{
    #pragma omp parallel for schedule(static) \
        shared(lokal, global, n_lokal, N) default(none)
    for (int i = 0; i < n_lokal; i++) {
        double ax=0, ay=0, az=0;
        for (int j = 0; j < N; j++) {
            double dx = global[j].x - lokal[i].x;
            double dy = global[j].y - lokal[i].y;
            double dz = global[j].z - lokal[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            /* skip self secara efisien lewat softening (tidak perlu cek eksplisit) */
            double inv_r3 = 1.0 / (r2 * sqrt(r2));
            double fac = G_CONST * global[j].mass * inv_r3;
            ax += fac * dx;
            ay += fac * dy;
            az += fac * dz;
        }
        lokal[i].ax = ax;
        lokal[i].ay = ay;
        lokal[i].az = az;
    }
}

/* ================================================================
   leapfrog_kick   → update kecepatan setengah step
   leapfrog_drift  → update posisi satu step penuh
   Skema KDK: Kick(dt/2) → Drift(dt) → hitung_gaya → Kick(dt/2)
   Keunggulan: time-reversible, konservasi energi jauh lebih baik
   ================================================================ */
void leapfrog_kick(Particle *p, int n, double half_dt) {
    for (int i=0;i<n;i++){
        p[i].vx += p[i].ax * half_dt;
        p[i].vy += p[i].ay * half_dt;
        p[i].vz += p[i].az * half_dt;
    }
}
void leapfrog_drift(Particle *p, int n, double dt) {
    for (int i=0;i<n;i++){
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/* ================================================================
   hitung_energi — total energi sistem (kinetik + potensial)
   Dijalankan hanya oleh rank 0 pada all_particles.
   ================================================================ */
double hitung_energi(const Particle *p, int N) {
    double E_kin = 0.0, E_pot = 0.0;
    for (int i=0;i<N;i++){
        double v2 = p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz;
        E_kin += 0.5 * p[i].mass * v2;
        for (int j=i+1;j<N;j++){
            double dx=p[j].x-p[i].x, dy=p[j].y-p[i].y, dz=p[j].z-p[i].z;
            double r=sqrt(dx*dx+dy*dy+dz*dz+SOFTENING*SOFTENING);
            E_pot -= G_CONST * p[i].mass * p[j].mass / r;
        }
    }
    return E_kin + E_pot;
}

/* ================================================================
   MAIN
   ================================================================ */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    /* --- parameter dari argumen atau default --- */
    int    N        = (argc > 1) ? atoi(argv[1]) : 512;
    int    N_steps  = (argc > 2) ? atoi(argv[2]) : 200;
    int    n_thread = (argc > 3) ? atoi(argv[3]) : 1;
    double dt       = DT_DEF;
    int    out_freq = 10;

    omp_set_num_threads(n_thread);

    if (N % n_procs != 0) {
        if (rank==0) fprintf(stderr,"[ERROR] N=%d harus habis dibagi P=%d\n",N,n_procs);
        MPI_Finalize(); return 1;
    }
    int n_lokal = N / n_procs;

    /* --- alokasi --- */
    Particle *all   = malloc(N       * sizeof(Particle));
    Particle *lokal = malloc(n_lokal * sizeof(Particle));

    /* --- inisialisasi (rank 0 saja) --- */
    if (rank == 0) {
        inisialisasi_plummer(all, N, 42);
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║   N-Body Paralel — FIXED (MPI + OpenMP)          ║\n");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  N partikel      : %-29d ║\n", N);
        printf("║  N steps         : %-29d ║\n", N_steps);
        printf("║  dt (adimensional): %-28.4f ║\n", dt);
        printf("║  MPI proses      : %-29d ║\n", n_procs);
        printf("║  OpenMP threads  : %-29d ║\n", n_thread);
        printf("║  Softening ε     : %-29.3f ║\n", SOFTENING);
        printf("╚══════════════════════════════════════════════════╝\n\n");
    }

    /* broadcast dan scatter */
    MPI_Bcast(all, N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Scatter(all,   n_lokal * sizeof(Particle), MPI_BYTE,
                lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                0, MPI_COMM_WORLD);

    /* Energi awal */
    double E0 = 0.0;
    if (rank == 0) {
        E0 = hitung_energi(all, N);
        printf("Energi awal E0 = %.6e\n\n", E0);
        printf("  %-6s  %-14s  %-14s  %-14s  %-10s\n",
               "Step","E_kin","E_pot","E_tot","ΔE/E0 (%)");
        printf("  %s\n", "──────────────────────────────────────────────────────────");
    }
    MPI_Bcast(&E0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ================================================================
       Hitung akselerasi awal sebelum loop (butuh untuk KDK pertama)
       ================================================================ */
    hitung_akselerasi_omp(lokal, n_lokal, all, N);
    MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                  all,   n_lokal * sizeof(Particle), MPI_BYTE,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double t_force=0, t_comm=0, t_integ=0;

    /* ================================================================
       MAIN LOOP — Leapfrog KDK
       K = Kick (update v setengah step)
       D = Drift (update posisi penuh)
       K = Kick (update v setengah step lagi)
       ================================================================ */
    for (int step = 1; step <= N_steps; step++) {

        /* --- KICK pertama: v += a * dt/2 --- */
        double t0 = MPI_Wtime();
        leapfrog_kick(lokal, n_lokal, 0.5 * dt);

        /* --- DRIFT: x += v * dt --- */
        leapfrog_drift(lokal, n_lokal, dt);
        t_integ += MPI_Wtime() - t0;

        /* --- KOMUNIKASI: kumpulkan posisi baru ke semua rank --- */
        t0 = MPI_Wtime();
        MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                      all,   n_lokal * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - t0;

        /* --- Hitung akselerasi baru (dengan posisi yang sudah di-drift) --- */
        t0 = MPI_Wtime();
        hitung_akselerasi_omp(lokal, n_lokal, all, N);
        t_force += MPI_Wtime() - t0;

        /* --- KICK kedua: v += a_baru * dt/2 --- */
        t0 = MPI_Wtime();
        leapfrog_kick(lokal, n_lokal, 0.5 * dt);
        t_integ += MPI_Wtime() - t0;

        /* --- Output & validasi energi --- */
        if (step % out_freq == 0 || step == N_steps) {
            /* Kumpulkan partikel final dulu */
            MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                          all,   n_lokal * sizeof(Particle), MPI_BYTE,
                          MPI_COMM_WORLD);
            if (rank == 0) {
                double E_kin=0, E_pot=0;
                for(int i=0;i<N;i++){
                    double v2=all[i].vx*all[i].vx+all[i].vy*all[i].vy+all[i].vz*all[i].vz;
                    E_kin+=0.5*all[i].mass*v2;
                }
                /* potensial hanya sample O(N) biar cepat untuk monitoring */
                for(int i=0;i<N;i++)
                    for(int j=i+1;j<N;j++){
                        double dx=all[j].x-all[i].x,dy=all[j].y-all[i].y,dz=all[j].z-all[i].z;
                        double r=sqrt(dx*dx+dy*dy+dz*dz+SOFTENING*SOFTENING);
                        E_pot -= G_CONST*all[i].mass*all[j].mass/r;
                    }
                double E_tot = E_kin + E_pot;
                double dE    = (E_tot - E0) / fabs(E0) * 100.0;
                printf("  %-6d  %-14.4e  %-14.4e  %-14.4e  %+.4f%%\n",
                       step, E_kin, E_pot, E_tot, dE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_start;

    /* Agregasi waktu */
    double tf_max, tc_max, ti_max;
    MPI_Reduce(&t_force, &tf_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,  &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_integ, &ti_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double E_final = hitung_energi(all, N);
        double dE_pct  = (E_final - E0) / fabs(E0) * 100.0;
        printf("\n╔══════════════════════════════════════════════════╗\n");
        printf("║  HASIL AKHIR                                     ║\n");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  Energi awal      : %-28.4e ║\n", E0);
        printf("║  Energi akhir     : %-28.4e ║\n", E_final);
        printf("║  Error ΔE/E₀      : %-27.4f%% ║\n", dE_pct);
        printf("║  Status           : %-28s ║\n",
               fabs(dE_pct)<1.0 ? "STABIL ✓ (<1%)" : fabs(dE_pct)<5.0 ?
               "OK (<5%)" : "PERINGATAN (>5%)");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  PROFIL WAKTU                                    ║\n");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  Total            : %-28.4f ║\n", t_total);
        printf("║  Hitung gaya      : %.4f s (%.1f%%)%*s║\n",
               tf_max, 100.*tf_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",tf_max,100.*tf_max/t_total)),"");
        printf("║  Komunikasi MPI   : %.4f s (%.1f%%)%*s║\n",
               tc_max, 100.*tc_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",tc_max,100.*tc_max/t_total)),"");
        printf("║  Integrasi        : %.4f s (%.1f%%)%*s║\n",
               ti_max, 100.*ti_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",ti_max,100.*ti_max/t_total)),"");
        printf("╚══════════════════════════════════════════════════╝\n");
    }

    free(all); free(lokal);
    MPI_Finalize();
    return 0;
}
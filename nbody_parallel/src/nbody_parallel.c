/*
 * nbody_parallel.c
 * ============================================================
 * Implementasi PARALEL simulasi N-Body gravitasi
 * menggunakan MPI (distribusi proses) + OpenMP (threading)
 *
 * Strategi paralelisasi:
 *   - MPI: setiap rank memiliki N/P partikel lokal
 *   - OpenMP: komputasi gaya di-parallelize dengan #pragma omp parallel for
 *   - MPI_Allgather: sinkronisasi posisi global setelah setiap step
 *
 * Kompleksitas waktu: O(N^2 / P) per timestep per proses
 * Kompleksitas komunikasi: O(N) per MPI_Allgather
 *
 * Kompilasi:
 *   mpicc -O2 -fopenmp -o nbody_parallel nbody_parallel.c nbody_utils.c -lm
 *
 * Penggunaan:
 *   mpirun -np 4 ./nbody_parallel -n 256 -t 50 -dt 0.01
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_parallel -n 256 -t 50
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include "../include/nbody.h"

/* ============================================================
   Struktur flat untuk MPI communication (lebih efisien daripada
   kirim struct Particle langsung karena padding)
   ============================================================ */
typedef struct {
    double x, y, z;       /* posisi */
    double vx, vy, vz;    /* kecepatan */
    double mass;           /* massa */
} ParticleFlat;

/* ============================================================
   Pack partikel ke flat struct untuk dikirim lewat MPI
   ============================================================ */
static void pack_particles(const Particle *src, ParticleFlat *dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i].x    = src[i].x;
        dst[i].y    = src[i].y;
        dst[i].z    = src[i].z;
        dst[i].vx   = src[i].vx;
        dst[i].vy   = src[i].vy;
        dst[i].vz   = src[i].vz;
        dst[i].mass = src[i].mass;
    }
}

/* ============================================================
   Unpack flat struct kembali ke Particle
   ============================================================ */
static void unpack_particles(const ParticleFlat *src, Particle *dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i].x    = src[i].x;
        dst[i].y    = src[i].y;
        dst[i].z    = src[i].z;
        dst[i].vx   = src[i].vx;
        dst[i].vy   = src[i].vy;
        dst[i].vz   = src[i].vz;
        dst[i].mass = src[i].mass;
        dst[i].fx   = 0.0;
        dst[i].fy   = 0.0;
        dst[i].fz   = 0.0;
    }
}

/* ============================================================
   Hitung gaya partikel lokal terhadap SEMUA partikel global
   Dijalankan paralel dengan OpenMP per partikel lokal.

   Catatan desain:
   - Tiap thread menghitung gaya untuk subset partikel lokal
   - Tidak ada race condition karena tiap thread menulis ke
     indeks fx[i] yang berbeda (private per partikel)
   - Global particles (all_particles) hanya dibaca, tidak ditulis
   ============================================================ */
static void compute_forces_parallel_omp(
    Particle       *local_particles,   /* partikel lokal (tulis gaya) */
    int             n_local,           /* jumlah partikel lokal */
    const Particle *all_particles,     /* semua partikel global (baca saja) */
    int             n_global)          /* jumlah total partikel */
{
    /* Reset gaya lokal */
    for (int i = 0; i < n_local; i++) {
        local_particles[i].fx = 0.0;
        local_particles[i].fy = 0.0;
        local_particles[i].fz = 0.0;
    }

    /*
     * #pragma omp parallel for:
     *   - Iterasi loop i di-split ke beberapa thread
     *   - schedule(dynamic) bagus kalau beban tidak merata
     *   - schedule(static) lebih efisien untuk beban seragam (kita pakai ini)
     *   - reduction tidak dibutuhkan: tiap i menulis ke index berbeda
     */
    #pragma omp parallel for schedule(static) default(none) \
        shared(local_particles, all_particles, n_local, n_global)
    for (int i = 0; i < n_local; i++) {
        double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;

        /* Hitung gaya dari SEMUA partikel global terhadap partikel lokal i */
        for (int j = 0; j < n_global; j++) {
            /* Skip self-interaction */
            double dx = all_particles[j].x - local_particles[i].x;
            double dy = all_particles[j].y - local_particles[i].y;
            double dz = all_particles[j].z - local_particles[i].z;

            double dist2 = dx*dx + dy*dy + dz*dz;

            /* Skip jika partikel identik (self atau posisi persis sama) */
            if (dist2 < 1e-30) continue;

            dist2 += SOFTENING * SOFTENING;
            double inv_dist  = 1.0 / sqrt(dist2);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;

            double f_mag = G_CONST * local_particles[i].mass
                         * all_particles[j].mass * inv_dist3;

            fx_i += f_mag * dx;
            fy_i += f_mag * dy;
            fz_i += f_mag * dz;
        }

        /* Akumulasi ke partikel lokal (aman: tidak ada race) */
        local_particles[i].fx = fx_i;
        local_particles[i].fy = fy_i;
        local_particles[i].fz = fz_i;
    }
}

/* ============================================================
   Update posisi dan kecepatan lokal (Velocity Verlet)
   ============================================================ */
static void update_local(Particle *local, int n_local, double dt)
{
    for (int i = 0; i < n_local; i++) {
        double inv_m = 1.0 / local[i].mass;
        double ax = local[i].fx * inv_m;
        double ay = local[i].fy * inv_m;
        double az = local[i].fz * inv_m;

        local[i].vx += ax * dt;
        local[i].vy += ay * dt;
        local[i].vz += az * dt;

        local[i].x  += local[i].vx * dt;
        local[i].y  += local[i].vy * dt;
        local[i].z  += local[i].vz * dt;
    }
}

/* ============================================================
   MAIN PROGRAM PARALEL
   ============================================================ */
int main(int argc, char *argv[])
{
    /* ---- Inisialisasi MPI ---- */
    MPI_Init(&argc, &argv);

    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    /* ---- Parse parameter ---- */
    SimParams params;
    parse_args(argc, argv, &params);

    /* ==============================================================
       PERBAIKAN: arahkan output trajektori ke ../results/
       ============================================================== */
    if (rank == 0) {
        if (strcmp(params.output_file, "output_trajectory.csv") == 0 ||
            strncmp(params.output_file, "../results/", 11) != 0) {
            snprintf(params.output_file, sizeof(params.output_file),
                     "../results/parallel_trajectory.csv");
        }
    }

    int    N  = params.n_particles;
    int    T  = params.n_steps;
    double dt = params.dt;

    /* ---- Validasi: N harus habis dibagi n_procs ---- */
    if (N % n_procs != 0) {
        if (rank == 0) {
            fprintf(stderr,
                "[ERROR] N=%d tidak habis dibagi oleh jumlah proses P=%d.\n"
                "        Pilih N yang merupakan kelipatan %d.\n",
                N, n_procs, n_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n_local = N / n_procs;  /* Partikel per proses */

    /* ---- Info thread OpenMP ---- */
    int n_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║         Simulasi N-Body Paralel (MPI + OpenMP)       ║\n");
        printf("╠══════════════════════════════════════════════════════╣\n");
        printf("║  Partikel (N)       : %-31d ║\n", N);
        printf("║  Timestep (T)       : %-31d ║\n", T);
        printf("║  dt                 : %-31.4f ║\n", dt);
        printf("║  Proses MPI (P)     : %-31d ║\n", n_procs);
        printf("║  Thread OpenMP/rank : %-31d ║\n", n_threads);
        printf("║  Total core logis   : %-31d ║\n", n_procs * n_threads);
        printf("║  Partikel per rank  : %-31d ║\n", n_local);
        printf("╚══════════════════════════════════════════════════════╝\n\n");
    }

    /* ---- Alokasi memori ---- */
    /* Semua partikel (inisialisasi di rank 0, lalu di-broadcast) */
    Particle *all_particles   = (Particle *)malloc(N * sizeof(Particle));
    /* Partikel lokal milik rank ini */
    Particle *local_particles = (Particle *)malloc(n_local * sizeof(Particle));
    /* Buffer flat untuk komunikasi MPI */
    ParticleFlat *flat_all   = (ParticleFlat *)malloc(N * sizeof(ParticleFlat));
    ParticleFlat *flat_local = (ParticleFlat *)malloc(n_local * sizeof(ParticleFlat));

    if (!all_particles || !local_particles || !flat_all || !flat_local) {
        fprintf(stderr, "[RANK %d ERROR] Gagal alokasi memori\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ---- Inisialisasi partikel (hanya rank 0) ---- */
    if (rank == 0) {
        init_particles(all_particles, N, params.seed);
        printf("[INFO] Rank 0: inisialisasi %d partikel selesai\n", N);
    }

    /* ---- Broadcast semua partikel ke semua rank ----
     * Kita broadcast struct langsung (karena compiler sama, padding sama).
     * Untuk portabilitas produksi, gunakan MPI derived type atau serialisasi.
     */
    MPI_Bcast(all_particles, N * sizeof(Particle), MPI_BYTE,
              0, MPI_COMM_WORLD);

    /* ---- Distribusi partikel lokal ke masing-masing rank ----
     * Rank 0 scatter, setiap rank menerima slice n_local partikel
     */
    MPI_Scatter(all_particles,   n_local * sizeof(Particle), MPI_BYTE,
                local_particles, n_local * sizeof(Particle), MPI_BYTE,
                0, MPI_COMM_WORLD);

    /* ---- Hitung energi awal (rank 0 saja) ---- */
    Energy E_init = {0.0, 0.0, 0.0};
    if (rank == 0) {
        E_init = compute_energy(all_particles, N);
        printf("[VALIDASI] Energi awal: E_tot = %.6e J\n", E_init.total);
    }
    /* Broadcast energi awal ke semua rank untuk validasi lokal */
    MPI_Bcast(&E_init, sizeof(Energy), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ---- Buka file output (rank 0 saja) ---- */
    FILE *fp_out = NULL;
    if (rank == 0) {
        fp_out = fopen(params.output_file, "w");
        if (fp_out) {
            fprintf(fp_out, "step,time,particle_id,x,y,z,vx,vy,vz,mass\n");
            save_state(fp_out, all_particles, N, 0, 0.0);
        }
    }

    /* ================================================================
       MAIN LOOP SIMULASI PARALEL
       ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_total = MPI_Wtime();
    double t_force_total = 0.0;
    double t_comm_total  = 0.0;
    double t_integ_total = 0.0;

    if (rank == 0) {
        printf("\n[SIMULASI] Mulai %d timestep...\n", T);
        printf("─────────────────────────────────────────────────────────────────\n");
    }

    for (int step = 1; step <= T; step++) {
        double sim_time = step * dt;

        /* ============================================================
           FASE 1: Hitung gaya (MPI x OpenMP)
           - Tiap rank menghitung gaya lokal terhadap SEMUA partikel global
           - OpenMP parallelkan loop partikel lokal
           ============================================================ */
        double t0 = MPI_Wtime();
        compute_forces_parallel_omp(local_particles, n_local,
                                    all_particles, N);
        double t1 = MPI_Wtime();
        t_force_total += (t1 - t0);

        /* ============================================================
           FASE 2: Update posisi & kecepatan lokal (Velocity Verlet)
           ============================================================ */
        double t2 = MPI_Wtime();
        update_local(local_particles, n_local, dt);
        double t3 = MPI_Wtime();
        t_integ_total += (t3 - t2);

        /* ============================================================
           FASE 3: Sinkronisasi global (MPI_Allgather)
           - Setiap rank mengirim partikel lokalnya
           - Semua rank menerima semua partikel (global state diperbarui)
           ============================================================ */
        double t4 = MPI_Wtime();
        MPI_Allgather(local_particles, n_local * sizeof(Particle), MPI_BYTE,
                      all_particles,  n_local * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);
        double t5 = MPI_Wtime();
        t_comm_total += (t5 - t4);

        /* ============================================================
           FASE 4: Output dan validasi (rank 0 saja)
           ============================================================ */
        if (step % params.output_freq == 0 || step == T) {
            if (rank == 0) {
                double elapsed = MPI_Wtime() - t_start_total;
                Energy E_now = compute_energy(all_particles, N);
                print_energy(step, E_now, elapsed);

                /* Validasi konservasi energi */
                if (!validate_energy_conservation(E_init.total, E_now.total, 0.05)) {
                    printf("  [WARNING] Energi menyimpang >5%% pada step %d!\n", step);
                }

                if (fp_out) save_state(fp_out, all_particles, N, step, sim_time);
            }
        }
    }

    /* ---- Barrier untuk sinkronisasi waktu akhir ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_start_total;

    /* ---- Agregasi statistik waktu dari semua rank ---- */
    double t_force_max, t_comm_max, t_integ_max;
    MPI_Reduce(&t_force_total, &t_force_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_total,  &t_comm_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_integ_total, &t_integ_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- Cetak ringkasan (rank 0) ---- */
    if (rank == 0) {
        Energy E_final = compute_energy(all_particles, N);
        double rel_err = fabs((E_final.total - E_init.total) / E_init.total) * 100.0;

        printf("─────────────────────────────────────────────────────────────────\n");
        printf("\n[HASIL AKHIR]\n");
        printf("  Energi awal  : %.6e J\n", E_init.total);
        printf("  Energi akhir : %.6e J\n", E_final.total);
        printf("  Error relatif: %.4f %%\n", rel_err);
        if (validate_energy_conservation(E_init.total, E_final.total, 0.05))
            printf("  Status       : VALID (konservasi energi terjaga)\n");
        else
            printf("  Status       : PERINGATAN (energi menyimpang)\n");

        printf("\n[PROFIL WAKTU PARALEL (rank terlambat)]\n");
        printf("  Total waktu    : %.4f detik\n", t_total);
        printf("  Waktu gaya     : %.4f detik (%.1f%%)\n",
               t_force_max, 100.0 * t_force_max / t_total);
        printf("  Waktu komunikasi: %.4f detik (%.1f%%)\n",
               t_comm_max, 100.0 * t_comm_max / t_total);
        printf("  Waktu integrasi: %.4f detik (%.1f%%)\n",
               t_integ_max, 100.0 * t_integ_max / t_total);
        printf("  Throughput     : %.2f partikel-pasang/detik\n",
               (double)N * (double)N * T / t_total);
        printf("  Konfigurasi    : %d MPI proses x %d thread = %d core\n",
               n_procs, n_threads, n_procs * n_threads);

        /* ============================================================
           PERBAIKAN: simpan timing ke ../results/
           ============================================================ */
        FILE *fp_time = fopen("../results/parallel_timing.txt", "w");
        if (fp_time) {
            fprintf(fp_time, "mode=parallel\n");
            fprintf(fp_time, "N=%d\n", N);
            fprintf(fp_time, "T=%d\n", T);
            fprintf(fp_time, "mpi_procs=%d\n", n_procs);
            fprintf(fp_time, "omp_threads=%d\n", n_threads);
            fprintf(fp_time, "total_time=%.6f\n", t_total);
            fprintf(fp_time, "force_time=%.6f\n", t_force_max);
            fprintf(fp_time, "comm_time=%.6f\n", t_comm_max);
            fprintf(fp_time, "energy_error_pct=%.6f\n", rel_err);
            fclose(fp_time);
            printf("\n[INFO] Timing disimpan ke: ../results/parallel_timing.txt\n");
        }

        if (fp_out) {
            fclose(fp_out);
            printf("[INFO] Trajektori disimpan ke: %s\n", params.output_file);
        }
    }

    /* ---- Bersihkan memori ---- */
    free(all_particles);
    free(local_particles);
    free(flat_all);
    free(flat_local);

    MPI_Finalize();

    if (rank == 0)
        printf("\n[SELESAI] Simulasi paralel berhasil.\n\n");

    return EXIT_SUCCESS;
}
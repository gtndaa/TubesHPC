/*
 * nbody_serial.c
 * ============================================================
 * Implementasi SERIAL simulasi N-Body gravitasi
 *
 * Versi ini digunakan sebagai:
 *   1. Baseline pembanding kecepatan vs versi paralel
 *   2. Referensi kebenaran hasil (ground truth)
 *   3. Validasi silang dengan versi MPI+OpenMP
 *
 * Kompleksitas waktu: O(N^2) per timestep
 * Kompleksitas ruang: O(N)
 *
 * Kompilasi:
 *   gcc -O2 -o nbody_serial nbody_serial.c nbody_utils.c -lm
 *
 * Penggunaan:
 *   ./nbody_serial -n 256 -t 50 -dt 0.01 -seed 42
 * ============================================================
 */

#include "../include/nbody.h"

/* Timer portabel menggunakan clock_gettime */
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[])
{
    SimParams params;
    parse_args(argc, argv, &params);

    /* Override nama file output agar tidak konflik dengan paralel */
    if (strcmp(params.output_file, "output_trajectory.csv") == 0)
        strncpy(params.output_file, "../results/serial_trajectory.csv",
                sizeof(params.output_file)-1);

    print_params(&params, "SERIAL (single-core, no MPI/OpenMP)");

    int    N  = params.n_particles;
    int    T  = params.n_steps;
    double dt = params.dt;

    /* ---- Alokasi memori ---- */
    Particle *particles = (Particle *)malloc(N * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "[ERROR] Gagal alokasi memori untuk %d partikel\n", N);
        return EXIT_FAILURE;
    }

    /* ---- Inisialisasi partikel ---- */
    init_particles(particles, N, params.seed);
    printf("[INFO] Inisialisasi %d partikel selesai\n", N);

    /* ---- Hitung energi awal (untuk validasi) ---- */
    Energy E_init = compute_energy(particles, N);
    printf("[VALIDASI] Energi awal:\n");
    print_energy(0, E_init, 0.0);

    /* ---- Buka file output ---- */
    FILE *fp_out = fopen(params.output_file, "w");
    if (!fp_out) {
        fprintf(stderr, "[WARNING] Tidak bisa buka file output: %s\n",
                params.output_file);
    } else {
        /* Header CSV */
        fprintf(fp_out, "step,time,particle_id,x,y,z,vx,vy,vz,mass\n");
        save_state(fp_out, particles, N, 0, 0.0);
    }

    /* ---- Mulai simulasi ---- */
    printf("\n[SIMULASI] Mulai %d timestep...\n", T);
    printf("─────────────────────────────────────────────────────────────\n");

    double t_start_total = get_time_sec();
    double t_force_total = 0.0;
    double t_integ_total = 0.0;

    /* ================================================================
       MAIN LOOP SIMULASI SERIAL
       Setiap timestep:
         1. Hitung gaya O(N^2)
         2. Update posisi & kecepatan (Velocity Verlet)
         3. Validasi energi secara periodik
       ================================================================ */
    for (int step = 1; step <= T; step++) {
        double sim_time = step * dt;

        /* --- Step 1: Hitung gaya gravitasi (serial O(N^2)) --- */
        double t0 = get_time_sec();
        compute_forces_serial(particles, N);
        double t1 = get_time_sec();
        t_force_total += (t1 - t0);

        /* --- Step 2: Update posisi dan kecepatan (Velocity Verlet) --- */
        double t2 = get_time_sec();
        velocity_verlet_step(particles, N, dt);
        double t3 = get_time_sec();
        t_integ_total += (t3 - t2);

        /* --- Step 3: Output periodik dan validasi --- */
        if (step % params.output_freq == 0 || step == T) {
            double elapsed = get_time_sec() - t_start_total;
            Energy E_now = compute_energy(particles, N);
            print_energy(step, E_now, elapsed);

            /* Validasi konservasi energi (toleransi 5%) */
            if (!validate_energy_conservation(E_init.total, E_now.total, 0.05)) {
                printf("[WARNING] Energi menyimpang >5%% pada step %d! "
                       "Pertimbangkan dt lebih kecil.\n", step);
            }

            /* Simpan ke file */
            if (fp_out) save_state(fp_out, particles, N, step, sim_time);
        }
    }

    /* ---- Hitung energi akhir (validasi konservasi) ---- */
    Energy E_final = compute_energy(particles, N);
    double t_total = get_time_sec() - t_start_total;

    /* ---- Cetak ringkasan ---- */
    printf("─────────────────────────────────────────────────────────────\n");
    printf("\n[HASIL AKHIR]\n");
    printf("  Energi awal  : %.6e J\n", E_init.total);
    printf("  Energi akhir : %.6e J\n", E_final.total);
    double rel_err = fabs((E_final.total - E_init.total) / E_init.total) * 100.0;
    printf("  Error relatif: %.4f %%\n", rel_err);
    if (validate_energy_conservation(E_init.total, E_final.total, 0.05))
        printf("  Status       : VALID (konservasi energi terjaga)\n");
    else
        printf("  Status       : PERINGATAN (energi menyimpang)\n");

    printf("\n[PROFIL WAKTU SERIAL]\n");
    printf("  Total waktu    : %.4f detik\n", t_total);
    printf("  Waktu gaya     : %.4f detik (%.1f%%)\n",
           t_force_total, 100.0 * t_force_total / t_total);
    printf("  Waktu integrasi: %.4f detik (%.1f%%)\n",
           t_integ_total, 100.0 * t_integ_total / t_total);
    printf("  Throughput     : %.2f partikel-pasang/detik\n",
           (double)N * (N-1) / 2.0 * T / t_total);

    /* ---- Simpan waktu ke file untuk perbandingan ---- */
    FILE *fp_time = fopen("../results/serial_timing.txt", "w");
    if (fp_time) {
        fprintf(fp_time, "mode=serial\n");
        fprintf(fp_time, "N=%d\n", N);
        fprintf(fp_time, "T=%d\n", T);
        fprintf(fp_time, "total_time=%.6f\n", t_total);
        fprintf(fp_time, "force_time=%.6f\n", t_force_total);
        fprintf(fp_time, "energy_error_pct=%.6f\n", rel_err);
        fclose(fp_time);
        printf("\n[INFO] Timing disimpan ke: ../results/serial_timing.txt\n");
    }

    if (fp_out) {
        fclose(fp_out);
        printf("[INFO] Trajektori disimpan ke: %s\n", params.output_file);
    }

    free(particles);
    printf("\n[SELESAI] Simulasi serial berhasil.\n\n");
    return EXIT_SUCCESS;
}
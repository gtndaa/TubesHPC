/*
 * Compilation:
 *   gcc -O2 -o nbody_serial nbody_serial.c -lm
 *
 * Usage:
 *   ./nbody_serial [N_particle] [N_steps] [output_freq]
 *   ./nbody_serial 1024 200 10
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* adimensional constants */
#define G_CONST   1.0
#define SOFTENING 0.05
#define DT_DEF    0.001

/* Particle structure */
typedef struct {
    double x,  y,  z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
} Particle;

/* timer */
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Plummer initialize: to distribute galaxy particles */
static void inisialisasi_plummer(Particle *p, int N, unsigned int seed)
{
    srand(seed);
    double inv_N = 1.0 / N;

    for (int i = 0; i < N; i++) {
        /* Inverse CDF: r = 1/sqrt(u^(-2/3) - 1), u ~ Uniform(0,1)*/
        double u   = 0.001 + 0.998 * ((double)rand() / RAND_MAX);
        double r   = 1.0 / sqrt(pow(u, -2.0/3.0) - 1.0);
        double cos_theta = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        double phi       = 2.0 * M_PI * ((double)rand() / RAND_MAX);

        p[i].x = r * sin_theta * cos(phi);
        p[i].y = r * sin_theta * sin(phi);
        p[i].z = r * cos_theta;

        /* v_esc = sqrt(2 * |phi(r)|), phi(r) = -G*M/sqrt(r²+1) */
        double phi_r = -G_CONST * N / sqrt(r * r + 1.0);
        double v_max = sqrt(2.0 * fabs(phi_r)) * 0.5;

        /* rejection sampling for velocity distribution */
        double v, g;
        do {
            v = ((double)rand() / RAND_MAX) * v_max;
            g = v * v * pow(1.0 - v * v / (v_max * v_max + 1e-10), 3.5);
        } while (((double)rand() / RAND_MAX) > g / (v_max * v_max * 0.1 + 1e-10));

        double cos_tv = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double sin_tv = sqrt(1.0 - cos_tv * cos_tv);
        double phi_v  = 2.0 * M_PI * ((double)rand() / RAND_MAX);

        p[i].vx = v * sin_tv * cos(phi_v);
        p[i].vy = v * sin_tv * sin(phi_v);
        p[i].vz = v * cos_tv;

        p[i].ax   = 0.0;
        p[i].ay   = 0.0;
        p[i].az   = 0.0;
        p[i].mass = 1.0;
    }

    /* Center of mass correction: shift to zero */
    double cx=0,cy=0,cz=0, cvx=0,cvy=0,cvz=0;
    for (int i = 0; i < N; i++) {
        cx += p[i].x;  cy += p[i].y;  cz += p[i].z;
        cvx+= p[i].vx; cvy+= p[i].vy; cvz+= p[i].vz;
    }
    cx*=inv_N; cy*=inv_N; cz*=inv_N;
    cvx*=inv_N; cvy*=inv_N; cvz*=inv_N;
    for (int i = 0; i < N; i++) {
        p[i].x  -= cx;  p[i].y  -= cy;  p[i].z  -= cz;
        p[i].vx -= cvx; p[i].vy -= cvy; p[i].vz -= cvz;
    }
}

/* serial acceleration calculation */
static void hitung_akselerasi_serial(Particle *p, int N)
{
    /* reset acceleration */
    for (int i = 0; i < N; i++)
        p[i].ax = p[i].ay = p[i].az = 0.0;

    /* Calculate accelerations for each particle */
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = p[j].x - p[i].x;
            double dy = p[j].y - p[i].y;
            double dz = p[j].z - p[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + SOFTENING * SOFTENING;
            double inv_r3 = 1.0 / (r2 * sqrt(r2));

            /* a_i += G*m_j * r_ij / |r_ij|^3 */
            double fac_i = G_CONST * p[j].mass * inv_r3;
            p[i].ax += fac_i * dx;
            p[i].ay += fac_i * dy;
            p[i].az += fac_i * dz;

            /* a_j += -G*m_i * r_ij / |r_ij|^3  (Newton III) */
            double fac_j = G_CONST * p[i].mass * inv_r3;
            p[j].ax -= fac_j * dx;
            p[j].ay -= fac_j * dy;
            p[j].az -= fac_j * dz;
        }
    }
}

/* Leapfrog Integration with KDK scheme */
static void leapfrog_kick(Particle *p, int N, double half_dt)
{
    for (int i = 0; i < N; i++) {
        p[i].vx += p[i].ax * half_dt;
        p[i].vy += p[i].ay * half_dt;
        p[i].vz += p[i].az * half_dt;
    }
}
static void leapfrog_drift(Particle *p, int N, double dt)
{
    for (int i = 0; i < N; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/* energy calculation */
static double hitung_energi(const Particle *p, int N,
                             double *out_ekin, double *out_epot)
{
    double E_kin = 0.0, E_pot = 0.0;
    for (int i = 0; i < N; i++) {
        double v2 = p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz;
        E_kin += 0.5 * p[i].mass * v2;
        for (int j = i + 1; j < N; j++) {
            double dx = p[j].x - p[i].x;
            double dy = p[j].y - p[i].y;
            double dz = p[j].z - p[i].z;
            double r  = sqrt(dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING);
            E_pot -= G_CONST * p[i].mass * p[j].mass / r;
        }
    }
    if (out_ekin) *out_ekin = E_kin;
    if (out_epot) *out_epot = E_pot;
    return E_kin + E_pot;
}

/* Main function */
int main(int argc, char *argv[])
{
    int N        = (argc > 1) ? atoi(argv[1]) : 512;
    int N_steps  = (argc > 2) ? atoi(argv[2]) : 200;
    int out_freq = (argc > 3) ? atoi(argv[3]) : 10;
    double dt    = DT_DEF;

    printf("\n");
    printf("Simulasi N-Body SERIAL\n");
    printf("N particle       : %-24d\n", N);
    printf("N steps          : %-24d\n", N_steps);
    printf("dt (adimensional): %-24.4f\n", dt);
    printf("Softening ε      : %-24.3f\n", SOFTENING);
    printf("Distribution     : %-24s\n", "Plummer (seed=42)");
    printf("Integration      : %-24s\n", "Leapfrog KDK");

    /* memory allocation */
    Particle *p = (Particle *)malloc(N * sizeof(Particle));
    if (!p) {
        fprintf(stderr, "[ERROR] Gagal alokasi memori untuk %d partikel\n", N);
        return EXIT_FAILURE;
    }

    /* plummer initialization */
    inisialisasi_plummer(p, N, 42);
    printf("[INFO] Inisialisasi %d partikel (Plummer) selesai\n", N);

    /* initial energy */
    double E_kin0, E_pot0;
    double E0 = hitung_energi(p, N, &E_kin0, &E_pot0);
    printf("[VALIDASI] Energi awal E0 = %.6e\n", E0);
    printf("           E_kin = %.4e, E_pot = %.4e\n\n", E_kin0, E_pot0);

    /* initial acceleration */
    hitung_akselerasi_serial(p, N);

    /* output */
    printf("  %-6s  %-14s  %-14s  %-14s  %-12s\n",
           "Step", "E_kin", "E_pot", "E_tot", "ΔE/E0 (%)");
    printf("  %s\n",
           "──────────────────────────────────────────────────────────────");

    /* time profiling */
    double t_start  = get_time_sec();
    double t_force  = 0.0;
    double t_integ  = 0.0;

    /* KDK loop */
    for (int step = 1; step <= N_steps; step++) {

        /* first Kick v += a * dt/2 --- */
        double t0 = get_time_sec();
        leapfrog_kick(p, N, 0.5 * dt);

        /* DRIFT: x += v * dt --- */
        leapfrog_drift(p, N, dt);
        t_integ += get_time_sec() - t0;

        /* update acceleration */
        t0 = get_time_sec();
        hitung_akselerasi_serial(p, N);
        t_force += get_time_sec() - t0;

        /* second Kick v += a_baru * dt/2 --- */
        t0 = get_time_sec();
        leapfrog_kick(p, N, 0.5 * dt);
        t_integ += get_time_sec() - t0;


        if (step % out_freq == 0 || step == N_steps) {
            double E_kin, E_pot;
            double E_tot = hitung_energi(p, N, &E_kin, &E_pot);
            double dE    = (E_tot - E0) / fabs(E0) * 100.0;

            printf("  %-6d  %-14.4e  %-14.4e  %-14.4e  %+.4f%%\n",
                   step, E_kin, E_pot, E_tot, dE);

            if (fabs(dE) > 5.0)
                printf("  [WARNING] |ΔE/E0| = %.2f%% > 5%% pada step %d! "
                       "Coba dt lebih kecil.\n", fabs(dE), step);
        }
    }

    double t_total = get_time_sec() - t_start;

    /* calculate final energy */
    double E_kin_f, E_pot_f;
    double E_final = hitung_energi(p, N, &E_kin_f, &E_pot_f);
    double dE_pct  = (E_final - E0) / fabs(E0) * 100.0;

    /* final results */
    printf("HASIL AKHIR\n");
    printf("Energi awal   : %-27.4e \n", E0);
    printf("Energi akhir  : %-27.4e \n", E_final);
    printf("Error ΔE/E₀   : %-26.4f%%    \n", dE_pct);
    printf("Status        : %-27s \n",
           fabs(dE_pct) < 1.0 ? "STABLE ✓ (<1%)" :
           fabs(dE_pct) < 5.0 ? "OK (<5%)" : "WARNING (>5%)");
    printf("\nTime Profiling\n");
    printf("Total             : %-27.4f \n", t_total);
    printf("Force Calculation : %-6.4f s (%-6.1f%%)\n",
           t_force, 100.0 * t_force / t_total);
    printf("Integration       : %-6.4f s (%-6.1f%%)\n",
           t_integ, 100.0 * t_integ / t_total);

    FILE *fp = fopen("../results/serial_timing.txt", "w");
    if (fp) {
        fprintf(fp, "mode=serial\n");
        fprintf(fp, "N=%d\n", N);
        fprintf(fp, "T=%d\n", N_steps);
        fprintf(fp, "total_time=%.6f\n", t_total);
        fprintf(fp, "force_time=%.6f\n", t_force);
        fprintf(fp, "energy_error_pct=%.6f\n", dE_pct);
        fclose(fp);
    }

    free(p);
    return EXIT_SUCCESS;
}
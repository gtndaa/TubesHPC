/*
 * nbody_utils.c
 * Fungsi utilitas bersama untuk simulasi N-Body
 * (dipakai oleh versi serial maupun paralel)
 */

#include "../include/nbody.h"

/* ============================================================
   Inisialisasi Partikel
   Mengisi N partikel dengan posisi, kecepatan, dan massa acak.
   Seed digunakan agar hasil serial dan paralel identik.
   ============================================================ */
void init_particles(Particle *particles, int n, int seed)
{
    srand((unsigned int)seed);

    for (int i = 0; i < n; i++) {
        /* Posisi dalam kubus [-50, 50] meter */
        particles[i].x    = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        particles[i].y    = ((double)rand() / RAND_MAX - 0.5) * 100.0;
        particles[i].z    = ((double)rand() / RAND_MAX - 0.5) * 100.0;

        /* Kecepatan awal [-1, 1] m/s */
        particles[i].vx   = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        particles[i].vy   = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        particles[i].vz   = ((double)rand() / RAND_MAX - 0.5) * 2.0;

        /* Gaya diinisialisasi nol */
        particles[i].fx   = 0.0;
        particles[i].fy   = 0.0;
        particles[i].fz   = 0.0;

        /* Massa dalam rentang [1e10, 1e12] kg */
        particles[i].mass = (1.0 + (double)rand() / RAND_MAX * 99.0) * 1.0e10;
    }
}

/* ============================================================
   Hitung gaya gravitasi satu pasang partikel (i terhadap j)
   F_ij = G * m_i * m_j * (r_j - r_i) / (|r_ij|^2 + eps^2)^(3/2)
   Softening (eps) mencegah singularitas saat r -> 0
   ============================================================ */
void compute_force_pair(const Particle *pi, const Particle *pj,
                        double *fx, double *fy, double *fz)
{
    double dx = pj->x - pi->x;
    double dy = pj->y - pi->y;
    double dz = pj->z - pi->z;

    double dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
    double dist  = sqrt(dist2);
    double dist3 = dist2 * dist;  /* |r|^3 */

    double f_mag = G_CONST * pi->mass * pj->mass / dist3;

    *fx = f_mag * dx;
    *fy = f_mag * dy;
    *fz = f_mag * dz;
}

/* ============================================================
   Hitung semua gaya (versi serial O(N^2))
   Digunakan oleh nbody_serial.c dan juga sebagai referensi
   validasi untuk versi paralel.
   ============================================================ */
void compute_forces_serial(Particle *particles, int n)
{
    /* Reset gaya semua partikel */
    for (int i = 0; i < n; i++) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }

    /* Hitung pasangan (i, j) -- pakai simetri Newton: F_ij = -F_ji */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double fx, fy, fz;
            compute_force_pair(&particles[i], &particles[j], &fx, &fy, &fz);

            particles[i].fx += fx;
            particles[i].fy += fy;
            particles[i].fz += fz;

            /* Hukum Newton III: gaya balasan */
            particles[j].fx -= fx;
            particles[j].fy -= fy;
            particles[j].fz -= fz;
        }
    }
}

/* ============================================================
   Integrasi Velocity Verlet (satu langkah waktu)
   
   Algoritma:
     1. x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
     2. a(t+dt) = F(t+dt) / m   [dihitung ulang setelah posisi update]
     3. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
   
   Di sini kita implementasikan versi sederhana (Leapfrog-like):
     v_half = v(t) + 0.5*(F/m)*dt
     x(t+dt) = x(t) + v_half*dt
     [hitung F baru]
     v(t+dt) = v_half + 0.5*(F_new/m)*dt
   
   Catatan: fungsi ini hanya melakukan half-step update posisi+kecepatan.
   Step lengkap dilakukan di main loop.
   ============================================================ */
void velocity_verlet_step(Particle *particles, int n, double dt)
{
    double half_dt = 0.5 * dt;

    for (int i = 0; i < n; i++) {
        double inv_m = 1.0 / particles[i].mass;
        double ax = particles[i].fx * inv_m;
        double ay = particles[i].fy * inv_m;
        double az = particles[i].fz * inv_m;

        /* Update kecepatan: v += a * dt */
        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
        particles[i].vz += az * dt;

        /* Update posisi: x += v * dt */
        particles[i].x  += particles[i].vx * half_dt;
        particles[i].y  += particles[i].vy * half_dt;
        particles[i].z  += particles[i].vz * half_dt;
    }
}

/* ============================================================
   Hitung energi mekanik total sistem
   
   E_kin  = Σ 0.5 * m_i * |v_i|^2
   E_pot  = Σ_{i<j} -G * m_i * m_j / r_ij
   E_total = E_kin + E_pot
   ============================================================ */
Energy compute_energy(const Particle *particles, int n)
{
    Energy e = {0.0, 0.0, 0.0};

    /* Energi kinetik */
    for (int i = 0; i < n; i++) {
        double v2 = particles[i].vx * particles[i].vx
                  + particles[i].vy * particles[i].vy
                  + particles[i].vz * particles[i].vz;
        e.kinetic += 0.5 * particles[i].mass * v2;
    }

    /* Energi potensial gravitasi */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r  = sqrt(dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING);
            e.potential -= G_CONST * particles[i].mass * particles[j].mass / r;
        }
    }

    e.total = e.kinetic + e.potential;
    return e;
}

/* ============================================================
   Validasi konservasi energi
   Mengecek apakah perubahan energi masih dalam batas toleransi.
   Toleransi relatif: |ΔE/E0| < tol
   Return 1 jika valid, 0 jika tidak.
   ============================================================ */
int validate_energy_conservation(double E0, double E1, double tolerance)
{
    if (fabs(E0) < 1e-30) return 1; /* Hindari division by zero */
    double rel_error = fabs((E1 - E0) / E0);
    return (rel_error < tolerance) ? 1 : 0;
}

/* ============================================================
   Print statistik energi
   ============================================================ */
void print_energy(int step, Energy e, double elapsed)
{
    printf("  Step %5d | E_kin=%12.4e | E_pot=%12.4e | E_tot=%12.4e | t=%.3f s\n",
           step, e.kinetic, e.potential, e.total, elapsed);
}

/* ============================================================
   Simpan state partikel ke file CSV
   ============================================================ */
void save_state(FILE *fp, const Particle *particles, int n, int step, double time)
{
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d,%.6f,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                step, time, i,
                particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz,
                particles[i].mass);
    }
}

/* ============================================================
   Print parameter simulasi
   ============================================================ */
void print_params(const SimParams *p, const char *mode)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        Simulasi N-Body Gravitasi             ║\n");
    printf("║  Mode: %-37s║\n", mode);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Partikel (N)    : %-25d ║\n", p->n_particles);
    printf("║  Timestep (T)    : %-25d ║\n", p->n_steps);
    printf("║  dt              : %-25.4f ║\n", p->dt);
    printf("║  Random seed     : %-25d ║\n", p->seed);
    printf("║  Output file     : %-25s ║\n", p->output_file);
    printf("╚══════════════════════════════════════════════╝\n\n");
}

/* ============================================================
   Set nilai default parameter simulasi
   ============================================================ */
void default_params(SimParams *p)
{
    p->n_particles = DEFAULT_N;
    p->n_steps     = DEFAULT_T;
    p->dt          = DEFAULT_DT;
    p->seed        = DEFAULT_SEED;
    p->output_freq = 10;
    strncpy(p->output_file, "output_trajectory.csv", sizeof(p->output_file)-1);
}

/* ============================================================
   Parse argumen baris perintah
   Penggunaan: program -n <N> -t <T> -dt <dt> -seed <seed> -o <file>
   ============================================================ */
void parse_args(int argc, char *argv[], SimParams *p)
{
    default_params(p);
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-n") == 0)    p->n_particles = atoi(argv[i+1]);
        if (strcmp(argv[i], "-t") == 0)    p->n_steps     = atoi(argv[i+1]);
        if (strcmp(argv[i], "-dt") == 0)   p->dt          = atof(argv[i+1]);
        if (strcmp(argv[i], "-seed") == 0) p->seed        = atoi(argv[i+1]);
        if (strcmp(argv[i], "-freq") == 0) p->output_freq = atoi(argv[i+1]);
        if (strcmp(argv[i], "-o") == 0)    strncpy(p->output_file, argv[i+1],
                                                   sizeof(p->output_file)-1);
    }
}
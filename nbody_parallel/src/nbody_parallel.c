/*
 * Compile:
 *   mpicc -O2 -fopenmp -o nbody_parallel nbody_parallel.c -lm
 *
 * Usage:
 *   mpirun --oversubscribe --allow-run-as-root \
 *          --mca plm_rsh_agent "" -np 4 \
 *          ./nbody_parallel 512 200 4
 *   (arg: N_particle  N_steps  N_threads)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// Adimensional constants
#define G_CONST   1.0      // gravitasi adimensional
#define SOFTENING 0.05     // ~5% dari radius sistem, cegah singularity
#define DT_DEF    0.001    // timestep untuk unit adimensional

// Particle structure
typedef struct {
    double x,  y,  z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
} Particle;

// Plummer initialize: to distribute galaxy particles
void inisialisasi_plummer(Particle *p, int N, unsigned int seed) {
    srand(seed);
    double inv_N = 1.0 / N;

    for (int i = 0; i < N; i++) {
        // Inverse CDF: r = 1/sqrt(u^(-2/3) - 1), u ~ Uniform(0,1)
        double u   = 0.001 + 0.998 * ((double)rand() / RAND_MAX);
        double r   = 1.0 / sqrt(pow(u, -2.0/3.0) - 1.0);
        double cos_theta = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        double phi       = 2.0 * M_PI * ((double)rand() / RAND_MAX);

        p[i].x = r * sin_theta * cos(phi);
        p[i].y = r * sin_theta * sin(phi);
        p[i].z = r * cos_theta;

        // v_esc = sqrt(2 * |phi(r)|), phi(r) = -G*M/sqrt(r²+1)
        double phi_r = -G_CONST * N / sqrt(r*r + 1.0);
        double v_max = sqrt(2.0 * fabs(phi_r)) * 0.5;

        // rejection sampling for velocity distribution
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
        p[i].mass = 1.0;   // adimensional mass
    }

    // Center of mass correction: shift to zero
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

// Local acceleration calculation
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

/*   Leapfrog Integration with KDK scheme
   → leapfrog_kick: update velocity of half step
   → leapfrog_drift: update position of full step */
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

// Energy calculation
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

// Main function
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // default parameters
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

    // memory allocation
    Particle *all   = malloc(N       * sizeof(Particle));
    Particle *lokal = malloc(n_lokal * sizeof(Particle));

    // initialization
    if (rank == 0) {
        inisialisasi_plummer(all, N, 42);
        printf("N-Body Paralel  (MPI + OpenMP Implementation)\n");
        printf("N particle\t\t: %-29d \n", N);
        printf("N steps\t\t\t: %-29d \n", N_steps);
        printf("dt (adimensional)\t: %-28.4f \n", dt);
        printf("MPI processors\t\t: %-29d \n", n_procs);
        printf("OpenMP threads\t\t: %-29d \n", n_thread);
        printf("Softening ε\t\t: %-29.3f \n\n", SOFTENING);
    }

    // broadcast dan scatter
    MPI_Bcast(all, N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Scatter(all,   n_lokal * sizeof(Particle), MPI_BYTE,
                lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // initial energy
    double E0 = 0.0;
    if (rank == 0) {
        E0 = hitung_energi(all, N);
        printf("Energi awal E0 = %.6e\n\n", E0);
        printf("  %-6s  %-14s  %-14s  %-14s  %-10s\n",
               "Step","E_kin","E_pot","E_tot","ΔE/E0 (%)");
        printf("  %s\n", "──────────────────────────────────────────────────────────");
    }
    MPI_Bcast(&E0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initial acceleration
    hitung_akselerasi_omp(lokal, n_lokal, all, N);
    MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                  all,   n_lokal * sizeof(Particle), MPI_BYTE,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double t_force=0, t_comm=0, t_integ=0;

    // KDK loop
    for (int step = 1; step <= N_steps; step++) {

        // first Kick v += a * dt/2
        double t0 = MPI_Wtime();
        leapfrog_kick(lokal, n_lokal, 0.5 * dt);

        // drift x += v * dt
        leapfrog_drift(lokal, n_lokal, dt);
        t_integ += MPI_Wtime() - t0;

        // MPI communication
        t0 = MPI_Wtime();
        MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                      all,   n_lokal * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - t0;

        // update acceleration
        t0 = MPI_Wtime();
        hitung_akselerasi_omp(lokal, n_lokal, all, N);
        t_force += MPI_Wtime() - t0;

        // second Kick: v += a_baru * dt/2
        t0 = MPI_Wtime();
        leapfrog_kick(lokal, n_lokal, 0.5 * dt);
        t_integ += MPI_Wtime() - t0;

        // validation
        if (step % out_freq == 0 || step == N_steps) {
            // collect all particles
            MPI_Allgather(lokal, n_lokal * sizeof(Particle), MPI_BYTE,
                          all,   n_lokal * sizeof(Particle), MPI_BYTE,
                          MPI_COMM_WORLD);
            if (rank == 0) {
                double E_kin=0, E_pot=0;
                for(int i=0;i<N;i++){
                    double v2=all[i].vx*all[i].vx+all[i].vy*all[i].vy+all[i].vz*all[i].vz;
                    E_kin+=0.5*all[i].mass*v2;
                }
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

    // time aggregation
    double tf_max, tc_max, ti_max;
    MPI_Reduce(&t_force, &tf_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,  &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_integ, &ti_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double E_final = hitung_energi(all, N);
        double dE_pct  = (E_final - E0) / fabs(E0) * 100.0;
        printf("\nFinal Results\n");
        printf("Initial energy      : %-28.4e \n", E0);
        printf("Final energy        : %-28.4e \n", E_final);
        printf("Error ΔE/E₀         : %-27.4f %% \n", dE_pct);
        printf("Status              : %-28s \n",
               fabs(dE_pct)<1.0 ? "STABLE ✓ (<1%)" : fabs(dE_pct)<5.0 ?
               "OK (<5%)" : "WARNING (>5%)");
        printf("\nTime Profiling\n");
        printf("Total            : %-28.4f \n", t_total);
        printf("Force Calculation: %.4f s (%.1f%%)%*s\n",
               tf_max, 100.*tf_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",tf_max,100.*tf_max/t_total)),"");
        printf("MPI Communication: %.4f s (%.1f%%)%*s\n",
               tc_max, 100.*tc_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",tc_max,100.*tc_max/t_total)),"");
        printf("Integration      : %.4f s (%.1f%%)%*s\n",
               ti_max, 100.*ti_max/t_total,
               (int)(18-snprintf(NULL,0,"%.4f s (%.1f%%)",ti_max,100.*ti_max/t_total)),"");
    }

    free(all); free(lokal);
    MPI_Finalize();
    return 0;
}
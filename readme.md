# N-Body Simulation

Simulasi N-Body gravitasi dalam dua mode:

* **Serial**
* **Paralel (MPI + OpenMP)**


## Kebutuhan

* GCC atau Clang
* MPI (`mpicc`, `mpirun`)
* OpenMP
* Linux / WSL direkomendasikan

## Build

Jalankan:

```bash
make
```

Hasil build:

* `nbody_serial`
* `nbody_parallel`

## Menjalankan Versi Serial

Contoh:

```bash
./nbody_serial -n 512 -t 100 -dt 0.01 -seed 42
```

Parameter:

* `-n` jumlah partikel
* `-t` jumlah timestep
* `-dt` ukuran timestep
* `-seed` seed random
* `-freq` frekuensi output
* `-o` nama file output CSV

Contoh output:

* `result/serial_trajectory.csv`
* `result/serial_timing.txt`

## Menjalankan Versi Paralel

Contoh:

```bash
OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_parallel -n 512 -t 100 -dt 0.01 -seed 42
```

Keterangan:

* `mpirun -np 2` = 2 proses MPI
* `OMP_NUM_THREADS=4` = 4 thread OpenMP per proses

Contoh output:

* `result/parallel_trajectory.csv`
* `result/parallel_timing.txt`

## Melihat Hasil

File hasil ada di folder `result/`.

Biasanya yang dicek:

* trajektori partikel (`*.csv`)
* waktu eksekusi (`*_timing.txt`)
* grafik dari script di `scripts/`

## Membersihkan Build

```bash
make clean
```

## Catatan

* Mode serial dipakai sebagai pembanding dan validasi.
* Mode paralel memakai MPI untuk pembagian proses dan OpenMP untuk paralelisasi loop gaya.
* Hasil serial dan paralel sebaiknya dibandingkan untuk memastikan simulasi konsisten.

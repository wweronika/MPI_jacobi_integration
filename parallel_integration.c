 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

void print_mat(double* mat, int width, int height) {
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            printf("%lf ", mat[i*width+j]);
        }
        printf("\n");
    }
}

void print_vec_double(double* vec, int size) {
    for(int i=0; i<size; i++) {
        printf("%lf ", vec[i]);
    }
    printf("\n");
}
    
void print_vec_int(int* vec, int size) {
    for(int i=0; i<size; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

void initialize_zero_matrix(double* mat, int width, int height) {
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            mat[i*width+j] = 0.0; 
        }
    }
}

void initialize_matrix(double* mat, bool is_first, bool is_last, int size, double inv_dx2) {

    if(is_first) {

        mat[0] = 1.0;

        for(int i=1; i<size; i++) {
            mat[i*(size+1)+i] = -2.0 * inv_dx2;
            mat[i*(size+1)+i-1] = mat[i*(size+1)+i+1] = inv_dx2;
        }
    }

    else if(is_last) {

        mat[size*(size+1)-1] = 1.0;

        for(int i=0; i<size-1; i++) {
            mat[i*(size+1)+i+1] = -2.0 * inv_dx2;
            mat[i*(size+1)+i] = mat[i*(size+1)+i+2] = inv_dx2;
        }
    }

    else {
        
        for(int i=0; i<size; i++) {
            mat[i*(size+2)+i+1] = -2.0 * inv_dx2;
            mat[i*(size+2)+i] = mat[i*(size+2)+i+2] = inv_dx2;
        }
    }

}

void initialize_xsol(double* xsol, int size) {
    for(int i=0; i<size; i++) {
    //    if(i ==0 || i == size-1) {xsol[i] = -12.0;}
    //    else {xsol[i] = 1.0;}
          xsol[i] = 1.0; 
    }
}

void initialize_rhs(double* rhs, int size, bool is_first, bool is_last) {
    for(int i=0; i<size; i++) {
        rhs[i] = -1.0;
    }

    if(is_first) {
        rhs[0] = 0.0;
    }

    else if(is_last) {
        rhs[size-1] = 0.0;
    }
}

// Initialize the 1D mesh
void initialize_x_mesh(double* x, int n_points) {
  
  for (int i = 0 ; i < n_points; i++) {
    x[i] = (double)i / (n_points - 1);  // i * dx
  }
}

// Main computational iteration
void jacobi_iteration(double* mat, double* xsol, double* rhs, int size, bool is_first, bool is_last) {

    double sigma = 0.0;

    if(is_first) {
        for(int i=0; i<size; i++) {
            sigma = 0.0;

            for(int j=0; j<size+1; j++) {
                if(j != i) {
                    sigma += xsol[j] * mat[i*(size+1)+j];
                }
            }
            xsol[i] = (rhs[i] - sigma) / mat[i*(size+1)+i];

        }
    }

    else if(is_last) {
        for(int i=0; i<size; i++) {
            sigma = 0.0;

            for(int j=0; j<size+1; j++) {
                if(j != i+1) {
                    sigma += xsol[j] * mat[i*(size+1)+j];
                }
            }
            xsol[i+1] = (rhs[i] - sigma) / mat[i*(size+1)+i+1];
        }

    }

    else {
        
        for(int i=0; i<size; i++) {
            sigma = 0.0;

            for(int j=0; j<size+2; j++) {
                if(j != i+1) {
                    sigma += xsol[j] * mat[i*(size+2)+j];
                }
            }
            xsol[i+1] = (rhs[i] - sigma) / mat[i*(size+2)+i+1];

        }
    }

}

// Get arrays of sizes and starting points for each processor
void get_starts_and_sizes(int* size_array, int* start_array, int n_points, int world_size) {

    int remainder;
    int size;
    int start;

    for (int i=0; i<world_size; i++) {

    remainder = n_points % world_size;
    size = n_points / world_size;
    start = 0;

    if (i < remainder) {
        size ++;
        start = size * i;
        }

    else {
    
        start = (size + 1) * remainder + size * (i - remainder);
    }

    size_array[i] = size;
    start_array[i] = start;

    }
}

// Pass ghost points to neighbouring processors
void pass_ghost_points(int world_rank, int world_size, int size, double* xsol, bool is_first, bool is_last) {

    if(is_first) {
     //   if (world_rank == 0) printf("xsol[ %d ]  %lf \n", size-1, xsol[0]);
        MPI_Send(&xsol[size - 1], 1, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD);
    //    printf("sending xsol[size-1]  %.16lf  from rank %d to %d \n", xsol[size+1], world_rank, world_rank+1);
        MPI_Recv(&xsol[size], 1, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    else if(is_last) {
        MPI_Send(&xsol[1], 1, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD);
        MPI_Recv(&xsol[0], 1, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive the first ghost point from previous process
      //  printf("receiving to xsol[0]  %.16lf  from rank %d to %d \n", xsol[size+1], world_rank-1, world_rank);
    }

    else {

     //   if (world_rank == 1) printf("xsol[0]  %lf \n", xsol[0]);

        MPI_Send(&xsol[1], 1, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD);  // send the first calculated point to the previous process
        MPI_Send(&xsol[size], 1, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD); // send the last calculated point to the next process
     //   printf("sending xsol[size]  %.16lf  from rank %d to %d \n", xsol[size+1], world_rank, world_rank+1);
 //    if (world_rank == 1) printf("xsol[0]  %lf \n", xsol[0]);
        MPI_Recv(&xsol[0], 1, MPI_DOUBLE, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive the first ghost point from previous process
     //   printf("receiving to xsol[0]  %.16lf  from rank %d to %d \n", xsol[size+1], world_rank-1, world_rank);
  //   if (world_rank == 1) printf("xsol[0]  %lf \n", xsol[0]);

        MPI_Recv(&xsol[size+1], 1, MPI_DOUBLE, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive the last ghost point from next process 
    //    if (world_rank == 1) print_vec_double(xsol, world_size+1);
     //   printf("^ to bylo xsol dla rank 1 po send-recv \n");
    }

}

// Cut the xsol vector to exclude ghost points
void extract_xsol(double* xsol, double* xsol_extracted, int width, bool is_first, bool is_last) {

    if(is_first) {
        for(int i=0; i<width-1; i++) {
            xsol_extracted[i] = xsol[i];
        }
    }
    else if(is_last) {
        for(int i=0; i<width-1; i++) {
            xsol_extracted[i] = xsol[i+1];
        }
    }
    else {
        for(int i=0; i<width-2; i++) {
            xsol_extracted[i] = xsol[i+1];
        }
    }
}

// Calculate the solution analytically using f(x) = 1/2x * (1-x)
double analytical_solution(double x) {

  double asol = 0.0;

  asol = 0.5 * x * (1.0 - x);

  return asol;
}

// Save to file
void save_to_file(double* x, double* total_xsol, int n_points) {
    FILE *fp;
    fp = fopen("output.dat", "w");

    for (int i = 0 ; i < n_points; ++i) {
        double asol = analytical_solution(x[i]);
        fprintf(fp, "%.16lf %.16lf %.16lf %.16lf\n", x[i], total_xsol[i], asol, total_xsol[i] - asol);
  }

  fclose(fp);
}

// Log time to file
void append_time_to_file(double time, double world_size, double n_points, char* msg) {
    FILE *fp;
    fp = fopen("log.dat", "a");
    fprintf(fp, "%lf %lf %lf %s \n", n_points, world_size, time, msg);
}

// Run simulation once
void simulate(int n_points, int world_size, int world_rank, bool has_barrier) {

    // Initialize size and start point array for gathering the data later
    int start_array[world_size];
    int size_array[world_size];
    get_starts_and_sizes(size_array, start_array, n_points, world_size);

    // Get own size and is_first, is_last
    int size = size_array[world_rank];
    int width = size + 2;
    bool is_first = (world_rank == 0);
    bool is_last = (world_rank == world_size-1);


    // Special width for first and last
    if(is_first || is_last) {
        width = size+1;
    }
  
    //Calculate inverse dx^2
    double inv_dx2 = (double)(n_points-1)*(n_points-1); 

    // Initialize all matrices and vetors
    double mat [size * width];
    double xsol [width];
    double rhs [size];
    double x[n_points];
    double t1, t2;

    initialize_zero_matrix(mat, width, size);
    initialize_matrix(mat, is_first, is_last, size, inv_dx2);
    initialize_xsol(xsol, width);
    initialize_rhs(rhs, size, is_first, is_last);
    initialize_x_mesh(x, n_points);

    t1 = MPI_Wtime();

    for(int l=0; l<10000; l++) {

        pass_ghost_points(world_rank, world_size, size, xsol, is_first, is_last);
        jacobi_iteration(mat, xsol, rhs, size, is_first, is_last);
        if(has_barrier) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    t2 = MPI_Wtime();

    double extracted_xsol[size];
    extract_xsol(xsol, extracted_xsol, width, is_first, is_last);

    double total_xsol[n_points];
    MPI_Allgatherv(&extracted_xsol, size, MPI_DOUBLE, total_xsol, size_array, start_array, MPI_DOUBLE, MPI_COMM_WORLD);

    char msg[12];

    if(has_barrier) {
        strcpy(msg, "barrier");
    }
    else {
        strcpy(msg, "no barrier");
    }
    if(world_rank == 0) {
        append_time_to_file(t2-t1, world_size, n_points, msg);
        save_to_file(x, total_xsol, n_points);
    }
}

int main(int argc, char** argv) {

/*
    if (argc != 2) {
        fprintf(stderr, "Usage: n_points\n");
        exit(1);
    }

    // Input n of points on 1D grid
    int n_points = atoi(argv[1]); 

*/

    // MPI initial setup
    MPI_Init(NULL, NULL);



    for(int n_points=20; n_points<100; n_points++) {

            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            simulate(n_points, world_size, world_rank, false);
            MPI_Barrier(MPI_COMM_WORLD);
            simulate(n_points, world_size, world_rank, true);
    }


    // Finish MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
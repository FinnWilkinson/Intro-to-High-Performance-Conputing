#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int work_start, int work_end);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);

void collect(int myRank, float* image, int nx, int ny, int width, int height, int nprocs, int MASTER_nx, int process_nx);

void exchangeData(int myRank, float* image, int nprocs, int nx, int ny, int width, int height,
                  int MASTER_nx, int process_nx, int work_start, int work_end);


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);       //gets number of cores/processes using in job script
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);         //gets which rank this processes is

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in stencil
  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set up the input image
  init_image(nx, ny, width, height, image, tmp_image);

  int MASTER_nx = (nx/nprocs) + (nx % nprocs);
  int process_nx = (nx / nprocs);

  int work_start = (rank == MASTER) ? 1 : MASTER_nx + 1 + ((rank - 1) * process_nx);
  int work_end = (rank == MASTER) ? MASTER_nx : work_start + process_nx - 1;

  if(nprocs == 1){
    // Call the stencil kernel
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      stencil(nx, ny, width, height, image, tmp_image, work_start, work_end);
      stencil(nx, ny, width, height, tmp_image, image, work_start, work_end);
    }
    double toc = wtime();

    // Output
    if(rank == MASTER){
      printf("------------------------------------\n");
      printf(" runtime: %lf s\n", toc - tic);
      printf("------------------------------------\n");
    }
  }
  else{
    // Call the stencil kernel
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      exchangeData(rank, image, nprocs, nx, ny, width, height, MASTER_nx, process_nx, work_start, work_end);
      stencil(nx, ny, width, height, image, tmp_image, work_start, work_end);
      exchangeData(rank, tmp_image, nprocs, nx, ny, width, height, MASTER_nx, process_nx, work_start, work_end);
      stencil(nx, ny, width, height, tmp_image, image, work_start, work_end);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double toc = wtime();

    // Output
    if(rank == MASTER){
      printf("------------------------------------\n");
      printf(" runtime: %lf s\n", toc - tic);
      printf("------------------------------------\n");
    }
  }

  //collect data from all processes
  if(nprocs != 1) collect(rank, image, nx, ny, width, height, nprocs, MASTER_nx, process_nx);
  //output final image
  if(rank == MASTER) output_image(OUTPUT_FILE, nx, ny, width, height, image);
  free(image);
  free(tmp_image);
  MPI_Finalize();
}

void exchangeData(int myRank, float* image, int nprocs, int nx, int ny, int width, int height,
                  int MASTER_nx, int process_nx, int work_start, int work_end)
{

  int leftRank;
  int rightRank;
  int tag = 0;

  float* sendArr = malloc(sizeof(float) * (ny+2));

  float* recvArr = malloc(sizeof(float) * (ny+2));

  if(myRank == MASTER){
    leftRank = MPI_PROC_NULL;
    rightRank = myRank + 1;
    //send and recieve from MPI_PROC_NULL when left    //send and recieve from myRank + 1 when right

    //send to left, recieve from right
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, leftRank, tag, recvArr, ny + 2, MPI_FLOAT,
                rightRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //Copy sent data into image
    for(int i = 0; i < ny + 2; i++){
      image[i + (MASTER_nx+1) * height] = recvArr[i];
    }

    //Initiliase send array
    for(int i = 0; i < ny + 2; i++){
      sendArr[i] = image[i + MASTER_nx * height];
    }
    //send to right, recieve from left
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, rightRank, tag, recvArr, ny + 2, MPI_FLOAT,
                leftRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else if (myRank == (nprocs - 1)){
    leftRank = myRank - 1;
    rightRank = MPI_PROC_NULL;
    //send and recieve from nyRank - 1 when left    //send and recieve from MPI_PROC_NULL when right
    //Initiliase send array
    for(int i = 0; i < ny + 2; i++){
      sendArr[i] = image[(work_start * height) + i];
    }
    //send to left, recieve from right
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, leftRank, tag, recvArr, ny + 2, MPI_FLOAT,
                rightRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //send to right, recieve from left
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, rightRank, tag, recvArr, ny + 2, MPI_FLOAT,
                leftRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //copy sent data into image
    for(int i = 0; i < ny + 2; i++){
      image[((work_start - 1) * height) + i] = recvArr[i];
    }
  }
  else{
    leftRank = myRank - 1;
    rightRank = myRank + 1;
    //send and recieve from myRank - 1 when left    //send and recieve from myRank + 1 when right
    //Initiliase send array
    for(int i = 0; i < ny + 2; i++){
      sendArr[i] = image[(work_start * height) + i];
    }
    //send to left, recieve from right
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, leftRank, tag, recvArr, ny + 2, MPI_FLOAT,
                rightRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //Copy sent data into image
    for(int i = 0; i < ny + 2; i++){
      image[i + (work_end+1) * height] = recvArr[i];
    }
    //Initiliase send array
    for(int i = 0; i < ny + 2; i++){
      sendArr[i] = image[i + (work_end * height)];
    }
    //send to right, recieve from left
    MPI_Sendrecv(sendArr, ny + 2, MPI_FLOAT, rightRank, tag, recvArr, ny + 2, MPI_FLOAT,
                leftRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //copy sent data into image
    for(int i = 0; i < ny + 2; i++){
      image[((work_start - 1) * height) + i] = recvArr[i];
    }
  }
  free(sendArr);
  free(recvArr);
}

void collect(int myRank, float* image, int nx, int ny, int width, int height, int nprocs, int MASTER_nx, int process_nx)
{
  int iterator;
  if(myRank == MASTER){

    //Initiliase array to recieve data from workers
    float* recv_image = malloc(sizeof(float) * width * height);
    // Zero everything
    for (int j = 0; j < ny + 2; j++) {
      for (int i = 0; i < nx + 2; i++) {
        recv_image[j + i * height] = 0.0;
      }
    }

    //recieve data from all other nodes
    //need to remove padding (first and last columns)
    for(int src = 1; src < nprocs; src++){ //for each process not including MASTER
      MPI_Recv(recv_image, width * height, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int start_pos = MASTER_nx + 1 + ((src - 1)*process_nx);

      for(int j = 1; j < ny + 1; j++){
        for(int i = start_pos; i < start_pos + process_nx; i++){
          image[j + i * height] = recv_image[j + i * height];
        }
      }
    }
    free(recv_image);
  }
  else {
    MPI_Send(image, width * height, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
  }
}

//ORIGINAL
void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image, int work_start, int work_end)
{
  for (int i = work_start; i < work_end + 1; i++) {
    for (int j = 1; j < ny + 1; j++) {
      int currentPos = j+i*height;
      tmp_image[currentPos] =  ((image[currentPos-height] +  image[currentPos-1] +  image[currentPos+1] +  image[currentPos+height]) * 0.1f)
                            + image[currentPos] * 0.6f;
    }
  }
}

//ORIGINAL
/*void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 3.0 / 5.0;
      tmp_image[j + i * height] += image[j     + (i - 1) * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j     + (i + 1) * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j - 1 + i       * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j + 1 + i       * height] * 0.5 / 5.0;
    }
  }
}*/
//new and working
/*void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int i = 1; i < nx + 1; i++) {
    int currentPos = i * height;
    for (int j = 1; j < ny + 1; j++) {
      currentPos++;
      tmp_image[currentPos] = ((image[currentPos - height] + image[currentPos - 1] + image[currentPos + 1] + image[currentPos + height]) * 0.1)
                            + (image[currentPos] * 0.6);

    }
  }
}*/

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

#include <algorithm>
#include <tuple>
#include <cstdint>
#include <cstdlib>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

using namespace std;




#pragma region CONFIGURATION
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#pragma endregion




#pragma region METHODS
#pragma region MEMORY MAPPED FILE
/**
 * Map file to memory.
 * @param pth file path
 * @param ADV use early madvise (MADV_WILLNEED)?
 * @returns file descriptor, mapped data, and file size
 */
inline tuple<int, void*, size_t> mapFileToMemory(const char *pth, bool ADV=false) {
  // Open file as read-only.
  int fd = open(pth, O_RDONLY);
  if (fd==-1) return {-1, nullptr, 0};
  // Get file size.
  struct stat sb;
  if (fstat(fd, &sb)==-1) return {-1, nullptr, 0};
  // Map file to memory.
  void *addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);  // MAP_SHARED?
  if (addr==MAP_FAILED) return {-1, nullptr, 0};
  if (ADV) madvise(addr, sb.st_size, MADV_WILLNEED);  // MADV_SEQUENTIAL?
  // Return file descriptor, mapped data, and file size.
  return {fd, addr, sb.st_size};
}


/**
 * Unmap file from memory.
 * @param fd file descriptor
 * @param addr mapped data
 * @param size file size
 */
inline void unmapFileFromMemory(int fd, void *addr, size_t size) {
  munmap(addr, size);
  close(fd);
}
#pragma endregion



#pragma region PERFORM EXPERIMENT
/**
 * Calculate the sum of all bytes in a file.
 * @param data file data
 * @param N file size
 * @param BLOCK block size
 * @param MODE 0=none, 1=madvise, 2=mmap
 * @returns sum of all bytes
 */
inline size_t byteSum(const uint8_t *data, size_t N, size_t BLOCK, int MODE) {
  size_t a = 0;
  for (size_t b=0; b<N; b+=BLOCK) {
    size_t I = min(b + BLOCK, N);
    if (MODE==1) madvise(data + b, I-b, MADV_WILLNEED | MADV_POPULATE_READ);
    if (MODE==2) mmap   (data + b, I-b, PROT_READ, MAP_PRIVATE | MAP_POPULATE, -1, 0);
    for (size_t i=b; i<I; ++i)
      a += data[i];
  }
  return a;
}


/**
 * Calculate the sum of all bytes in a file, using OpenMP.
 * @param data file data
 * @param N file size
 * @param BLOCK block size
 * @param MODE 0=none, 1=madvise, 2=mmap
 * @returns sum of all bytes
 */
inline size_t byteSumOmp(const uint8_t *data, size_t N, size_t BLOCK, int MODE) {
  size_t a = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+:a)
  for (size_t b=0; b<N; b+=BLOCK) {
    size_t I = min(b + BLOCK, N);
    if (MODE==1) madvise(data + b, I-b, MADV_WILLNEED | MADV_POPULATE_READ);
    if (MODE==2) mmap   (data + b, I-b, PROT_READ, MAP_PRIVATE | MAP_POPULATE, -1, 0);
    for (size_t i=b; i<I; ++i)
      a += data[i];
  }
  return a;
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  char  *file  = argv[1];
  bool   PAR   = argc>2 ? atoi(argv[2]) : 0;     // 0=serial, 1=parallel
  bool   ADV   = argc>3 ? atoi(argv[3]) : 0;     // 0=none, 1=early madvise
  size_t BLOCK = argc>4 ? atol(argv[4]) : 4096;  // block size
  int    MODE  = argc>5 ? atoi(argv[5]) : 0;     // 0=none, 1=madvise, 2=mmap
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  auto [fd, addr, size] = mapFileToMemory(file, ADV);
  LOG("Finding byte sum of file %s ...\n", file);
  float tr = measureDuration([&]() {
    if (PAR) byteSumOmp((uint8_t*) addr, size, BLOCK, MODE);
    else     byteSum   ((uint8_t*) addr, size, BLOCK, MODE);
  });
  printf("{%09.1fms} %s\n", tr, PAR? "byteSumOmp" : "byteSum");
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__
#define VECTOR_LENGTH_IN_BYTES 32
void shm_initialize(int size, int rank, char* addr_string, char* port_string);
#endif

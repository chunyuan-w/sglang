#include <errno.h> 
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shm.h"

// states for collectives
enum coll_state {
    coll_begin = 0,
    coll_allreduce_naive__copy_in_done,
    coll_allreduce_naive__reduce_done,
    // alternative state when allreduce is working on alternative buffer
    // of the double buffer.
    coll_alt1_allreduce_naive__copy_in_done,
    coll_alt2_allreduce_naive__copy_in_done,
    coll_alt1_allreduce_naive__reduce_done,
};

// SHM building blocks
struct SharedData {
    const char* name;
    int descriptor;
    void* bytes;
    size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes)
{
    int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
        data->name = name;
        data->descriptor = d;
        data->bytes = bytes;
        data->nbytes = nbytes;
    } else {
        if (errno != ENOENT) {
            // don't print if shm can not be found because we want to loop over from
            // caller again until the other ranks created the shm
            printf("shared_open %s failed, errno=%d\n", name, errno);
        }
        data->descriptor = -1;
    }
}


void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes)
{
    int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        if (nbytes = write(d, bytes, nbytes)) { shared_open(data, name, nbytes); }
    } else {
        printf("shared_create %s failed\n", name);
    }
}

static int world_size;

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
    enum coll_state states[2];  // idx=0 -- state for symmetric_naive_all_reduce
                                // idx=1 -- state for distributed_naive_all_reduce
    // double buffer to avoid syncing between rounds
    // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for symmetric_naive_all_reduce
    // after that : buffer for distributed_naive_all_reduce
    char buffer[2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) 2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE

struct allreduce_workspace** workspace;

// buffer for small messages, double buffer
char** symmetric_buffer[2];
// buffer for large messages, double buffer
char** distributed_buffer[2];

static bool is_initialized = 0;
static int world_rank;

void shm_initialize(int size, int rank, char* addr_string, char* port_string)
{
    if (is_initialized) return;
    is_initialized = 1;

    world_size = size;
    world_rank = rank;

    char shm_name_prefix[NAME_BUF_SIZE];
    char shm_name[NAME_BUF_SIZE];
    snprintf(shm_name_prefix,
             NAME_BUF_SIZE,
             "%s_%d_%s_%s",
             SHM_BUFFER_NAME,
             getuid(),
             addr_string,
             port_string);
    // create shared workspace for SHM based allreduce
    SharedData allreduce_buffer;
    // allocate workspace_buf for current rank
    struct allreduce_workspace* workspace_buf;
    struct allreduce_workspace* workspace_buf_other;
    workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
    snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, rank);
    shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
    workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
    workspace_buf->states[0] = coll_alt2_allreduce_naive__copy_in_done;
    workspace_buf->states[1] = coll_begin;

    // create the workspace pointer list
    workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
    symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
    symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
    distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
    distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

    // map shm of all ranks
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, i);
            // printf("open %s, %d\n", shm_name, rank);
            do {
                shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
            } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
            workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
            workspace[i] = workspace_buf_other;
        } else {
            workspace[i] = workspace_buf;
        }
        symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
        symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
        distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
        distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);
    }
}

void all_reduce(
    torch::Tensor input,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
  std::vector<torch::Tensor> tensors = {input};
  process_group->allreduce(tensors)->wait();
  return;
}
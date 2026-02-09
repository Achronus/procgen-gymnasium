/*
 * libenv.h - C interface for vectorized environments
 *
 * Vendored from gym3 (https://github.com/openai/gym3)
 * Original license: MIT
 *
 * This header defines the C ABI that the compiled procgen shared library
 * exposes and that the Python-side loader calls via cffi.
 */

#ifndef LIBENV_H
#define LIBENV_H

#include <stdint.h>

#define LIBENV_VERSION 3
#define LIBENV_MAX_NAME_LEN 128
#define LIBENV_MAX_NDIM 16

#ifdef _WIN32
#define LIBENV_API __declspec(dllexport)
#else
#define LIBENV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum libenv_dtype {
    LIBENV_DTYPE_UNUSED  = 0,
    LIBENV_DTYPE_UINT8   = 1,
    LIBENV_DTYPE_INT32   = 2,
    LIBENV_DTYPE_FLOAT32 = 3,
};

enum libenv_scalar_type {
    LIBENV_SCALAR_TYPE_UNUSED   = 0,
    LIBENV_SCALAR_TYPE_REAL     = 1,
    LIBENV_SCALAR_TYPE_DISCRETE = 2,
};

enum libenv_space_name {
    LIBENV_SPACE_UNUSED      = 0,
    LIBENV_SPACE_OBSERVATION = 1,
    LIBENV_SPACE_ACTION      = 2,
    LIBENV_SPACE_INFO        = 3,
};

union libenv_value {
    uint8_t uint8;
    int32_t int32;
    float float32;
};

struct libenv_tensortype {
    char name[LIBENV_MAX_NAME_LEN];
    enum libenv_scalar_type scalar_type;
    enum libenv_dtype dtype;
    int ndim;
    int shape[LIBENV_MAX_NDIM];
    union libenv_value low;
    union libenv_value high;
};

struct libenv_option {
    char name[LIBENV_MAX_NAME_LEN];
    enum libenv_dtype dtype;
    int count;
    void *data;
};

struct libenv_options {
    struct libenv_option *items;
    int count;
};

struct libenv_buffers {
    void **ob;
    void **ac;
    void **info;
    float *rew;
    uint8_t *first;
};

typedef struct libenv_env_s libenv_env;

LIBENV_API int libenv_version(void);
LIBENV_API libenv_env *libenv_make(int num_envs, const struct libenv_options options);
LIBENV_API int libenv_get_tensortypes(libenv_env *handle, enum libenv_space_name name, struct libenv_tensortype *out_types);
LIBENV_API void libenv_set_buffers(libenv_env *handle, struct libenv_buffers *bufs);
LIBENV_API void libenv_observe(libenv_env *handle);
LIBENV_API void libenv_act(libenv_env *handle);
LIBENV_API void libenv_close(libenv_env *handle);

#ifdef __cplusplus
}
#endif

#endif /* LIBENV_H */

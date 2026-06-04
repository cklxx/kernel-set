/*
 * kernel-set — symbol visibility / linkage macros.
 *
 * Every public C ABI symbol is annotated with KS_API. On Windows this toggles
 * between dllexport (when building the library) and dllimport (when consuming
 * it); on Unix-likes it forces default visibility so the symbol survives
 * -fvisibility=hidden builds.
 */
#ifndef KERNEL_SET_EXPORT_H_
#define KERNEL_SET_EXPORT_H_

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef KERNEL_SET_BUILD
#    define KS_API __declspec(dllexport)
#  else
#    define KS_API __declspec(dllimport)
#  endif
#else
#  define KS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#  define KS_BEGIN_EXTERN_C extern "C" {
#  define KS_END_EXTERN_C }
#else
#  define KS_BEGIN_EXTERN_C
#  define KS_END_EXTERN_C
#endif

#endif /* KERNEL_SET_EXPORT_H_ */

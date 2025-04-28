/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/runtime/omp/omp_phys_mem.hpp"

// Mac OSX implementation:
#if defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>

std::optional<std::size_t> hipsycl::rt::getPhysicalMemory() {

int mib[]     = {CTL_HW, HW_MEMSIZE};
int64_t value = 0;
size_t length = sizeof(value);

if (-1 == sysctl(mib, 2, &value, &length, NULL, 0)) {
    return std::nullopt;
}
return value;
}

// Linux/BSD implementation:
#elif (defined(linux) || defined(__linux__) || defined(__linux))                                   \
|| (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__)                      \
    || defined(__OpenBSD__))

#include <sys/sysinfo.h>

std::optional<std::size_t> hipsycl::rt::getPhysicalMemory() {
struct sysinfo info;
sysinfo(&info);
return info.totalram;
}

#else

std::optional<std::size_t> hipsycl::rt::getPhysicalMemory() { return std::nullopt; }

#endif
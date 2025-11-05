// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <cassert>
#include <sycl/sycl.hpp>
#include "common.hpp"


#include <sycl/sycl.hpp>

int main() {
  sycl::queue q = get_queue();


  auto const x = sycl::malloc_shared<int>(1, q);
  *x = 32;

  q.submit(
      [&](sycl::handler &cgh) { cgh.single_task([=]() { assert(x[0] == 32); x[0] += 1; }); });

  q.wait();
  // CHECK: 33
  std::cout << *x << std::endl;

  sycl::free(x, q);
  return 0;
}

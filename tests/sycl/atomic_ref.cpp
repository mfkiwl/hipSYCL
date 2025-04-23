#define BOOST_TEST_MODULE my_sycl_tests
#include <boost/test/included/unit_test.hpp>
#include "sycl_test_suite.hpp"

using namespace cl;

// NOTE: creates a fixed test-case-env with param reset_device_fixture`
BOOST_FIXTURE_TEST_SUITE(unsigned_values, reset_device_fixture)

BOOST_AUTO_TEST_CASE(fetch_op) {

   // Create a queue.
   ::sycl::queue q;

   // Allocate some memory. For the target device
   ::sycl::buffer<unsigned int> buf(1);
   // Allows for our CPU to have access to the allocated memory
   auto host_acc = buf.get_access<::sycl::access::mode::read_write>();
   // First entry of memory is 0 unsigned
   host_acc[0] = 0u;

   // Submit a command group to the queue, which would increment the buffer
   // value atomically.
   q.submit([&](::sycl::handler &cgh)
            {
      // Get an accessor to the buffer.
      auto ptr = buf.get_access<::sycl::access::mode::read_write>(cgh);

      // Increment the value in the buffer atomically.
      cgh.single_task<class kernel>([ptr]() {
         ::sycl::atomic_ref<unsigned int,
                            ::sycl::memory_order::relaxed,
                            ::sycl::memory_scope::device,
                            ::sycl::access::address_space::global_space>
                            ref(ptr[0]);
         ref.fetch_add(1);
      }); })
       .wait_and_throw();
  
  // now read it back on the host
  auto host_acc_read = buf.get_access<::sycl::access::mode::read_write>();
  BOOST_CHECK_EQUAL(host_acc_read[0], 1u);
}

BOOST_AUTO_TEST_SUITE_END()

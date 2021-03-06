/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

using namespace cl;

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(group_x_of_local) {
  using T = char;

  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size;
  const size_t offset_divisor = 1;
  const size_t buffer_size    = global_size;
  const auto data_generator   = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_any_of(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, false, true, true},
                                               "any_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_all_of(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "all_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_none_of(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "none_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

BOOST_AUTO_TEST_CASE(group_x_of_ptr) {
  using T = char;

  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size * 2;
  const size_t offset_divisor = 1;
  const size_t buffer_size    = global_size * 3;
  const auto data_generator   = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size * 2, global_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::any_of(g, start.get(), end.get());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, false, true, true},
                                               "any_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::all_of(g, start.get(), end.get());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "all_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::none_of(g, start.get(), end.get());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "none_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HIP)
BOOST_AUTO_TEST_CASE(sub_group_x_of_local) {
  using T = char;

  const size_t local_size      = 256;
  const size_t global_size     = 1024;
  const size_t offset_margin   = global_size;
  const size_t offset_divisor  = 1;
  const size_t buffer_size     = global_size;
  const uint32_t subgroup_size = static_cast<uint32_t>(warpSize);

  const auto data_generator = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_any_of(sg, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, false, true, true},
                                               "any_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_all_of(sg, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "all_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_none_of(sg, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "none_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }
}
#endif

BOOST_AUTO_TEST_CASE(group_x_of_ptr_function) {
  using T = char;

  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size;
  const size_t offset_divisor = 1;
  const size_t buffer_size    = global_size * 3;
  const auto data_generator   = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size * 2, global_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::any_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, true, true, false},
                                               "any_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::all_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "all_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::detail::none_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "none_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

BOOST_AUTO_TEST_CASE(group_x_of_function) {
  using T = char;

  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size;
  const size_t offset_divisor = 1;
  const size_t buffer_size    = global_size;
  const auto data_generator   = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_any_of(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, true, true, false},
                                               "any_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_all_of(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "all_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_none_of(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "none_of");
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HIP)
BOOST_AUTO_TEST_CASE(sub_group_x_of_function) {
  using T = char;

  const size_t local_size      = 256;
  const size_t global_size     = 1024;
  const size_t offset_margin   = global_size;
  const size_t offset_divisor  = 1;
  const size_t buffer_size     = global_size;
  const uint32_t subgroup_size = static_cast<uint32_t>(warpSize);

  const auto data_generator = [](std::vector<T> &v) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_any_of(sg, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{true, true, true, false},
                                               "any_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_all_of(sg, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, true, false, false},
                                               "all_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_none_of(sg, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      detail::check_binary_reduce<T, __LINE__>(vIn, local_size, global_size,
                                               std::vector<bool>{false, false, false, true},
                                               "none_of", subgroup_size);
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }
}
#endif
BOOST_AUTO_TEST_SUITE_END()

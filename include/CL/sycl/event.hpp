/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef SYCU_EVENT_HPP
#define SYCU_EVENT_HPP

#include "types.hpp"
#include "backend/backend.hpp"
#include "exception.hpp"
#include "info/event.hpp"
#include "detail/task_graph.hpp"

namespace cl {
namespace sycl {

// ToDo: Replace with detail::task_graph_node
class event {

public:
  event()
    : _is_null_event{true}
  {}

  event(const detail::task_graph_node_ptr& evt)
    : _is_null_event{false}, _evt{evt}
  {}


  /* CL Interop is not supported
  event(cl_event clEvent, const context& syclContext);

  cl_event get();
  */

  vector_class<event> get_wait_list()
  {
    return vector_class<event>{};
  }

  void wait()
  {
    if(!this->_is_null_event)
      this->_evt->wait();
  }

  static void wait(const vector_class<event> &eventList)
  {
    for(const event& evt: eventList)
      const_cast<event&>(evt).wait();
  }

  void wait_and_throw()
  {
    wait();
  }

  static void wait_and_throw(const vector_class<event> &eventList)
  {
    wait(eventList);
  }

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const
  { throw unimplemented{"event::get_info() is unimplemented."}; }

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type get_profiling_info() const
  { throw unimplemented{"event::get_profiling_info() is unimplemented."}; }

  bool operator ==(const event& rhs) const
  { return _evt == rhs._evt; }

  bool operator !=(const event& rhs) const
  { return !(*this == rhs); }

private:

  bool _is_null_event;
  detail::task_graph_node_ptr _evt;
};


} // namespace sycl
} // namespace cl

#endif

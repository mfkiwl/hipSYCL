# hipSYCL buffer-USM interoperability

hipSYCL supports interoperability between `buffer` objects and USM pointers. All `buffer` memory allocations are USM pointers internally that can be extracted and be used with USM operations.
Similarly, `buffer` objects can be constructed on top of existing USM pointers.

hipSYCL follows its own [specification](runtime-spec.md) for the hipSYCL buffer-accessor model. Refer to this document to understand the memory management and allocation behavior of hipSYCL buffer objects. Using buffer-USM interoperability without a solid understanding of hipSYCL's buffer model is not recommended.

## Buffer introspection

USM pointers that a buffer currently manages can be queried, and some aspects can be modified.

## API reference

```c++
namespace sycl {

/// Describes a buffer memory allocation represented by a USM pointer 
/// and additional meta-nformation.
template <class T> struct buffer_allocation {
  /// the USM pointer
  T *ptr;
  /// the device for which this allocation is used.
  /// Note that the runtime may only maintain
  /// a single allocation for all host devices.
  device dev;
  /// If true, the runtime will delete this allocation
  /// at buffer destruction.
  bool is_owned;
};

template <typename T, int dimensions,
          typename AllocatorT>
class buffer {
public:

  /// Iterate over all allocations used by this buffer, and invoke
  /// a handler object for each allocation.
  /// \param h Handler that will be invoked for each allocation.
  ///  Signature of \c h: void(const buffer_allocation<T>&)
  template <class Handler>
  void for_each_allocation(Handler &&h) const;

  /// Get USM pointer for the buffer allocation of the specified device.
  /// \return The USM pointer associated with the device, or nullptr if
  /// the buffer does not contain an allocation for the device.
  T *get_pointer(const device &dev) const;

  /// \return Whether the buffer contains an allocation for the given device.
  bool has_allocation(const device &dev) const;

  /// \return the buffer allocation object associated with the provided
  /// device. If the buffer does not contain an allocation for the specified
  /// device, throws \c invalid_parameter_error.
  buffer_allocation<T> get_allocation(const device &dev) const;

  /// \return the buffer allocation object associated with the provided pointer.
  /// If the buffer does not contain an allocation described by ptr,
  /// throws \c invalid_parameter_error.
  buffer_allocation<T> get_allocation(const T *ptr) const;

  /// Instruct buffer to free the allocation on the specified device at buffer
  /// destruction.
  /// Throws \c invalid_parameter_error if no allocation for specified device
  /// exists.
  void own_allocation(const device &dev);

  /// Instruct buffer to free the allocation at buffer destruction.
  /// \c ptr must be an existing allocation managed by the buffer.
  /// If \c ptr cannot be found among the managed memory allocations,
  /// \c invalid_parameter_error is thrown.
  void own_allocation(const T *ptr);

  /// Instruct buffer to not free the allocation on the specified device
  /// at buffer destruction.
  /// Throws \c invalid_parameter_error if no allocation for specified device
  /// exists.
  void disown_allocation(const device &dev);

  /// Instruct buffer to not free the allocation associated with the provided
  /// pointer.
  /// Throws \c invalid_parameter_error if no allocation managed by the buffer
  /// is described by \c ptr.
  void disown_allocation(const T *ptr);

}; // class buffer

} // namespace sycl

```

## Constructing buffer objects on top of USM pointers

TBD

## Example code

TBD
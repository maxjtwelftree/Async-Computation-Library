# Async-Computation-Library

This library is offering a simple, light-weight and easy-to-use components to write asynchronous code. The components offered include Lazy (based on C++20 stackless coroutine), Uthread (based on stackful coroutine) and the traditional Future/Promise.

The main idea, was heavily inspired by [async-simple](https://github.com/alibaba/async_simple/tree/main/async_simple).

## Quick Experience

We can try async_simple online in [compiler-explorer](compiler-explorer.com): https://compiler-explorer.com/z/Tdaesqsqj . Note that `Uthread` cannot be use in compiler-explorer since it requires installation.

## Compiler Requirement
Required Compiler: clang (>= 10.0.0) or gcc (>= 10.3) or Apple-clang (>= 14)

Note that we need to add the -Wno-maybe-uninitialized option when we use gcc 12 due to a false positive diagnostic message by gcc 12

Note that when using clang 15 it may be necessary to add the -Wno-unsequenced option, which is a false positive of clang 15. See llvm/llvm-project#56768 for details.

If you meet any problem about MSVC Compiler Error C4737. Try to add the /EHa option to fix the problem.
## Build

### cmake
```bash
$ mkdir build && cd build
# Specify [-DASYNC_SIMPLE_ENABLE_TESTS=OFF] to skip tests.
# Specify [-DASYNC_SIMPLE_BUILD_DEMO_EXAMPLE=OFF] to skip build demo example.
# Specify [-DASYNC_SIMPLE_DISABLE_AIO=ON] to skip the build libaio
CXX=clang++ CC=clang cmake ../ -DCMAKE_BUILD_TYPE=[Release|Debug] [-DASYNC_SIMPLE_ENABLE_TESTS=OFF] [-DASYNC_SIMPLE_BUILD_DEMO_EXAMPLE=OFF] [-DASYNC_SIMPLE_DISABLE_AIO=ON]
# for gcc, use CXX=g++ CC=gcc
make -j4
make test # optional
make install # sudo if required
```

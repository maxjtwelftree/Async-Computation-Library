#pragma once

#include <coroutine>
#include <memory>
#include <optional>
#include <exception>

template<typename T>
class Lazy {
public:
    Lazy(std::coroutine_handle<PromiseType> handle) : coroHandle(handle) {}

    ~Lazy() { if (coroHandle) coroHandle.destroy(); }

    T get() {
        if (!coroHandle.done()) {
            coroHandle.resume();
        }
        return coroHandle.promise().value();
    }

private:
    struct PromiseType {
        std::optional<T> result;
        std::exception_ptr exception;

        Lazy<T> get_return_object() {
            return Lazy<T>{std::coroutine_handle<PromiseType>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void return_value(T value) {
            result = std::move(value);
        }

        void unhandled_exception() {
            exception = std::current_exception();
        }

        T value() {
            if (exception) {
                std::rethrow_exception(exception);
            }
            return *result;
        }
    };

    std::coroutine_handle<PromiseType> coroHandle;
};

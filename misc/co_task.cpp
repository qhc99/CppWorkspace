#include <concepts>
#include <coroutine>
#include <exception>
#include <optional>

template <typename T>
    requires std::movable<T>
struct TaskResult {

    explicit TaskResult() = default;

    explicit TaskResult(T&& value)
        : _value(value)
    {
    }

    explicit TaskResult(std::exception_ptr&& exception_ptr)
        : _exception_ptr(exception_ptr)
    {
    }

    T get_or_throw()
    {
        if (_exception_ptr) {
            std::rethrow_exception(_exception_ptr);
        }
        return _value;
    }

private:
    T _value {};
    std::exception_ptr _exception_ptr;
};

template <typename T>
struct Task;

template <typename T>
struct TaskAwaiter;

template <typename ResultType>
    requires std::movable<ResultType>
struct TaskPromise {

    std::suspend_never initial_suspend() { return {}; }

    std::suspend_always final_suspend() noexcept { return {}; }

    Task<ResultType> get_return_object()
    {
        return { std::coroutine_handle<TaskPromise>::from_promise(*this) };
    }

    void unhandled_exception()
    {
        result = TaskResult<ResultType>(std::current_exception());
    }

    void return_value(ResultType value)
    {
        result = TaskResult<ResultType>(std::move(value));
    }

    template <typename TResultType>
    TaskAwaiter<TResultType> await_transform(Task<TResultType>&& task)
    {
        return TaskAwaiter<TResultType>(std::move(task));
    }

private:
    std::optional<TaskResult<ResultType>> result;
};

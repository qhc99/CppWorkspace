#include <concepts>
#include <coroutine>
#include <iostream>
#include <type_traits>
#include <utility>

template <typename T>
    requires std::copyable<T>
struct Generator {
    struct promise_type;
    class ExhaustedException : std::exception { };
    std::coroutine_handle<Generator::promise_type> handle;

    struct promise_type {
        T value {};
        bool is_ready { false };

        /**
         * @brief Not run on constructing awaiter
         * 
         * @return std::suspend_always 
         */
        static std::suspend_always initial_suspend() { return {}; };

        /**
         * @brief Not clean 
         * 
         * @return std::suspend_always 
         */
        static std::suspend_always final_suspend() noexcept { return {}; }

        /**
         * @brief No exception handling
         * 
         */
        void unhandled_exception() { }

        /**
         * @brief Return awaiter
         * 
         * @return Generator 
         */
        Generator get_return_object()
        {
            return std::move(Generator { std::coroutine_handle<promise_type>::from_promise(*this) });
        }

        /**
         * @brief Transform to promise
         * 
         * @param value 
         * @return std::suspend_always 
         */
        std::suspend_always await_transform(T value)
        {
            this->value = value;
            is_ready = true;
            return {};
        }

        /**
         * @brief yield value to promise
         * 
         * @param value 
         * @return std::suspend_always 
         */
        std::suspend_always yield_value(T value)
        {
            this->value = value;
            is_ready = true;
            return {};
        }

        /**
         * @brief co_await no return
         * 
         */
        void return_void() { }
    };

    [[nodiscard]] bool has_next() const
    {
        if (handle.done()) {
            return false;
        }

        if (!handle.promise().is_ready) {
            handle.resume();
        }

        return !handle.done();
    }

    [[nodiscard]] T next() const
    {
        if (has_next()) {
            handle.promise().is_ready = false;
            return handle.promise().value;
        }
        throw ExhaustedException();
    }

    template <typename F>
        requires std::is_invocable_v<F, T>
    Generator<std::invoke_result_t<F, T>> map(F f)
    {
        while (has_next()) {
            co_yield f(next());
        }
    }

    explicit Generator(std::coroutine_handle<Generator::promise_type> h)
        : handle(h)
    {
    }

    Generator(const Generator& o) = delete;

    Generator& operator=(const Generator& o) = delete;

    Generator(Generator&& o) noexcept
        : handle(std::exchange(o.handle, nullptr)) {};

    Generator& operator=(Generator&& o) noexcept
    {
        std::swap(handle, o.handle);
        return *this;
    };

    ~Generator()
    {
        if (handle) {
            handle.destroy();
        }
    }
};

// NOLINTNEXTLINE(readability-static-accessed-through-instance)
Generator<int> sequence()
{
    int i = 0;
    // create promise
    // get return object
    while (true) {
        // transform promise
        co_await i++;
    }
}

// NOLINTNEXTLINE(readability-static-accessed-through-instance)
Generator<int> sequence_yield()
{
    int i = 0;
    while (true) {
        co_yield i++;
    }
}

int main()
{
    auto generator = sequence();
    for (int i = 0; i < 15; ++i) {
        if (generator.has_next()) {
            std::cout << generator.next() << std::endl;
        } else {
            break;
        }
    }
    auto origin{sequence_yield()}; // cannot be rvalue
    auto yield_half { origin.map([](int i) { return i / 2.; }) };
    for (int i = 0; i < 15; ++i) {
        if (yield_half.has_next()) {
            std::cout << yield_half.next() << std::endl;
        } else {
            break;
        }
    }
    return 0;
}
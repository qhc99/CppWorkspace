#include <coroutine>
#include <iostream>
#include <utility>

struct Generator {
    struct promise_type;
    class ExhaustedException : std::exception { };
    std::coroutine_handle<Generator::promise_type> handle;

    struct promise_type {
        int value {};
        bool is_ready { false };

        static std::suspend_always initial_suspend() { return {}; };

        static std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() { }

        Generator get_return_object()
        {
            return std::move(Generator { std::coroutine_handle<promise_type>::from_promise(*this) });
        }

        std::suspend_always await_transform(int value)
        {
            this->value = value;
            is_ready = true;
            return {};
        }

        std::suspend_always yield_value(int value)
        {
            this->value = value;
            is_ready = true;
            return {};
        }

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

    [[nodiscard]] int next() const
    {
        if (has_next()) {
            handle.promise().is_ready = false;
            return handle.promise().value;
        }
        throw ExhaustedException();
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
Generator sequence()
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
Generator sequence_yield()
{
    int i = 0;
    // create promise
    // get return object
    while (true) {
        // transform promise
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

    generator = sequence_yield();
    for (int i = 0; i < 15; ++i) {
        if (generator.has_next()) {
            std::cout << generator.next() << std::endl;
        } else {
            break;
        }
    }
    return 0;
}
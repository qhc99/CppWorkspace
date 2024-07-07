#include <concepts>
#include <coroutine>
#include <iostream>
#include <type_traits>
#include <utility>

struct NothingAwaiter {
    static bool await_ready()  noexcept { return true; }
    void await_suspend(std::coroutine_handle<> /*unused*/) const noexcept {}
    void await_resume() const noexcept {}
};


struct Continuation {
    struct promise_type;
    class ExhaustedException : std::exception { };
    std::coroutine_handle<Continuation::promise_type> handle;

    struct promise_type {

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
        Continuation get_return_object()
        {
            return std::move(Continuation { std::coroutine_handle<promise_type>::from_promise(*this) });
        }

        /**
         * @brief Transform to promise
         * 
         * @param value 
         * @return std::suspend_always 
         */
        static std::suspend_always await_transform(NothingAwaiter /*unused*/)
        {
            return {};
        }

        /**
         * @brief co_await no return
         * 
         */
        void return_void() { }
    };

    void run() const{
        handle.resume();
    }

    explicit Continuation(std::coroutine_handle<Continuation::promise_type> h)
        : handle(h)
    {
    }

    Continuation(const Continuation& o) = delete;

    Continuation& operator=(const Continuation& o) = delete;

    Continuation(Continuation&& o) noexcept
        : handle(std::exchange(o.handle, nullptr)) {};

    Continuation& operator=(Continuation&& o) noexcept
    {
        std::swap(handle, o.handle);
        return *this;
    };

    ~Continuation()
    {
        if (handle) {
            handle.destroy();
        }
    }
};

// NOLINTNEXTLINE(readability-static-accessed-through-instance)
Continuation run()
{
    int i = 0;
    // create promise
    // get return object
    while (true) {
        std::cout << "continue" << std::endl;
        // NOLINTNEXTLINE(readability-static-accessed-through-instance)
        co_await NothingAwaiter{};
    }
}



int main()
{
    
    auto continuation{run()};
    for (int i{0}; i < 10; ++i){
        std::cout << "main" << std::endl;
        continuation.run();
    }
    return 0;
}
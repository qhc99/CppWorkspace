#include "workspace_pch.h"

// forward
template <typename T>
    requires std::copyable<T>
struct Generator;

namespace {
// is generator template utility
template <typename T>
struct is_generator : std::false_type { };

template <typename Any>
struct is_generator<Generator<Any>> : std::true_type { };

template <typename T>
inline constexpr bool is_generator_v = is_generator<T>::value;
}

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

    /**
     * @brief This generator should not be destroyed before returned new generator
     * 
     * @tparam F function type
     * @param f function
     * @return requires 
     */
    template <typename F>
        requires std::is_invocable_v<F, T>
    Generator<std::invoke_result_t<F, T>> map_view(F f)
    {
        while (has_next()) {
            co_yield f(next());
        }
    }

    template <typename F>
        requires std::is_invocable_v<F, T> && is_generator_v<std::invoke_result_t<F, T>>
    std::invoke_result_t<F, T> flat_map_view(F f)
    {
        while (has_next()) {
            auto generator = f(next());
            while (generator.has_next()) {
                co_yield generator.next();
            }
        }
    }

    template <typename F>
        requires std::is_invocable_v<F, T>
    void for_each_view(F f)
    {
        while (has_next()) {
            f(next());
        }
    }

    template <typename R, typename F>
        requires std::is_invocable_v<F, R, R>
    R fold_view(R initial, F f)
    {
        R acc = initial;
        while (has_next()) {
            acc = f(acc, next());
        }
        return acc;
    }

    template <typename F>
        requires std::is_invocable_r_v<bool, F, T>
    Generator filter_view(F f)
    {
        while (has_next()) {
            T value = next();
            if (f(value)) {
                co_yield value;
            }
        }
    }

    Generator take_view(int n)
    {
        int i = 0;
        while (i++ < n && has_next()) {
            co_yield next();
        }
    }

    template <typename F>
        requires std::is_invocable_r_v<bool, F, T>
    Generator take_while_view(F f)
    {
        while (has_next()) {
            T value = next();
            if (f(value)) {
                co_yield value;
            } else {
                break;
            }
        }
    }

    template <typename... TArgs>
        requires(std::is_same_v<TArgs, T> && ...)
    Generator static from(TArgs... args)
    {
        (co_yield args, ...);
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
    // initial suspend
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

void print_test();

int main()
{
    print_test();
    return 0;
}

void print_test()
{
    auto generator = sequence();
    for (int i = 0; i < 15; ++i) {
        if (generator.has_next()) {
            std::cout << generator.next() << '\n';
        } else {
            break;
        }
    }
    //
    auto origin { sequence_yield() }; // cannot be rvalue
    auto yield_half { origin.map_view([](int i) { return i / 2.; }) };
    for (int i = 0; i < 15; ++i) {
        if (yield_half.has_next()) {
            std::cout << yield_half.next() << '\n';
        } else {
            break;
        }
    }
    //
    generator = Generator<int>::from(11, 22, 33, 44);
    for (int i = 0; i < 15; ++i) {
        if (generator.has_next()) {
            std::cout << generator.next() << '\n';
        } else {
            break;
        }
    }
    //
    Generator<int>::from(1, 2, 3, 4, 5)
        .flat_map_view([](auto i) -> Generator<int> {
            for (int j = 0; j < i; ++j) {
                co_yield j;
            }
        })
        .for_each_view([](auto i) {
            if (i == 0) {
                std::cout << '\n';
            }
            std::cout << "* ";
        });
    std::cout << '\n';
    //
    std::cout << (Generator<int>::from(1, 2, 3, 4, 5, 6)
                      .fold_view(1, [](auto acc, auto i) {
                          return acc * i; // factorial
                      }))
              << '\n';
    //
    Generator<int>::from(1, 2, 3, 4, 5).take_while_view([](auto i) {
                                           return i < 3;
                                       })
        .for_each_view([](auto i) {
            std::cout << i << " ";
        });
    std::cout << '\n';
    //
    Generator<int>::from(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        .filter_view([](auto i) {
            std::cout << "filter: " << i << std::endl;
            return i % 2 == 0;
        })
        .map_view([](auto i) {
            std::cout << "map: " << i << std::endl;
            return i * 3;
        })
        .flat_map_view([](auto i) -> Generator<int> {
            std::cout << "flat_map: " << i << std::endl;
            for (int j = 0; j < i; ++j) {
                co_yield j;
            }
        })
        .take_view(3)
        .for_each_view([](auto i) {
            std::cout << "for_each: " << i << std::endl;
        });
}
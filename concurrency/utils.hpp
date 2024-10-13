#ifndef CONCURRENCY_JOIN_THREADS_UTILS_HPP
#define CONCURRENCY_JOIN_THREADS_UTILS_HPP
class join_threads {
    std::vector<std::thread>* threads;

public:
    explicit join_threads(std::vector<std::thread>& threads_)
        : threads(&threads_)
    {
    }
    join_threads(const join_threads&) = delete;
    join_threads(join_threads&&) = default;
    join_threads& operator=(const join_threads&) = delete;
    join_threads& operator=(join_threads&&) = default;

    ~join_threads()
    {
        if (threads != nullptr) {
            for (auto& thread : *threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
    }
};

#endif
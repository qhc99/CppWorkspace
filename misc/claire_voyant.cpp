
class Main {
public:
    static std::vector<double> bigDecimalArray(int start, int endInclusive, int step, int scale)
    {
        std::vector<double> result;
        for (int i = start; i <= endInclusive; i += step) {
            result.push_back(static_cast<double>(i) / scale);
        }
        return result;
    }

    static double generate(const std::vector<double>& a)
    {
        struct Solver {
            double total = 0.0;
            std::vector<double> minFutures;
            std::vector<double> maxFutures;

            explicit Solver(const std::vector<double>& a)
            {
                for (size_t i = 0; i < a.size(); ++i) {
                    auto t = getFutures(i, a);
                    minFutures.push_back(t[0]);
                    maxFutures.push_back(t[1]);
                }
            }

            static std::vector<double> getFutures(size_t idx, const std::vector<double>& arr)
            {
                double min = std::accumulate(arr.begin() + idx, arr.end(), 1.0, [](double acc, double x) {
                    return acc * std::min(x, 1.0 - x);
                });

                double max = std::accumulate(arr.begin() + idx, arr.end(), 1.0, [](double acc, double x) {
                    return acc * std::max(x, 1.0 - x);
                });

                return { min, max };
            }

            void generateTableRecursive(size_t current_idx, double prev_choice_1_result, double prev_choice_2_result, const std::vector<double>& arr)
            {
                if (current_idx == arr.size()) {
                    total += std::max(prev_choice_1_result, prev_choice_2_result);
                } else {
                    double minFuture = minFutures[current_idx];
                    double maxFuture = maxFutures[current_idx];
                    double minChoice = std::min(prev_choice_1_result, prev_choice_2_result);
                    double maxChoice = std::max(prev_choice_2_result, prev_choice_1_result);

                    if (minChoice * maxFuture <= maxChoice * minFuture) {
                        total += maxChoice;
                        return;
                    }

                    generateTableRecursive(current_idx + 1,
                        prev_choice_1_result * arr[current_idx],
                        prev_choice_2_result * (1.0 - arr[current_idx]),
                        arr);

                    generateTableRecursive(current_idx + 1,
                        prev_choice_1_result * (1.0 - arr[current_idx]),
                        prev_choice_2_result * arr[current_idx],
                        arr);
                }
            }
        };

        Solver solver(a);
        solver.generateTableRecursive(1, a[0], 1.0 - a[0], a);
        return solver.total;
    }

    static void main()
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> arr = bigDecimalArray(25, 58, 1, 100);
        std::cout << std::setprecision(15) << generate(arr) << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << duration.count() << " seconds" << std::endl;
    }
};

int main()
{
    Main::main();
    return 0;
}

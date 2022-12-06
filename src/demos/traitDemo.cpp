//
// Created by Nathan on 2021/3/15.
//

#include "demos/traitDemo.h"

namespace org::nathan::traitDemo {
    template<class T>
    struct MyIter {
        using value_type = T; //
        T *ptr;

        explicit MyIter(T *p = 0) : ptr(p) {}

        auto operator*() const -> T & { return *ptr; }
    };

// class type
    template<class T>
    struct iterator_traits {
        using value_type = typename T::value_type;
    };
    template<class T>
    struct iterator_traits<T *> {
        using value_type = T;
    };
    template<class T>
    struct iterator_traits<const T *> {
        using value_type = T;
    };

    template<class I>
    typename iterator_traits<I>::value_type func(I ite) {
        std::cout << typeid(I).name() << std::endl;
        std::cout << "normal version" << std::endl;
        return *ite;
    }

    [[maybe_unused]] void traitDemo() {
        MyIter<int> ite(new int(8));
        std::cout << func(ite) << std::endl;
        int *p = new int(52);
        std::cout << func(p) << std::endl;
        const int k = 3;
        std::cout << func(&k) << std::endl;
    }
}

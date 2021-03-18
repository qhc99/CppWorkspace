//
// Created by Nathan on 2021/3/15.
//

#include <iostream>

namespace org::nathan::traitDemo {
    template<class T>
    struct MyIter {
        [[maybe_unused]] typedef T value_type; // ��Ƕ�ͱ�����
        T *ptr;

        explicit MyIter(T *p = 0) : ptr(p) {}

        T &operator*() const { return *ptr; }
    };

// class type
    template<class T>
    struct iterator_traits {
        [[maybe_unused]] typedef typename T::value_type value_type;
    };
    template<class T>
    struct iterator_traits<T *> {
        [[maybe_unused]] typedef T value_type;
    };
    template<class T>
    struct iterator_traits<const T *> {
        [[maybe_unused]] typedef T value_type;
    };

    template<class I>
    typename iterator_traits<I>::value_type
    func(I ite) {
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

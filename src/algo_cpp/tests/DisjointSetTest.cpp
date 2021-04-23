//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE DisjointSetTest

#include "algo_cpp/DisjointSet.h"
#include <boost/test/unit_test.hpp>

using org::nathan::algo_cpp::DisjointSet;

BOOST_AUTO_TEST_SUITE(DisjointSetAPITest) // NOLINT

    BOOST_AUTO_TEST_CASE(find_set) // NOLINT
    {
        DisjointSet p{};
        DisjointSet a{};
        a.unionSet(p);
        DisjointSet b{};
        b.unionSet(p);
        DisjointSet f{};
        BOOST_CHECK_EQUAL(&a.findGroupRep(), &b.findGroupRep());
        BOOST_CHECK_NE(&a.findGroupRep(), &f.findGroupRep());
    }

BOOST_AUTO_TEST_SUITE_END() // NOLINT

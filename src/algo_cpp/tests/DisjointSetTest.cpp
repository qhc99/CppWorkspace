//
// Created by Nathan on 2021/4/4.
//

//#include <gtest/gtest.h>
#include "algo_cpp/structures/DisjointSet.h"

#define BOOST_TEST_MODULE DisjonintSetTest

#include <boost/test/unit_test.hpp>

using org::nathan::algo_cpp::structures::DisjointSet;

BOOST_AUTO_TEST_SUITE(DisjointSetAPITest) // NOLINT

    BOOST_AUTO_TEST_CASE(findSet) // NOLINT
    {
        DisjointSet p{};
        DisjointSet a{};
        a.unionSet(p);
        DisjointSet b{};
        b.unionSet(p);
        BOOST_CHECK_EQUAL(&a.findGroupRep(), &b.findGroupRep());
    }

BOOST_AUTO_TEST_SUITE_END() // NOLINT
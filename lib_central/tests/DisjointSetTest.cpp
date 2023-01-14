//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE DisjointSetTest

#include "lib_central/DisjointSet.h"
#include "boost/test/unit_test.hpp"

using dev::qhc::lib_central::DisjointSet;

BOOST_AUTO_TEST_SUITE(DisjointSetAPITest) // NOLINT

    BOOST_AUTO_TEST_CASE(find_set)
    {
        DisjointSet p{};
        DisjointSet a{};
        a.unionSet(p);
        DisjointSet b{};
        b.unionSet(p);
        DisjointSet f{};
        BOOST_CHECK_EQUAL(&a.groupRep(), &b.groupRep());
        BOOST_CHECK_NE(&a.groupRep(), &f.groupRep());
    }

BOOST_AUTO_TEST_SUITE_END() // NOLINT

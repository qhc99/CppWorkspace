//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE DisjointSetTest

#include "lib_central/DisjointSet.h"
#include <gtest/gtest.h>
#include <iostream>

using dev::qhc::lib_central::DisjointSet;



TEST(DisjointSetAPITest, find_set){
    DisjointSet p{};
    DisjointSet a{};
    a.unionSet(p);
    DisjointSet b{};
    b.unionSet(p);
    DisjointSet f{};
    EXPECT_EQ(&a.groupRep(), &b.groupRep());
    EXPECT_NE(&a.groupRep(), &f.groupRep());
}


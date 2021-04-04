//
// Created by Nathan on 2021/4/4.
//

#include <gtest/gtest.h>
#include "algo_cpp_structures/DisjointSet.h"


class DisjointSetTests : public ::testing::Test
{
};

TEST_F(DisjointSetTests, findSet)
{
    DisjointSet p{};
    DisjointSet a{};
    a.unionSet(p);
    DisjointSet b{};
    b.unionSet(p);
    EXPECT_TRUE(&a.findGroupRep() == &b.findGroupRep());
}
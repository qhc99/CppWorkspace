//
// Created by Nathan on 2021/4/4.
//

#include <gtest/gtest.h>
#include "algo_cpp/structures/DisjointSet.h"

using org::nathan::algo_cpp::structures::DisjointSet;

class DisjointSetFixture : public ::testing::Test
{
};

TEST_F(DisjointSetFixture, findSet) // NOLINT
{
    DisjointSet p{};
    DisjointSet a{};
    a.unionSet(p);
    DisjointSet b{};
    b.unionSet(p);
    EXPECT_TRUE(&a.findGroupRep() == &b.findGroupRep()) << "findGroupRep()";
}

TEST_F(DisjointSetFixture, foo) // NOLINT
{
    EXPECT_EQ(1, 1) << "foo";
}

//int main(int argc, char **argv)
//{
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
//}
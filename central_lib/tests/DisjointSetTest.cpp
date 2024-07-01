//
// Created by QC on 2021/4/4.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib_central/DisjointSet.h"
#include "doctest.h"

using dev::qhc::central_lib::DisjointSet;

TEST_CASE("DisjointSetAPITest.find_set")
{

    DisjointSet p {};
    DisjointSet a {};
    a.unionSet(p);
    DisjointSet b {};
    b.unionSet(p);
    DisjointSet f {};
    REQUIRE(&a.groupRep() == &b.groupRep());
    REQUIRE(&a.groupRep() != &f.groupRep());
}

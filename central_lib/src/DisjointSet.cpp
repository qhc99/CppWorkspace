#include "lib_central/DisjointSet.h"

using namespace dev::qhc::central_lib;

void DisjointSet::link(DisjointSet& x, DisjointSet& y)
{
    if (x.rank > y.rank) {
        y.parent = &x;
    } else {
        x.parent = &y;
        if (x.rank == y.rank) {
            y.rank = y.rank + 1;
        }
    }
}

auto DisjointSet::findGroupRep(DisjointSet& x)
    -> DisjointSet& // NOLINT(misc-no-recursion)
{
    if (&x != x.parent) {
        x.parent = &findGroupRep(*x.parent);
    }
    return *x.parent;
}

auto DisjointSet::groupRep() -> DisjointSet&
{
    if (this != this->parent) {
        this->parent = &findGroupRep(*this->parent);
    }
    return *this->parent;
}

[[maybe_unused]] void DisjointSet::unionSet(DisjointSet& a)
{
    link(groupRep(), findGroupRep(a));
}

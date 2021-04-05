#include "algo_cpp/structures/DisjointSet.h"

namespace org::nathan::algo_cpp::structures
{
    void DisjointSet::link(DisjointSet &x, DisjointSet &y)
    {
        if (x.rank > y.rank)
        {
            y.parent = &x;
        }
        else
        {
            x.parent = &y;
            if (x.rank == y.rank)
            {
                y.rank = y.rank + 1;
            }
        }
    }

    DisjointSet &DisjointSet::findGroupRep(DisjointSet &x) //NOLINT recursive call
    {
        if (&x != x.parent)
        {
            x.parent = &findGroupRep(*x.parent);
        }
        return *x.parent;
    }

    DisjointSet &DisjointSet::findGroupRep()
    {
        if (this != this->parent)
        {
            this->parent = &findGroupRep(*this->parent);
        }
        return *this->parent;
    }


    [[maybe_unused]] void DisjointSet::unionSet(DisjointSet &a)
    {
        link(findGroupRep(), findGroupRep(a));
    }
}

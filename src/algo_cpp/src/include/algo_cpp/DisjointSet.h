#ifndef ORG_NATHAN_CPP_ALL_IN_ONE_ALGO_CPP_STRUCTURES_DISJOINT_SET_H
#define ORG_NATHAN_CPP_ALL_IN_ONE_ALGO_CPP_STRUCTURES_DISJOINT_SET_H

namespace org::nathan::algo_cpp
{
    class [[maybe_unused]] DisjointSet
    {
    private:
        int rank{0};
        DisjointSet *parent{this};

        static void link(DisjointSet &x, DisjointSet &y);

        static DisjointSet &findGroupRep(DisjointSet &x);

    public:

        DisjointSet &findGroupRep();

        [[maybe_unused]] void unionSet(DisjointSet &a);

        virtual ~DisjointSet() = default;
    };
}


#endif

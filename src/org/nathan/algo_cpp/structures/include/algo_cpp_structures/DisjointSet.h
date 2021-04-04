#ifndef ORG_NATHAN_CPP_ALL_IN_ONE_ALGO_CPP_STRUCTURES_DISJOINT_SET_H
#define ORG_NATHAN_CPP_ALL_IN_ONE_ALGO_CPP_STRUCTURES_DISJOINT_SET_H


class [[maybe_unused]] DisjointSet
{
private:
    int rank{0};
    DisjointSet *parent{nullptr};

    static void link(DisjointSet &x, DisjointSet &y);

public:
    static DisjointSet &findSet(DisjointSet &x);

    DisjointSet &findSet();

    [[maybe_unused]] static void unionSet(DisjointSet &a, DisjointSet &b);

    [[maybe_unused]] void unionSet(DisjointSet &a);
};


#endif

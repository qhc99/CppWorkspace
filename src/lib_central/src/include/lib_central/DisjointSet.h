#ifndef ORG_QHC_LIB_CENTRAL_DISJOINT_SET_H
#define ORG_QHC_LIB_CENTRAL_DISJOINT_SET_H

namespace org::qhc::lib_central {
    class DisjointSet {
    private:
        int rank{0};
        DisjointSet *parent{this};


        static void link(DisjointSet &x, DisjointSet &y);

        static auto findGroupRepOf(DisjointSet &x) -> DisjointSet &;

    public:

        DisjointSet() = default;

        DisjointSet(const DisjointSet &other) = default;

        DisjointSet(DisjointSet &&other) = default;

        auto groupRep() -> DisjointSet &;

        auto operator=(const DisjointSet &other) -> DisjointSet & = default;

        auto operator=(DisjointSet &&other) -> DisjointSet & = default;

        void unionSet(DisjointSet &a);

        virtual ~DisjointSet() = default;
    };
}


#endif

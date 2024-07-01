#ifndef ORG_QC_CPP_CENTRAL_LIB_DISJOINT_SET
#define ORG_QC_CPP_CENTRAL_LIB_DISJOINT_SET

namespace dev::qhc::central_lib {
class DisjointSet {
private:
    int rank { 0 };
    DisjointSet* parent { this };

    static void link(DisjointSet& x, DisjointSet& y);

    static auto findGroupRep(DisjointSet& x) -> DisjointSet&;

public:
    DisjointSet() = default;

    DisjointSet(const DisjointSet& other) = default;

    DisjointSet(DisjointSet&& other) = default;

    auto groupRep() -> DisjointSet&;

    auto operator=(const DisjointSet& other) -> DisjointSet& = default;

    auto operator=(DisjointSet&& other) -> DisjointSet& = default;

    void unionSet(DisjointSet& a);

    virtual ~DisjointSet() = default;
};
} // namespace dev::qhc::central_lib

#endif

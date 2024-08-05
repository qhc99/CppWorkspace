#include <cmath>

#include <iostream>
using namespace std;

class DefaultPolicy1 { };
class DefaultPolicy2 { };
class DefaultPolicy3 { };
class DefaultPolicy4 { };
class DefaultPolicies {
public:
    using P1 = DefaultPolicy1;
    using P2 = DefaultPolicy2;
    using P3 = DefaultPolicy3;
    using P4 = DefaultPolicy4;
};
class DefaultPolicyArgs : virtual public DefaultPolicies { };
template <typename Policy>
class Policy1_is : virtual public DefaultPolicies {
public:
    using P1 = Policy;
};
template <typename Policy>
class Policy2_is : virtual public DefaultPolicies {
public:
    using P2 = Policy;
};
template <typename Policy>
class Policy3_is : virtual public DefaultPolicies {
public:
    using P3 = Policy;
};
template <typename Policy>
class Policy4_is : virtual public DefaultPolicies {
public:
    using P4 = Policy;
};
template <typename Base, int D>
class CanMultiDerivedFromOneClass : public Base { };
template <typename Setter1, typename Setter2, typename Setter3, typename Setter4>
class PolicySelector : public CanMultiDerivedFromOneClass<Setter1, 1>,
                       public CanMultiDerivedFromOneClass<Setter2, 2>,
                       public CanMultiDerivedFromOneClass<Setter3, 3>,
                       public CanMultiDerivedFromOneClass<Setter4, 4> { };
template <typename PolicySet1 = DefaultPolicyArgs,
    typename PolicySet2 = DefaultPolicyArgs,
    typename PolicySet3 = DefaultPolicyArgs,
    typename PolicySet4 = DefaultPolicyArgs>
class MyClass {
public:
    using Policies = PolicySelector<PolicySet1, PolicySet2, PolicySet3, PolicySet4>;
};
int main()
{
    typedef MyClass<Policy2_is<int>, Policy4_is<double>> MyClassDef;
    MyClassDef::Policies::P1 p1;
    MyClassDef::Policies::P2 p2 = 0;
    MyClassDef::Policies::P3 p3;
    MyClassDef::Policies::P4 p4 = NAN;
    cout << typeid(p1).name() << '\n';
    cout << typeid(p2).name() << '\n';
    cout << typeid(p3).name() << '\n';
    cout << typeid(p4).name() << '\n';
    return 0;
}
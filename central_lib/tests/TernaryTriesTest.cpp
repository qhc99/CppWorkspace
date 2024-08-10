#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib_central/TernaryTries.hpp"
#include "doctest/doctest.h"
#include <string>
#include <unordered_set>

class TernaryTriesTestFixture {

public:
    TernaryTries<int> case1;
    std::unordered_set<std::string> keys;
    TernaryTries<int> case2;
    TernaryTries<int> case3;
    TernaryTries<int> case4;

    TernaryTriesTestFixture()
    {
        case1.insert("by", 0);
        case1.insert("she", 1);
        case1.insert("shells", 2);
        case1.insert("sea", 3);
        case1.insert("sells", 4);
        case1.insert("shore", 5);
        case1.insert("the", 6);

        keys.insert("by");
        keys.insert("she");
        keys.insert("shells");
        keys.insert("sea");
        keys.insert("sells");
        keys.insert("shore");
        keys.insert("the");

        case2.insert("shells", 0);
        case2.insert("by", 1);
        case2.insert("the", 2);
        case2.insert("sells", 3);
        case2.insert("shore", 4);
        case2.insert("she", 5);
        case2.insert("sea", 6);

        case4.insert("a", 0);
    }
};

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "TestLongestPrefix")
{
    REQUIRE("she" == case1.longestPrefixOf("shell"));
    REQUIRE("shells" == case1.longestPrefixOf("shellsort"));
    REQUIRE("" == case1.longestPrefixOf("a"));
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "KeysTest")
{
    auto ks { case1.keys() };
    for (const std::string& k : ks) {
        REQUIRE(keys.contains(k));
    }
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "KeysWithPrefixTest")
{
    std::deque<std::string> d {};
    d.emplace_back("she");
    d.emplace_back("shells");
    d.emplace_back("shore");
    auto res = case1.keysWithPrefix("sh");
    for (const auto& i : res) {
        REQUIRE((std::find(d.begin(), d.end(), i) != d.end()) == true);
    }

    std::deque<std::string> d2 {};
    d2.emplace_back("by");
    auto res2 = case1.keysWithPrefix("b");
    for (const auto& i : res2) {
        REQUIRE((std::find(d2.begin(), d2.end(), i) != d2.end()) == true);
    }
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "RemoveTest")
{
    REQUIRE(7 == case2.getCount());

    int val {};
    REQUIRE(case2.remove("by", &val) == true);
    REQUIRE(case2.contain_key("by") == false);
    REQUIRE(case2.contain_key("she") == true);
    REQUIRE(case2.contain_key("shells") == true);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(6 == case2.getCount());
    REQUIRE(1 == val);

    REQUIRE(case2.remove("she", &val) == true);
    REQUIRE(case2.contain_key("she") == false);
    REQUIRE(case2.contain_key("shells") == true);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(5 == case2.getCount());
    REQUIRE(5 == val);

    REQUIRE(case2.remove("shells", &val) == true);
    REQUIRE(case2.contain_key("shells") == false);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(4 == case2.getCount());
    REQUIRE(0 == val);

    REQUIRE(case2.remove("aaaa", nullptr) == false);

    REQUIRE(case2.remove("sea", &val) == true);
    REQUIRE(case2.contain_key("sea") == false);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(3 == case2.getCount());
    REQUIRE(6 == val);

    REQUIRE(case2.remove("sells", &val) == true);
    REQUIRE(case2.contain_key("sells") == false);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(2 == case2.getCount());
    REQUIRE(3 == val);

    REQUIRE(case2.remove("shore", &val) == true);
    REQUIRE(case2.contain_key("shore") == false);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(1 == case2.getCount());
    REQUIRE(4 == val);

    REQUIRE(case2.remove("the", &val) == true);
    REQUIRE(case2.contain_key("the") == false);
    REQUIRE(2 == val);

    REQUIRE(nullptr == case2.getRoot());
    REQUIRE(0 == case2.getCount());
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "TryGetTest")
{
    REQUIRE(7 == case2.getCount());

    REQUIRE(case2.remove("by", nullptr) == true);
    REQUIRE(case2.remove("she", nullptr) == true);
    REQUIRE(case2.remove("shells", nullptr) == true);
    REQUIRE(case2.remove("aaaa", nullptr) == false);
    REQUIRE(case2.remove("sea", nullptr) == true);

    int val {};
    REQUIRE(case2.try_get("by", &val) == false);
    REQUIRE(case2.try_get("she", &val) == false);
    REQUIRE(case2.try_get("shells", &val) == false);
    REQUIRE(case2.try_get("sea", &val) == false);

    REQUIRE(case2.try_get("sells", &val) == true);
    REQUIRE(3 == val);
    REQUIRE(case2.try_get("shore", &val) == true);
    REQUIRE(4 == val);
    REQUIRE(case2.try_get("the", &val) == true);
    REQUIRE(2 == val);
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "CloneTest")
{
    auto caseClone { case2.clone() };
    REQUIRE(7 == caseClone.getCount());

    int val {};
    REQUIRE(caseClone.remove("by", nullptr) == true);
    REQUIRE(caseClone.remove("she", nullptr) == true);
    REQUIRE(caseClone.remove("shells", nullptr) == true);
    REQUIRE(caseClone.remove("aaaa", nullptr) == false);
    REQUIRE(caseClone.remove("sea", nullptr) == true);

    REQUIRE(caseClone.try_get("by", &val) == false);
    REQUIRE(caseClone.try_get("she", &val) == false);
    REQUIRE(caseClone.try_get("shells", &val) == false);
    REQUIRE(caseClone.try_get("sea", &val) == false);

    REQUIRE(caseClone.try_get("sells", &val) == true);
    REQUIRE(3 == val);
    REQUIRE(caseClone.try_get("shore", &val) == true);
    REQUIRE(4 == val);
    REQUIRE(caseClone.try_get("the", &val) == true);
    REQUIRE(2 == val);
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "CornerCaseTest")
{
    int val {};
    // empty strings
    REQUIRE(case1.try_get("", &val) == false);
    REQUIRE(case1.remove("", &val) == false);
    REQUIRE("" == case1.longestPrefixOf(""));

    REQUIRE(case1.remove("shellsFalse", &val) == false);
    case1.insert("by", 1, false);

    // empty and single string
    REQUIRE(0 == case3.keysWithPrefix("a").size());
    REQUIRE(1 == case4.keysWithPrefix("a").size());

    // has prefix but not contain
    REQUIRE(case1.remove("sh", nullptr) == false);
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "KeyOnlyTest")
{
    TernaryTries t{};
    t.insert("a");
    REQUIRE(true == t.contain_key("a"));
    t.remove("a");
    REQUIRE(false == t.contain_key("a"));
}
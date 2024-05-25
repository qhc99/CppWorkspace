//
// Created by QC on 2023-01-21.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib_central/TernaryTries.hpp"
#include "doctest.h"
#include <string>
#include <unordered_set>


class TernaryTriesTestFixture {

public:
    TernaryTries<int> case1{};
    std::unordered_set<std::string> keys{};
    TernaryTries<int> case2{};

    TernaryTriesTestFixture() {
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
    }


};

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "TestLongestPrefix") {
    REQUIRE("she" == case1.longestPrefixOf("shell"));
    REQUIRE("shells" == case1.longestPrefixOf("shellsort"));
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "KeysTest") {
    auto ks{case1.keys()};
    for (const std::string &k: ks) {
        REQUIRE(keys.contains(k));
    }
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "KeysWithPrefixTest") {
    std::deque<std::string> d{};
    d.emplace_back("she");
    d.emplace_back("shells");
    d.emplace_back("shore");
    auto res = case1.keysWithPrefix("sh");
    for (const auto &i: res) {
        REQUIRE((std::find(d.begin(), d.end(), i) != d.end()) == true);
    }
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "RemoveTest") {
    REQUIRE(7 == case2.getCount());


    REQUIRE(case2.remove("by", nullptr) == true);
    REQUIRE(case2.contain_key("by") == false);
    REQUIRE(case2.contain_key("she") == true);
    REQUIRE(case2.contain_key("shells") == true);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(6 == case2.getCount());

    REQUIRE(case2.remove("she", nullptr) == true);
    REQUIRE(case2.contain_key("she") == false);
    REQUIRE(case2.contain_key("shells") == true);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(5 == case2.getCount());

    REQUIRE(case2.remove("shells", nullptr) == true);
    REQUIRE(case2.contain_key("shells") == false);
    REQUIRE(case2.contain_key("sea") == true);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(4 == case2.getCount());

    REQUIRE(case2.remove("aaaa", nullptr) == false);

    REQUIRE(case2.remove("sea", nullptr) == true);
    REQUIRE(case2.contain_key("sea") == false);
    REQUIRE(case2.contain_key("sells") == true);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(3 == case2.getCount());

    REQUIRE(case2.remove("sells", nullptr) == true);
    REQUIRE(case2.contain_key("sells") == false);
    REQUIRE(case2.contain_key("shore") == true);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(2 == case2.getCount());

    REQUIRE(case2.remove("shore", nullptr) == true);
    REQUIRE(case2.contain_key("shore") == false);
    REQUIRE(case2.contain_key("the") == true);
    REQUIRE(1 == case2.getCount());

    REQUIRE(case2.remove("the", nullptr) == true);
    REQUIRE(case2.contain_key("the") == false);

    REQUIRE(nullptr == case2.getRoot());
    REQUIRE(0 == case2.getCount());
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture, "TryGetTest") {
    REQUIRE(7 == case2.getCount());


    REQUIRE(case2.remove("by", nullptr) == true);
    REQUIRE(case2.remove("she", nullptr) == true);
    REQUIRE(case2.remove("shells", nullptr) == true);
    REQUIRE(case2.remove("aaaa", nullptr) == false);
    REQUIRE(case2.remove("sea", nullptr) == true);

    int val{};
    REQUIRE(case2.try_get("by", &val) == false);
    REQUIRE(case2.try_get("she", &val) == false);
    REQUIRE(case2.try_get("shells", &val) == false);
    REQUIRE(case2.try_get("sea", &val) == false);

    REQUIRE(case2.try_get("sells", &val) == true);
    REQUIRE(3 == val);
    REQUIRE(case2.try_get("shore",&val) == true);
    REQUIRE(4 == val);
    REQUIRE(case2.try_get("the",&val) == true);
    REQUIRE(2 == val);
}

TEST_CASE_FIXTURE(TernaryTriesTestFixture,"CloneTest"){
    auto case3 {case2.clone()};
    REQUIRE(7 == case3.getCount());


    REQUIRE(case3.remove("by", nullptr) == true);
    REQUIRE(case3.remove("she", nullptr) == true);
    REQUIRE(case3.remove("shells", nullptr) == true);
    REQUIRE(case3.remove("aaaa", nullptr) == false);
    REQUIRE(case3.remove("sea", nullptr) == true);

    int val{};
    REQUIRE(case3.try_get("by", &val) == false);
    REQUIRE(case3.try_get("she", &val) == false);
    REQUIRE(case3.try_get("shells", &val) == false);
    REQUIRE(case3.try_get("sea", &val) == false);

    REQUIRE(case3.try_get("sells", &val) == true);
    REQUIRE(3 == val);
    REQUIRE(case3.try_get("shore",&val) == true);
    REQUIRE(4 == val);
    REQUIRE(case3.try_get("the",&val) == true);
    REQUIRE(2 == val);
}
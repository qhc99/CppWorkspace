//
// Created by Nathan on 2023-01-21.
//
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include "lib_central/TernaryTries.hpp"

using std::string, std::unordered_set, std::deque;

class TernaryTriesTest : public ::testing::Test {

public:
    TernaryTries<int> case1{};
    unordered_set<string> keys{};
    TernaryTries<int> case2{};

    TernaryTriesTest() {
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

TEST_F(TernaryTriesTest, TestLongestPrefix) {
    EXPECT_EQ("she", case1.longestPrefixOf("shell"));
    EXPECT_EQ("shells", case1.longestPrefixOf("shellsort"));
}

TEST_F(TernaryTriesTest, KeysTest) {
    auto ks{case1.keys()};
    for (const auto &k: ks) {
        EXPECT_TRUE(keys.contains(k));
    }
}

TEST_F(TernaryTriesTest, KeysWithPrefixTest) {
    deque<string> d{};
    d.emplace_back("she");
    d.emplace_back("shells");
    d.emplace_back("shore");
    auto res = case1.keysWithPrefix("sh");
    for (const auto &i: res) {
        EXPECT_TRUE(std::find(d.begin(), d.end(), i) != d.end());
    }
}

TEST_F(TernaryTriesTest, RemoveTest) {
    EXPECT_EQ(7, case2.getCount());


    EXPECT_TRUE(case2.remove("by", nullptr));
    EXPECT_FALSE(case2.contain_key("by"));
    EXPECT_TRUE(case2.contain_key("she"));
    EXPECT_TRUE(case2.contain_key("shells"));
    EXPECT_TRUE(case2.contain_key("sea"));
    EXPECT_TRUE(case2.contain_key("sells"));
    EXPECT_TRUE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(6, case2.getCount());

    EXPECT_TRUE(case2.remove("she", nullptr));
    EXPECT_FALSE(case2.contain_key("she"));
    EXPECT_TRUE(case2.contain_key("shells"));
    EXPECT_TRUE(case2.contain_key("sea"));
    EXPECT_TRUE(case2.contain_key("sells"));
    EXPECT_TRUE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(5, case2.getCount());

    EXPECT_TRUE(case2.remove("shells", nullptr));
    EXPECT_FALSE(case2.contain_key("shells"));
    EXPECT_TRUE(case2.contain_key("sea"));
    EXPECT_TRUE(case2.contain_key("sells"));
    EXPECT_TRUE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(4, case2.getCount());

    EXPECT_FALSE(case2.remove("aaaa", nullptr));

    EXPECT_TRUE(case2.remove("sea", nullptr));
    EXPECT_FALSE(case2.contain_key("sea"));
    EXPECT_TRUE(case2.contain_key("sells"));
    EXPECT_TRUE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(3, case2.getCount());

    EXPECT_TRUE(case2.remove("sells", nullptr));
    EXPECT_FALSE(case2.contain_key("sells"));
    EXPECT_TRUE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(2, case2.getCount());

    EXPECT_TRUE(case2.remove("shore", nullptr));
    EXPECT_FALSE(case2.contain_key("shore"));
    EXPECT_TRUE(case2.contain_key("the"));
    EXPECT_EQ(1, case2.getCount());

    EXPECT_TRUE(case2.remove("the", nullptr));
    EXPECT_FALSE(case2.contain_key("the"));

    EXPECT_EQ(nullptr, case2.getRoot());
    EXPECT_EQ(0, case2.getCount());
}

TEST_F(TernaryTriesTest, TryGetTest) {
    EXPECT_EQ(7, case2.getCount());


    EXPECT_TRUE(case2.remove("by", nullptr));
    EXPECT_TRUE(case2.remove("she", nullptr));
    EXPECT_TRUE(case2.remove("shells", nullptr));
    EXPECT_FALSE(case2.remove("aaaa", nullptr));
    EXPECT_TRUE(case2.remove("sea", nullptr));

    int val{};
    EXPECT_FALSE(case2.try_get("by", &val));
    EXPECT_FALSE(case2.try_get("she", &val));
    EXPECT_FALSE(case2.try_get("shells", &val));
    EXPECT_FALSE(case2.try_get("sea", &val));

    EXPECT_TRUE(case2.try_get("sells", &val));
    EXPECT_EQ(3,val);
    EXPECT_TRUE(case2.try_get("shore",&val));
    EXPECT_EQ(4,val);
    EXPECT_TRUE(case2.try_get("the",&val));
    EXPECT_EQ(2,val);
}
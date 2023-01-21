//
// Created by Nathan on 2023-01-21.
//
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include "lib_central/TernaryTries.hpp"

using std::string, std::unordered_set,std::deque;

class TernaryTriesTest : public ::testing::Test {

public:
    TernaryTries<int> case1{};
    unordered_set<string> keys{};
    TernaryTries<int> case2{};

    TernaryTriesTest() {
        case1.insert("by", 9);
        case1.insert("she", 0);
        case1.insert("shells", 0);
        case1.insert("sea", 0);
        case1.insert("sells", 0);
        case1.insert("shore", 0);
        case1.insert("the", 0);

        keys.insert("by");
        keys.insert("she");
        keys.insert("shells");
        keys.insert("sea");
        keys.insert("sells");
        keys.insert("shore");
        keys.insert("the");

        case2.insert("shells", 0);
        case2.insert("by", 0);
        case2.insert("the", 0);
        case2.insert("sells", 0);
        case2.insert("shore", 0);
        case2.insert("she", 0);
        case2.insert("sea", 0);
    }


};

TEST_F(TernaryTriesTest, TestLongestPrefix) {
    EXPECT_EQ("she", case1.longestPrefixOf("shell"));
    EXPECT_EQ("shells", case1.longestPrefixOf("shellsort"));
}

TEST_F(TernaryTriesTest, KeysTest) {
    auto ks{case1.keys()};
    for(const auto& k : ks){
        EXPECT_TRUE(keys.contains(k));
    }
}

TEST_F(TernaryTriesTest, KeysWithPrefixTest) {
    deque<string> d{};
    d.emplace_back("she");
    d.emplace_back("shells");
    d.emplace_back("shore");
    auto res = case1.keysWithPrefix("sh");
    for(const auto& i : res){
        EXPECT_TRUE(std::find(d.begin(), d.end(), i) != d.end());
    }
}

TEST_F(TernaryTriesTest, RemoveTest) {

}
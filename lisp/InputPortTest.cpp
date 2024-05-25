#include <gtest/gtest.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <array>
#include "InputPort.h"
#include "values.h"

using std::stringstream,std::make_shared,std::shared_ptr;

TEST(InputPortTest, test_next_token_simple){
    auto s{make_shared<stringstream>("(define x 0)")};
    auto i{InputPort{*s}};
    shared_ptr<Value> token{};
    std::array<string,5> ans {{"(", "define", "x","0",")"}};
    int idx{};
    do{
        token = i.next_token();
        if(typeid(*token) == typeid(String)){
            EXPECT_EQ(ans[idx++], std::dynamic_pointer_cast<String>(token)->val);
        }
    }while(typeid(*token) != typeid(Symbol));
    EXPECT_EQ(idx, ans.size());
}
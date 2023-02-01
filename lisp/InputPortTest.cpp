#include <gtest/gtest.h>
#include <memory>
#include <iostream>
#include <sstream>
#include "InputPort.h"
#include "values.h"

using std::unique_ptr,std::stringstream,std::make_unique,std::shared_ptr;
TEST(InputPortTest, test_next_token_simple){
    auto s{make_unique<stringstream>("(define x 0)")};
    auto i{InputPort{std::move(s)}};
    shared_ptr<Value> token{};
    string ans[] = {"(", "define", "x","0",")"};
    int idx{};
    do{
        token = i.next_token();
        if(typeid(*token) == typeid(String)){
            EXPECT_EQ(ans[idx++], std::dynamic_pointer_cast<String>(token)->val);
        }
    }while(typeid(*token) != typeid(Symbol));
}
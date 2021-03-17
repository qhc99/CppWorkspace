//
// Created by Nathan on 2021/3/16.
//

#ifndef LEARNCPP_ALIASES_H
#define LEARNCPP_ALIASES_H

#include <string>
#include <iostream>
#include <ostream>
#include <iosfwd>
#include <vector>
#include <set>
#include <map>
#include <memory>

namespace org::nathan::aliases {

    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;
    using std::set;
    using std::map;
    using std::shared_ptr;
    using std::weak_ptr;
    using std::unique_ptr;
    using std::make_unique;
    using std::make_shared;

    // TODO Newline has bug in operator<<
    template<typename CharT, typename Traits>
    [[maybe_unused]] constexpr auto Newline = endl<CharT, Traits>;
}

#endif //LEARNCPP_ALIASES_H

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace Jakube {
template <typename seed_t>
class Kiss64Random {
    static_assert(std::is_unsigned<seed_t>::value,
                  "Kiss64Random seed type must be an unsigned integer.");
    
public:
    
    
    static constexpr uint64_t default_seed = 1234567890987654321ULL;
    
    Kiss64Random(seed_t seed):
        x{seed},
        y{362436362436362436ULL},
        z{1066149217761810ULL},
        c{123456123456123456ULL}
    {}

    Kiss64Random(): Kiss64Random(static_cast<seed_t>(default_seed)){}

    auto kiss();

    inline auto flip();

    inline std::size_t index(std::size_t);

    inline void set_seed(seed_t);


private:
    std::uint64_t x, y, z, c;

};
}

template <typename seed_t>
auto Jakube::Kiss64Random<seed_t>::kiss(){
        z = 6906969069LL*z+1234567;

        y ^= (y<<13);
        y ^= (y>>17);
        y ^= (y<<43);

        auto t = (x<<58)+c;
        c = (x>>6);
        x += t;
        c += (x<t);

        return x + y + z;
}

template <typename seed_t>
inline auto Jakube::Kiss64Random<seed_t>::flip(){
    return kiss() & 1;
}

template <typename seed_t>
inline std::size_t Jakube::Kiss64Random<seed_t>::index(std::size_t n){
    if (n == 0){
        return 0;
    }
    return kiss() % n;
}

template <typename seed_t>
inline void Jakube::Kiss64Random<seed_t>::set_seed(seed_t seed){
    x = seed;
    y = 362436362436362436ULL;
    z = 1066149217761810ULL;
    c = 123456123456123456ULL;
}

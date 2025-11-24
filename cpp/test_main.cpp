// COMPILE WITH: g++ -std=c++17 -O2 -o test_runner test_jakube.cpp

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "kissrandom.h"
#include "jakubelib.h"
#include <vector>
#include <cstdint>
#include <cstdio>

// We'll use the Kiss64Random you provided as the RNG
using MyRandom = Jakube::Kiss64Random<uint64_t>;

// Only keep SingleThreadPolicy.
using SingleThreadPolicy = Jakube::JakubeIndexSingleThreadedBuildPolicy;
// Only define HammingIndex for this project.
using HammingIndex = Jakube::JakubeIndex<int, int32_t, Jakube::Hamming, MyRandom>;

TEST_CASE("JakubeIndex - Hamming Workflow") {
    int f = 1; // 1 int32_t vector element
    HammingIndex index(f);

    std::vector<int32_t> v0 = {0b0011}; // 0x3
    std::vector<int32_t> v1 = {0b0110}; // 0x6
    std::vector<int32_t> v2 = {0b1111}; // 0xF

    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());

    CHECK(index.get_distance(0, 1) == 2); // popcount(0x3 ^ 0x6 = 0x5)
    CHECK(index.get_distance(0, 2) == 2); // popcount(0x3 ^ 0xF = 0xC)
    CHECK(index.get_distance(1, 2) == 2); // popcount(0x6 ^ 0xF = 0x9)

    index.build(5);

    std::vector<int32_t> q = {0b0000};
    std::vector<int> result;
    std::vector<int32_t> distances;
    index.get_nns_by_vector(q.data(), 3, -1, &result, &distances);

    REQUIRE(result.size() == 3);
    REQUIRE(distances.size() == 3);
}

TEST_CASE("Kiss64Random - Behavior") {
    using TestRandom = Jakube::Kiss64Random<uint64_t>;
    TestRandom rand1, rand2;
    for(int i = 0; i < 10; ++i)
        CHECK(rand1.kiss() == rand2.kiss());

    TestRandom rand3(12345);
    TestRandom rand4(12345);
    for(int i = 0; i < 5; ++i)
        CHECK(rand3.kiss() == rand4.kiss());
}

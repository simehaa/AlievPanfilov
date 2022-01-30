#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "options.hpp"

unsigned index(unsigned x, unsigned y, unsigned z, unsigned width, unsigned depth);
unsigned block_low(unsigned id, unsigned p, unsigned n);
unsigned block_high(unsigned id, unsigned p, unsigned n);
unsigned block_size(unsigned id, unsigned p, unsigned n);
std::size_t volume(std::vector<std::size_t> shape);
void work_division(Options &options);
void test_upper_dt(Options &options);
void print_results(Options &options);
#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "options.hpp"

std::size_t index(std::size_t x, std::size_t y, std::size_t z, std::size_t width, std::size_t depth);
std::size_t block_low(std::size_t id, std::size_t p, std::size_t n);
std::size_t block_high(std::size_t id, std::size_t p, std::size_t n);
std::size_t volume(std::vector<std::size_t> shape);
std::vector<std::size_t> work_division_3d(std::size_t, std::size_t, std::size_t, std::size_t);
void test_upper_dt(Options &options);
void test_against_cpu(
  std::vector<float> initial_e, 
  std::vector<float> initial_r, 
  const std::vector<float> ipu_e, 
  const std::vector<float> ipu_r, 
  Options &options
);
void print_results_and_options(Options &options);
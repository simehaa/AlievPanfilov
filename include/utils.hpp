#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "options.hpp"

std::size_t index(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
std::size_t block_low(std::size_t, std::size_t, std::size_t);
std::size_t block_high(std::size_t, std::size_t, std::size_t);
std::size_t volume(std::vector<std::size_t>);
std::size_t surface_area(std::vector<std::size_t>);
std::vector<std::size_t> work_division_3d(std::size_t, std::size_t, std::size_t, std::size_t);
void test_upper_dt(Options&);
void test_against_cpu(
  std::vector<float>, 
  std::vector<float>, 
  const std::vector<float>, 
  const std::vector<float>, 
  Options&
);
void print_pde_problem(Options&);
void print_data_exchange_volumes(Options&);
void print_results(Options&);
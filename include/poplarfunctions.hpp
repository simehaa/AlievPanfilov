#pragma once
#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/TargetType.hpp>
#include <poplar/Tensor.hpp>
#include <vector>
#include "options.hpp"
#include "utils.hpp"

poplar::Device get_device(std::size_t n);
poplar::ComputeSet create_compute_set(
  poplar::Graph &graph,
  poplar::Tensor &e_in,
  poplar::Tensor &e_out,
  poplar::Tensor &r,
  Options &options,
  const std::string& compute_set_name
);
std::vector<poplar::program::Program> create_ipu_programs(
  poplar::Graph &graph, 
  Options &options
);
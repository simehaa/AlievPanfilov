#pragma once
#include <iostream>
#include <vector>
#include <boost/program_options.hpp>

// Command line arguments
struct Options {
	float my1;
	float my2;
	float delta;
	float epsilon;
	float a;
	float b;
	float k;
	float dx;
	float dt;
	float b_plus_1; // function of b
	float dtk; // function of dt and k
	float lambda; // function of delta, dx, dt
	float gamma; // function of delta, dx, dt
	std::size_t height;
	std::size_t width;
	std::size_t depth;
	std::string vertex;
	std::size_t num_ipus;
	std::size_t num_iterations;
	std::size_t tiles_per_ipu = 0;
	std::size_t num_tiles_available = 0;
	std::vector<std::size_t> tile_splits = {0,0,0};
	std::vector<std::size_t> ipu_splits = {1,1,1};
	std::vector<std::size_t> worker_splits = {3,2,1}; // 6 workers
	std::vector<std::size_t> smallest_slice = {
		std::numeric_limits<std::size_t>::max(),1,1
	};
	std::vector<std::size_t> largest_slice = {0,0,0};
	double wall_time;
	bool cpu = false;
	bool hierarchical_partitioning = false;
};

Options parse_options(int argc, char** argv);
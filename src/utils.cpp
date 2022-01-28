#include "utils.hpp"

unsigned index(unsigned x, unsigned y, unsigned z, unsigned width, unsigned depth) {
	// 1D index corresponding to a flattened 3D variable. 
  return (z) + (y)*(depth) + (x)*(width)*(depth);
}

unsigned block_low(unsigned id, unsigned p, unsigned n) {
	// low index of block id=0,...,p, when dividing the total length n into p pieces.
  return (id*n)/p; 
}

unsigned block_high(unsigned id, unsigned p, unsigned n) {
	// high index (non-inclusive) of block id=0,...,p, when dividing the total length n into p pieces.
  return block_low(id+1, p, n); 
}

unsigned block_size(unsigned id, unsigned p, unsigned n) {
	// length of block id=0,...,p, when dividing the total length n into p pieces.
  return block_high(id, p, n) - block_low(id, p, n); 
}

std::size_t volume(std::vector<std::size_t> shape) {
  // return volume of shape vector (3D)
  return shape[0]*shape[1]*shape[2];
}

void work_division(Options &options) {
  /* Function UPDATES options.splits
   * 1) all tiles will be used, hence options.num_tiles_available must
   *    previously be updated by using the target object
   * 2) the average resulting slice will have shape [height/nh, width/nw, depth/nd'] and
   *    this function chooses nh, nw, nd, so that the surface area is minimized.
   */
  float smallest_surface_area = std::numeric_limits<float>::max();
  std::size_t height = (options.height - 2);
  std::size_t width = (options.width - 2);
  std::size_t depth = (options.depth - 2) / options.num_ipus;
  std::size_t tile_count = options.num_tiles_available / options.num_ipus;
  for (std::size_t i = 1; i <= tile_count; ++i) {
    if (tile_count % i == 0) { // then i is a factor
      // Further, find two other factors, to obtain exactly three factors
      std::size_t other_factor = tile_count/i;
      for (std::size_t j = 1; j <= other_factor; ++j) {
        if (other_factor % j == 0) { // then j is a second factor
          std::size_t k = other_factor/j; // and k is the third factor
          std::vector<std::size_t> splits = {i,j,k}; 
          if (i*j*k != tile_count) {
            throw std::runtime_error("workDivision(), factorization does not work.");
          }
          for (std::size_t l = 0; l < 3; ++l) {
            for (std::size_t m = 0; m < 3; ++m) {
              for (std::size_t n = 0; n < 3; ++n) {
                if (l != m && l != n && m != n) {
                  float slice_height = float(height)/float(splits[l]);
                  float slice_width = float(width)/float(splits[m]);
                  float slice_depth = float(depth)/float(splits[n]);
                  float surface_area = 2.0*(slice_height*slice_width + slice_depth*slice_width + slice_depth*slice_height);
                  if (surface_area <= smallest_surface_area) {
                    smallest_surface_area = surface_area;
                    options.splits = splits;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_upper_dt(Options &options) {
  float ka = options.k*options.a;
  float k_1_a = options.k*(1 - options.a);
  float max = (ka > k_1_a) ? ka : k_1_a;
  float lambda = options.delta/(options.dx*options.dx);
  float r_plus = options.k*(options.b+1)*(options.b+1)/4.0;
  float upper_bound_dt = 1.0/(4*lambda + max + r_plus);
  if (options.dt > upper_bound_dt) 
    throw std::runtime_error(
      "Forward Euler method is not stable, because dt ("+std::to_string(options.dt)+
      ") > upper bound ("+std::to_string(upper_bound_dt)+")."
    );
}

void print_results(Options &options, double wall_time) {
  // Calculate metrics
  double mesh_volume = (double) options.height * (double) options.width * (double) options.depth;
  double flops_per_element = 29.0;
  double flops = mesh_volume * (double) options.num_iterations * flops_per_element / wall_time;
  double tflops = flops*1e-12;

  std::cout 
    <<   "3D Aliev-Panfilov model"
    << "\n-----------------------"
    << "\nNo. IPUs           = " << options.num_ipus
    << "\nNo. Tiles          = " << options.num_tiles_available
    << "\nTotal Grid         = " << options.height << "*" << options.width << "*" << options.depth << " = "
                                 << mesh_volume*1e-6 << " million elements"
    << "\nSmallest Sub-grid  = " << options.smallest_slice[0] << "*" << options.smallest_slice[1] << "*" << options.smallest_slice[2] 
    << "\nLargest Sub-grid   = " << options.largest_slice[0] << "*" << options.largest_slice[1] << "*" << options.largest_slice[2] 
    << "\nNo. Iterations     = " << options.num_iterations
    << "\nTFLOPS             = " << std::setprecision(4) << tflops
    << "\n";
}
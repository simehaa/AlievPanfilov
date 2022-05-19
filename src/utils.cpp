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
  std::size_t height = options.height;
  std::size_t width = options.width;
  std::size_t depth = options.depth / options.num_ipus;
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
            throw std::runtime_error("work_division(), factorization does not work.");
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

void test_against_cpu(
  std::vector<float> initial_e, 
  std::vector<float> initial_r, 
  const std::vector<float> ipu_e, 
  const std::vector<float> ipu_r, 
  Options &options
) {
  // Test against CPU code
  std::size_t h = options.height;
  std::size_t w = options.width;
  std::size_t d = options.depth;
  std::size_t hp = h + 2;
  std::size_t wp = w + 2;
  std::size_t dp = d + 2;

  std::size_t volume = h*w*d;
  std::size_t padded_volume = hp*wp*dp;
  std::vector<float> cpu_e(padded_volume);
  /* 
   * Note, cpu_e_temp and cpu_r are not padded
   * To access index [x,y,z] one must use width and depth
   * Remember to use h,w,d (for height, width, depth, respectively) for these
   * However, for cpu_e, use hp,wp,dp (height padded, width padded, depth padded)
   */
  std::vector<float> cpu_e_temp(volume);
  std::vector<float> cpu_r(volume);
  float e_center, r_center;

  // Initialize cpu_e/cpu_r
  for (std::size_t x = 1; x <= h; ++x) {
    for (std::size_t y = 1; y <= w; ++y) {
      for (std::size_t z = 1; z <= d; ++z) {
        cpu_e[index(x,y,z,wp,dp)] = initial_e[index(x-1,y-1,z-1,w,d)];
        cpu_r[index(x-1,y-1,z-1,w,d)] = initial_r[index(x-1,y-1,z-1,w,d)];
      }
    }
  }

  // Perform PDE model
  for (std::size_t t = 0; t < options.num_iterations; ++t) {
    /*
     * Boundary condition: net-zero gradient
     * Copy inner neighbour surfaces to padded boundaries
     * Each dimention goes from 0, ..., h-1 (or w or d)
     * Padded dimentions goes from 0, ..., h+1 (or w or d)
     * Immediate inner surfaces are at index 2 and h-1 (or w or d)
     * The destiation index for those surfaces are at index 0 and h+1 (or w or d)
     */
    for (std::size_t x = 1; x <= h; ++x) {
      for (std::size_t y = 1; y <= w; ++y) {
        cpu_e[index(x,y,0,wp,dp)] = cpu_e[index(x,y,2,wp,dp)]; // front surface
        cpu_e[index(x,y,d+1,wp,dp)] = cpu_e[index(x,y,d-1,wp,dp)]; // back surface
      }
    }
    for (std::size_t x = 1; x <= h; ++x) {
      for (std::size_t z = 1; z <= d; ++z) {
        cpu_e[index(x,0,z,wp,dp)] = cpu_e[index(x,2,z,wp,dp)]; // left surface
        cpu_e[index(x,w+1,z,wp,dp)] = cpu_e[index(x,w-1,z,wp,dp)]; // right surface
      }
    }
    for (std::size_t y = 1; y <= w; ++y) {
      for (std::size_t z = 1; z <= d; ++z) {
        cpu_e[index(0,y,z,wp,dp)] = cpu_e[index(2,y,z,wp,dp)]; // top surface
        cpu_e[index(h+1,y,z,wp,dp)] = cpu_e[index(h-1,y,z,wp,dp)]; // bottom surface
      }
    }

    // PDE computation by sliding stencil over inner volume 
    // (inner volume of padded mesh corresponds to full volume of unpadded mesh)
    for (std::size_t x = 1; x <= h; ++x) {
      for (std::size_t y = 1; y <= w; ++y) {
        for (std::size_t z = 1; z <= d; ++z) {
          e_center = cpu_e[index(x,y,z,wp,dp)]; // reusable variable
          r_center = cpu_r[index(x-1,y-1,z-1,w,d)];

          // New e_out_center
          cpu_e_temp[index(x-1,y-1,z-1,w,d)] = options.lambda*(
              cpu_e[index(x-1,y,z,wp,dp)] + cpu_e[index(x+1,y,z,wp,dp)] +
              cpu_e[index(x,y-1,z,wp,dp)] + cpu_e[index(x,y+1,z,wp,dp)] +
              cpu_e[index(x,y,z-1,wp,dp)] + cpu_e[index(x,y,z+1,wp,dp)]
            ) + options.gamma*e_center
            - options.dtk*e_center*(e_center - options.a)*(e_center - 1) 
            - options.dt*e_center*r_center;

          // New r_center
          cpu_r[index(x-1,y-1,z-1,w,d)] -= options.dt*(
            (options.epsilon + options.my1*r_center/(options.my2 + e_center))*
            (r_center + options.k*e_center*(e_center - options.b_plus_1))
          );
        }
      }
    }

    // re-update cpu_e from cpu_e_temp
    for (std::size_t x = 1; x <= h; ++x) {
      for (std::size_t y = 1; y <= w; ++y) {
        for (std::size_t z = 1; z <= d; ++z) {
          cpu_e[index(x,y,z,wp,dp)] = cpu_e_temp[index(x-1,y-1,z-1,w,d)];
        }
      }
    }
  }

  // Evaluate MSE
  double e_error, r_error, e_MSE=0, r_MSE=0;
  double scale = 1.0 / (double) volume;
  for (std::size_t x = 1; x < h + 1; ++x) {
    for (std::size_t y = 1; y < w + 1; ++y) {
      for (std::size_t z = 1; z < d + 1; ++z) {
        e_error = ipu_e[index(x-1,y-1,z-1,w,d)] - cpu_e[index(x,y,z,wp,dp)];
        r_error = ipu_r[index(x-1,y-1,z-1,w,d)] - cpu_r[index(x-1,y-1,z-1,w,d)];
        e_MSE += scale*e_error*e_error;
        r_MSE += scale*r_error*r_error;
      }
    }
  }
  std::cout
    << "Test IPU results vs. CPU results\n"
    << "--------------------------------\n"
    << "e_MSE = " << e_MSE << "\n" 
    << "r_MSE = " << r_MSE << "\n\n";
}

void test_upper_dt(Options &options) {
  float ka = options.k*options.a;
  float k_1_a = options.k*(1 - options.a);
  float max = (ka > k_1_a) ? ka : k_1_a;
  float r_plus = options.k*options.b_plus_1*options.b_plus_1/4.0;
  float upper_bound_dt = 1.0/(4*options.lambda + max + r_plus);
  if (options.dt > upper_bound_dt) 
    throw std::runtime_error(
      "Forward Euler method is not stable, because dt ("+std::to_string(options.dt)+
      ") > upper bound ("+std::to_string(upper_bound_dt)+")."
    );
}

void print_results_and_options(Options &options) {
  // Calculate metrics
  double mesh_volume = (double) options.height * (double) options.width * (double) options.depth;
  double flops_per_element = 28.0;
  double flops = mesh_volume * (double) options.num_iterations * flops_per_element / options.wall_time;
  double tflops = flops*1e-12;
  double padded_mesh = (double) (options.height + 2) * (double) (options.width + 2) * (double) (options.depth + 2);
  double minimum_MB = (2*padded_mesh + mesh_volume)*4*1e-6; // 4 bytes per element and converted to _mega_ (bytes)

  std::cout 
    << "3D Aliev-Panfilov model\n"
    << "-----------------------\n"
    << "No. IPUs = " << options.num_ipus << "\n"
    << "No. tiles = " << options.num_tiles_available << "\n"
    << "Total mesh = " << options.height << "*" << options.width << "*" << options.depth << " elements\n"
    << "3 Tensor's on-tile memory usage = " << minimum_MB << " MB\n"
    << "Available on-tile memory = " << options.total_memory_avail_MB << " MB\n" 
    << "Smallest tile partition = " << options.smallest_slice[0] << "*" << options.smallest_slice[1] << "*" << options.smallest_slice[2]  << " elements\n"
    << "Largest tile partition = " << options.largest_slice[0] << "*" << options.largest_slice[1] << "*" << options.largest_slice[2]  << " elements\n"
    << "No. iterations = " << options.num_iterations << "\n"
    << "Wall Time = " << std::setprecision(5) << options.wall_time << " s\n"
    << "Computational throughput = " << std::setprecision(5) << tflops << " TFLOPS\n\n";
}
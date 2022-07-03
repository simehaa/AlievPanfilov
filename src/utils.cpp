#include "utils.hpp"

std::size_t index(std::size_t x, std::size_t y, std::size_t z, std::size_t height, std::size_t width) {
	// 1D index corresponding to a flattened 3D variable. 
  return x + y*height + z*height*width;
}

std::size_t block_low(std::size_t id, std::size_t p, std::size_t n) {
	// low index of block id=0,...,p, when dividing the total length n into p pieces.
  return (id*n)/p; 
}

std::size_t block_high(std::size_t id, std::size_t p, std::size_t n) {
	// high index (non-inclusive) of block id=0,...,p, when dividing the total length n into p pieces.
  return block_low(id+1, p, n); 
}

std::size_t volume(std::vector<std::size_t> shape) {
  // return volume of shape vector (3D)
  return shape[0]*shape[1]*shape[2];
}

std::size_t surface_area(std::vector<std::size_t> shape) {
  // return volume of shape vector (3D)
  return 2*(shape[0]*shape[1] + shape[0]*shape[2] + shape[1]*shape[2]);
}

std::vector<std::size_t> work_division_3d(
  std::size_t height,
  std::size_t width,
  std::size_t depth,
  std::size_t num_partitions
) {
  /* 
   * Find the partition configuration that minimizes surface area
   * and uses num_partitions number of partitions.
   */
  std::vector<std::size_t> splits = {0, 0, 0};
  float smallest_surface_area = std::numeric_limits<float>::max();
  for (std::size_t i = 1; i <= num_partitions; ++i) {
    if (num_partitions % i == 0) { // then i is a factor
      // Further, find two other factors, to obtain exactly three factors
      std::size_t other_factor = num_partitions/i;
      for (std::size_t j = 1; j <= other_factor; ++j) {
        if (other_factor % j == 0) { // then j is a second factor
          std::size_t k = other_factor/j; // and k is the third factor
          std::vector<std::size_t> test_splits = {i,j,k}; 
          if (i*j*k != num_partitions) {
            throw std::runtime_error("work_division(), factorization does not work.");
          }
          for (std::size_t l = 0; l < 3; ++l) {
            for (std::size_t m = 0; m < 3; ++m) {
              for (std::size_t n = 0; n < 3; ++n) {
                if (l != m && l != n && m != n) {
                  float slice_height = float(height)/float(test_splits[l]);
                  float slice_width = float(width)/float(test_splits[m]);
                  float slice_depth = float(depth)/float(test_splits[n]);
                  float surface_area = 2.0*(slice_height*slice_width + slice_depth*slice_width + slice_depth*slice_height);
                  if (surface_area <= smallest_surface_area) {
                    smallest_surface_area = surface_area;
                    splits = test_splits;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return splits;
}

void hierarchical_tile_mapping(Options &options) {
  for (std::size_t i = 0; i < options.num_tiles_available; ++i) {
    options.mapping.push_back(0);
  }
  options.tile_splits[0] = options.partitions[0]/options.ipu_splits[0];
  options.tile_splits[1] = options.partitions[1]/options.ipu_splits[1];
  options.tile_splits[2] = options.partitions[2]/options.ipu_splits[2];
  std::size_t tiles_per_xy_slice = options.tile_splits[0]*options.tile_splits[1]; 
  for (std::size_t ipu_id = 0; ipu_id < options.num_ipus; ++ipu_id) {
    std::size_t ipu_y = ipu_id % options.ipu_splits[1];
    std::size_t ipu_z = ipu_id / options.ipu_splits[1];
    std::size_t tile_offset_y = ipu_y*options.tile_splits[1];
    std::size_t tile_offset_z = ipu_z*options.tile_splits[2];

    for (std::size_t local_tile_id = 0; local_tile_id < options.tiles_per_ipu; ++local_tile_id) {
      std::size_t global_tile_id = ipu_id*options.tiles_per_ipu + local_tile_id;
      std::size_t global_tile_x = (local_tile_id % tiles_per_xy_slice) % options.tile_splits[0];
      std::size_t global_tile_y = ((local_tile_id % tiles_per_xy_slice) / options.tile_splits[0]) + tile_offset_y;
      std::size_t global_tile_z = (local_tile_id / tiles_per_xy_slice) + tile_offset_z;
      options.mapping[index(global_tile_x, global_tile_y, global_tile_z, options.partitions[0], options.partitions[1])] = global_tile_id;
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
  for (std::size_t z = 1; z <= d; ++z) {
    for (std::size_t y = 1; y <= w; ++y) {
      for (std::size_t x = 1; x <= h; ++x) {
        cpu_e[index(x,y,z,hp,wp)] = initial_e[index(x-1,y-1,z-1,h,w)];
        cpu_r[index(x-1,y-1,z-1,h,w)] = initial_r[index(x-1,y-1,z-1,h,w)];
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
        cpu_e[index(x,y,0,hp,wp)] = cpu_e[index(x,y,2,hp,wp)]; // front surface
        cpu_e[index(x,y,d+1,hp,wp)] = cpu_e[index(x,y,d-1,hp,wp)]; // back surface
      }
    }
    for (std::size_t x = 1; x <= h; ++x) {
      for (std::size_t z = 1; z <= d; ++z) {
        cpu_e[index(x,0,z,hp,wp)] = cpu_e[index(x,2,z,hp,wp)]; // left surface
        cpu_e[index(x,w+1,z,hp,wp)] = cpu_e[index(x,w-1,z,hp,wp)]; // right surface
      }
    }
    for (std::size_t y = 1; y <= w; ++y) {
      for (std::size_t z = 1; z <= d; ++z) {
        cpu_e[index(0,y,z,hp,wp)] = cpu_e[index(2,y,z,hp,wp)]; // top surface
        cpu_e[index(h+1,y,z,hp,wp)] = cpu_e[index(h-1,y,z,hp,wp)]; // bottom surface
      }
    }

    // PDE computation by sliding stencil over inner volume 
    // (inner volume of padded mesh corresponds to full volume of unpadded mesh)
    for (std::size_t z = 1; z <= d; ++z) {
      for (std::size_t y = 1; y <= w; ++y) {
        for (std::size_t x = 1; x <= h; ++x) {
          e_center = cpu_e[index(x,y,z,hp,wp)]; // reusable variable
          r_center = cpu_r[index(x-1,y-1,z-1,h,w)];

          // New e_out_center
          cpu_e_temp[index(x-1,y-1,z-1,h,w)] = options.lambda*(
              cpu_e[index(x-1,y,z,hp,wp)] + cpu_e[index(x+1,y,z,hp,wp)] +
              cpu_e[index(x,y-1,z,hp,wp)] + cpu_e[index(x,y+1,z,hp,wp)] +
              cpu_e[index(x,y,z-1,hp,wp)] + cpu_e[index(x,y,z+1,hp,wp)]
            ) + options.gamma*e_center
            - options.dtk*e_center*(e_center - options.a)*(e_center - 1) 
            - options.dt*e_center*r_center;

          // New r_center
          cpu_r[index(x-1,y-1,z-1,h,w)] -= options.dt*(
            (options.epsilon + options.my1*r_center/(options.my2 + e_center))*
            (r_center + options.k*e_center*(e_center - options.b_plus_1))
          );
        }
      }
    }

    // re-update cpu_e from cpu_e_temp
    for (std::size_t z = 1; z <= d; ++z) {
      for (std::size_t y = 1; y <= w; ++y) {
        for (std::size_t x = 1; x <= h; ++x) {
          cpu_e[index(x,y,z,hp,wp)] = cpu_e_temp[index(x-1,y-1,z-1,h,w)];
        }
      }
    }
  }

  // Evaluate MSE
  double e_error, r_error, e_MSE=0, r_MSE=0;
  double scale = 1.0 / (double) volume;
  for (std::size_t z = 1; z < d + 1; ++z) {
    for (std::size_t y = 1; y < w + 1; ++y) {
      for (std::size_t x = 1; x < h + 1; ++x) {
        e_error = ipu_e[index(x-1,y-1,z-1,h,w)] - cpu_e[index(x,y,z,hp,wp)];
        r_error = ipu_r[index(x-1,y-1,z-1,h,w)] - cpu_r[index(x-1,y-1,z-1,h,w)];
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

void print_pde_problem(Options &options) {
  double padded_mesh = (double) (options.height + 2) * (double) (options.width + 2) * (double) (options.depth + 2);
  double minimum_MB = 4*3*padded_mesh*1e-6; // 3 tensors, 4 bytes per element, converted to MB

  std::cout
    << "3D Aliev-Panfilov Model\n"
    << "-----------------------\n"
    << "Full 3D mesh: " << options.height << "*" << options.width << "*" << options.depth << " elements\n"
    << "Number of iterations: " << options.num_iterations << "\n"
    << "Minimum memory usage (two copies of e, one copy of r): " << minimum_MB << " MB\n";
}

void print_data_exchange_volumes(Options &options) {
  float one_ipus_mesh_height = options.height / options.ipu_splits[0];
  float one_ipus_mesh_width = options.width / options.ipu_splits[1];
  float one_ipus_mesh_depth = options.depth / options.ipu_splits[2];
  float total_communication_volume = 2*(
    (options.partitions[0] - 1)*options.width*options.depth +
    (options.partitions[1] - 1)*options.height*options.depth +
    (options.partitions[2] - 1)*options.height*options.width
  );
  float ipu_height = options.height/options.ipu_splits[0];
  float ipu_width = options.width/options.ipu_splits[1];
  float ipu_depth = options.depth/options.ipu_splits[2];
  float inter_ipu_communication_volume = 2*(
    (options.ipu_splits[0] - 1)*ipu_width*ipu_depth +
    (options.ipu_splits[1] - 1)*ipu_height*ipu_depth +
    (options.ipu_splits[2] - 1)*ipu_height*ipu_width
  );
  float intra_ipu_communication_volume = total_communication_volume - inter_ipu_communication_volume;

  std::cout
    << "Number of IPUs: " << options.num_ipus << "\n"
    << "Number of tiles: " << options.num_tiles_available << "\n"
    << "Mesh partitioning: " << options.partitions[0] << "*" << options.partitions[1] << "*" << options.partitions[2] << " partitions\n"
    << "IPU partitioning: " << options.ipu_splits[0] << "*" << options.ipu_splits[1] << "*" << options.ipu_splits[2] << " partitions\n"
    << "Tile partitioning: " << options.tile_splits[0] << "*" << options.tile_splits[1] << "*" << options.tile_splits[2] << " partitions\n"
    << "Total communication volume: " << std::setprecision(5) << total_communication_volume*4*1e-6 << " MB\n"
    << "Inter-IPU communication volume: " << std::setprecision(5) << inter_ipu_communication_volume*4*1e-6 << " MB\n"
    << "Intra-IPU communication volume: " << std::setprecision(5) << intra_ipu_communication_volume*4*1e-6 << " MB\n";
}

void print_results(Options &options) {
  // Calculate metrics
  double mesh_volume = (double) options.height * (double) options.width * (double) options.depth;
  double flops_per_element = 28.0;
  double flops = mesh_volume * (double) options.num_iterations * flops_per_element / options.wall_time;
  double tflops = flops*1e-12;

  std::cout 
    << "Smallest tile partition: " << options.smallest_slice[0] << "*" << options.smallest_slice[1] << "*" << options.smallest_slice[2]  << " elements\n"
    << "Smallest tile communication volume: " << 4*surface_area(options.smallest_slice) << " Bytes\n"
    << "Largest tile partition: " << options.largest_slice[0] << "*" << options.largest_slice[1] << "*" << options.largest_slice[2]  << " elements\n"
    << "Largest tile communication volume: " << 4*surface_area(options.largest_slice) << " Bytes\n"
    << "Wall Time: " << std::setprecision(5) << options.wall_time << " s\n"
    << "Computational throughput: " << std::setprecision(5) << tflops << " TFLOPS\n\n";
}
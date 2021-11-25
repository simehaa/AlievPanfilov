#include <chrono>
#include <math.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

#include "utils.hpp"

poplar::ComputeSet createComputeSet(
  poplar::Graph &graph,
  poplar::Tensor &in,
  poplar::Tensor &out,
  utils::Options &options,
  const std::string& compute_set_name) {

  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.splits[0]; // No. partitions along height per IPU mesh
  unsigned nw = options.splits[1]; // No. partitions along width per IPU mesh
  unsigned nd = options.splits[2]; // No. partitions along depth per IPU mesh
  unsigned nwh = 2; // No. partitions along height per tile
  unsigned nww = 3; // No. partitions along width per tile

  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    // Ensure overlapping grids among the IPUs
    std::size_t offset_back = 2;
    auto ipu_e_in_slice = in.slice(
      {
        0, 
        0, 
        block_low(ipu, options.num_ipus, options.depth-2)
      },
      {
        options.height, 
        options.width, 
        block_high(ipu, options.num_ipus, options.depth-2) + offset_back
      }
    );
    auto ipu_e_out_slice = out.slice(
      {
        0, 
        0, 
        block_low(ipu, options.num_ipus, options.depth-2)
      },
      {
        options.height, 
        options.width, 
        block_high(ipu, options.num_ipus, options.depth-2) + offset_back
      }
    );
    auto ipu_r_slice = out.slice(
      {
        0, 
        0, 
        block_low(ipu, options.num_ipus, options.depth-2)
      },
      {
        options.height, 
        options.width, 
        block_high(ipu, options.num_ipus, options.depth-2) + offset_back
      }
    );
    std::size_t inter_depth = ipu_in_slice.shape()[2];

    // Work division per IPU (amongst tiles)
    for (std::size_t x = 0; x < nh; ++x) {
      for (std::size_t y = 0; y < nw; ++y) {
        for (std::size_t z = 0; z < nd; ++z) {

          // Find indices and side lengths for this tile's mesh
          unsigned tile_id = index(x, y, z, nw, nd) + ipu*options.tiles_per_ipu;
          unsigned tile_x = block_low(x, nh, options.height-2) + 1;
          unsigned tile_y = block_low(y, nw, options.width-2) + 1;
          unsigned tile_height = block_size(x, nh, options.height-2);
          unsigned tile_width = block_size(y, nw, options.width-2);
          unsigned z_low = block_low(z, nd, inter_depth-2) + 1;
          unsigned z_high = block_high(z, nd, inter_depth-2) + 1;

          // Record some metrics
          std::vector<std::size_t> shape = {tile_height, tile_width, z_high - z_low};
          if (volume(shape) < volume(options.smallest_slice))
            options.smallest_slice = shape;
          if (volume(shape) > volume(options.largest_slice)) 
            options.largest_slice = shape;

          // Work division per tile (amongst threads)
          for (std::size_t worker_xi = 0; worker_xi < nwh; ++worker_xi) {
            for (std::size_t worker_yi = 0; worker_yi < nww; ++worker_yi) {
              
              // Dividing tile work among workers
              unsigned x_low = tile_x + block_low(worker_xi, nwh, tile_height);
              unsigned x_high = tile_x + block_high(worker_xi, nwh, tile_height);
              unsigned y_low = tile_y + block_low(worker_yi, nww, tile_width);
              unsigned y_high = tile_y + block_high(worker_yi, nww, tile_width);

              // NOTE: include overlap for input slice
              auto in_slice = ipu_in_slice.slice(
                {
                  x_low-1, 
                  y_low-1, 
                  z_low-1
                },
                {
                  x_high+1, 
                  y_high+1, 
                  z_high+1
                }
              );

              // No overlap on the output slice
              auto out_slice = ipu_out_slice.slice(
                {
                  x_low, 
                  y_low, 
                  z_low
                },
                {
                  x_high, 
                  y_high, 
                  z_high
                }
              );

              // Assign vertex to graph 
              // (six vertices per tile, which will be solved by six different threads)
              auto v = graph.addVertex(compute_set, "AlievPanfilovVertex");
              graph.connect(v["e_in"], in_slice.flatten(0,2));
              graph.connect(v["e_out"], out_slice.flatten(0,2));
              graph.connect(v["r"], in_slice.flatten(0,2));
              graph.setInitialValue(v["worker_height"], x_high - x_low);
              graph.setInitialValue(v["worker_width"], y_high - y_low);
              graph.setInitialValue(v["worker_depth"], z_high - z_low);
              graph.setInitialValue(v["delta"], options.delta);
              graph.setInitialValue(v["epsilon"], options.epsilon);
              graph.setInitialValue(v["my1"], options.my1);
              graph.setInitialValue(v["my2"], options.my2);
              graph.setInitialValue(v["dx"], options.dx);
              graph.setInitialValue(v["dt"], options.dt);
              graph.setInitialValue(v["k"], options.k);
              graph.setInitialValue(v["a"], options.a);
              graph.setInitialValue(v["b"], options.b);
              graph.setTileMapping(v, tile_id);
            }
          }
        }
      }
    }
  }

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph, 
  utils::Options &options) { 

  // Allocate tensors (pad e_a and e_b in order to handle boundary condition)
  auto e_a = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_b");
  auto r = graph.addVariable(poplar::FLOAT, {options.height, options.width, options.depth}, "r");
  
  // Top-level partitioning amongst IPUs
  std::vector<std::size_t> partitions(3) = {1, 1, options.num_ipus};
  for (std::size_t ipu_x = 0; ipu_x < partitions[0]; ++ipu_x) {
    for (std::size_t ipu_y = 0; ipu_y < partitions[1]; ++ipu_y) {
      for (std::size_t ipu_z = 0; ipu_z < partitions[2]; ++ipu_z) {
        auto ipu_slice = r.slice(
          {
            block_low(ipu_x, partitions[0], options.height),
            block_low(ipu_y, partitions[1], options.width),
            block_low(ipu_z, partitions[2], options.depth)
          },
          {
            block_high(ipu_x, partitions[0], options.height)
            block_high(ipu_y, partitions[1], options.width)
            block_high(ipu_z, partitions[2], options.depth)
          }
        );

        // Fine-level partitioning amongst tiles
        for (std::size_t tile_x = 0; tile_x < options.splits[0]; ++tile_x) {
          for (std::size_t tile_y = 0; tile_y < options.splits[1]; ++tile_y) {
            for (std::size_t tile_z = 0; tile_z < options.splits[2]; ++tile_z) {

              unsigned tile_id = index(tile_x, tile_y, tile_z, options.splits[1], options.splits[2]) + ipu*options.tiles_per_ipu;

              // Evaluate offsets in all dimensions (avoid overlap at edges)
              std::size_t offset_top = (tile_x == 0) ? 0 : 1;
              std::size_t offset_left = (tile_y == 0) ? 0 : 1;
              std::size_t inter_offset_front = (tile_z == 0) ? 0 : 1;
              std::size_t offset_bottom = (tile_x == options.splits[0] - 1) ? 2 : 1;
              std::size_t offset_right = (tile_y == options.splits[1] - 1) ? 2 : 1;
              std::size_t inter_offset_back = (tile_z == options.splits[2] - 1) ? 2 : 1;

              auto tile_slice = ipu_slice.slice(
                {
                  block_low(tile_x, options.splits[0], options.height-2) + offset_top, 
                  block_low(tile_y, options.splits[1], options.width-2) + offset_left,
                  block_low(tile_z, options.splits[2], inter_depth-2) + inter_offset_front
                },
                {
                  block_high(tile_x, options.splits[0], options.height-2) + offset_bottom, 
                  block_high(tile_y, options.splits[1], options.width-2) + offset_right,
                  block_high(tile_z, options.splits[2], inter_depth-2) + inter_offset_back
                }
              );
              
              graph.setTileMapping(tile_slice, tile_id);
            }
          }
        }
      }
    }
  }

  // Apply the tile mapping of "a" to be the same for "b"
  const auto& tile_mapping = graph.getTileMapping(e_a);
  graph.setTileMapping(e_b, tile_mapping);
  graph.setTileMapping(r, tile_mapping);

  // Define data streams
  std::size_t volume = options.height*options.width*options.depth;
  auto host_to_device_e = graph.addHostToDeviceFIFO("host_to_device_stream_e", poplar::FLOAT, volume);
  auto host_to_device_r = graph.addHostToDeviceFIFO("host_to_device_stream_r", poplar::FLOAT, volume);
  auto device_to_host_e = graph.addDeviceToHostFIFO("device_to_host_stream_e", poplar::FLOAT, volume);
  auto device_to_host_r = graph.addDeviceToHostFIFO("device_to_host_stream_r", poplar::FLOAT, volume);

  std::vector<poplar::program::Program> programs;

  // Program 0: move content of initial_values into both device variables a and b
  programs.push_back(
    poplar::program::Sequence{
      poplar::program::Copy(host_to_device, e_a),
      poplar::program::Copy(e_a, e_b),
    }
  );

  // Create compute sets
  auto compute_set_b_to_a = createComputeSet(graph, b, a, options, "compute_set_b_to_a");
  auto compute_set_a_to_b = createComputeSet(graph, a, b, options, "compute_set_a_to_b");
  poplar::program::Sequence execute_this_compute_set;

  if (options.num_iterations % 2 == 1) { // if num_iterations is odd: add one extra iteration
    execute_this_compute_set.add(poplar::program::Execute(compute_set_a_to_b));
  }

  // add iterations 
  execute_this_compute_set.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence{
        poplar::program::Execute(compute_set_b_to_a),
        poplar::program::Execute(compute_set_a_to_b)
      }
    )
  );

  programs.push_back(execute_this_compute_set);
  programs.push_back(poplar::program::Copy(b, device_to_host));

  return programs;
}

int main (int argc, char** argv) {
  try {
    // Get options from command line arguments / defaults. (see utils.hpp)
    auto options = utils::parseOptions(argc, argv);
    testUpperBoundDt(options);

    // Set up of 3D mesh properties
    std::size_t base_length = 320;
    options.side = side_length(options.num_ipus, base_length);

    // Attach to IPU device
    auto device = getDevice(options);
    auto &target = device.getTarget();
    options.num_tiles_available = target.getNumTiles();
    options.tiles_per_ipu = options.num_tiles_available/options.num_ipus;
    workDivision(options);
    
    std::size_t h = options.height;
    std::size_t w = options.width;
    std::size_t d = options.depth;
    std::size_t volume = h*w*d;
    std::vector<float> host_e(volume);
    std::vector<float> host_r(volume);
    std::vector<float> ipu_e(volume); 
    std::vector<float> ipu_r(volume); 

    // initial values
    // e: left half=0, right half=1
    // r: bottom half=0, top half=1
    for (std::size_t x = 0; x < h; ++x) {
      for (std::size_t y = 0; y < w; ++y) {
        for (std::size_t z = 0; z < d; ++z) {
          host_e[index(x,y,z,w,d)] = (y < w/2) ? 0.0 : 1.0;
          host_r[index(x,y,z,w,d)] = (x < h/2) ? 1.0 : 0.0;
        }
      }
    }
    
    // Setup of programs, graph and engine
    poplar::Graph graph{target};
    graph.addCodelets("codelets.gp");
    auto programs = createIpuPrograms(graph, options); // Custom function to construct vector of programs
    auto exe = poplar::compileGraph(graph, programs);
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream_e", &initial_e[0], &initial_e[volume]);
    engine.connectStream("host_to_device_stream_r", &initial_r[0], &initial_r[volume]);
    engine.connectStream("device_to_host_stream_e", &ipu_results_e[0], &ipu_results_e[volume]);
    engine.connectStream("device_to_host_stream_r", &ipu_results_r[0], &ipu_results_r[volume]);
    engine.load(device);

    std::size_t num_program_steps = programs.size();
    engine.run(0); // stream data to device
    auto start = std::chrono::steady_clock::now();
    engine.run(1); // Compute set execution
    auto stop = std::chrono::steady_clock::now();
    engine.run(2); // Stream of results

    // Report
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    double wall_time = 1e-9*diff.count();
    printResults(options, wall_time);

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return 0;
}


#include <chrono>
#include <iostream>
#include <vector>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include "poplarfunctions.hpp"
#include "options.hpp"
#include "utils.hpp"

int main (int argc, char** argv) {
  try {
    // Get options from command line arguments
    auto options = parse_options(argc, argv);
    test_upper_dt(options);

    // Initialize meshes
    std::size_t h = options.height;
    std::size_t w = options.width;
    std::size_t d = options.depth;
    std::size_t volume = h*w*d;
    std::vector<float> initial_e(volume);
    std::vector<float> initial_r(volume);
    std::vector<float> ipu_e(volume); 
    std::vector<float> ipu_r(volume); 

    // initial values
    // e: left half=0, right half=1
    // r: bottom half=0, top half=1
    for (std::size_t x = 0; x < h; ++x) {
      for (std::size_t y = 0; y < w; ++y) {
        for (std::size_t z = 0; z < d; ++z) {
          initial_e[index(x,y,z,w,d)] = (y < w/2) ? 0.0 : 1.0;
          initial_r[index(x,y,z,w,d)] = (x < h/2) ? 1.0 : 0.0;
        }
      }
    }
    
    // Attach to IPU device
    auto device = get_device(options.num_ipus);
    auto &target = device.getTarget();
    options.num_tiles_available = target.getNumTiles();
    options.tiles_per_ipu = target.getTilesPerIPU();
    options.total_memory_avail_MB = (double) target.getBytesPerTile() * options.num_tiles_available * 1e-6;
    work_division(options);
    
    // Setup of programs, graph and engine
    poplar::Graph graph{target};
    graph.addCodelets("obj/codelets.gp");
    auto programs = create_ipu_programs(graph, options); // Custom function to construct vector of programs
    auto exe = poplar::compileGraph(graph, programs);
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream_e", &initial_e[0], &initial_e[volume]);
    engine.connectStream("host_to_device_stream_r", &initial_r[0], &initial_r[volume]);
    engine.connectStream("device_to_host_stream_e", &ipu_e[0], &ipu_e[volume]);
    engine.connectStream("device_to_host_stream_r", &ipu_r[0], &ipu_r[volume]);
    engine.load(device);

    std::size_t num_program_steps = programs.size();
    engine.run(0); // stream data to device
    auto start = std::chrono::steady_clock::now();
    engine.run(1); // Compute set execution
    auto stop = std::chrono::steady_clock::now();
    engine.run(2); // Stream of results

    if (options.cpu)
      test_against_cpu(initial_e, initial_r, ipu_e, ipu_r, options);

    // Report
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    options.wall_time = 1e-9*diff.count();
    print_results_and_options(options);

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return 0;
}
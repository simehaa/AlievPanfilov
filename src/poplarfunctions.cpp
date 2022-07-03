#include "poplarfunctions.hpp"

poplar::Device get_device(std::size_t n) {
  /* Attach to a Poplar device consisting of n IPUs, where 
   * n is between 1 and 64 IPUs (must be a power-of-two).
   */
  if (n!=1 && n!=2 && n!=4 && n!=8 && n!=16 && n!=32 && n!=64)
    throw std::runtime_error("Invalid number of IPUs.");

  // Create device manager
  auto manager = poplar::DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, n);

  // Use the first available device
  for (auto &device : devices)
    if (device.attach()) 
      return std::move(device);

  throw std::runtime_error("No hardware device available.");
}

poplar::ComputeSet create_compute_set(
  poplar::Graph &graph,
  poplar::Tensor &e_in,
  poplar::Tensor &e_out,
  poplar::Tensor &r,
  Options &options,
  const std::string& compute_set_name) {
  /* Add compute set to graph */
  auto compute_set = graph.addComputeSet(compute_set_name);

  // Hierarchical loop: among IPUs -> among tiles -> among worker contexts
  for (std::size_t z = 0; z < options.partitions[2]; ++z) {
    for (std::size_t y = 0; y < options.partitions[1]; ++y) {
      for (std::size_t x = 0; x < options.partitions[0]; ++x) {
        std::size_t offset_top = 1;
        std::size_t offset_left = 1;
        std::size_t offset_front = 1;
        std::size_t offset_bottom = 1;
        std::size_t offset_right = 1;
        std::size_t offset_back = 1;
        std::size_t x_low = block_low(x, options.partitions[0], options.height) + offset_top;
        std::size_t y_low = block_low(y, options.partitions[1], options.width) + offset_left;
        std::size_t z_low = block_low(z, options.partitions[2], options.depth) + offset_front;
        std::size_t x_high = block_high(x, options.partitions[0], options.height) + offset_bottom;
        std::size_t y_high = block_high(y, options.partitions[1], options.width) + offset_right;
        std::size_t z_high = block_high(z, options.partitions[2], options.depth) + offset_back;
        std::size_t tile_height = x_high - x_low;
        std::size_t tile_width = y_high - y_low;
        std::size_t tile_depth = z_high - z_low;
        std::size_t tile_id = options.mapping[index(x, y, z, options.partitions[0], options.partitions[1])];

        // Record some metrics
        std::vector<std::size_t> shape = {tile_height, tile_width, tile_depth};
        if (volume(shape) < volume(options.smallest_slice))
          options.smallest_slice = shape;
        if (volume(shape) > volume(options.largest_slice)) 
          options.largest_slice = shape;

        // Work division per tile (amongst threads)
        for (std::size_t worker_z = 0; worker_z < options.worker_splits[2]; ++worker_z) {
          for (std::size_t worker_y = 0; worker_y < options.worker_splits[1]; ++worker_y) {
            for (std::size_t worker_x = 0; worker_x < options.worker_splits[0]; ++worker_x) {
              
              // Dividing tile work among workers
              std::size_t worker_x_low = x_low + block_low(worker_x, options.worker_splits[0], tile_height);
              std::size_t worker_y_low = y_low + block_low(worker_y, options.worker_splits[1], tile_width);
              std::size_t worker_z_low = z_low + block_low(worker_z, options.worker_splits[2], tile_depth);
              std::size_t worker_x_high = x_low + block_high(worker_x, options.worker_splits[0], tile_height);
              std::size_t worker_y_high = y_low + block_high(worker_y, options.worker_splits[1], tile_width);
              std::size_t worker_z_high = z_low + block_high(worker_z, options.worker_splits[2], tile_depth);
              std::size_t worker_height = worker_x_high - worker_x_low;
              std::size_t worker_width = worker_y_high - worker_y_low;
              std::size_t worker_depth = worker_z_high - worker_z_low;

              // Vertex' r slice (offset of +1 because of the padding)
              auto r_slice = r.slice(
                {worker_x_low, worker_y_low, worker_z_low},
                {worker_x_high, worker_y_high, worker_z_high}
              );

              // Vertex' e_out slice (offset of +1 because of the padding)
              auto e_out_slice = e_out.slice(
                {worker_x_low, worker_y_low, worker_z_low},
                {worker_x_high, worker_y_high, worker_z_high}
              );

              // Vertex' e_in slice (notice padding wrt to both e_out and r)
              auto e_in_slice = e_in.slice(
                {worker_x_low-1, worker_y_low-1, worker_z_low-1},
                {worker_x_high+1, worker_y_high+1, worker_z_high+1}
              );

              // Assign vertex to graph 
              // (six vertices per tile, which will be solved by six different threads)
              auto v = graph.addVertex(compute_set, "AlievPanfilovVertex");
              graph.connect(v["e_in"], e_in_slice.flatten(0,2));
              graph.connect(v["e_out"], e_out_slice.flatten(0,2));
              graph.connect(v["r"], r_slice.flatten(0,2));
              graph.setInitialValue(v["worker_height"], worker_height);
              graph.setInitialValue(v["worker_width"], worker_width);
              graph.setInitialValue(v["worker_depth"], worker_depth);
              graph.setInitialValue(v["epsilon"], options.epsilon);
              graph.setInitialValue(v["my1"], options.my1);
              graph.setInitialValue(v["my2"], options.my2);
              graph.setInitialValue(v["dt"], options.dt);
              graph.setInitialValue(v["k"], options.k);
              graph.setInitialValue(v["a"], options.a);
              graph.setInitialValue(v["lambda"], options.lambda);
              graph.setInitialValue(v["gamma"], options.gamma);
              graph.setInitialValue(v["dtk"], options.dtk);
              graph.setInitialValue(v["b_plus_1"], options.b_plus_1);
              graph.setTileMapping(v, tile_id);
              if (tile_id > options.num_tiles_available) {
                std::cout << "Error,. tile_id = " << tile_id << "\n";
              }
            }
          }
        }
      }
    }
  }

  return compute_set;
}

poplar::program::Program copyForBoundaryCondition(
  poplar::Tensor &e,
  Options &options,
  const std::string& compute_set_name) {
  // "rename" variables to shorter names
  std::size_t h = options.height;
  std::size_t w = options.width;
  std::size_t d = options.depth;

  auto copy_all_surfaces = poplar::program::Sequence({
    poplar::program::Copy(
      e.slice({2,1,1},{2+1,w+1,d+1}), // from inner top
      e.slice({0,1,1},{0+1,w+1,d+1}) // to outer top
    ),
    poplar::program::Copy(
      e.slice({h-1,1,1},{h-1+1,w+1,d+1}), // from inner bottom
      e.slice({h+1,1,1},{h+1+1,w+1,d+1}) // to outer bottom
    ),
    poplar::program::Copy(
      e.slice({1,1,2},{h+1,w+1,2+1}), // from inner front
      e.slice({1,1,0},{h+1,w+1,0+1}) // to outer front
    ),
    poplar::program::Copy(
      e.slice({1,1,d-1},{h+1,w+1,d-1+1}), // from inner back
      e.slice({1,1,d+1},{h+1,w+1,d+1+1}) // to outer back
    ),
    poplar::program::Copy(
      e.slice({1,2,1},{h+1,2+1,d+1}), // from inner left
      e.slice({1,0,1},{h+1,0+1,d+1}) // to outer left
    ),
    poplar::program::Copy(
      e.slice({1,w-1,1},{h+1,w-1+1,d+1}), // from inner right
      e.slice({1,w+1,1},{h+1,w+1+1,d+1}) // to outer right
    )
  });

  return copy_all_surfaces;
}

std::vector<poplar::program::Program> create_ipu_programs(
  poplar::Graph &graph, 
  Options &options) {

  // Allocate tensors (pad e_a and e_b in order to handle boundary condition)
  auto e_a = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_b");
  auto r = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "r"); // padding for simplicity, not necessary

  int counter = 0;
  for (std::size_t z = 0; z < options.partitions[2]; ++z) { 
    for (std::size_t y = 0; y < options.partitions[1]; ++y) {
      for (std::size_t x = 0; x < options.partitions[0]; ++x) {
        std::size_t offset_top = (x == 0) ? 0 : 1;
        std::size_t offset_left = (y == 0) ? 0 : 1;
        std::size_t offset_front = (z == 0) ? 0 : 1;
        std::size_t offset_bottom = (x == options.partitions[0] - 1) ? 2 : 1;
        std::size_t offset_right = (y == options.partitions[1] - 1) ? 2 : 1;
        std::size_t offset_back = (z == options.partitions[2] - 1) ? 2 : 1;
        std::size_t x_low = block_low(x, options.partitions[0], options.height) + offset_top;
        std::size_t y_low = block_low(y, options.partitions[1], options.width) + offset_left;
        std::size_t z_low = block_low(z, options.partitions[2], options.depth) + offset_front;
        std::size_t x_high = block_high(x, options.partitions[0], options.height) + offset_bottom;
        std::size_t y_high = block_high(y, options.partitions[1], options.width) + offset_right;
        std::size_t z_high = block_high(z, options.partitions[2], options.depth) + offset_back;
        std::size_t tile_id = options.mapping[index(x, y, z, options.partitions[0], options.partitions[1])];
        auto tile_slice = e_a.slice({x_low, y_low, z_low}, {x_high, y_high, z_high});
        if (tile_id > options.num_tiles_available) {
          std::cout << "Error,. tile_id = " << tile_id << "\n";
        }
        graph.setTileMapping(tile_slice, tile_id);
      }
    }
  }

  // Apply the tile mapping of "e_a" to be the same for "e_b"
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

  // Set up data streams to the unpadded areas of the three tensors
  auto e_a_inner = e_a.slice({1, 1, 1}, {options.height+1, options.width+1, options.depth+1});
  auto e_b_inner = e_b.slice({1, 1, 1}, {options.height+1, options.width+1, options.depth+1});
  auto r_inner = r.slice({1, 1, 1}, {options.height+1, options.width+1, options.depth+1});

  // Program 0: move initial values onto all device tensors
  programs.push_back(
    poplar::program::Sequence({
      poplar::program::Copy(host_to_device_r, r_inner),
      poplar::program::Copy(host_to_device_e, e_a_inner),
      poplar::program::Copy(e_a_inner, e_b_inner) // on-device copy (much faster)
    })
  );

  // Define (1) compute sets, (2) prepare boundary copies, and (3) full iteration steps
  auto compute_set_b_to_a = create_compute_set(graph, e_b, e_a, r, options, "compute_set_b_to_a");
  auto compute_set_a_to_b = create_compute_set(graph, e_a, e_b, r, options, "compute_set_a_to_b");
  auto prepare_a_boundary = copyForBoundaryCondition(e_a, options, "prepare_a_boundary");
  auto prepare_b_boundary = copyForBoundaryCondition(e_b, options, "prepare_b_boundary");
  auto iteration_a_to_b = poplar::program::Sequence({
    prepare_a_boundary, poplar::program::Execute(compute_set_a_to_b)
  });
  auto iteration_b_to_a = poplar::program::Sequence({
    prepare_b_boundary, poplar::program::Execute(compute_set_b_to_a)
  });

  poplar::program::Sequence all_iterations;

  // If num_iterations is odd: add one extra iteration
  if (options.num_iterations % 2 == 1) { 
    all_iterations.add(iteration_a_to_b);
  }

  // Add remaining iterations 
  all_iterations.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence({
        iteration_b_to_a, iteration_a_to_b
      })
    )
  );
  programs.push_back(all_iterations);

  // Copy results back to host (e_b holds last e)
  programs.push_back(
    poplar::program::Sequence({
      poplar::program::Copy(r_inner, device_to_host_r),
      poplar::program::Copy(e_b_inner, device_to_host_e),
    })
  );

  return programs;
}
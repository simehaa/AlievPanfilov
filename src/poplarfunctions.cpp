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

poplar::ComputeSet createComputeSet(
  poplar::Graph &graph,
  poplar::Tensor &e_in,
  poplar::Tensor &e_out,
  poplar::Tensor &r,
  Options &options,
  const std::string& compute_set_name) {

  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.splits[0]; // No. partitions along height per IPU mesh
  unsigned nw = options.splits[1]; // No. partitions along width per IPU mesh
  unsigned nd = options.splits[2]; // No. partitions along depth per IPU mesh
  unsigned nwh = 2; // No. partitions along height per tile
  unsigned nww = 3; // No. partitions along width per tile
  unsigned nwd = 1; // No. partitions along depth per tile

  // Work division per IPU (amongst tiles)
  for (std::size_t x = 0; x < nh; ++x) {
    for (std::size_t y = 0; y < nw; ++y) {
      for (std::size_t z = 0; z < nd; ++z) {

        // Find indices and side lengths for this tile's mesh
        unsigned tile_id = index(x, y, z, nw, nd); // + ipu*options.tiles_per_ipu;
        
        unsigned tile_x = block_low(x, nh, options.height);
        unsigned tile_y = block_low(y, nw, options.width);
        unsigned tile_z = block_low(z, nd, options.depth);
        unsigned tile_height = block_size(x, nh, options.height);
        unsigned tile_width = block_size(y, nw, options.width);
        unsigned tile_depth = block_size(z, nd, options.depth);

        // Record some metrics
        std::vector<std::size_t> shape = {tile_height, tile_width, tile_depth};
        if (volume(shape) < volume(options.smallest_slice))
          options.smallest_slice = shape;
        if (volume(shape) > volume(options.largest_slice)) 
          options.largest_slice = shape;

        // Work division per tile (amongst threads)
        for (std::size_t worker_xi = 0; worker_xi < nwh; ++worker_xi) {
          for (std::size_t worker_yi = 0; worker_yi < nww; ++worker_yi) {
            for (std::size_t worker_zi = 0; worker_zi < nwd; ++worker_zi) {
              
              // Dividing tile work among workers
              unsigned x_low = tile_x + block_low(worker_xi, nwh, tile_height);
              unsigned x_high = tile_x + block_high(worker_xi, nwh, tile_height);
              unsigned y_low = tile_y + block_low(worker_yi, nww, tile_width);
              unsigned y_high = tile_y + block_high(worker_yi, nww, tile_width);
              unsigned z_low = tile_z + block_low(worker_zi, nwd, tile_depth);
              unsigned z_high = tile_z + block_high(worker_zi, nwd, tile_depth);

              // Vertex' r slice
              auto r_slice = r.slice({x_low,y_low,z_low}, {x_high, y_high, z_high});

              // Vertex' e_out slice (offset of +1 because of the padding)
              auto e_out_slice = e_out.slice({x_low+1, y_low+1, z_low+1}, {x_high+1, y_high+1, z_high+1});

              // Vertex' e_in slice (padding of 1 wrt. e_out slice)
              auto e_in_slice = e_in.slice({x_low, y_low, z_low}, {x_high+2, y_high+2, z_high+2});

              // Assign vertex to graph 
              // (six vertices per tile, which will be solved by six different threads)
              auto v = graph.addVertex(compute_set, "AlievPanfilovVertex");
              graph.connect(v["e_in"], e_in_slice.flatten(0,2));
              graph.connect(v["e_out"], e_out_slice.flatten(0,2));
              graph.connect(v["r"], r_slice.flatten(0,2));
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
  Options &options) { 

  // Allocate tensors (pad e_a and e_b in order to handle boundary condition)
  auto e_a = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_b");
  auto r = graph.addVariable(poplar::FLOAT, {options.height, options.width, options.depth}, "r");

  // Fine-level partitioning amongst tiles
  for (std::size_t tile_x = 0; tile_x < options.splits[0]; ++tile_x) {
    for (std::size_t tile_y = 0; tile_y < options.splits[1]; ++tile_y) {
      for (std::size_t tile_z = 0; tile_z < options.splits[2]; ++tile_z) {
        // Running index over all tiles
        unsigned tile_id = index(tile_x, tile_y, tile_z, options.splits[1], options.splits[2]); // + ipu*options.tiles_per_ipu;

        // split up r (easiest, because neither padding nor overlap)
        auto r_slice = r.slice(
          {
            block_low(tile_x, options.splits[0], options.height), 
            block_low(tile_y, options.splits[1], options.width),
            block_low(tile_z, options.splits[2], options.depth)
          },
          {
            block_high(tile_x, options.splits[0], options.height), 
            block_high(tile_y, options.splits[1], options.width),
            block_high(tile_z, options.splits[2], options.depth)
          }
        );

        // Evaluate offsets in all dimensions: the paddings should be included for boundary partitions
        std::size_t offset_top = (tile_x == 0) ? 0 : 1;
        std::size_t offset_left = (tile_y == 0) ? 0 : 1;
        std::size_t offset_front = (tile_z == 0) ? 0 : 1;
        std::size_t offset_bottom = (tile_x == options.splits[0] - 1) ? 2 : 1;
        std::size_t offset_right = (tile_y == options.splits[1] - 1) ? 2 : 1;
        std::size_t offset_back = (tile_z == options.splits[2] - 1) ? 2 : 1;

        auto e_a_slice = e_a.slice(
          {
            block_low(tile_x, options.splits[0], options.height) + offset_top, 
            block_low(tile_y, options.splits[1], options.width) + offset_left,
            block_low(tile_z, options.splits[2], options.depth) + offset_front
          },
          {
            block_high(tile_x, options.splits[0], options.height) + offset_bottom, 
            block_high(tile_y, options.splits[1], options.width) + offset_right,
            block_high(tile_z, options.splits[2], options.depth) + offset_back
          }
        );
        
        graph.setTileMapping(r_slice, tile_id);
        graph.setTileMapping(e_a_slice, tile_id);
      }
    }
  }

  // Apply the tile mapping of "e_a" to be the same for "e_b"
  const auto& tile_mapping = graph.getTileMapping(e_a);
  graph.setTileMapping(e_b, tile_mapping);

  // Define data streams
  std::size_t volume = options.height*options.width*options.depth;
  auto host_to_device_e = graph.addHostToDeviceFIFO("host_to_device_stream_e", poplar::FLOAT, volume);
  auto host_to_device_r = graph.addHostToDeviceFIFO("host_to_device_stream_r", poplar::FLOAT, volume);
  auto device_to_host_e = graph.addDeviceToHostFIFO("device_to_host_stream_e", poplar::FLOAT, volume);
  auto device_to_host_r = graph.addDeviceToHostFIFO("device_to_host_stream_r", poplar::FLOAT, volume);

  std::vector<poplar::program::Program> programs;

  // Acutal e_a and e_b are only the inner slice which excludes the padding of 1 (in all directions)
  auto e_a_inner = e_a.slice({1, 1, 1}, {options.height+1, options.width+1, options.depth+1});
  auto e_b_inner = e_b.slice({1, 1, 1}, {options.height+1, options.width+1, options.depth+1});

  // Program 0: move initial values onto all device tensors
  programs.push_back(
    poplar::program::Sequence({
      poplar::program::Copy(host_to_device_r, r),
      poplar::program::Copy(host_to_device_e, e_a_inner),
      poplar::program::Copy(e_a_inner, e_b_inner) // on-device copy (much faster)
    })
  );

  // Create compute sets
  auto compute_set_b_to_a = createComputeSet(graph, e_b, e_a, r, options, "compute_set_b_to_a");
  auto compute_set_a_to_b = createComputeSet(graph, e_a, e_b, r, options, "compute_set_a_to_b");
  poplar::program::Sequence execute_this_compute_set;

  if (options.num_iterations % 2 == 1) { // if num_iterations is odd: add one extra iteration
    execute_this_compute_set.add(poplar::program::Execute(compute_set_a_to_b));
  }

  // add iterations 
  execute_this_compute_set.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence({
        poplar::program::Execute(compute_set_b_to_a),
        poplar::program::Execute(compute_set_a_to_b)
      })
    )
  );
  programs.push_back(execute_this_compute_set);

  // Copy results back to host (e_b holds last e)
  programs.push_back(
    poplar::program::Sequence({
      poplar::program::Copy(r, device_to_host_r),
      poplar::program::Copy(e_b_inner, device_to_host_e),
    })
  );

  return programs;
}
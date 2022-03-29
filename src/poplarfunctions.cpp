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
  unsigned nwh = 1; // No. partitions along height per tile
  unsigned nww = 2; // No. partitions along width per tile
  unsigned nwd = 3; // No. partitions along depth per tile

  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    unsigned offset_front = 0;
    unsigned offset_back = 2;
    unsigned ipu_x_lo = 0;
    unsigned ipu_y_lo = 0;
    unsigned ipu_z_lo = block_low(ipu, options.num_ipus, options.depth) + offset_front;
    unsigned ipu_x_hi = options.height + 2;
    unsigned ipu_y_hi = options.width + 2;
    unsigned ipu_z_hi = block_high(ipu, options.num_ipus, options.depth) + offset_back;
    unsigned ipu_depth = ipu_z_hi - ipu_z_lo;

    // std::cout 
    //   << "{"
    //   << ipu_x_lo << ","
    //   << ipu_y_lo << ","
    //   << ipu_z_lo
    //   << "} {"
    //   << ipu_x_hi << ","
    //   << ipu_y_hi << ","
    //   << ipu_z_hi << "}\n";

    auto e_in_ipu_slice = e_in.slice(
      {ipu_x_lo, ipu_y_lo, ipu_z_lo}, 
      {ipu_x_hi, ipu_y_hi, ipu_z_hi}
    );
    auto e_out_ipu_slice = e_out.slice(
      {ipu_x_lo, ipu_y_lo, ipu_z_lo}, 
      {ipu_x_hi, ipu_y_hi, ipu_z_hi}
    );
    auto r_ipu_slice = r.slice(
      {ipu_x_lo, ipu_y_lo, ipu_z_lo}, 
      {ipu_x_hi, ipu_y_hi, ipu_z_hi}
    );

    // Work division per IPU (amongst tiles)
    for (std::size_t tile_x = 0; tile_x < nh; ++tile_x) {
      for (std::size_t tile_y = 0; tile_y < nw; ++tile_y) {
        for (std::size_t tile_z = 0; tile_z < nd; ++tile_z) {

          // Find indices and side lengths for this tile's mesh
          unsigned tile_id = index(tile_x, tile_y, tile_z, nw, nd) + ipu*options.tiles_per_ipu;
          unsigned tile_x_lo = block_low(tile_x, nh, options.height) + 1;
          unsigned tile_y_lo = block_low(tile_y, nw, options.width) + 1;
          unsigned tile_z_lo = block_low(tile_z, nd, ipu_depth-2) + 1;
          unsigned tile_x_hi = block_high(tile_x, nh, options.height) + 1;
          unsigned tile_y_hi = block_high(tile_y, nw, options.width) + 1;
          unsigned tile_z_hi = block_high(tile_z, nd, ipu_depth-2) + 1;
          unsigned tile_height = block_size(tile_x, nh, options.height);
          unsigned tile_width = block_size(tile_y, nw, options.width);
          unsigned tile_depth = block_size(tile_z, nd, ipu_depth-2);

          // if (ipu == 0 && tile_x == tile_y && tile_y == tile_z) {
          //   std::cout 
          //   << "{"
          //   << tile_x_lo << ","
          //   << tile_y_lo << ","
          //   << tile_z_lo
          //   << "} {"
          //   << tile_x_hi << ","
          //   << tile_y_hi << ","
          //   << tile_z_hi << "}\n";
          // }

          // Record some metrics
          std::vector<std::size_t> shape = {tile_height, tile_width, tile_depth};
          if (volume(shape) < volume(options.smallest_slice))
            options.smallest_slice = shape;
          if (volume(shape) > volume(options.largest_slice)) 
            options.largest_slice = shape;

          // Work division per tile (amongst threads)
          for (std::size_t worker_x = 0; worker_x < nwh; ++worker_x) {
            for (std::size_t worker_y = 0; worker_y < nww; ++worker_y) {
              for (std::size_t worker_z = 0; worker_z < nwd; ++worker_z) {
                
                // Dividing tile work among workers
                unsigned x_lo = tile_x_lo + block_low(worker_x, nwh, tile_height);
                unsigned y_lo = tile_y_lo + block_low(worker_y, nww, tile_width);
                unsigned z_lo = tile_z_lo + block_low(worker_z, nwd, tile_depth);
                unsigned x_hi = tile_x_lo + block_high(worker_x, nwh, tile_height);
                unsigned y_hi = tile_y_lo + block_high(worker_y, nww, tile_width);
                unsigned z_hi = tile_z_lo + block_high(worker_z, nwd, tile_depth);

                // Vertex' r slice (offset of +1 because of the padding)
                auto r_slice = r_ipu_slice.slice({x_lo, y_lo, z_lo}, {x_hi, y_hi, z_hi});

                // Vertex' e_out slice (offset of +1 because of the padding)
                auto e_out_slice = e_out_ipu_slice.slice({x_lo, y_lo, z_lo}, {x_hi, y_hi, z_hi});

                // Vertex' e_in slice (notice padding wrt to both e_out and r)
                auto e_in_slice = e_in_ipu_slice.slice({x_lo-1, y_lo-1, z_lo-1}, {x_hi+1, y_hi+1, z_hi+1});

                // if (ipu == 0 && tile_x == tile_y && tile_y == tile_z) {
                //   std::cout 
                //   << "{"
                //   << x_lo << ","
                //   << y_lo << ","
                //   << z_lo
                //   << "} {"
                //   << x_hi << ","
                //   << y_hi << ","
                //   << z_hi << "}\n";
                // }

                // Assign vertex to graph 
                // (six vertices per tile, which will be solved by six different threads)
                auto v = graph.addVertex(compute_set, "AlievPanfilovVertex");
                graph.connect(v["e_in"], e_in_slice.flatten(0,2));
                graph.connect(v["e_out"], e_out_slice.flatten(0,2));
                graph.connect(v["r"], r_slice.flatten(0,2));
                graph.setInitialValue(v["worker_height"], x_hi - x_lo);
                graph.setInitialValue(v["worker_width"], y_hi - y_lo);
                graph.setInitialValue(v["worker_depth"], z_hi - z_lo);
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
  }

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph, 
  Options &options) {

  // Allocate tensors (pad e_a and e_b in order to handle boundary condition)
  auto e_a = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "e_b");
  auto r = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2, options.depth + 2}, "r"); // padding for simplicity, not necessary

  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    unsigned offset_front = (ipu == 0) ? 0 : 1;
    unsigned offset_back = (ipu == options.num_ipus - 1) ? 2 : 1;
    unsigned ipu_z_low = block_low(ipu, options.num_ipus, options.depth) + offset_front;
    unsigned ipu_z_high = block_high(ipu, options.num_ipus, options.depth) + offset_back;

    // +2 in high indices for e, because of padding
    auto ipu_slice = e_a.slice(
      {0, 0, ipu_z_low}, 
      {options.height+2, options.width+2, ipu_z_high}
    );
    unsigned ipu_depth = ipu_slice.shape()[2];

    // Fine-level partitioning amongst tiles
    for (std::size_t tile_x = 0; tile_x < options.splits[0]; ++tile_x) {
      for (std::size_t tile_y = 0; tile_y < options.splits[1]; ++tile_y) {
        for (std::size_t tile_z = 0; tile_z < options.splits[2]; ++tile_z) {
          // Running index over all tiles
          unsigned tile_id = index(tile_x, tile_y, tile_z, options.splits[1], options.splits[2]) + ipu*options.tiles_per_ipu;
          
          // Evaluate offsets in all dimensions (avoid overlap at edges)
          std::size_t offset_top = (tile_x == 0) ? 0 : 1;
          std::size_t offset_left = (tile_y == 0) ? 0 : 1;
          std::size_t inter_offset_front = (tile_z == 0) ? 0 : 1;
          std::size_t offset_bottom = (tile_x == options.splits[0] - 1) ? 2 : 1;
          std::size_t offset_right = (tile_y == options.splits[1] - 1) ? 2 : 1;
          std::size_t inter_offset_back = (tile_z == options.splits[2] - 1) ? 2 : 1;
          std::size_t x_lo = block_low(tile_x, options.splits[0], options.height) + offset_top;
          std::size_t y_lo = block_low(tile_y, options.splits[1], options.width) + offset_left;
          std::size_t z_lo = block_low(tile_z, options.splits[2], ipu_depth-2) + inter_offset_front;
          std::size_t x_hi = block_high(tile_x, options.splits[0], options.height) + offset_bottom;
          std::size_t y_hi = block_high(tile_y, options.splits[1], options.width) + offset_right;
          std::size_t z_hi = block_high(tile_z, options.splits[2], ipu_depth-2) + inter_offset_back;

          // if (tile_x == tile_y && tile_y == tile_z) {
          //   std::cout 
          //   << "{"
          //   << x_lo << ","
          //   << y_lo << ","
          //   << z_lo
          //   << "} {"
          //   << x_hi << ","
          //   << y_hi << ","
          //   << z_hi << "}\n";
          // }

          auto tile_slice = ipu_slice.slice({x_lo, y_lo, z_lo}, {x_hi, y_hi, z_hi});
          graph.setTileMapping(tile_slice, tile_id);
        }
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
      poplar::program::Copy(r_inner, device_to_host_r),
      poplar::program::Copy(e_b_inner, device_to_host_e),
    })
  );

  return programs;
}
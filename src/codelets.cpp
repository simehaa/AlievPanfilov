#include <poplar/Vertex.hpp>

using namespace poplar;

class AlievPanfilovVertex : public Vertex {
public:
  AlievPanfilovVertex();

  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> e_in;
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> e_out;
  Vector<InOut<Vector<float, VectorLayout::SPAN, 4, false>>> r;
  const std::size_t worker_height;
  const std::size_t worker_width;
  const std::size_t worker_depth;
  const float epsilon;
  const float my1;
  const float my2;
  const float dt;
  const float k;
  const float a;
  const float lambda; // delta*dt/(dx*dx)
  const float gamma; // 1 - 6*lambda
  const float dtk; // dt*k
  const float b_plus_1; // b + 1

  std::size_t idx(std::size_t x, std::size_t y, std::size_t w) {
    /* The index corresponding to [x,y] in for a row-wise flattened 2D variable*/
    return y + x*w;
  } 

  bool compute () {
    const std::size_t w = worker_width;
    const std::size_t pw = worker_width + 2;
    float e_center, r_center;

    for (std::size_t x = 1; x < worker_height + 1; ++x) {
      for (std::size_t y = 1; y < worker_width + 1; ++y) {
        for (std::size_t z = 1; z < worker_depth + 1; ++z) {
          e_center = e_in[idx(x,y,pw)][z];
          r_center = r[idx(x-1,y-1,w)][z-1];

          // New e_out_center
          e_out[idx(x-1,y-1,w)][z-1] = lambda*(
              e_in[idx(x+1,y,pw)][z] + e_in[idx(x-1,y,pw)][z] +
              e_in[idx(x,y+1,pw)][z] + e_in[idx(x,y-1,pw)][z] +
              e_in[idx(x,y,pw)][z+1] + e_in[idx(x,y,pw)][z-1]
            ) + gamma*e_center
            - dtk*e_center*(e_center - a)*(e_center - 1) 
            - dt*e_center*r_center; // 17 FLOPs

          // New r_center
          r[idx(x-1,y-1,w)][z-1] -= dt*(
            (epsilon + my1*r_center/(my2 + e_center))*
            (r_center + k*e_center*(e_center - b_plus_1))
          ); // 11 FLOPs
        }
      }
    }

    return true;
  }
};
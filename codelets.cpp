#include <poplar/Vertex.hpp>

using namespace poplar;

class AlievPanfilovVertex : public Vertex {
public:
  AlievPanfilovVertex();

  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> e_in;
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> e_out;
  Vector<InOut<Vector<float, VectorLayout::SPAN, 4, false>>> r;
  const unsigned worker_height;
  const unsigned worker_width;
  const unsigned worker_depth;
  const float delta;
  const float epsilon;
  const float my1;
  const float my2;
  const float dx;
  const float dt;
  const float k;
  const float a;
  const float b;

  unsigned idx(unsigned x, unsigned y, unsigned w) {
    /* The index corresponding to [x,y] in for a row-wise flattened 2D variable*/
    return y + x*w;
  } 

  bool compute () {
    const unsigned w = worker_width;
    const unsigned pw = worker_width + 2;
    const float d_dx2 = delta/(dx*dx);
    const float b_plus_1 = b + 1;
    float e_center;

    for (std::size_t x = 1; x < worker_height + 1; ++x) {
      for (std::size_t y = 1; y < worker_width + 1; ++y) {
        for (std::size_t z = 1; z < worker_depth + 1; ++z) {
          e_center = e_in[idx(x,y,pw)][z];

          // New e_out_center
          e_out[idx(x-1,y-1,w)][z-1] = e_center + dt*(
            d_dx2*(-6*e_center + 
              e_in[idx(x+1,y,pw)][z] + e_in[idx(x-1,y,pw)][z] +
              e_in[idx(x,y+1,pw)][z] + e_in[idx(x,y-1,pw)][z] +
              e_in[idx(x,y,pw)][z+1] + e_in[idx(x,y,pw)][z-1]
            ) 
            - k*e_center*(e_center - a)*(e_center - 1) 
            - e_center*r[idx(x-1,y-1,w)][z-1]
          );

          // New r_center
          r[idx(x-1,y-1,w)][z-1] += dt*(
            (-epsilon - my1*r[idx(x-1,y-1,w)][z-1]/(my2 + e_center))*
            (r[idx(x-1,y-1,w)][z-1] + k*e_center*(e_center - b_plus_1))
          );
        }
      }
    }

    return true;
  }
};
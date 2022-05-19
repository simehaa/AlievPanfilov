#include "options.hpp"

Options parse_options(int argc, char** argv) {
	Options options;
	namespace po = boost::program_options;
	po::options_description desc("Flags");
	desc.add_options()
	("help", "Show command help.")
	(
		"num-ipus",
		po::value<std::size_t>(&options.num_ipus)->default_value(1),
		"Number of IPUs (must be a power of 2)"
	)
	(
		"num-iterations",
		po::value<std::size_t>(&options.num_iterations)->default_value(10),
		"PDE: number of iterations to execute on grid."
	)
	(
		"height",
		po::value<std::size_t>(&options.height)->default_value(200),
		"Heigth of a custom 3D grid"
	)
	(
		"width",
		po::value<std::size_t>(&options.width)->default_value(200), 
		"Width of a custom 3D grid"
	)
	(
		"depth",
		po::value<std::size_t>(&options.depth)->default_value(200),
		"Depth of a custom 3D grid"
	)
	(
		"my1",
		po::value<float>(&options.my1)->default_value(0.07),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"my2",
		po::value<float>(&options.my2)->default_value(0.3),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"k",
		po::value<float>(&options.k)->default_value(8.0),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"epsilon",
		po::value<float>(&options.epsilon)->default_value(0.01),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"b",
		po::value<float>(&options.b)->default_value(0.1),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"a",
		po::value<float>(&options.a)->default_value(0.1),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"dt",
		po::value<float>(&options.dt)->default_value(0.0001),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"dx",
		po::value<float>(&options.dx)->default_value(0.001),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"delta",
		po::value<float>(&options.delta)->default_value(5.0e-5),
		"A constant in the forward Euler Aliev-Panfilov equations."
	)
	(
		"cpu-test",
		po::bool_switch(&options.cpu)->default_value(false),
		"Whether to test (MSE of meshes) against a slow CPU code."
	);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		throw std::runtime_error("Show help");
	}
	po::notify(vm);

	options.b_plus_1 = options.b + 1; // function of b
	options.dtk = options.dt*options.k; // function of dt and k
	options.lambda = options.delta*options.dt/(options.dx * options.dx); // function of delta, dx, dt
	options.gamma = 1 - 6*options.lambda; // function of delta, dx, dt

	return options;
}
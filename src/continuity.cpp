#include "asgard.hpp"

using namespace asgard;

using P = default_precision;

// exact solution in 1D
void f1(std::vector<P> const &x, P /* time */, std::vector<P> &fx)
{
  assert(fx.size() == x.size()); // this is a guarantee, do not resize fx
  for (size_t i = 0; i < x.size(); i++)
    fx[i] = std::sin(x[i]);
}
// derivative of the exact solution in 1D
void df1(std::vector<P> const &x, P /* time */, std::vector<P> &fx)
{
  for (size_t i = 0; i < x.size(); i++)
    fx[i] = std::cos(x[i]);
}
// time component of the solution
P tf(P t) {
  return std::exp(-t);
}
// derivative of the time component
P dtf(P t) {
  return -std::exp(-t);
}
// constant one
P const_one(P, P) { return P{1}; }

int main(int argc, char** argv)
{
  prog_opts options(argc, argv);

  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f = s(t, x)\n";
    std::cout << " with periodic boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-dims            -dm     int        accepts: 1 - 6\n";
    std::cout << "                                    the number of dimensions\n\n";
    return 0;
  }

  std::optional<int> opt_dims = options.extra_cli_value<int>("-dims");
  if (not opt_dims)
    opt_dims = options.extra_cli_value<int>("-dm");

  int const num_dims = opt_dims.value_or(3);

  if (not opt_dims)
    std::cout << "no -dims provided, setting a default 3D problem\n";
  else
    std::cout << "setting a " << num_dims << "D problem\n";

  // some defaults, if not provided in the cli options
  options.default_degree = 2;
  options.default_start_levels = {4, };

  // basic cfl condition (TODO: this must be cleaned)
  int max_level = options.default_start_levels.front();
  if (not options.start_levels.empty())
    max_level = std::max(
        max_level, *std::max_element(options.start_levels.begin(), options.start_levels.end()));

  options.default_dt = 0.5 * 0.1 * (4 * PI) / fm::ipow2(max_level);

  options.default_stop_time = 1.0; // untegrate until T = 1

  options.title = "Continuity " + std::to_string(num_dims) + "D";

  // the domain will have range -2 * PI to 2 * PI in each direction
  std::vector<domain_range<P>> ranges(num_dims, {-2 * PI, 2 * PI});

  pde_domain<P> domain(ranges); // can use move, but copy is cheap enough

  // create a pde from the given options and domain
  // we can read the variables using pde.options() and pde.domain() (both return const-refs)
  // the option entries may have been populated or updated with default values
  PDEv2<P> pde(std::move(options), std::move(domain));

  // one dimensional divergence term using upwind flux
  // note the type change, we are creating pterm_chain1d from a singe partial_term
  // multiple terms can be chained to obtain higher order derivatives
  pterm_chain1d<P> div = partial_term<P>(pt_div_periodic, flux_type::upwind, const_one);

  // the multi-dimensional divergence, initially set to identity in md
  std::vector<pterm_chain1d<P>> ops(num_dims);
  for (int d = 0; d < num_dims; d++)
  {
    ops[d] = div; // using derivative in the d-direction
    pde += term_md<P>(ops);
    ops[d] = pterm_chain1d<P>{pt_identity}; // reset back to identity
  }

  // define a vector containing the entries for the separable known solution
  // the time component is tf
  std::vector<svector_func1d<P>> sep(num_dims, f1);

  separable_func<P> exact(sep, tf);

  pde.add_initial(exact); // the initial condition is the exact solution

  pde.add_source({sep, dtf}); // derivative in time
  // differentiating the product, one direction at a time
  for (int d = 0; d < num_dims; d++)
  {
    sep[d] = df1; // set derivative in direction d
    pde.add_source({sep, tf});
    sep[d] = f1; // revert the vector
  }

  discretization_manager<P> disc(std::move(pde), verbosity_level::high);

  if (tools::timer.enabled() and disc.high_verbosity())
    std::cout << tools::timer.report() << '\n';

  P sum{0};
  for (auto s : disc.current_state())
    sum += s * s;
  sum = std::sqrt(sum);

  std::cout << "\n";
  std::cout << std::scientific;
  std::cout.precision(8);
  std::cout << " sum = " << sum << "  " << std::pow(2 * PI, 0.5 * num_dims) << "\n";
  std::cout << " norm error: " << std::abs(sum - std::pow(2 * PI, 0.5 * num_dims) ) << "\n";

  return 0;
};

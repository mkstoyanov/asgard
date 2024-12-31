#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// 3D test case using relaxation problem
//
//  df/dt == div_v( (v-u(x))f + theta(x)\grad_v f)
//
//  where the domain is (x,v_1,v_2).  The moments of f are constant x.
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_relaxation_1x2v : public PDE<P>
{
public:
  PDE_relaxation_1x2v(prog_opts const &cli_input)
  {
    term_set<P> terms;
    add_lenard_bernstein_collisions_1x2v(nu, terms);

    this->initialize(cli_input, num_dims_, num_sources_, dimensions_,
                     terms, sources_, exact_vector_funcs_, get_dt_,
                     has_analytic_soln_, do_collision_operator_);
  }

private:
  static int constexpr num_dims_    = 3;
  static int constexpr num_sources_ = 0;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = true;
  static bool constexpr has_analytic_soln_     = true;
  static int constexpr default_degree          = 2;

  static P constexpr nu = 1e3;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx[i] = 0.5;
    }
    return fx;
  }

  // Test 3 - 2 Maxwellians
  static fk::vector<P>
  initial_condition_dim_v_0_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    const P theta       = 0.5;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          const P ux = 3.0;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - ux, 2));
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_1_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const theta       = 0.5;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          P const uy = 0.0;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - uy, 2));
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    const P theta       = 0.5;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          const P ux = 0.0;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - ux, 2));
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_1_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const theta       = 0.5;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          P const uy = 3.0;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - uy, 2));
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      fx[i] = 0.5;
    }
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 = dimension<P>(
      -0.5, 0.5, 4, default_degree,
      {initial_condition_dim_x_0, initial_condition_dim_x_1}, nullptr, "x");

  inline static dimension<P> const dim_1 =
      dimension<P>(-8.0, 12.0, 3, default_degree,
                   {initial_condition_dim_v_0_0, initial_condition_dim_v_0_1},
                   nullptr, "v1");

  inline static dimension<P> const dim_2 =
      dimension<P>(-8.0, 12.0, 3, default_degree,
                   {initial_condition_dim_v_1_0, initial_condition_dim_v_1_1},
                   nullptr, "v2");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1,
                                                               dim_2};

  static fk::vector<P> exact_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      ignore(x_v);
      return 1.0;
    });
    return fx;
  }

  static fk::vector<P> exact_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P const theta       = 2.75;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          const P u1 = 1.5;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - u1, 2));
        });
    return fx;
  }

  static fk::vector<P> exact_dim_v_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const theta       = 2.75;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * theta);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, theta](P const x_v) -> P {
          const P u2 = 1.5;
          return coefficient * std::exp(-(0.5 / theta) * std::pow(x_v - u2, 2));
        });
    return fx;
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_dim_x_0, exact_dim_v_0, exact_dim_v_1};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    /* return dx; this will be scaled by CFL from command line */
    // return std::pow(0.25, dim.get_level());

    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL
    return (6.0 - (-6.0)) / std::pow(2, 3);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits.h>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "pde/pde_base.hpp"

namespace asgard
{

template<typename precision>
std::vector<dimension_description<precision>>
cli_apply_level_degree_correction(parser const &cli_input,
                                  std::vector<dimension_description<precision>> const dimensions)
{
  size_t num_dims = dimensions.size();
  std::vector<int> levels(dimensions.size()), degrees(dimensions.size());
  for(size_t i=0; i<num_dims; i++)
  {
      levels[i] = dimensions[i].level;
      degrees[i] = dimensions[i].degree;
  }

  // modify for appropriate level/degree
  // if default lev/degree not used
  auto const user_levels = cli_input.get_starting_levels().size();
  if (user_levels != 0 && user_levels != static_cast<int>(num_dims))
  {
    throw std::runtime_error(
        std::string("failed to parse dimension-many starting levels - parsed ")
        + std::to_string(user_levels) + " levels");
  }
  if (user_levels == static_cast<int>(num_dims))
  {
    auto counter = 0;
    for (int &l : levels)
    {
      l = cli_input.get_starting_levels()(counter++);
      expect(l > 1);
    }
  }
  auto const cli_degree = cli_input.get_degree();
  if (cli_degree != parser::NO_USER_VALUE)
  {
    expect(cli_degree > 0);
    for (int &d : degrees) d = cli_degree;
  }

  // check all dimensions
  for(size_t i=0; i<dimensions.size(); i++)
  {
    expect(degrees[i] > 0);
    expect(levels[i] > 1);
  }

  std::vector<dimension_description<precision>> result;
  result.reserve(num_dims);
  for(size_t i=0; i<num_dims; i++)
  {
    result.push_back(
      dimension_description<precision>(dimensions[i].d_min, dimensions[i].d_max,
                                       levels[i], degrees[i],
                                       dimensions[i].name)
                     );
  }
  return result;
}

/*!
 * \brief Throws an exception if there are repeated entries among the names.
 */
inline void verify_unique_strings(std::vector<std::string> const &names) {
  size_t num_dims = names.size();
  for(size_t i=0; i<num_dims; i++)
  {
    for(size_t j=i+1; j<num_dims; j++)
    {
      if (names[i] == names[j])
        throw std::runtime_error("Dimension names should be unique");
    }
  }
}

template<typename precision>
struct dimension_set {
  dimension_set(parser const &cli_input, std::vector<dimension_description<precision>> const dimensions)
    : list(cli_apply_level_degree_correction(cli_input, dimensions))
  {
    std::vector<std::string> names(list.size());
    for(size_t i=0; i<list.size(); i++)
      names[i] = list[i].name;

    verify_unique_strings(names);
  }

  dimension_description<precision> operator() (std::string const &name) const
  {
    for(size_t i=0; i<list.size(); i++)
    {
      if (list[i].name == name)
        return list[i];
    }
    throw std::runtime_error(std::string("invalid dimension name: '") + name + "', has not been defined.");
  }

  std::vector<dimension_description<precision>> const list;
};


template<typename precision>
struct field_description
{
  field_description(std::string const dimension,
                    vector_func<precision> const initial_condition,
                    vector_func<precision> const exact_solution,
                    g_func_type<precision> const volume_jacobian_dV_in, // MIRO: maybe this should be part of the dimension_description
                    std::string const field_name
                    )
    : field_description(std::vector<std::string>{dimension}, {initial_condition}, {exact_solution}, {volume_jacobian_dV_in}, field_name)
    {}

  field_description(std::vector<std::string> const dimensions,
                    std::vector<vector_func<precision>> const initial_conditions,
                    std::vector<vector_func<precision>> const exact_solution,
                    std::vector<g_func_type<precision>> const volume_jacobian_dV_in, // MIRO: maybe this should be part of the dimension_description
                    std::string const field_name
                    )
      : d_names(dimensions),
        init_cond(initial_conditions), exact(exact_solution),
        jacobian(volume_jacobian_dV_in), name(field_name)
  {
    static_assert(std::is_same<precision, float>::value
                  or std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template parameters for precision.");

    expect(dimensions.size() > 0);
    expect(dimensions.size() == initial_conditions.size());
    expect(exact_solution.size() == 0 or dimensions.size() == initial_conditions.size());
    expect(dimensions.size() == volume_jacobian_dV_in.size());
    verify_unique_strings(d_names);
  }

  size_t num_dimensions() const { return d_names.size(); }
  bool has_exact_solution() const { return (exact.size() > 0); }

  std::vector<std::string> const d_names;
  std::vector<vector_func<precision>> init_cond;
  std::vector<vector_func<precision>> exact;
  std::vector<g_func_type<precision>> jacobian;
  std::string const name;
};

template<typename precision>
class field
{
public:
  std::string const name;

  field(dimension_set<precision> const &dimensions,
        field_description<precision> const &description
        )
    : name(description.name)
  {
    size_t num_dims = description.num_dimensions();
    dims.reserve(num_dims);
    if (description.has_exact_solution())
      exact_solution.reserve(num_dims);

    for(size_t i=0; i<num_dims; i++)
    {
      // load the dimensions
      dims.push_back(
        dimension<precision>(dimensions(description.d_names[i]),
                             description.init_cond[i],
                             description.jacobian[i]
                            )
                    );

        if (description.has_exact_solution())
          exact_solution.push_back(description.exact[i]);
    }
  }

  std::vector<dimension<precision>> const &get_dimensions() const
  {
    return dims;
  }

  // not sure if this is needed
  void update_level(int const dim_index, int const new_level)
  {
    assert(dim_index >= 0);
    assert(dim_index < dims.size());
    assert(new_level >= 0);

    dims[dim_index].set_level(new_level);
  }

  bool has_exact_solution() const { return not exact_solution.empty(); }

private:
  std::vector<dimension<precision>> dims;
  std::vector<vector_func<precision>> exact_solution;
};

}

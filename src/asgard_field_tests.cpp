#include "asgard_field.hpp"

using namespace asgard;

static fk::vector<float> ic_x(fk::vector<float> const &x,
                              float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = 1.0;
  return fx;
}
static fk::vector<float> ic_y(fk::vector<float> const &x,
                             float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = x[i];
  return fx;
}
static fk::vector<float> ic_vel(fk::vector<float> const &x,
                             float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = x[i] * x[i];
  return fx;
}

static float vol_jac_dV(float const, float const)
{
  return 1.0;
}

int main(int argc, char *argv[])
{

  parser const cli_input(argc, argv);

  float min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<float> dim_0 =
      dimension_description<float>(min0, min1, level, degree, "x");
  dimension_description<float> dim_1 =
      dimension_description<float>(min0, min1, level, degree, "y");
  dimension_description<float> dim_2 =
      dimension_description<float>(min0, min1, level, degree, "vx", vol_jac_dV);
  dimension_description<float> dim_3 =
      dimension_description<float>(min0, min1, level, degree, "vy");

  field_description<float> field_1d(field_mode::evolution, "x", ic_x, ic_x, "projected to 1d");
  field_description<float> pos_field(field_mode::closure, {"x", "y"}, {ic_x, ic_y}, {ic_x, ic_y}, "position");
  field_description<float> vel_only_field(field_mode::evolution, {"vx", "vy"}, {ic_vel, ic_vel}, {ic_vel, ic_vel}, "vel-only");
  field_description<float> mixed_field(field_mode::closure,
                                       std::vector<std::string>{"x", "y", "vx", "vy"},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       "mixed-field");

  return 0;
}

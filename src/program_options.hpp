#pragma once
#include "asgard_tools.hpp"

namespace asgard
{
// implemented solvers for implicit stepping
enum class solve_opts
{
  direct,
  gmres,
  bicgstab,
  scalapack
};

// the choices for supported PDE types
enum class PDE_opts
{
  custom = 0, // user provided pde
  advection_1,
  continuity_1,
  continuity_2,
  continuity_3,
  continuity_6,
  fokkerplanck_1d_pitch_E_case1,
  fokkerplanck_1d_pitch_E_case2,
  fokkerplanck_1d_pitch_C,
  fokkerplanck_1d_4p3,
  fokkerplanck_1d_4p4,
  fokkerplanck_1d_4p5,
  fokkerplanck_2d_complete_case1,
  fokkerplanck_2d_complete_case2,
  fokkerplanck_2d_complete_case3,
  fokkerplanck_2d_complete_case4,
  diffusion_1,
  diffusion_2,
  vlasov_lb_full_f,
  vlasov_two_stream,
  relaxation_1x1v,
  relaxation_1x2v,
  relaxation_1x3v,
  riemann_1x2v,
  riemann_1x3v,
  collisional_landau,
  collisional_landau_1x2v,
  collisional_landau_1x3v
};

// Not very informative, maybe rename these of handle in a different wat
enum class PDE_case_opts
{
  case0,
  case1,
  case2,
  case3,
  case4,
  case_count
  // FIXME will need to add the user supplied PDE cases choice
};

/*!
 * \brief Indicates whether we should be using sparse or dense kronmult.
 *
 * Used by the local kronmult, global kronmult (block or non-block case)
 * will ignore these options.
 */
enum class kronmult_mode
{
  //! \brief Using a dense matrix (assumes everything is connected).
  dense,
  //! \brief Using a sparse matrix (checks for connectivity).
  sparse
};

/*!
 * \brief Type of discretization grid
 */
enum class grid_type
{
  //! Standard spars grid
  sparse,
  //! Dense grid
  dense,
  //! Dense tensor of two sparse grids
  mixed
};

namespace time_advance
{
//! types of time time advance methods, declared here to be used in the program options
enum class method
{
  imp,
  exp, // explicit is reserved keyword
  imex
};
} // namespace time_advance

enum adapt_norm
{
  l2,
  linf
};

namespace solver
{
//! indicates the values is unspecified
int constexpr novalue = -1;
//! indicates the value is unspecified
double constexpr notolerance = -1.0;
} // namespace solver

/*!
 * \internal
 * \brief Internal use (mostly)
 *
 * Allows constructing prog_opts directly from a vector of string_view.
 * Works around the ambiguity in the constructor between using a filename
 * and a list of views.
 * \endinternal
 */
struct vecstrview
{
  explicit vecstrview(std::vector<std::string_view> const &s) : s_(s)
  {}

  operator std::vector<std::string_view> const &() const { return s_; }

  std::vector<std::string_view> const &s_;
};

/*!
 * \internal
 * \brief Internal use only
 *
 * Takes ownership of a vector of strings and creates an associated vector
 * of string_view that can be used until this object is destroyed.
 *
 * Used in conjuction with split_argv to hold the temporary result.
 * \endinternal
 */
struct split_views
{
  split_views(std::vector<std::string> &&own)
      : own_(std::move(own)), strview(views_)
  {
    views_.reserve(own_.size() + 1);
    views_.push_back("test");
    for (auto &s : own_)
      views_.emplace_back(s);
  }
  operator std::vector<std::string_view> const &()
  {
    return views_;
  }
  operator vecstrview const &()
  {
    return strview;
  }
  std::vector<std::string> own_;
  std::vector<std::string_view> views_;
  vecstrview strview;
};

/*!
 * \internal
 * \brief (testing) splits a single string into multiple strings by spaces
 *
 * The method is intended for testing where it is much easier to write
 * a single string, e.g., "-p continuity_1 -d 3 -l 4" as opposed to multiple
 * lines setting pde_choice, degree and start_levels.
 *
 * However, the parsing of the string has little to no robustness,
 * especially when it comes to passing in lists.
 *
 * The use of this method in production is strongly discouraged.
 * \endinternal
 */
split_views split_argv(std::string_view const &opts);

/*!
 * \brief Reads options from the command line and input files
 *
 * The file and command line capabilities are provided for
 * convenience and are entirely optional.
 * The asgsrd::prog_opts objects can be default-constructed as empty,
 * i.e., no options provided, then each of the values can be set manually
 * before passing into other ASGarD objects.
 *
 * Reading from the command line example:
 * \code
 *   int main(int argc, char **argv) {
 *
 *     prog_opts options(argc, argv);
 *
 *     // for the use of imex, all options can be adjusted
 *     options.step_method = time_advance::method::imex;
 *
 *     // make new PDE with these options
 *     auto pde = asgard::make_custom_pde<mypde>(options);
 *
 * \endcode
 * This will process the inputs from argv and will also include any inputs
 * from a file. If an option is present multiple times, the last option
 * will take precedence, this options hard-coded in the file can be adjusted
 * from the command line, but only if they appear after the -if option.
 *
 * List of the standard ASGarD options can be seen with:
 * \code
 *   ./asgard --help # from the command line
 *   or
 *   asgard::prog_opts::print_help(); // from C++
 * \endcode
 *
 * Reading extra options from the file:
 * \code
 *   std::optional<int> foo = options.file_value<int>("foo");
 *
 *   std::optional<double> bar = options.file_value<double>("bar");
 *
 *   std::optional<std::string> myname = options.file_value<std::string>("myname");
 * \endcode
 * Supported types are bool, int, float, double, and std::string.
 *
 * Reading from a hard-coded filename reduces flexibility but can improve
 * reproducibility:
 * \code
 *   prog_opts options("intput_filename.txt")
 * \endcode
 *
 * The file format consists of simple pairs of keys-values separated by colon ":"
 * Keys that match command line input options will be used as if provided by
 * the command line, other values can be read and retrieved
 * \code
 *  # ASGarD standard options
 *  -tile         : read from test file
 *  -start_levels : 4 5
 *  -max_levels   : 7 8
 *
 *  # user specific options
 *  my keyname 1  : 1.E-4
 *  my keyname 2  : 5
 *  my keyname 3  : enable
 * \endcode
 * The 3 keys can be retrieved as double, int and bool respectively, or they can
 * all be read as strings. A good practice is to provide meaningful names for
 * the keys, e.g., "temperature".
 *
 * Notes about the API:
 * - If the keyword is missing or it is missing a value, the optional will be empty.
 * - The file_value() method may throw conversion error, e.g., trying to read
 *   an int from a sting describing a double.
 * - Boolean values interpreted as true are "true", "on", "enable", "1", "yes"
 * - Boolean values interpreted as false are "false", "off", "disable", "0", "no"
 * - ASGarD will not access or use any of the extra options provided in the file
 *   or the command line, those are responsibility of the user code.
 */
struct prog_opts
{
  //! if provided, the title helps organize projects with multiple files
  std::string title;
  //! if provided, the subtitile is an addition to the main title
  std::string subtitle;

  //! if set, one of the builtin PDEs will be used, keep unset for custom PDE projects
  std::optional<PDE_opts> pde_choice;

  //! read from -start_levels
  std::vector<int> start_levels;
  //! read from -max_levels
  std::vector<int> max_levels;

  //! sparse, dense or mixed grid
  std::optional<grid_type> grid;
  //! if using mixed group, the size of the first mixed group
  std::optional<int> mgrid_group;
  //! degree of the polynomial basis
  std::optional<int> degree;

  //! if set, enables grid adaptivity and provides the tolerance threshold
  std::optional<double> adapt_threshold;
  //! adaptivity norm, either l2 or linf
  std::optional<adapt_norm> anorm;

  //! time stepping method, explicit, implicit or imex
  std::optional<time_advance::method> step_method;
  //! fixed time step, if missing the default cfl condition will be used
  std::optional<double> dt;
  //! number of fixed time steps to take
  std::optional<int> num_time_steps;

  //! enable/disable the Poisson solver
  std::optional<bool> set_electric;

  //! output frequency of wavelet data used for restarts or python plotting
  std::optional<int> wavelet_output_freq;
  //! older method for plotting using MATLAB integration
  std::optional<int> realspace_output_freq;

  //! solver for implicit or imex methods: direct, gmres, bicgstab
  std::optional<solve_opts> solver;
  //! tolerance for the iterative solvers (gmres, bicgstab)
  std::optional<double> isolver_tolerance;
  //! max number of iterations (inner iterations for gmres)
  std::optional<int> isolver_iterations;
  //! max number of output gmres iterations
  std::optional<int> isolver_outer_iterations;

  //! local kron method only, mode dense or sparse (faster but memory hungry)
  std::optional<kronmult_mode> kron_mode;
  //! local kron method only, sparse mode, keeps less data on the device
  std::optional<int> memory_limit;

  //! restart the simulation from a file
  std::string restart_file;
  //! filename for the last time step
  std::filesystem::path outfile;

  //! indicates if the --help option was selected
  bool show_help = false;
  //! indicates if the --version option was selected
  bool show_version = false;
  //! indicates if the -pde? options was selected
  bool show_pde_help = false;
  //! indicates if the exact solution should be ignored
  bool ignore_exact = false;

  //! print list of ASGarD specific options
  static void print_help(std::ostream &os = std::cout);
  //! print version and build (cmake) options
  static void print_version_help(std::ostream &os = std::cout);
  //! print information about the builtin PDEs
  static void print_pde_help(std::ostream &os = std::cout);
  //! print the current set of options
  void print_options(std::ostream &os = std::cout) const;

  //! argv input values unrecognized by ASGarD
  std::vector<std::string> externals;

  //! converts the start_levels to a human readable string
  std::string start_levels_str() const { return vect_to_str(start_levels); }
  //! converts the max_levels to a human readable string
  std::string max_levels_str() const { return vect_to_str(max_levels); }

  //! create empty options, allows to manually fill the options later
  prog_opts() = default;

  //! process the command line arguments
  prog_opts(int const argc, char const *const *argv,
            bool ignore_unknown = true);

  //! process from a file
  explicit prog_opts(std::filesystem::path const &filename)
      : infile(filename)
  {
    process_file("<executable>");
  }

  //! mostly for testing
  explicit prog_opts(vecstrview const &argv)
  {
    process_inputs(argv, handle_mode::ignore_unknown);
  }

  //! read an extra option from a file
  template<typename out_type>
  std::optional<out_type> file_value(std::string_view const &s) const
  {
    return get_val<out_type>(filedata, s);
  }
  //! read an extra option from the cli extras
  template<typename out_type>
  std::optional<out_type> extra_cli_value(std::string_view const &s) const
  {
    return get_val<out_type>(externals, s);
  }

private:
  //! mapping from cli options to variables and actions
  enum class optentry
  {
    show_help,
    version_help,
    pde_help,
    input_file,
    ignore_exact,
    title,
    subtitle,
    grid_mode,
    step_method,
    adapt_norm,
    set_electric,
    adapt_threshold,
    start_levels,
    max_levels,
    degree,
    num_time_steps,
    wavelet_output_freq,
    realspace_output_freq,
    output_file,
    dt,
    pde_choice,
    solver,
    memory_limit,
    kron_mode,
    isol_tolerance,
    isol_iterations,
    isol_outer_iterations,
    restart_file,
  };
  enum class handle_mode
  {
    warn_on_unknown, // print warning
    ignore_unknown,  // do nothing (user inputs)
    save_unknown     // reading from file
  };

  //! input filename
  std::filesystem::path infile;

  //! file inputs, ordered as pairs by line
  std::vector<std::string> filedata;
  //! process input file, exec_name is the name of the executable
  void process_file(std::string_view const &exec_name);

  //! not in the constructor so it can be reused when reading from file
  void process_inputs(std::vector<std::string_view> const &argv,
                      handle_mode mode);
  //! map pde options string to enum value
  static std::optional<PDE_opts> get_pde_opt(std::string_view const &pde_str);

  //! converts a string of ints into a vector of ints, limited to max_num_dimensions
  static std::vector<int> parse_ints(std::string const &number_string)
  {
    std::stringstream number_stream(number_string);
    std::vector<int> result;
    result.reserve(max_num_dimensions);
    while (!number_stream.eof())
    {
      std::string word;
      number_stream >> word;
      int temp_int = -1;

      // remove any leading or trailing '"'
      size_t pos = word.find_first_of('\"');
      if (pos != std::string::npos)
        word.erase(word.begin() + pos);

      if (std::stringstream(word) >> temp_int)
      {
        if (result.size() == max_num_dimensions)
        { // too many ints, return invalid result
          result.clear();
          return result;
        }
        else
          result.push_back(temp_int);
      }
    }

    return result;
  }

  template<typename out_type>
  std::optional<out_type> get_val(std::vector<std::string> const &strs,
                                  std::string_view const &s) const
  {
    static_assert(std::is_same_v<out_type, int> or std::is_same_v<out_type, bool>
                  or std::is_same_v<out_type, float> or std::is_same_v<out_type, double>
                  or std::is_same_v<out_type, std::string>,
                  "prog_opts can only process: int, float, double, bool or string");
    for (size_t i = 0; i < strs.size(); i += 2)
    {
      if (strs[i] == s)
      {
        if (i + 1 == strs.size())
          return {};
        const std::string &val = strs[i + 1];
        if constexpr (std::is_same_v<out_type, std::string>)
          return val;
        else if constexpr (std::is_same_v<out_type, int>)
          return std::stoi(val);
        else if constexpr (std::is_same_v<out_type, double>)
          return std::stod(val);
        else if constexpr (std::is_same_v<out_type, float>)
          return std::stof(val);
        else if constexpr (std::is_same_v<out_type, bool>)
        {
          if (val == "on" or val == "yes" or val == "enable" or val == "true"
              or val == "1")
            return true;
          else if (val == "off" or val == "no" or val == "disable"
                   or val == "false" or val == "0")
            return false;
          else
            return {};
        }
      }
    }
    return {};
  }

  //! converts vector of ints into a string
  static std::string vect_to_str(std::vector<int> const &ints)
  {
    std::string s = "";
    for (auto i : ints)
      s += ((i < 10) ? "  " : " ") + std::to_string(i);
    return s;
  }
};

/*!
 * \internal
 * \brief Makes a prog_opts object from a single sting, see split_argv
 *
 * \endinternal
 */
inline prog_opts make_opts(std::string const &cli)
{
  return prog_opts(split_argv(cli));
}

//! overload for writing options to a stream
inline std::ostream &operator<<(std::ostream &os, prog_opts const &options)
{
  options.print_options(os);
  return os;
}

} // namespace asgard

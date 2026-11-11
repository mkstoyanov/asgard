#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

namespace scratch {

  struct scratch_space {
    scratch_space() {
      all_data_.push_back(std::vector<std::byte>{1024});
      pcurrent = all_data_.front().data();

      current_capacity = initial_size;
    }

    static size_t constexpr initial_size = 1024;

    std::vector<std::vector<std::byte>> all_data_;

    std::byte *pcurrent = nullptr;

    size_t current_capacity  = 0;
    size_t current_size      = 0;

    template<typename T, typename... Args>
    T *make(Args... the_args) {
      auto make_pntr_ = [&]() -> T *
        {
          T* result = new (pcurrent) T(the_args...);
          current_size = pcurrent - all_data_.back().data();
          pcurrent = reinterpret_cast<std::byte *>(result) + sizeof(T);
          return result;
        };
      if (current_size + sizeof(T) + alignof(T) < current_capacity) {
        return make_pntr_();
      } else {
        expand_();
        assert(sizeof(T) + alignof(T) < current_capacity);
        return make_pntr_();
      }
    }

    template<typename T>
    T *make_arr(size_t num) {
      auto make_arr_ = [&]() -> T *
        {
          T* result = new (pcurrent) T[num];
          current_size = pcurrent - all_data_.back().data();
          pcurrent = reinterpret_cast<std::byte *>(result) + sizeof(T) * num;
          return result;
        };
      if (current_size + sizeof(T) * num + alignof(T) < current_capacity) {
        return make_arr_();
      } else {
        expand_();
        assert(sizeof(T) * num + alignof(T) < current_capacity);
        return make_arr_();
      }
    }

    void clear() {
      if (all_data_.size() > 1) {
        size_t const usage = std::accumulate(all_data_.begin(), all_data_.end(), size_t{0},
                                             [](size_t c, auto const &vec) -> size_t { return c + vec.size(); });
        all_data_.clear();
        all_data_.push_back(std::vector<std::byte>{usage});
      }
      current_capacity = all_data_.front().size();
      pcurrent = all_data_.front().data();
      current_size = 0;
    }

    void expand_() {
      all_data_.push_back(std::vector<std::byte>{all_data_.front().size()});
      pcurrent = all_data_.back().data();
      std::cout << "expanding scratch\n";
    }

  };

  std::optional<scratch_space> space_;

  void init() { space_.emplace(); }

  template<typename T, typename... Args>
  T *make_ptr(Args... the_args) {
    return space_->make<T>(the_args...);
  }

  template<typename T, typename... Args>
  T &make(Args... the_args) {
    return *space_->make<T>(the_args...);
  }

  template<typename T>
  T *make_arr(size_t num) {
    return space_->make_arr<T>(num);
  }

  void clear() { space_->clear(); }

}

int main(int argc, char **argv)
{
  std::ignore = argc;
  std::ignore = argv;

  scratch::init();

  double *pd = scratch::make_ptr<double>(1);
  std::cout << " double = " << *pd << '\n';

  auto &d = scratch::make<double>(2);
  std::cout << " double-ref = " << d << '\n';

  auto &arr = scratch::make<std::array<double, 10>>();
  arr.fill(30);
  for (auto v : arr) std::cout << v << "  ";
  std::cout << '\n';

  size_t n = 40;
  auto a = scratch::make_arr<double>(n);
  for (size_t i = 0; i < n; i++)
    a[i] = i + 1;

  for (size_t i = 0; i < n; i++)
    std::cout << a[i] << "  ";
  std::cout << '\n';

  std::cout << "  ==== NEW FRAME ====\n";

  std::vector<double *> vecs;
  for (int s = 0; s < 10; s++)
    vecs.push_back( scratch::make_arr<double>(120) );

  scratch::clear();
  vecs.clear();
  std::cout << "  ==== NEW FRAME ====\n";

  for (int s = 0; s < 20; s++)
    vecs.push_back( scratch::make_arr<double>(120) );

  double *p = vecs.back();
  for (int i = 0; i < 64; i++) p[i] = 1.0;

  scratch::clear();
  vecs.clear();
  std::cout << "  ==== NEW FRAME ====\n";

  for (int s = 0; s < 10; s++)
    vecs.push_back( scratch::make_arr<double>(120) );

  p = vecs.back();
  for (int i = 0; i < 64; i++) p[i] = 1.0;

  std::cout << "  ==== END ====\n";

  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}

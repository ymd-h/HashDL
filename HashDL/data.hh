#ifndef DATA_HH
#define DATA_HH

#include <algorithm>
#include <vector>
#include <iterator>

namespace HashDL {
  using hashcode_t = std::uint64_t;
  using idx_t = std::vector<std::size_t>;

  inline auto index_vec(std::size_t N){
    idx_t idx{};

    idx.reserve(N);
    std::generate_n(std::back_inserter(idx), N, [i=0]() mutable { return i++; });

    return idx;
  }


  template<typename T> class Data {
  private:
    std::size_t _size;
    std::vector<T> data;

  public:
    Data(): Data{1} {}
    Data(std::size_t size): _size{size}, data(size) {}
    Data(const std::vector<T>& data): _size{data.size()}, data{data} {}
    Data(std::vector<T>&& data): _size{data.size()}, data{data} {}
    template<typename I> Data(I&& begin, I&& end)
      : _size{(std::size_t)std::distance(begin, end)}, data{begin, end} {}
    template<typename I, typename F> Data<T>(I&& begin, I&& end, F&& f)
      : _size{(std::size_t)std::distance(begin, end)}, data{}
    {
      data.reserve(_size);
      std::transform(begin, end, std::back_inserter(data), f);
    }
    Data(const Data<T>&) = default;
    Data(Data<T>&&) = default;
    Data& operator=(const Data<T>&) = default;
    Data& operator=(Data<T>&&) = default;
    ~Data() = default;

    std::size_t size() const noexcept { return _size; }
    auto begin(){ return data.begin(); }
    auto end(){ return data.end(); }
    const auto begin() const { return data.begin(); }
    const auto end() const { return data.end(); }
    const auto operator[](std::size_t n) const { return data[n]; }
    auto& operator[](std::size_t n){ return data[n]; }
  };

  template<typename T> inline auto begin(Data<T>& d){ return d.begin(); }
  template<typename T> inline auto end(Data<T>& d){ return d.end(); }
  template<typename T> inline auto begin(const Data<T>& d){ return d.begin(); }
  template<typename T> inline auto end(const Data<T>& d){ return d.end(); }


  template<typename T> class BatchData {
  private:
    std::size_t data_size;
    std::vector<T> data;
  public:
    BatchData() = default;
    BatchData(std::size_t data_size): data_size{data_size}, data{} {}
    BatchData(std::size_t data_size, const std::vector<T>& data)
      : data_size{data_size}, data{data} {}
    BatchData(std::size_t data_size, std::vector<T>&& data)
      : data_size{data_size}, data{data} {}
    BatchData(std::size_t data_size, std::size_t batch_size, T v)
      : data_size{data_size}, data(data_size * batch_size, v) {}
    BatchData(const BatchData&) = default;
    BatchData(BatchData&&) = default;
    BatchData& operator=(const BatchData&) = default;
    BatchData& operator=(BatchData&&) = default;
    ~BatchData() = default;

    auto begin(){ return data.begin(); }
    auto end(){ return data.end(); }
    auto begin(std::size_t i){ return data.begin() + data_size * i; }
    auto end(std::size_t i){ return data.begin() + data_size * (i+1); }

    auto get_data_size() const noexcept { return data_size; }
    auto get_batch_size() const noexcept { return data.size() / data_size; }

    void push_back(const std::vector<T>& d){
      if(d.size() % data_size){
	throw std::runtime_error("Input data size is not compatible with data_size");
      }
      data.reserve(data.size() + d.size());
      std::copy(d.begin(), d.end(), std::back_inserter(data));
    }

    void push_back(std::vector<T>&& d){
      if(d.size() % data_size){
	throw std::runtime_error("Input data size is not compatible with data_size");
      }
      data.reserve(data.size() + d.size());
      std::move(d.begin(), d.end(), std::back_inserter(data));
    }
  };


  template<typename T> class BatchView {
  private:
    std::size_t data_size;
    std::size_t batch_size;
    T* data_ptr;
  public:
    BatchView() = default;
    BatchView(std::size_t data_size, std::size_t batch_size, T* data_ptr)
      : data_size{data_size}, batch_size{batch_size}, data_ptr{data_ptr} {}
    BatchView(const BatchView&) = default;
    BatchView(BatchView&&) = default;
    BatchView& operator=(const BatchView&) = default;
    BatchView& operator=(BatchView&&) = default;
    ~BatchView() = default;

    T* begin(){ return data_ptr; }
    T* end(){ return data_ptr + data_size * batch_size; }
    T* begin(std::size_t i){ return data_ptr + data_size * i; }
    T* end(std::size_t i){ return data_ptr + data_size * (i+1); }
    const T* begin(std::size_t i) const { return data_ptr + data_size * i; }
    const T* end(std::size_t i) const { return data_ptr + data_size * (i+1); }

    auto get_data_size() const noexcept { return data_size; }
    auto get_batch_size() const noexcept { return batch_size; }
  };
}

#endif

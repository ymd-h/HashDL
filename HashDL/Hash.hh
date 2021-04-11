#ifndef HASH_HH
#define HASH_HH

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>


namespace HashDL {

  using data_t = float;
  using hashcode_t = std::uint64_t;

  template<typename T> class Data {
  private:
    std::size_t size;
    std::vector<T> data;

  public:
    Data<T>(): Data<T>{1};
    Data<T>(std::size_t size): size{size}, data(size) {}
    Data<T>(const Data<T>&) = default;
    Data<T>(Data<T>&&) = default;
    Data<T>& operator=(const Data<T>&) = default;
    Data<T>& operator=(Data<T>&&) = default;
    ~Data<T>() = default;

    const auto size() const { return size; }
    auto begin(){ data.begin(); }
    auto end(){ data.end(); }
    const auto operator[](std::size_t n) const { return data[n]; }
    auto& operator[](std::size_t n){ return data[n]; }
  };

  inline auto begin(Data<T>& d){ return d.begin(); }
  inline auto end(Data<T>& d){ return d.end(); }


  class Hash {
  public:
    Hash() = default;
    Hash(const Hash&) = default;
    Hash(Hash&&) = default;
    Hash& operator=(const Hash&) = default;
    Hash& operator=(Hash&&) = default;
    ~Hash() = default;
    using Data_t = Data<data_t>;

    virtual hashcode_t operator()(Data_t data) = 0;
  };


  class WTA : public Hash {
  private:
    const std::size_t bin_size;
    const std::size_t data_size;
    const std::size_t sample_size;
    std::size_t sample_bits;
    std::vector<std::vector<std::size_t>> theta; // [bin_size, sample_size]
  public:
    WTA(): WTA{8, 16, 4} {}
    WTA(std::size_t bin_size, std::size_t data_size, std::size_t sample_size)
      : bin_size{bin_size},
	data_size{data_size},
	sample_size{sample_size},
	sample_bits{1},
	theta{}
    {
      if(data_size < sample_size){
	throw std::runtime_error("sample_size must be smaller than data_size");
      }

      std::size_t power = 2;
      while(sample_size > power){
	++sample_bits;
	power *= 2;
      }

      if(bin_size*sample_bits > 64){
	throw std::runtime_error("sample_size and bin_size is too large "
				 "for 64bit hash code");
      }

      std::vector<std::size_t> index{};
      index.reserve(data_size);
      std::generate_n(std::back_inserter(index), data_size,
		      [i=0]() mutable { return i++; });

      std::mt19937 generator{std::random_device{}()};
      theta.reserve(bin_size);
      for(std::size_t i=0; i<bin_size; ++i){
	std::shuffle(index.begin(), index.end(), generator);
	theta.emplace_back(index.begin(), index.begin()+sample_size);
      }
    }
    WTA(const WTA&) = default;
    WTA(WTA&&) = default;
    WTA& operator=(const WTA&) = default;
    WTA& operator=(WTA&&) = default;
    ~WTA() = default;
    using Data_t = Data<data_t>;

    virtual hashcode_t operator()(const Data_t& data) override {
      if(data.size() != data_size){ throw std::runtime_error("Data size mismuch!"); }
      hashcode_t hash = 0;

      for(const auto& th : theta){
	auto max_v = std::numeric_limits<data_t>::lowest();
	std::size_t max_i = 0;
	for(std::size_t i=0; i<sample_size; ++i){
	  if(const auto v = data[th[i]]; v > max_v){ max_v = v; max_i = i; }
	}

	hash = (hash << sample_bits) | hashcode_t{max_i};
      }

      return hash;
    }
  };



  class DWTA : public Hash {
  private:


  public:
    DWTA() = default;
    DWTA(const DWTA&) = default;
    DWTA(DWTA&&) = default;
    DWTA& operator=(const DWTA&) = default;
    DWTA& operator=(DWTA&&) = default;
    ~DWTA() = default;
    using Data_t = Data<data_t>;

    virtual hashcode_t operator()(Data_t data) override {

    }
  };

};
#endif

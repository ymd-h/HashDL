#ifndef HASH_HH
#define HASH_HH

#include <cstdint>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "data.hh"

namespace HashDL {
  template<typename T> class Hash {
  public:
    Hash() = default;
    Hash(const Hash&) = default;
    Hash(Hash&&) = default;
    Hash& operator=(const Hash&) = default;
    Hash& operator=(Hash&&) = default;
    ~Hash() = default;
    using Data_t = Data<T>;

    virtual hashcode_t encode()(Data_t data) = 0;
  };


  template<typename T> class WTA : public Hash<T> {
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
    using Data_t = Data<T>;

    virtual hashcode_t encode()(const Data_t& data) override {
      if(data.size() != data_size){ throw std::runtime_error("Data size mismuch!"); }
      hashcode_t hash = 0;

      for(const auto& th : theta){
	auto max_v = std::numeric_limits<T>::lowest();
	std::size_t max_i = 0;
	for(std::size_t i=0; i<sample_size; ++i){
	  if(const auto v = data[th[i]]; v > max_v){ max_v = v; max_i = i; }
	}

	hash = (hash << sample_bits) | hashcode_t{max_i};
      }

      return hash;
    }
    static std::function<WTA*()> make_factory(std::size_t bin_size,
					      std::size_t data_size,
					      std::size_t sample_size){
      return [=](){ return new WTA{bin_size, data_size, sample_size}; };
    }
  };



  template<T> class DWTA : public Hash<T> {
  private:
    const std::size_t bin_size;
    const std::size_t data_size;
    const std::size_t sample_size;
    std::size_t sample_bits;
    const std::max_attempt;
    std::size_t attempt_bits;
    std::vector<std::vector<std::size_t>> theta; // [bin_size, sample_size]
    std::size_t coprime;
  public:
    DWTA() : DWTA{8, 16, 4} {}
    DWTA(std::size_t bin_size, std::size_t data_size, std::size_t sample_size,
	 std::size_t max_attempt=100)
      : bin_size{bin_size},
	data_size{data_size},
	sample_size{sample_size},
	sample_bits{1},
	max_attempt{max_attempt},
	attempt_bits{1},
	theta{},
	coprime{}
    {
      if(data_size < sample_size){
	throw std::runtime_error("sample_size must be smaller than data_size");
      }

      std::size_t power = 2;
      while(sample_size > power){
	++sample_bits;
	power *= 2;
      }

      power = 2;
      while(max_attempt > power){
	++attempt_bits;
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

      std::uniform_int_distribution<std::size_t> dist(0, std::numeric_limits<std::size_t>::max());
      coprime = dist(generator);
      while(std::gcd(sample_size, coprime) != 1){ coprime = dist(gen); }
    }
    DWTA(const DWTA&) = default;
    DWTA(DWTA&&) = default;
    DWTA& operator=(const DWTA&) = default;
    DWTA& operator=(DWTA&&) = default;
    ~DWTA() = default;
    using Data_t = Data<T>;

    virtual hashcode_t encode()(Data_t data) override {
      if(data.size() != data_size){ throw std::runtime_error("Data size mismuch!"); }

      std::vector<T> max_vs{};
      max_vs.reserve(sample_size);

      std::vector<std::size_t> max_is{};
      max_is.reserve(sample_size);

      for(const auto& th : theta){
	auto max_v = std::numeric_limits<T>::lowest();
	std::size_t max_i = 0;
	for(std::size_t i=0; i<sample_size; ++i){
	  if(const auto v = data[th[i]]; v > max_v){ max_v = v; max_i = i; }
	}

	max_vs.push_back(max_v);
	max_is.push_back(max_i);
      }

      hashcode_t hash = 0;
      for(std::size_t i=0; i<sample_size; ++i){
	if(max_vs[i]){ // != 0.0
	  hash = (hash << sample_bits) | hashcode_t{max_is[i]};
	}else{ // == 0.0
	  std::size_t next = i;
	  for(std::size_t attempt=0; attempt<max_attempt; ++attempt){
	    next = univarsal_hash(i, attempt);
	    if(max_vs[next]){ break; }
	  }
	  // Original DWTA adds "attempt + C", however, SLIDE doesn't.
	  // http://auai.org/uai2018/proceedings/papers/321.pdf
	  hash = (hash << sample_bits) | hashcode_t{max_is[next]};
	}
      }

      return hash;
    }

    std::size_t universal_hash(std::size_t i, std::size_t attempt){
      auto x = (i << attempt_bits) + attempt;
      return (x * coprime) % sample_size;
    }
    static std::function<DWTA*()> make_factory(std::size_t bin_size,
					       std::size_t data_size,
					       std::size_t sample_size,
					       std::size_t max_attempt=100){
      return [=](){ return new DWTA{bin_size,data_size,sample_size,max_attempt}; };
    }
  };

}
#endif

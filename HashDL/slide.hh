#ifndef SLIDE_HH
#define SLIDE_HH

#include <execution>
#include <unordered_map>

#include "Hash.hh"

class HashTable {
private:
  const std::size_t L;
  std::function<Hash*()> hash_factory;
  std::vector<std::unique_ptr<Hash>> hash;
  std::vector<std::unordered_multimap<hashcode_t, std::size_t>> backet;
  std::vector<std::size_t> idx;
  std::size_t neuron_size;
public:
  HashTable(): HashTable(50, DWTA::make_factory(8, 16, 8)){}
  HashTable(std::size_t L, std::function<Hash*()> hash_factory)
    : L{L}, hash_factory{hash_factory}, hash{}, backet(L), idx{}, neuron_size{}
  {
    hash.reserve(L);
    std::generate_n(std::back_inserter(hash), L,
		    [&](){ return std::unique_ptr<Hash>{hash_factory()}; });

    idx.reserve(L);
    std::generate_n(std::back_inserter(idx), L, [i=0]() mutable { return i++; });
  }
  HashTable(const HashTable&) = default;
  HashTable(HashTable&&) = default;
  HashTable& operator=(const HashTable&) = default;
  HashTable& operator=(HashTable&&) = default;
  ~HashTable() = default;

  void reset(){
    for(auto& h : hash){ h.reset(hash_factory); }

    backet.clear();
    backet.resize(L);

    neuron_size = 0;
  }

  void add(const std::vector<Data>& W){
    std::for_each(std::execution::par, idx.begin(), idx.end(),
		  [&W,this](auto i){
		    for(std::size_t n=0, size=W.size(); n<size; ++n){
		      this->backet[i].insert(this->hash[i]->operator(W[n]), n);
		    }
		  });
    neuron_size = W.size();
  }

  auto retrieve(const std::vector<Data>& X) const {
    std::vector<std::vector<std::size_t>> nids{}
    nids.reserve(X.size());

    for(auto& x : X){
      std::vector<std::size_t> neuron_id{};
      neuron_id.reserve(neuron_size);
      std::generate_n(std::back_inserter(neuron_id), neuron_size,
		      [i=0]() mutable { return i++; });

      for(auto i=0; i<L; ++i){
	auto [begin, end] = backet[i].equal_range(hash[i]->operator(x));
	std::remove_if(neuron_id.begin(), neuron_id.end(),
		       [=](auto n){ return std::find(begin, end, n) == end; });
      }

      nids.push_back(std::move(neuron_id));
    }

    return nids;
  }
};



class LSH {
public:
  LSH() = default;
  LSH(const size_t K, const size_t L){}
  LSH(const LSH&) = default;
  LSH(LSH&&) = default;
  LSH& operator=(const LSH&) = default;
  LSH& operator=(LSH&&) = default;
  ~LSH() = default;

private:
  std::vector<HashTable> hash;

public:

};



#endif

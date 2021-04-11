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
public:
  HashTable(): HashTable(50, DWTA::make_factory(8, 16, 8)){}
  HashTable(std::size_t L, std::function<Hash*()> hash_factory)
    : L{L}, hash_factory{hash_factory}, hash{}, backet(L), idx{}
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
  }

  void add(std::size_t n, const Data& w){
    std::for_each(std::execution::par, idx.begin(), idx.end(),
		  [&w,n,this](auto i){
		    this->backet[i].insert(this->hash[i]->operator(w), n);
		  });
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

#ifndef SLIDE_HH
#define SLIDE_HH

#include <execution>
#include <unordered_map>

#include "Activation.hh"
#include "Hash.hh"

namespace HashDL {
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

    auto get_data_size() const noexcept { return data_size; }
    auto get_batch_size() const noexcept { return batch_size; }
  };

  class LSH {
  private:
    const std::size_t L;
    std::function<Hash*()> hash_factory;
    std::vector<std::unique_ptr<Hash>> hash;
    std::vector<std::unordered_multimap<hashcode_t, std::size_t>> backet;
    std::vector<std::size_t> idx;
    std::size_t neuron_size;
  public:
    LSH(): LSH(50, DWTA::make_factory(8, 16, 8)){}
    LSH(std::size_t L, std::function<Hash*()> hash_factory)
      : L{L}, hash_factory{hash_factory}, hash{}, backet(L), idx{}, neuron_size{}
    {
      hash.reserve(L);
      std::generate_n(std::back_inserter(hash), L,
		      [&](){ return std::unique_ptr<Hash>{hash_factory()}; });

      idx.reserve(L);
      std::generate_n(std::back_inserter(idx), L, [i=0]() mutable { return i++; });
    }
    LSH(const LSH&) = default;
    LSH(LSH&&) = default;
    LSH& operator=(const LSH&) = default;
    LSH& operator=(LSH&&) = default;
    ~LSH() = default;

    void reset(){
      for(auto& h : hash){ h.reset(hash_factory); }

      backet.clear();
      backet.resize(L);

      neuron_size = 0;
    }

    void add(const Data<data_t>& W){
      std::for_each(std::execution::par, idx.begin(), idx.end(),
		    [&W,this](auto i){
		      for(std::size_t n=0, size=W.size(); n<size; ++n){
			this->backet[i].insert(this->hash[i]->encode(W[n]), n);
		      }
		    });
      neuron_size = W.size();
    }

    auto retrieve(const Data<data_t>& X) const {
      std::vector<std::size_t> neuron_id{};
      neuron_id.reserve(neuron_size);
      std::generate_n(std::back_inserter(neuron_id), neuron_size,
		      [i=0]() mutable { return i++; });

      for(auto i=0; i<L; ++i){
	auto [begin, end] = backet[i].equal_range(hash[i]->encode(X));
	std::remove_if(neuron_id.begin(), neuron_id.end(),
		       [=](auto n){ return std::find(begin, end, n) == end; });
      }

      return neuron_id;
    }
  };


  class Neuron {
  private:
    std::vector<int> is_active;
    std::vector<data_t> data;
    std::vector<data_t> gradient;
    std::vector<data_t> weight;
    data_t bias;
  public:
    Neuron(): Neuron{16};
    Neuron(std::size_t prev_units,
	   std::function<data_t()> weight_initializer = [](){ return 0; })
      : is_active{}, data{}, gradient{}, weight{}, bias{}
    {
      weight.reserve(prev_units);
      std::generate_n(std::back_inserter(weight), prev_units, weight_initializer);
    }
    Neuron(const Neuron&) = default;
    Neuron(Neuron&&) = default;
    Neuron& operator=(const Neuron&) = default;
    Neuron& operator=(Neuron&&) = default;
    ~Neuron() = default;

    void reset_batch(std::size_t batch_size){
      is_active.clear();
      is_active.resize(batch_size, 0);

      data.clear();
      data.resize(batch_size, 0);

      gradient.clear();
      gradient.resize(batch_size, 0);
    }

    void activate(std::size_t i_batch){
      is_active[i_batch] = 1;
    }

    const auto forward(std::size_t batch_i, const Data<data_t>& X,
			const std::unique_ptr<Activation<data_t>>& f){
      if(!is_active[batch_i]){
	data[batch_i] = 0;
	return data[batch_i];
      }

      for(std::size_t j=0, data_size=X.size(); j<data_size; ++j){
	data[batch_i] += weights[j]*data[j];
      }
      data[batch_i] += bias;

      return f->call(data[batch_i]);
    }

    const auto& get_weight() const noexcept { return weight; }
  };


  class Layer {
  private:
    const std::size_t neuron_size;
    std::vector<Neuron> neuron;
    LSH hash;
    std::unique_ptr<Activation<data_t>> activation;
  public:
    Layer(): Layer{30}{}
    Layer(std::size_t prev_units, std::size_t units, Activation<data_t>* f)
      : neuron_size{units}, neuron(units,Neuron{prev_units}), activation{f()} {}
    Layer(const Layer&) = default;
    Layer(Layer&&) = default;
    Layer& operator=(const Layer&) = default;
    Layer& operator=(Layer&&) = default;
    ~Layer() = default;

    auto operator()(std::size_t batch_i, const Data<data_t>& X){
      for(const auto& nid: hash.retrieve(X)){ neuron[nid].activate(batch_i); }

      Data<data_t> Y{neuron_size};
      for(std::size n=0; n<neuron_size; ++n){ Y[n] = neuron[n].forward(batch_i, X); }

      return Y;
    }
  };


  class Network {
  private:
    std::size_t input_dim;
    std::size_t output_dim;
    std::vector<Layer> layer;
  public:
    Network() = default;
    Network(const Network&) = default;
    Network(Network&&) = default;
    Network& operator=(const Network&) = default;
    Network& operator=(Network&&) = default;
    ~Network() = default;

    auto operator()(const DataView<data_t>& X){
      const auto batch_size = X.get_batch_size();

      for(auto& L: layer){ L.reset(batch_size); }

      std::vector<std::size_t> batch_idx{}
      batch_idx.reserve(batch_size);
      std::generate_n(std::back_inserter(batch_idx), batch_size,
		      [i=0]() mutable { return i++; });


      // Parallel Feed-Forward over Batch
      BatchData Y{output_dim, std::vector<data_t>(output_dim * batch_size, 0)};
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<data_t>{X.begin(i), X.end(i)};
		      for(auto& L: this->layer){ d = L(i, d); }

		      std::copy(d.begin(), d.end(), Y.begin(i));
		    });

      return Y;
    }
  };

}
#endif

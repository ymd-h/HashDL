#ifndef SLIDE_HH
#define SLIDE_HH

#include <execution>
#include <unordered_map>

#include "Activation.hh"
#include "Hash.hh"

namespace HashDL {
  inline auto index_vec(std::size_t N){
    std::vector<std::size_t> idx{};

    idx.reserve(N);
    std::generate_n(std::back_inserter(idx), N, [i=0]() mutable { return i++; });

    return idx;
  }


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


  template<typename T> class Weight {
  private:
    std::vector<T> w;
    T b;
    std::vector<std::atomic<T>> w_diff;
    std::atomic<T> b_diff;
  public:
    Weight() = default;
    Weight(std::size_t N): w(N), b{}, w_diff(N), b_diff{} {}
    Weight(const Weight&) = default;
    Weight(Weight&&) = default;
    Weight& operator=(const Weight&) = default;
    Weight& operator=(Weight&&) = default;
    ~Weight() = default;

    const auto& weight() const noexcept { return w; }
    auto weight(std::size_t i) const { return w[i]; }
    auto bias() const noexcept { return b; }

    void update(){
      for(std::size_t i=0, size=w.size(); i<size; ++i){
	w[i] += w_diff[i].exchange(0);
      }
      b += b_diff.exchange(0);
    }

    void add_weight_diff(std::size_t i, T d){ w_diff.fetch_add(d); }
    void add_bias_diff(T d){ b_diff.fetch_add(d); }

    auto affine(const Data<T>& X, const std::vector<std::size_t>& prev_active) const {
      auto result = b;
      for(auto i : prev_active){
	result += w[i]*X[i];
      }
      return result;
    }
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
      for(auto& h : hash){ h.reset(hash_factory()); }

      backet.clear();
      backet.resize(L);

      neuron_size = 0;
    }

    void add(const std::vector<Neuron>& N){
      std::for_each(std::execution::par, idx.begin(), idx.end(),
		    [&W,this](auto i){
		      for(std::size_t n=0, size=N.size(); n<size; ++n){
			const auto& W = N[n].get_weight();
			this->backet[i].insert(this->hash[i]->encode(W), n);
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
    std::vector<data_t> data;
    std::vector<data_t> gradient;
    std::vector<data_t> weight;
    data_t bias;
  public:
    Neuron(): Neuron{16};
    Neuron(std::size_t prev_units,
	   std::function<data_t()> weight_initializer = [](){ return 0; })
      : data{}, gradient{}, weight{}, bias{}
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
      data.clear();
      data.resize(batch_size, 0);

      gradient.clear();
      gradient.resize(batch_size, 0);
    }

    const auto forward(std::size_t batch_i,
		       const Data<data_t>& X,
		       const std::vector<std::size_t>& prev_active,
		       const std::unique_ptr<Activation<data_t>>& f){
      data[batch_i] = weight.affine(X, prev_active);

      return f->call(data[batch_i]);
    }

    const auto backward(std::size_t batch_i,
			const Data<data_t>& dn_dy,
			const std::unique_ptr<Activation<data_t>>& f){
      return f->back(data[batch_i], dn_dy);
    }

    const auto& get_weight() const noexcept { return weight; }
  };

  class Layer {
  private:
    Layer* _next;
    Layer* _prev;
  public:
    auto next() const noexcept { return _next; }
    auto prev() const noexcept { return _prev; }
    void set_next(Layer* L){ _next = L; }
    void set_prev(Layer* L){ _prev = L; }
    virtual Data<data_t> forward(std::size_t,
				 const Data<data_t>&,
				 const std::vector<std::size_t>&) = 0;
    virtual void backward(std::size_t,
			  const Data<data_t>&,
			  const std::vector<std::size_t>&) = 0;
    virtual void reset(std::size_t batch_size){}
  };


  class InputLayer : public Layer {
  private:
    std::vector<std::size_t> idx;
  public:
    InputLayer() = default;
    InputLayer(std::size_t unit): idx{index_vec(unit)} {}
    InputLayer(const InputLayer&) = default;
    InputLayer(InputLayer&&) = default;
    InputLayer& operator=(const InputLayer&) = default;
    InputLayer& operator=(InputLayer&&) = default;
    ~InputLayer() = default;

    virtual Data<data_t> forward(std::size_t batch_i, const Data<data_t>& X,
				 std::vector<std::size_t>& prev_active) override {
      return next()->forward(batch_i, X, idx);
    }
    virtual void backward(std::size_t batch_i,
			  const Data<data_t>& dn_dy,
			  const std::vector<std::size_t>& next_active) override {}
  };


  class OutputLayer : public Layer {
  private:
    std::vector<std::size_t> idx;
  public:
    OutputLayer() = default;
    OutputLayer(std::size_t unit): idx{index_vec(unit)} {}
    OutputLayer(const OutputLayer&) = default;
    OutputLayer(OutputLayer&&) = default;
    OutputLayer& operator=(const OutputLayer&) = default;
    OutputLayer& operator=(OutputLayer&&) = default;
    ~OutputLayer() = default;

    virtual Data<data_t> forward(std::size_t batch_i,
				 const Data<data_t>& X,
				 std::vector<std::size_t>& prev_active) override {
      return X;
    }
    virtual void backward(std::size_t batch_i,
			  const Data<data_t>& dn_dy,
			  const std::vector<std::size_t>& next_active) override {
      prev()->backward(batch_i, dn_dy, idx);
    }
  };


  class DenseLayer : public Layer {
  private:
    const std::size_t neuron_size;
    std::vector<Neuron> neuron;
    std::vector<std::vector<std::size_t>> active_list;
    LSH hash;
    std::unique_ptr<Activation<data_t>> activation;
  public:
    DenseLayer(): DenseLayer{30}{}
    DenseLayer(std::size_t prev_units, std::size_t units, Activation<data_t>* f)
      : neuron_size{units}, neuron(units, Neuron{prev_units}), active_list{},
	hash{}, activation{f} {
      hash.add(neuron);
    }
    DenseLayer(const DenseLayer&) = default;
    DenseLayer(DenseLayer&&) = default;
    DenseLayer& operator=(const DenseLayer&) = default;
    DenseLayer& operator=(DenseLayer&&) = default;
    ~DenseLayer() = default;

    void rehash(){
      hash.reset();
      hash.add(neuron);
    }

    virtual Data<data_t> forward(std::size_t batch_i,
				 const Data<data_t>& X,
				 const std::vector<std::size_t>& prev_active) override {
      active_list[batch_i] = hash.retrieve(X);

      Data<data_t> Y{neuron_size};
      for(auto n : active_list[batch_i]){
	Y[n] = neuron[n].forward(batch_i, X, activation);
      }

      return next()->forward(batch_i, Y, active_list[batch_i]);
    }

    virtual void backward(std::size_t batch_i,
			  const Data<data_t>& dn_dy,
			  const std::vector<std::size_t>& next_active) override {
      Data<data_t> dn_dx{neuron_size};
      std::for_each(std::execution::par,
		    active_list[batch_i].begin(), active_list[batch_i].end(),
		    [&, this](auto n){
		      dn_dx[n] = this->neuron[n].backward(batch_i, dn_dy, next_active);
		    });

      prev()->backward(batch_i, dn_dx, active_list[batch_i]);
    }

    virtual void reset(std::size_t batch_size) override {
      active_list.clear();
      active_list.resize(batch_size);

      for(auto& n : neuron){ neuron.reset(batch_size); }
    }
  };


  class Network {
  private:
    std::size_t input_dim;
    std::size_t output_dim;
    std::vector<std::unique_ptr<Layer>> layer;
  public:
    Network() = default;
    Network(const Network&) = default;
    Network(Network&&) = default;
    Network& operator=(const Network&) = default;
    Network& operator=(Network&&) = default;
    ~Network() = default;

    auto operator()(const DataView<data_t>& X){
      const auto batch_size = X.get_batch_size();

      for(auto& L: layer){ L->reset(batch_size); }

      std::vector<std::size_t> batch_idx{}
      batch_idx.reserve(batch_size);
      std::generate_n(std::back_inserter(batch_idx), batch_size,
		      [i=0]() mutable { return i++; });


      // Parallel Feed-Forward over Batch
      BatchData Y{output_dim, std::vector<data_t>(output_dim * batch_size, 0)};
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<data_t>{X.begin(i), X.end(i)};
		      d = layer.data()->forward(i, d, std::vector<std::size_t>{});

		      std::move(d.begin(), d.end(), Y.begin(i));
		    });

      return Y;
    }

    auto backward(const DataView<data_t>& dLoss_dy){
      const auto batch_size = dLoss_dy.get_batch_size();

      auto batch_idx = index_vec(batch_size);
      std::for_each(batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<data_t>{dLoss_dy.begin(i), dLoss_dy.end(i)};
		      layer.back()->backward(i, d, std::vector<std::size_t>{});
		    });

      std::for_each();
    }
  };

}
#endif

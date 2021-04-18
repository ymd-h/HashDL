#ifndef SLIDE_HH
#define SLIDE_HH

#include <execution>
#include <unordered_map>

#include "data.hh"
#include "activation.hh"
#include "optimizer.hh"
#include "hash.hh"

namespace HashDL {
  template<typename T> class Param {
  private:
    T value;
    std::atomic<T> grad;
    std::unique_ptr<OptimizerClient<T>> opt;
  public:
    Param() = default;
    Param(const std::unique_ptr<Optimizer<T>>& o): value{}, grad{}, opt{o->client()} {}
    Param(const Param&) = default;
    Param(Param&&) = default;
    Param& operator=(const Param&) = default;
    Param& operator=(Param&&) = default;
    ~Param() = default;

    void add_grad(T g){ grad.fetch_add(g); }
    const auto& operator()() const noexcept { return value; }
    void update(){ value += opt->diff(grad.exchange(0)); }
  };


  template<typename T> class Weight {
  private:
    std::vector<Param<T>> w;
    Param<T> b;
  public:
    Weight() = delete;
    Weight(std::size_t N, const std::unique_ptr<Optimizer<T>>& o): w{}, b{o} {
      w.reserve(N);
      for(auto i=0; i<N; ++i){ w.emplace_back(o); }
    }
    Weight(const Weight&) = default;
    Weight(Weight&&) = default;
    Weight& operator=(const Weight&) = default;
    Weight& operator=(Weight&&) = default;
    ~Weight() = default;

    auto weight() const noexcept {
      return Data<T>{w.begin(), w.end(), [](auto& wi){ return wi(); }};
    }
    auto weight(std::size_t i) const { return w[i](); }
    auto bias() const noexcept { return b(); }

    void update(){
      for(auto& wi : w){ wi.update(); }
      b.update();
    }

    void add_weight_grad(std::size_t i, T g){ w[i].add_grad(g); }
    void add_bias_grad(T g){ b.add_grad(g); }

    auto affine(const Data<T>& X, const idx_t& prev_active) const {
      auto result = b();
      for(auto i : prev_active){
	result += w[i]()*X[i];
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
    idx_t idx;
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
      idx_t neuron_id{};
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
    std::vector<data_t> gradient;
    Weight<data_t> weight;
  public:
    Neuron(): Neuron{16};
    Neuron(std::size_t prev_units,
	   std::function<data_t()> weight_initializer = [](){ return 0; })
      : data{}, gradient{}, weight{prev_units}, bias{}
    {}
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
		       const idx_t& prev_active,
		       const std::unique_ptr<Activation<data_t>>& f){
      return f->call(weight.affine(X, prev_active));
    }

    const auto backward(std::size_t batch_i,
			const Data<data_t>& X, data_t y,
			data_t dL_dy, Data<data_t>& dL_dx,
			const std::unique_ptr<Activation<data_t>>& f){
      dL_dy = f->back(y, dL_dy);

      for(auto i : prev_active){
	dL_dx[i] += dL_dy * weight.weight(i);
	weight.add_weight_grad(i, dL_dy * X[i]);
      }
      weight.add_bias_grad(dL_dy);
    }

    const auto& get_weight() const noexcept { return weight; }
  };

  class Layer {
  private:
    Layer* _next;
    Layer* _prev;
  protected:
    std::vector<Data<data_t>> Y;
  public:
    auto next() const noexcept { return _next; }
    auto prev() const noexcept { return _prev; }
    void set_next(Layer* L){ _next = L; }
    void set_prev(Layer* L){ _prev = L; }
    const Data<data_t>& fx(std::size_t batch_i) const { return Y[batch_i]; }
    virtual Data<data_t> forward(std::size_t, const Data<data_t>&) = 0;
    virtual void backward(std::size_t, const Data<data_t>&) = 0;
    virtual const idx_t& active_id(std::size_t) const = 0;
    virtual void reset(std::size_t batch_size){
      Y.clear();
      Y.resize(batch_size);
    }
  };


  class InputLayer : public Layer {
  private:
    idx_t idx;
  public:
    InputLayer() = default;
    InputLayer(std::size_t unit): idx{index_vec(unit)}, Y{} {}
    InputLayer(const InputLayer&) = default;
    InputLayer(InputLayer&&) = default;
    InputLayer& operator=(const InputLayer&) = default;
    InputLayer& operator=(InputLayer&&) = default;
    ~InputLayer() = default;

    virtual Data<data_t> forward(std::size_t batch_i, const Data<data_t>& X) override {
      Y[batch_i] = X;
      return next()->forward(batch_i, X);
    }

    virtual void backward(std::size_t batch_i, const Data<data_t>& dL_dy) override {}

    virtual const idx_t& active_id(std::size_t batch_i) override const {
      return idx;
    }
  };


  class OutputLayer : public Layer {
  private:
    idx_t idx;
  public:
    OutputLayer() = default;
    OutputLayer(std::size_t unit): idx{index_vec(unit)} {}
    OutputLayer(const OutputLayer&) = default;
    OutputLayer(OutputLayer&&) = default;
    OutputLayer& operator=(const OutputLayer&) = default;
    OutputLayer& operator=(OutputLayer&&) = default;
    ~OutputLayer() = default;

    virtual Data<data_t> forward(std::size_t batch_i, const Data<data_t>& X) override {
      Y[batch_i] = X;
      return X;
    }

    virtual void backward(std::size_t batch_i, const Data<data_t>& dL_dy) override {
      prev()->backward(batch_i, dL_dy);
    }

    virtual const idx_t& active_id(std::size_t batch_i) override const {
      return idx;
    }
  };


  class DenseLayer : public Layer {
  private:
    const std::size_t neuron_size;
    std::vector<Neuron> neuron;
    std::vector<idx_t> active_list;
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
				 const Data<data_t>& X) override {
      active_list[batch_i] = hash.retrieve(X);

      for(auto n : active_list[batch_i]){
	Y[batch_i][n] = neuron[n].forward(batch_i, X, activation);
      }

      return next()->forward(batch_i, Y[batch_i]);
    }

    virtual void backward(std::size_t batch_i,
			  const Data<data_t>& dL_dy) override {
      const auto& prev_list = prev()->active_id(batch_i);
      const auto& X = prev()->fx(batch_i);

      Data<data_t> dL_dx{X.get_data_size()};
      for(auto n : active_list[batch_i]){
	this->neuron[n].backward(batch_i, X, Y[n], dL_dy[n], dL_dx, activation);
      }

      prev()->backward(batch_i, dL_dx);
    }

    virtual void reset(std::size_t batch_size) override {
      Layer::reset(batch_i);

      active_list.clear();
      active_list.resize(batch_size);

      for(auto& n : neuron){ neuron.reset(batch_size); }
    }

    virtual const idx_t& active_id(std::size_t batch_i) override const {
      return active_list[batch_i];
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

      idx_t batch_idx{}
      batch_idx.reserve(batch_size);
      std::generate_n(std::back_inserter(batch_idx), batch_size,
		      [i=0]() mutable { return i++; });


      // Parallel Feed-Forward over Batch
      BatchData Y{output_dim, std::vector<data_t>(output_dim * batch_size, 0)};
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<data_t>{X.begin(i), X.end(i)};
		      d = layer.data()->forward(i, d);

		      std::move(d.begin(), d.end(), Y.begin(i));
		    });

      return Y;
    }

    auto backward(const DataView<data_t>& dL_dy){
      const auto batch_size = dL_dy.get_batch_size();

      auto batch_idx = index_vec(batch_size);
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<data_t>{dL_dy.begin(i), dL_dy.end(i)};
		      layer.back()->backward(i, d);
		    });

      std::for_each(std::execution::par, layer.begin(), layer.end(),
		    [](auto& L){ L->update(); });
    }
  };

}
#endif

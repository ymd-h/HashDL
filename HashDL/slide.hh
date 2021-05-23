#ifndef SLIDE_HH
#define SLIDE_HH

#include <execution>
#include <unordered_map>

#include "data.hh"
#include "activation.hh"
#include "optimizer.hh"
#include "hash.hh"
#include "scheduler.hh"

namespace HashDL {
  template<typename T> class Param {
  private:
    T value;
    std::atomic<T> grad;
    std::unique_ptr<OptimizerClient<T>> opt;
  public:
    Param() = delete;
    Param(const std::unique_ptr<Optimizer<T>>& o) : Param{o, T{}} {}
    Param(const std::unique_ptr<Optimizer<T>>& o, T v)
      : value{v}, grad{}, opt{o->client()} {}
    Param(const Param&) = delete;
    Param(Param&&) = delete;
    Param& operator=(const Param&) = default;
    Param& operator=(Param&&) = default;
    ~Param() = default;

    void add_grad(T g){ grad.fetch_add(g); }
    const auto& operator()() const noexcept { return value; }
    void update(){ value += opt->diff(grad.exchange(0)); }
  };


  template<typename T> class Weight {
  private:
    using Param_t = std::unique_ptr<Param<T>>;
    std::vector<Param_t> w;
    Param_t b;
  public:
    Weight() = delete;
    Weight(std::size_t N, const std::unique_ptr<Optimizer<T>>& o)
      : w{}, b{new Param<T>{o}}
    {
      w.reserve(N);
      std::generate_n(std::back_inserter(w), N,
		      [&](){ return Param_t{new Param<T>{o}}; });
    }
    Weight(std::size_t N, const std::unique_ptr<Optimizer<T>>& o,
	   std::function<T(void)> f)
          : w{}, b{new Param<T>{o}}
    {
      w.reserve(N);
      std::generate_n(std::back_inserter(w), N,
		      [&](){ return Param_t{new Param<T>{o, f()}}; });
    }
    Weight(const Weight&) = default;
    Weight(Weight&&) = default;
    Weight& operator=(const Weight&) = default;
    Weight& operator=(Weight&&) = default;
    ~Weight() = default;

    auto weight() const noexcept {
      return Data<T>{w.begin(), w.end(), [](auto& wi){ return (*wi)(); }};
    }
    auto weight(std::size_t i) const { return (*w[i])(); }
    auto bias() const noexcept { return (*b)(); }

    void update(){
      for(auto& wi : w){ wi->update(); }
      b->update();
    }

    void add_weight_grad(std::size_t i, T g){ w[i]->add_grad(g); }
    void add_bias_grad(T g){ b->add_grad(g); }

    auto affine(const Data<T>& X, const idx_t& prev_active) const {
      auto result = (*b)();
      for(auto i : prev_active){
	result += (*w[i])()*X[i];
      }
      return result;
    }
  };


  template<typename T> class Neuron {
  private:
    Weight<T> weight;
  public:
    Neuron(): Neuron{16, new Adam<T>{}} {}
    Neuron(std::size_t prev_units,
	   const std::unique_ptr<Optimizer<T>>& optimizer,
	   std::function<T()> weight_initializer = [](){ return 0; })
      : weight{prev_units, optimizer, weight_initializer} {}
    Neuron(const Neuron&) = default;
    Neuron(Neuron&&) = default;
    Neuron& operator=(const Neuron&) = default;
    Neuron& operator=(Neuron&&) = default;
    ~Neuron() = default;

    const auto forward(const Data<T>& X,
		       const idx_t& prev_active,
		       const std::shared_ptr<Activation<T>>& f){
      return f->call(weight.affine(X, prev_active));
    }

    const auto backward(const Data<T>& X, T y,
			T dL_dy, Data<T>& dL_dx,
			const idx_t& prev_active,
			const std::shared_ptr<Activation<T>>& f){
      dL_dy = f->back(y, dL_dy);

      for(auto i : prev_active){
	dL_dx[i] += dL_dy * weight.weight(i);
	weight.add_weight_grad(i, dL_dy * X[i]);
      }
      weight.add_bias_grad(dL_dy);
    }

    const auto w() const noexcept { return weight.weight(); }

    void update(){ weight.update(); }
  };

  template<typename T> class LSH {
  private:
    const std::size_t L;
    const std::size_t data_size;
    HashFunc<T>* hash_factory;
    using hash_ptr = std::unique_ptr<Hash<T>>;
    std::vector<hash_ptr> hash;
    std::vector<std::unordered_multimap<hashcode_t, std::size_t>> backet;
    idx_t idx;
    std::size_t neuron_size;
  public:
    LSH(): LSH(50, DWTAFunc<T>{8, 8}) {}
    LSH(std::size_t L, std::size_t data_size, HashFunc<T>* hash_factory)
      : L{L}, data_size{data_size}, hash_factory{hash_factory}, hash{}, backet(L),
	idx{index_vec(L)}, neuron_size{}
    {
      hash.reserve(L);
      std::generate_n(std::back_inserter(hash), L,
		      [&](){
			return hash_ptr{hash_factory->GetHash(data_size)};
		      });
    }
    LSH(const LSH&) = default;
    LSH(LSH&&) = default;
    LSH& operator=(const LSH&) = default;
    LSH& operator=(LSH&&) = default;
    ~LSH() = default;

    void reset(){
      for(auto& h : hash){ h.reset(hash_factory->GetHash(data_size)); }

      backet.clear();
      backet.resize(L);

      neuron_size = 0;
    }

    void add(const std::vector<Neuron<T>>& N){
      std::for_each(std::execution::par, idx.begin(), idx.end(),
		    [&N,this](auto i){
		      for(std::size_t n=0, size=N.size(); n<size; ++n){
			this->backet[i].emplace(this->hash[i]->encode(N[n].w()), n);
		      }
		    });
      neuron_size = N.size();
    }

    auto retrieve(const Data<T>& X) const {
      auto neuron_id = index_vec(neuron_size);

      for(std::size_t i=0; i<L; ++i){
	auto [begin, end] = backet[i].equal_range(hash[i]->encode(X));
	std::remove_if(neuron_id.begin(), neuron_id.end(),
		       [=](auto n){
			 auto f = [=](auto& v){ return v.second == n; };
			 return std::find_if(begin, end, f) == end;
		       });
      }

      return neuron_id;
    }
  };


  template<typename T> class Layer {
  private:
    std::weak_ptr<Layer<T>> _next;
    std::weak_ptr<Layer<T>> _prev;
  protected:
    std::vector<Data<T>> Y;
  public:
    auto next() const noexcept { return _next.lock(); }
    auto prev() const noexcept { return _prev.lock(); }
    void set_next(const std::shared_ptr<Layer<T>>& L){ _next = L; }
    void set_prev(const std::shared_ptr<Layer<T>>& L){ _prev = L; }
    const Data<T>& fx(std::size_t batch_i) const { return Y[batch_i]; }
    virtual Data<T> forward(std::size_t, const Data<T>&) = 0;
    virtual void backward(std::size_t, const Data<T>&) = 0;
    virtual const idx_t& active_id(std::size_t) const = 0;
    virtual void reset(std::size_t batch_size){
      Y.clear();
      Y.resize(batch_size);
    }
    virtual void update(bool){}
    virtual std::string to_string() const {
      return "Layer";
    }
  };

  template<typename T> inline auto to_string(const Layer<T>& layer){
    return (&layer)->to_string();
  }

  template<typename T> class InputLayer : public Layer<T> {
  private:
    idx_t idx;
  public:
    InputLayer() = default;
    InputLayer(std::size_t units): idx{index_vec(units)} {}
    InputLayer(const InputLayer&) = default;
    InputLayer(InputLayer&&) = default;
    InputLayer& operator=(const InputLayer&) = default;
    InputLayer& operator=(InputLayer&&) = default;
    ~InputLayer() = default;

    Data<T> forward(std::size_t batch_i, const Data<T>& X) override {
      this->Y[batch_i] = X;
      return this->next()->forward(batch_i, X);
    }

    void backward(std::size_t /* batch_i */, const Data<T>& /* dL_dy */) override {}

    const idx_t& active_id(std::size_t /* batch_i */) const override { return idx; }
  };


  template<typename T> class OutputLayer : public Layer<T> {
  private:
    idx_t idx;
  public:
    OutputLayer() = default;
    OutputLayer(std::size_t units): idx{index_vec(units)} {}
    OutputLayer(const OutputLayer&) = default;
    OutputLayer(OutputLayer&&) = default;
    OutputLayer& operator=(const OutputLayer&) = default;
    OutputLayer& operator=(OutputLayer&&) = default;
    ~OutputLayer() = default;

    Data<T> forward(std::size_t batch_i, const Data<T>& X) override {
      this->Y[batch_i] = X;
      return X;
    }

    void backward(std::size_t batch_i, const Data<T>& dL_dy) override {
      this->prev()->backward(batch_i, dL_dy);
    }

    const idx_t& active_id(std::size_t /* batch_i */) const override { return idx; }
  };


  template<typename T> class DenseLayer : public Layer<T> {
  private:
    std::vector<Neuron<T>> neuron;
    std::vector<idx_t> active_idx;
    LSH<T> hash;
    std::shared_ptr<Activation<T>> activation;
  public:
    DenseLayer()
      : DenseLayer{30, new ReLU<T>{}, 50, new WTAFunc<T>{}, std::shared_ptr<Optimizer<T>>{new Adam<T>{}}}{}
    DenseLayer(std::size_t prev_units, std::size_t units,
	       std::shared_ptr<Activation<T>> f,
	       std::size_t L, HashFunc<T>* hash_factory,
	       const std::unique_ptr<Optimizer<T>>& optimizer)
      : neuron{}, active_idx{}, hash{L, prev_units, hash_factory}, activation{f}
    {
      neuron.reserve(units);
      std::generate_n(std::back_inserter(neuron), units,
		      [&](){ return Neuron<T>{prev_units, optimizer}; });

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

    Data<T> forward(std::size_t batch_i, const Data<T>& X) override {
      active_idx[batch_i] = hash.retrieve(X);

      for(auto n : active_idx[batch_i]){
	this->Y[batch_i][n] = neuron[n].forward(X, this->prev()->active_id(batch_i),
						activation);
      }

      return this->next()->forward(batch_i, this->Y[batch_i]);
    }

    void backward(std::size_t batch_i, const Data<T>& dL_dy) override {
      const auto& X = this->prev()->fx(batch_i);

      Data<T> dL_dx{X.size()};
      for(auto n : active_idx[batch_i]){
	this->neuron[n].backward(X, this->Y[batch_i][n], dL_dy[n], dL_dx,
				 this->prev()->active_id(batch_i), activation);
      }

      this->prev()->backward(batch_i, dL_dx);
    }

    void reset(std::size_t batch_size) override {
      Layer<T>::reset(batch_size);

      active_idx.clear();
      active_idx.resize(batch_size);
    }

    const idx_t& active_id(std::size_t batch_i) const override {
      return active_idx[batch_i];
    }

    void update(bool is_rehash) override {
      for(auto& n: neuron){ n.update(); }
      if(is_rehash){ rehash(); }
    }
  };


  template<typename T> class Network {
  private:
    std::size_t output_dim;
    std::vector<std::shared_ptr<Layer<T>>> layer;
    std::shared_ptr<Optimizer<T>> opt;
    std::shared_ptr<Scheduler> update_freq;
  public:
    Network() = delete;
    Network(std::size_t input_size, std::vector<std::size_t> units, std::size_t L,
	    HashFunc<T>* hash, std::shared_ptr<Optimizer<T>> opt,
	    std::shared_ptr<Scheduler> update_freq)
      : output_dim{units.size() > 0 ? units.back(): input_size}, layer{},
	opt{opt}, update_freq{update_freq}
    {
      layer.reserve(units.size() + 2);

      layer.emplace_back(new InputLayer<T>{input_size});
      auto prev_units = input_size;
      for(auto& u : units){
	layer.emplace_back(new DenseLayer<T>{prev_units, u,
					     std::shared_ptr<Activation<T>>{new ReLU<T>{}},
					     L, hash, this->opt});
	prev_units = u;
	auto last = layer.size() -1;
	layer[last]->set_prev(layer[last-1]);
	layer[last-1]->set_next(layer[last]);
      }
      layer.emplace_back(new OutputLayer<T>{prev_units});
    }
    Network(const Network&) = default;
    Network(Network&&) = default;
    Network& operator=(const Network&) = default;
    Network& operator=(Network&&) = default;
    ~Network() = default;

    auto operator()(const BatchView<T>& X){
      const auto batch_size = X.get_batch_size();

      for(auto& L: layer){ L->reset(batch_size); }

      auto batch_idx = index_vec(batch_size);

      // Parallel Feed-Forward over Batch
      BatchData<T> Y{output_dim, batch_size, 0};
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<T>{X.begin(i), X.end(i)};
		      d = layer.front()->forward(i, d);

		      std::move(d.begin(), d.end(), Y.begin(i));
		    });

      return Y;
    }

    auto backward(const BatchView<T>& dL_dy){
      const auto batch_size = dL_dy.get_batch_size();

      auto batch_idx = index_vec(batch_size);
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<T>{dL_dy.begin(i), dL_dy.end(i)};
		      this->layer.back()->backward(i, d);
		    });

      opt->step();

      auto is_rehash = (*update_freq)();
      std::for_each(std::execution::par, layer.begin(), layer.end(),
		    [=](auto& L){ L->update(is_rehash); });
    }
  };

}
#endif

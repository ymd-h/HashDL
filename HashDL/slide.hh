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


  template<typename T> class LSH {
  private:
    const std::size_t L;
    std::function<Hash*()> hash_factory;
    std::vector<std::unique_ptr<Hash>> hash;
    std::vector<std::unordered_multimap<hashcode_t, std::size_t>> backet;
    idx_t idx;
    std::size_t neuron_size;
  public:
    LSH(): LSH(50, DWTA<T>::make_factory(8, 16, 8)){}
    LSH(std::size_t L, std::function<Hash*()> hash_factory)
      : L{L}, hash_factory{hash_factory}, hash{}, backet(L),
	idx{index_vec(L)}, neuron_size{}
    {
      hash.reserve(L);
      std::generate_n(std::back_inserter(hash), L,
		      [&](){ return std::unique_ptr<Hash>{hash_factory()}; });
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

    void add(const std::vector<Neuron<T>>& N){
      std::for_each(std::execution::par, idx.begin(), idx.end(),
		    [&N,this](auto i){
		      for(std::size_t n=0, size=N.size(); n<size; ++n){
			this->backet[i].insert(this->hash[i]->encode(N[n].w()), n);
		      }
		    });
      neuron_size = N.size();
    }

    auto retrieve(const Data<T>& X) const {
      auto neuron_id = index_vec(neuron_size);

      for(auto i=0; i<L; ++i){
	auto [begin, end] = backet[i].equal_range(hash[i]->encode(X));
	std::remove_if(neuron_id.begin(), neuron_id.end(),
		       [=](auto n){ return std::find(begin, end, n) == end; });
      }

      return neuron_id;
    }
  };


  template<typename T> class Neuron {
  private:
    Weight<T> weight;
  public:
    Neuron(): Neuron{16};
    Neuron(std::size_t prev_units,
	   std::function<T()> weight_initializer = [](){ return 0; })
      : data{}, gradient{}, weight{prev_units}, bias{}
    {}
    Neuron(const Neuron&) = default;
    Neuron(Neuron&&) = default;
    Neuron& operator=(const Neuron&) = default;
    Neuron& operator=(Neuron&&) = default;
    ~Neuron() = default;

    const auto forward(const Data<T>& X,
		       const idx_t& prev_active,
		       const std::unique_ptr<Activation<T>>& f){
      return f->call(weight.affine(X, prev_active));
    }

    const auto backward(const Data<T>& X, T y,
			T dL_dy, Data<T>& dL_dx,
			const idx_t& prev_active,
			const std::unique_ptr<Activation<T>>& f){
      dL_dy = f->back(y, dL_dy);

      for(auto i : prev_active){
	dL_dx[i] += dL_dy * weight.weight(i);
	weight.add_weight_grad(i, dL_dy * X[i]);
      }
      weight.add_bias_grad(dL_dy);
    }

    const auto w() const noexcept { return weight.weight(); }

    void update(){ weight->update(); }
  };


  template<typename T> class Layer {
  private:
    Layer<T>* _next;
    Layer<T>* _prev;
  protected:
    std::vector<Data<T>> Y;
  public:
    auto next() const noexcept { return _next; }
    auto prev() const noexcept { return _prev; }
    void set_next(Layer<T>* L){ _next = L; }
    void set_prev(Layer<T>* L){ _prev = L; }
    const Data<T>& fx(std::size_t batch_i) const { return Y[batch_i]; }
    virtual Data<T> forward(std::size_t, const Data<T>&) = 0;
    virtual void backward(std::size_t, const Data<T>&) = 0;
    virtual const idx_t& active_id(std::size_t) const = 0;
    virtual void reset(std::size_t batch_size){
      Y.clear();
      Y.resize(batch_size);
    }
    virtual void update(){}
  };


  template<typename T> class InputLayer : public Layer<T> {
  private:
    idx_t idx;
  public:
    InputLayer() = default;
    InputLayer(std::size_t units): idx{index_vec(units)}, Y{} {}
    InputLayer(const InputLayer>&) = default;
    InputLayer(InputLayer&&) = default;
    InputLayer& operator=(const InputLayer&) = default;
    InputLayer& operator=(InputLayer&&) = default;
    ~InputLayer() = default;

    virtual Data<T> forward(std::size_t batch_i, const Data<T>& X) override {
      Y[batch_i] = X;
      return next()->forward(batch_i, X);
    }

    virtual void backward(std::size_t batch_i, const Data<T>& dL_dy) override {}

    virtual const idx_t& active_id(std::size_t batch_i) override const { return idx; }
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

    virtual Data<T> forward(std::size_t batch_i, const Data<T>& X) override {
      Y[batch_i] = X;
      return X;
    }

    virtual void backward(std::size_t batch_i, const Data<T>& dL_dy) override {
      prev()->backward(batch_i, dL_dy);
    }

    virtual const idx_t& active_id(std::size_t batch_i) override const { return idx; }
  };


  template<typename T> class DenseLayer : public Layer<T> {
  private:
    std::vector<Neuron<T>> neuron;
    std::vector<idx_t> active_idx;
    LSH<T> hash;
    std::unique_ptr<Activation<T>> activation;
  public:
    DenseLayer(): DenseLayer{30}{}
    DenseLayer(std::size_t prev_units, std::size_t units, Activation<T>* f,
	       std::size_t L, std::function<Hash*()> hash_factory)
      : neuron(units, Neuron{prev_units}), active_idx{},
	hash{L, hash_factory}, activation{f}
    {
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

    virtual Data<T> forward(std::size_t batch_i,
			    const Data<T>& X) override {
      active_idx[batch_i] = hash.retrieve(X);

      for(auto n : active_idx[batch_i]){
	Y[batch_i][n] = neuron[n].forward(X, prev()->active_id(batch_i), activation);
      }

      return next()->forward(batch_i, Y[batch_i]);
    }

    virtual void backward(std::size_t batch_i,
			  const Data<T>& dL_dy) override {
      const auto& X = prev()->fx(batch_i);

      Data<T> dL_dx{X.size()};
      for(auto n : active_idx[batch_i]){
	this->neuron[n].backward(X, Y[n], dL_dy[n], dL_dx,
				 prev()->active_id(batch_i), activation);
      }

      prev()->backward(batch_i, dL_dx);
    }

    virtual void reset(std::size_t batch_size) override {
      Layer::reset(batch_i);

      active_idx.clear();
      active_idx.resize(batch_size);
    }

    virtual const idx_t& active_id(std::size_t batch_i) override const {
      return active_idx[batch_i];
    }

    virtual void update() override { for(auto& n: neuron){ n->update(); } }
  };


  template<typename T> class Network {
  private:
    std::size_t output_dim;
    std::vector<std::unique_ptr<Layer<T>>> layer;
    std::unique_ptr<Optimizer<T>> opt;
  public:
    Network() = delete;
    Network(std::size_t input_size, std::vector<std::size_t> units,
	    Optimizer* opt)
      : output_dim{units.size() > 0 ? units.back(): input_size}, layer{}, opt{opt} {
      layer.reserve(units.size() + 2);

      layer.emplace_back(new InputLayer{input_size});
      auto prev_units = input_size;
      for(auto& u : units){
	layer.emplace_back(new DenseLayer{prev_units, u, new ReLU{}});
	prev_units = u;
      }
      layer.emplace_back(new OutputLayer{prev_units});
    }
    Network(const Network&) = default;
    Network(Network&&) = default;
    Network& operator=(const Network&) = default;
    Network& operator=(Network&&) = default;
    ~Network() = default;

    auto operator()(const DataView<T>& X){
      const auto batch_size = X.get_batch_size();

      for(auto& L: layer){ L->reset(batch_size); }

      auto batch_idx = index_vec(batch_size);

      // Parallel Feed-Forward over Batch
      BatchData Y{output_dim, batch_size, 0};
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<T>{X.begin(i), X.end(i)};
		      d = layer.data()->forward(i, d);

		      std::move(d.begin(), d.end(), Y.begin(i));
		    });

      return Y;
    }

    auto backward(const DataView<T>& dL_dy){
      const auto batch_size = dL_dy.get_batch_size();

      auto batch_idx = index_vec(batch_size);
      std::for_each(std::execution::par, batch_idx.begin(), batch_idx.end(),
		    [&, this](auto i){
		      auto d = Data<T>{dL_dy.begin(i), dL_dy.end(i)};
		      layer.back()->backward(i, d);
		    });

      opt->step();
      std::for_each(std::execution::par, layer.begin(), layer.end(),
		    [](auto& L){ L->update(); });
    }
  };

}
#endif

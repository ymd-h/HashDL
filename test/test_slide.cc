#include <slide.hh>

#include "unittest.hh"

int main(int, char**){
  using namespace HashDL;

  auto test = Test{};
  auto opt = std::shared_ptr<Optimizer<float>>(new SGD<float>{1});
  auto a = std::shared_ptr<Activation<float>>{new Linear<float>{}};
  auto wta = WTAFunc<float>{8, 1};
  auto sch = std::shared_ptr<Scheduler>{new ConstantFrequency{1}};

  test.Add([&](){
    auto p = Param<float>{opt};

    AssertEqual(p(), 0);

    p.add_grad(0.5);
    AssertEqual(p(), 0);

    p.update();
    AssertEqual(p(), -0.5);
  }, "Param");

  test.Add([&](){
    auto p = Param<float>{opt, 0.5};

    AssertEqual(p(), 0.5);

    p.add_grad(0.5);
    AssertEqual(p(), 0.5);

    p.update();
    AssertEqual(p(), 0);
  }, "Param with initialization");

  test.Add([&](){
    auto w = Weight<float>{1, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);
  }, "Weight");

  test.Add([&](){
    auto w = Weight<float>{1, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);

    w.add_weight_grad(0, 0.5);
    w.add_bias_grad(0.2);
    AssertEqual(w.weight(), std::vector<float>{0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), 0);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), 0);

    w.update();
    AssertEqual(w.weight(), std::vector<float>{-0.5});
    AssertEqual(w.weight(0), -0.5);
    AssertEqual(w.bias(), -0.2);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{0}), -0.2);
    AssertEqual(w.affine(Data<float>{1}, std::vector<std::size_t>{}), -0.2);
    AssertEqual(w.affine(Data<float>{std::vector<float>{1}},
			 std::vector<std::size_t>{0}), -0.7);
    AssertEqual(w.affine(Data<float>{std::vector<float>{1}},
			 std::vector<std::size_t>{}), -0.2);
  }, "Weight update");

  test.Add([&](){
    auto w = Weight<float>{2, opt};

    AssertEqual(w.weight(), std::vector<float>{0.0, 0.0});
    AssertEqual(w.weight(0), 0);
    AssertEqual(w.weight(1), 0);
    AssertEqual(w.bias(), 0);
    AssertEqual(w.affine(Data<float>{2}, std::vector<std::size_t>{0, 1}), 0);
    AssertEqual(w.affine(Data<float>{2}, std::vector<std::size_t>{}), 0);
  }, "Multi dimension");

  test.Add([&](){
    auto w = Weight<float>{1, opt, [](){ return 0.5; }};

    AssertEqual(w.weight(), std::vector<float>{0.5});
    AssertEqual(w.weight(0), 0.5);
    AssertEqual(w.bias(), 0);
  }, "Weight initialization");

  test.Add([&](){
    auto N = Neuron<float>{1, opt};

    AssertEqual(N.w(), Data<float>{1});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);
  }, "Neuron");

  test.Add([&](){
    auto N = Neuron<float>{3, opt};

    AssertEqual(N.w(), Data<float>{3});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0, 1}, a), 0);
  }, "Neuron multi prev");

  test.Add([&](){
    auto N = Neuron<float>{1, opt};

    AssertEqual(N.w(), Data<float>{1});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);

    auto dL_dx = Data<float>{1};
    N.backward(Data<float>{1}, 0, 1.0, dL_dx,
	       std::vector<std::size_t>{0}, a);
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), 0);
    AssertEqual(dL_dx, std::vector<float>{0.0});
    N.update();
    AssertEqual(N.w(), Data<float>{std::vector<float>{0}});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), -1.0);
  }, "Neuron backward");

  test.Add([&](){
    auto N = Neuron<float>{1, opt};

    auto x = Data<float>{std::vector<float>{1.0}};
    AssertEqual(N.w(), Data<float>{1});
    AssertEqual(N.forward(x, std::vector<std::size_t>{} , a), 0);
    AssertEqual(N.forward(x, std::vector<std::size_t>{0}, a), 0);

    auto dL_dx = Data<float>{1};
    auto dL_dy = 1.0;
    N.backward(x, 0, dL_dy, dL_dx, std::vector<std::size_t>{0}, a);
    AssertEqual(N.forward(x, std::vector<std::size_t>{0}, a), 0);
    AssertEqual(dL_dx, std::vector<float>{0.0});
    N.update();
    AssertEqual(N.w(), std::vector<float>{-1.0});
    AssertEqual(N.forward(Data<float>{1}, std::vector<std::size_t>{0}, a), -1.0);
    AssertEqual(N.forward(x, std::vector<std::size_t>{0}, a), -2.0);
  }, "Neuron backward with weight");

  test.Add([&](){
    auto L = 50;
    auto d = 2;
    auto func = new WTAFunc<float>{8, 1};
    auto lsh = LSH<float>{L, d, func};
    auto N = std::vector<Neuron<float>>{};
    N.emplace_back(d, opt);

    lsh.add(N);

    if(func){
      delete func;
      func = nullptr;
    }
  }, "LSH");

  test.Add([&](){
    auto L = 50;
    auto d = 2;
    auto func = new WTAFunc<float>{8, 1};
    auto lsh = LSH<float>{L, d, func};
    auto N = std::vector<Neuron<float>>{};
    N.emplace_back(d, opt);
    lsh.add(N);

    lsh.reset();
    lsh.add(N);

    if(func){
      delete func;
      func = nullptr;
    }

  }, "LSH reset");

  test.Add([&](){
    auto L = 50;
    auto d = 2;
    auto func = new WTAFunc<float>{8, 1};
    auto lsh = LSH<float>{L, d, func};
    auto N = std::vector<Neuron<float>>{};
    N.emplace_back(d, opt);
    lsh.add(N);

    auto x = Data<float>{d};
    AssertEqual(lsh.retrieve(x), lsh.retrieve(x));

    lsh.reset();
    lsh.add(N);

    AssertEqual(lsh.retrieve(x), lsh.retrieve(x));

    if(func){
      delete func;
      func = nullptr;
    }
  }, "LSH retrieve");

  test.Add([&](){
    auto dsize = 1;
    auto input = std::make_shared<InputLayer<float>>(dsize);
    auto output = std::make_shared<OutputLayer<float>>(dsize);

    input->set_next(output);
    output->set_prev(input);

    AssertEqual(input->next(), output);
    AssertEqual(output->prev(), input);
  }, "Layers set");

  test.Add([&](){
    auto dsize = 1;
    auto input = std::make_shared<InputLayer<float>>(dsize);
    auto output = std::make_shared<OutputLayer<float>>(dsize);

    input->set_next(output);
    output->set_prev(input);

    auto x = Data<float>{dsize};
    input->reset(x.size());
    output->reset(x.size());
    AssertEqual(output->forward(0, x), x);
    AssertEqual(input->forward(0, x), x);
  }, "Layers forward");

  test.Add([&](){
    auto dsize = 1;
    auto input = std::make_shared<InputLayer<float>>(dsize);
    auto output = std::make_shared<OutputLayer<float>>(dsize);

    input->set_next(output);
    output->set_prev(input);

    auto x = Data<float>{dsize};
    input->reset(x.size());
    output->reset(x.size());
    input->backward(0, x);
    output->backward(0, x);
  }, "Layers backward");

  test.Add([&](){
    auto dsize = 1;
    auto input = std::make_shared<InputLayer<float>>(dsize);
    auto output = std::make_shared<OutputLayer<float>>(dsize);

    auto x = Data<float>{dsize};
    input->reset(x.size());
    output->reset(x.size());

    AssertEqual(input->active_id(0), std::vector<std::size_t>{0});
    AssertEqual(output->active_id(0), std::vector<std::size_t>{0});
  }, "Layer active index");

  test.Add([&](){
    auto dsize = 1;
    auto L = 5;
    auto input = std::shared_ptr<Layer<float>>{new InputLayer<float>{dsize}};
    auto hidden = std::shared_ptr<Layer<float>>{new DenseLayer<float>{dsize, dsize, a, L, &wta, opt}};
    auto output = std::shared_ptr<Layer<float>>{new OutputLayer<float>{dsize}};

    input->set_next(hidden);
    hidden->set_prev(input);

    hidden->set_next(output);
    output->set_prev(hidden);

    auto x = Data<float>{dsize};
    input->reset(x.size());
    hidden->reset(x.size());
    output->reset(x.size());

    AssertEqual(input->forward(0, x), x);
    output->backward(0, x);
  }, "Dense Layer");

  test.Add([&](){
    auto Net = Network<float>(1, std::vector<std::size_t>{1}, 10, &wta, opt, sch);

    auto x = std::vector<float>{0};
    auto x1 = BatchView<float>{1, 1, x.data()};
    AssertEqual(x1, x1);
    AssertEqual(Net(x1), x1);

    auto xx = std::vector<float>{0, 0};
    auto x2 = BatchView<float>{1, 2, x.data()};
    AssertEqual(x2, x2);
    AssertEqual(Net(x2), x2);
  }, "Network");

  test.Add([&](){
    auto Net = Network<float>(1, std::vector<std::size_t>{1}, 10, &wta, opt, sch);
    auto x = std::vector<float>{0};
    auto x1 = BatchView<float>{1, 1, x.data()};
    AssertEqual(Net(x1), x1);

    auto y = std::vector<float>{1};
    auto y1 = BatchView<float>{1, 1, y.data()};
    Net.backward(y1);
    AssertEqual(Net(x1), std::vector<float>{-1});
  }, "Network backward");

  return test.Run();
}

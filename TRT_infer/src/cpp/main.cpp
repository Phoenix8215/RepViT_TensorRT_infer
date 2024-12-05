#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const* argv[]) {
    /*这么实现目的在于让调用的整个过程精简化*/
    string onnxPath = "models/onnx/repvit_m2_3_distill_450e.onnx";

    auto level  = logger::Level::VERB;
    auto params = model::Params();

    params.img     = {224, 224, 3};
    params.num_cls = 1000;
    params.task    = model::task_type::CLASSIFICATION;
    params.dev     = model::device::GPU;
    params.prec    = model::precision::FP32;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker = thread::create_worker(onnxPath, level, params);

    // 根据worker中的task类型进行推理
    worker->inference("data/cat.png");
    worker->inference("data/gazelle.png");
    worker->inference("data/eagle.png");
    worker->inference("data/fox.png");
    worker->inference("data/tiny-cat.png");
    worker->inference("data/wolf.png");

    return 0;
}

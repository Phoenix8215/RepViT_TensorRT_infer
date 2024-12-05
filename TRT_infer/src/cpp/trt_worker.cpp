#include "trt_worker.hpp"
#include "memory"
#include "trt_classifier.hpp"
#include "trt_logger.hpp"

using namespace std;

namespace thread {

Worker::Worker(string onnxPath, logger::Level level, model::Params params) {
    m_logger = logger::create_logger(level);

    // 这里根据task_type选择创建的trt_model的子类，今后会针对detection, segmentation扩充
    if (params.task == model::task_type::CLASSIFICATION)
        m_classifier = model::classifier::make_classifier(onnxPath, level, params);
}

void Worker::inference(string imagePath) {
    if (m_classifier != nullptr) {
        m_classifier->load_image(imagePath);
        m_classifier->inference();
    }
}

shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params) {
    // 使用智能指针来创建一个实例
    return make_shared<Worker>(onnxPath, level, params);
}

}; // namespace thread

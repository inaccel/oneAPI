#include <CL/sycl.hpp>

using namespace sycl;

void RunGradients(queue& q, buffer<int8,1> *buf_label, buffer<float16,1> *buf_features, buffer<float16,1> *buf_weights,
                 buffer<float16,1> *buf_gradients, const int numClasses, const int numFeatures, const int numExamples,
				 event &_event, size_t engineID);

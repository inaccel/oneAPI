#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Gradients.hpp"

#define numClassesMax 32
#define numFeaturesMax 1023

#define numFeaturesPlusOneMax 64 //16 * 64 = 1024

#define chunk 8
#define parallelism 8

#define NUM_ENGINES 2

// Forward declaration of the kernel name reduces name mangling
template <int engineID>
class Gradient;

template <int engineID>
void SubmitTaskGradient(queue& q, buffer<int8,1> *buf_label, buffer<float16,1> *buf_features, buffer<float16,1> *buf_weights,
                 buffer<float16,1> *buf_gradients, const int numClasses, const int numFeatures, const int numExamples,event &_event){

    // submit the kernel
    _event = q.submit([&](handler &h) {
      // Data accessors
      auto accessor_label = buf_label->get_access<cl::sycl::access::mode::read>(h);
      auto accessor_features = buf_features->get_access<cl::sycl::access::mode::read>(h);
      auto accessor_weights = buf_weights->get_access<cl::sycl::access::mode::read>(h);
      auto accessor_gradients = buf_gradients->get_access<access::mode::discard_write>(h);


      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that buffers do not alias.
      h.single_task<Gradient<engineID>>([=]() [[intel::kernel_args_restrict]] {
	// KERNEL CODE START
	auto _label = accessor_label.get_pointer();
      	auto _features = accessor_features.get_pointer();
      	auto _weights = accessor_weights.get_pointer();
      	auto _gradients = accessor_gradients.get_pointer();
//	float16 __attribute__((numbanks(8), bankwidth(64), singlepump, numreadports(1), numwriteports(1))) features[numFeaturesPlusOneMax][chunk];
	float16 weights[numFeaturesPlusOneMax][numClassesMax];
	float16 gradients[numFeaturesPlusOneMax][numClassesMax];

//	float16 __attribute__((numbanks(1), bankwidth(64), singlepump, numreadports(2), numwriteports(1))) dotproduct[numClassesMax][chunk];
//	float __attribute__((numbanks(chunk), bankwidth(64), singlepump, numreadports(1), numwriteports(1))) prediction[numClassesMax][chunk];

	float16 features[numFeaturesPlusOneMax][chunk];
	float16 dotproduct[numClassesMax][chunk];
	float prediction[numClassesMax][chunk];
	int roundTo_0 = (numFeatures + 1 + (16 - 1)) & (~(16 - 1));
	int numFeaturesPlusOne = roundTo_0 >> 4;
	int numClassesMin = (13 > numClasses) ? 13 : numClasses;
        int roundTo_1 = (numExamples + (chunk - 1)) & (~(chunk - 1));
	int numChunks = roundTo_1 >> 3;

	#pragma unroll 1
	for (int kj = 0, k = 0, j = 0; kj < numClasses * numFeaturesPlusOne; kj++, j++){
			if (j == numFeaturesPlusOne) {k++; j = 0;}
			weights[j][k] = _weights[kj];
			gradients[j][k] = 0.0f;
	}

	#pragma unroll 1
	for (int i = 0; i < numChunks; i++){

		int offset = (i * chunk) * numFeaturesPlusOne;

		#pragma unroll 1
		for (int c = 0; c < chunk; c++){
			#pragma unroll 1
			for (int j = 1; j < numFeaturesPlusOne; j++){
				 features[j][c] = _features[offset + c * numFeaturesPlusOne + j];
			}
		}

		#pragma unroll 1
		for (int k = 0; k < numClasses; k++){
			#pragma unroll
			for (int c = 0; c < chunk; c++){
				dotproduct[k][c] = 0.0f;
			}
		}

		#pragma unroll 1
		for (int j = 0; j < numFeaturesPlusOne; j++){
			#pragma unroll 1
			for (int k = 0; k < numClassesMin; k++){
				#pragma unroll
				for (int c = 0; c < chunk; c++){
					dotproduct[k][c] += features[j][c] * weights[j][k];
				}
			}
		}

		#pragma unroll 1
		for (int k = 0; k < numClasses; k++){
			#pragma unroll
			for (int c = 0; c < chunk; c++){
				float16 vector = dotproduct[k][c];

				float8 tmp0 = vector.odd() + vector.even();

				float4 tmp1 = tmp0.odd() + tmp0.even();

				float2 tmp2 = tmp1.odd() + tmp1.even();

				float scalar = tmp2.odd() + tmp2.even();
				prediction[k][c] = 1.0 / (1.0 + exp(-scalar));
			}
		}

		int8 labels = _label[i];
		#pragma unroll
		for (int c = 0; c < chunk; c++){
			prediction[labels[c]][c] -= 1.0;
		}

		#pragma unroll 1
		for (int j = 0; j < numFeaturesPlusOne; j++){
			#pragma unroll 1
			for (int k = 0; k < numClassesMin; k++){
				#pragma unroll
				for (int c = 0; c < chunk; c++){
					gradients[j][k] += prediction[k][c] * features[j][c];
				}
			}
		}
	}

	#pragma unroll 1
	for (int kj = 0, k = 0, j = 0; kj < numClasses * numFeaturesPlusOne; kj++, j++){
			if (j == numFeaturesPlusOne) {k++; j = 0;}
			_gradients[kj] = gradients[j][k];
	}

	// KERNEL CODE END
      });
    });
}

void RunGradients(queue& q, buffer<int8,1> *buf_label, buffer<float16,1> *buf_features, buffer<float16,1> *buf_weights, buffer<float16,1> *buf_gradients, const int numClasses, const int numFeatures, const int numExamples, event &_event, size_t engineID){
	if (engineID == 0) {
	    SubmitTaskGradient<0>(q, buf_label, buf_features, buf_weights, buf_gradients, numClasses, numFeatures, numExamples, _event);
	}

	#if NUM_ENGINES > 1
	  if (engineID == 1) {
	    SubmitTaskGradient<1>(q, buf_label, buf_features, buf_weights, buf_gradients, numClasses, numFeatures, numExamples, _event);
	  }
	#endif

	#if NUM_ENGINES > 2
	  if (engineID == 2) {
	    SubmitTaskGradient<2>(q, buf_label, buf_features, buf_weights, buf_gradients, numClasses, numFeatures, numExamples, _event);
	  }
	#endif

	#if NUM_ENGINES > 3
	  if (engineID == 3) {
	    SubmitTaskGradient<3>(q, buf_label, buf_features, buf_weights, buf_gradients, numClasses, numFeatures, numExamples, _event);
	  }
	#endif
}

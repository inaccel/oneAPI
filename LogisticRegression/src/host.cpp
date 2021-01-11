#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"
#include <chrono>

#include "Gradients.hpp"

//using namespace sycl;

#ifndef _TEST_
#define _accel_ 1
#else
#define _accel_ 0
#endif

#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <malloc.h>
#include <math.h>
#include <sstream>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iomanip>
#include <time.h>

//using namespace std;
using std::vector;
using std::cout;
using std::ios;
using std::endl;
using std::string;
using std::fstream;
using std::stringstream;
using std::ifstream;
using std::ofstream;


// Dataset specific options
// Change below definitions according to your input dataset
#define NUMCLASSES 26
#define NUMFEATURES 784
#define NUMEXAMPLES 124800
#define NUM_KERNELS 2

size_t SyclGetExecTimeNs(event e) {
  size_t start_time =
      e.get_profiling_info<info::event_profiling::command_start>();
  size_t end_time = e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}

// Function to allocate an aligned memory buffer
void *INalligned_malloc(size_t size) {
  void *ptr = memalign(4096, size);
  if (!ptr) {
    printf("Error: alligned_malloc\n");
    exit(EXIT_FAILURE);
  }

  return ptr;
}

// Function to split a string on specified delimiter
vector<string> split(const string &s) {
  vector<string> elements;
  stringstream ss(s);
  string item;

  while (getline(ss, item)) {
    size_t prev = 0;
    size_t pos;

    while ((pos = item.find_first_of(" (,[])=", prev)) != std::string::npos) {
      if (pos > prev)
        elements.push_back(item.substr(prev, pos - prev));
      prev = pos + 1;
    }

    if (prev < item.length())
      elements.push_back(item.substr(prev, std::string::npos));
  }

  return elements;
}

// Reads the input dataset and sets features and labels buffers accordingly
void read_input(string filename, float *features, int *labels, int numFeatures,
                int numExamples) {
  ifstream train;
  train.open(filename.c_str());
  if(!train) {
    cout << "Failed to open train file..." << std::endl;
    exit(0);
  }

  string line;
  int i;
  int n = 0;

  while (getline(train, line) && (n < numExamples)) {
    if (line.length()) {
      vector<string> tokens = split(line);
      features[n * (16 + numFeatures) + numFeatures] = 1.0;
      labels[n] = atoi(tokens[0].c_str());
      for (i = 0; i < numFeatures; i++) {
        features[n * (16 + numFeatures) + i] = atof(tokens[i + 1].c_str());
      }
      n++;
    }
  }

  train.close();
}

// Writes a trained model to the specified filename
void write_output(string filename, float *weights, int numClasses,
                  int numFeatures) {

  ofstream results;
  results.open(filename.c_str());
  if(!results) {
    cout << "Failed to open train file..." << std::endl;
    exit(0);
  }

  for (int k = 0; k < numClasses; k++) {
    results << weights[k * (16 + numFeatures)];
    for (int j = 1; j < (16 + numFeatures); j++) {
      results << "," << weights[k * (16 + numFeatures) + j];
    }
    results << std::endl;
  }

  results.close();
}

// A simple classifier. Given an point it matches the class with the greatest
// probability
int classify(float *features, float *weights, int numClasses, int numFeatures) {
  float prob = -1.0;
  int prediction = -1;

  for (int k = 0; k < numClasses; k++) {
    float dot = weights[k * (16 + numFeatures) + numFeatures];

    for (int j = 0; j < numFeatures; j++) {
      dot += features[j] * weights[k * (16 + numFeatures) + j];
    }

    if (1.0 / (1.0 + exp(-dot)) > prob) {
      prob = 1.0 / (1.0 + exp(-dot));
      prediction = k;
    }
  }

  return prediction;
}

// A simple prediction function to evaluate the accuracy of a trained model
void predict(string filename, float *weights, int numClasses, int numFeatures) {
  cout << "    * LogisticRegression Testing *" << std::endl;

  float tr = 0.0;
  float fls = 0.0;
  float example[numFeatures];
  string line;
  ifstream test;

  test.open(filename.c_str());
  if(!test) {
    cout << "Failed to open test file..." << std::endl;
    exit(0);
  }

  while (getline(test, line)) {
    if (line.length()) {
      if (line[0] != '#' && line[0] != ' ') {
        vector<string> tokens = split(line);

        int label = (int)atof(tokens[0].c_str());
        for (int j = 1; j < (1 + numFeatures); j++) {
          example[j - 1] = atof(tokens[j].c_str());
        }

        int prediction = classify(example, weights, numClasses, numFeatures);

        if (prediction == label)
          tr++;
        else
          fls++;
      }
    }
  }

  test.close();

  printf("     # accuracy:       %1.3f (%i/%i)\n", (tr / (tr + fls)), (int)tr,
         (int)(tr + fls));
  printf("     # true:           %i\n", (int)tr);
  printf("     # false:          %i\n", (int)fls);
}

// CPU implementation of Logistic Regression gradients calculation
void gradients_sw(int *labels, float *features, float *weights,
                  float *gradients, int numClasses, int numFeatures,
                  int numExamples) {
  for (int k = 0; k < numClasses; k++) {
    for (int j = 0; j < (16 + numFeatures); j++) {
      gradients[k * (16 + numFeatures) + j] = 0.0;
    }
  }

  for (int i = 0; i < numExamples; i++) {
    for (int k = 0; k < numClasses; k++) {
      float dot = weights[k * (16 + numFeatures) + numFeatures];

      for (int j = 0; j < numFeatures; j++) {
        dot += weights[k * (16 + numFeatures) + j] *
               features[i * (16 + numFeatures) + j];
      }

      float dif = 1.0 / (1.0 + exp(-dot));
      if (labels[i] == k)
        dif -= 1;

      for (int j = 0; j < (16 + numFeatures); j++) {
        gradients[k * (16 + numFeatures) + j] +=
            dif * features[i * (16 + numFeatures) + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <iterations>" << std::endl;
    exit(-1);
  }

  struct timeval start, end;

  float alpha = 0.3f;
  float gamma = 0.95f;
  int iter = atoi(argv[1]);

  // Set up the specifications of the model to be trained
  int numClasses = NUMCLASSES;
  int numFeatures = NUMFEATURES;
  int numExamples = NUMEXAMPLES;

  // Split the dataset among the availbale kernels
  int chunkSize = numExamples / NUM_KERNELS;

  // Allocate host buffers for lables and features of the dataset as well as
  // weights and gradients for the model to be trained and lastly velocity
  // buffer for accuracy optimization
  int *labels = (int *)INalligned_malloc(numExamples * sizeof(int));
  float *features = (float *)INalligned_malloc(
      numExamples * (16 + numFeatures) * sizeof(float));
  float *weights = (float *)INalligned_malloc(numClasses * (16 + numFeatures) *
                                              sizeof(float));
  float *gradients = (float *)INalligned_malloc(
      numClasses * (16 + numFeatures) * sizeof(float));
  float *velocity = (float *)INalligned_malloc(numClasses * (1 + numFeatures) *
                                               sizeof(float));

  // Specify train and test input files as well as output model file
  string trainFile = "data/letters_csv_train.dat";
  string testFile = "data/letters_csv_test.dat";
  string modelFile = "data/weights.out";

  // Read the input dataset
  cout << "! Reading train file..." << std::endl;
  read_input(trainFile, features, labels, numFeatures, numExamples);

  // Initialize model weights to zero
  for (int i = 0; i < numClasses * (16 + numFeatures); i++)
    weights[i] = 0.0;

  if (_accel_) {
    // Invoke the hardware accelerated implementation of the algorithm
	float *grads[NUM_KERNELS];
	event fevent[NUM_KERNELS];
	event dma_labels[NUM_KERNELS], dma_features[NUM_KERNELS], dma_weights[NUM_KERNELS], dma_gradients[NUM_KERNELS];
	std::vector<sycl::buffer<int8,1>*> buf_labels;
	std::vector<sycl::buffer<float16,1>*> buf_features;
	std::vector<sycl::buffer<float16,1>*> buf_weights;
	std::vector<sycl::buffer<float16,1>*> buf_gradients;

	sycl::range<1> buf_labels_size{(size_t)(chunkSize/8)};
	sycl::range<1> buf_features_size{(size_t)(chunkSize * (16 + numFeatures)/16)};
	sycl::range<1> buf_weights_size{(size_t)(numClasses * (16 + numFeatures)/16)};

  	for (int i = 0; i < NUM_KERNELS; i++) {
		sycl::buffer<int8,1> *buffer_labels = new sycl::buffer<int8,1>(buf_labels_size);
		buf_labels.push_back(buffer_labels);

		sycl::buffer<float16,1> *buffer_features = new sycl::buffer<float16,1>(buf_features_size);
		buf_features.push_back(buffer_features);

		sycl::buffer<float16,1> *buffer_weights = new sycl::buffer<float16,1>(buf_weights_size);
		buf_weights.push_back(buffer_weights);

		sycl::buffer<float16,1> *buffer_gradients = new sycl::buffer<float16,1>(buf_weights_size);
		buf_gradients.push_back(buffer_gradients);

		grads[i] = (float *) INalligned_malloc(numClasses * (numFeatures + 16) * sizeof(float));
	}

    gettimeofday(&start, NULL);

	try {
		// Select either the FPGA emulator or FPGA device
		#if defined(FPGA_EMULATOR)
		sycl::INTEL::fpga_emulator_selector device_selector;
		#else
		sycl::INTEL::fpga_selector device_selector;
		#endif
		// Create a queue bound to the chosen device.
		// If the device is unavailable, a SYCL runtime exception is thrown.
		auto prop_list = property_list{property::queue::enable_profiling()};
		sycl::queue q(device_selector, dpc_common::exception_handler, prop_list);
		//cout << "queue is in order: " << q.is_in_order() << std::endl;

		// Copy to the fpga card labels and features
		for (int i = 0; i < NUM_KERNELS; i++) {
			int* flabels = labels + i * chunkSize;
			float* ffeatures = features + (i * chunkSize * (16 + numFeatures));

			dma_labels[i] =  q.submit([&](handler &h) {
					auto in_data = buf_labels[i]->get_access<access::mode::discard_write>(h);
					h.copy(flabels, in_data);
				});

			dma_features[i] =  q.submit([&](handler &h) {
					auto in_data = buf_features[i]->get_access<access::mode::discard_write>(h);
					h.copy(ffeatures, in_data);
				});
		}

		// Start the iterative part for the training of the algorithm
		size_t exec_time = 0, copy_to_time = 0, copy_from_time = 0;
		for (int t = 0; t < iter; t++) {
			for (int i = 0; i < NUM_KERNELS; i++) {
				// Copy to the fpga card weights
				dma_weights[i] = q.submit([&](handler &h) {
						auto in_data = buf_weights[i]->get_access<access::mode::discard_write>(h);
						h.copy(weights, in_data);
				});

				// The definition of this function is in a different compilation unit,
				// so host and device code can be separately compiled.
				RunGradients(q, buf_labels[i], buf_features[i], buf_weights[i], buf_gradients[i], numClasses, numFeatures, chunkSize, fevent[i], i);

				// Copy from the fpga card gradients
				dma_gradients[i] = q.submit([&](handler &h) {
						auto out_data = buf_gradients[i]->get_access<access::mode::read>(h);
						h.copy(out_data, grads[i]);
				});
			}
			// Wait for all engines to be finished
			for (int i = 0; i < NUM_KERNELS; i++) {
				if(t == 0){
					dma_labels[i].wait();
					dma_features[i].wait();
					copy_to_time += SyclGetExecTimeNs(dma_labels[i]) + SyclGetExecTimeNs(dma_features[i]);
				}
				dma_weights[i].wait();
				fevent[i].wait();
				dma_gradients[i].wait();
				copy_to_time += SyclGetExecTimeNs(dma_weights[i]);
				exec_time += SyclGetExecTimeNs(fevent[i]);
				copy_from_time += SyclGetExecTimeNs(dma_gradients[i]);
			}

			// Aggregate the gradients from all kernels
			for (int j = 0; j < numClasses * (16 + numFeatures); j++) {
				gradients[j] = grads[0][j];
				for (int i = 1; i < NUM_KERNELS; i++) {
					gradients[j] += grads[i][j];
				}
			}

			// Compute the new weights of the model applying some software
			// optimizations for better model accuracy
			for (int k = 0; k < numClasses; k++) {
				for (int j = 0; j < (1 + numFeatures); j++) {
					velocity[k * (1 + numFeatures) + j] =
							gamma * velocity[k * (1 + numFeatures) + j] +
							 (alpha / numExamples) * gradients[k * (16 + numFeatures) + j];
					weights[k * (16 + numFeatures) + j] -=
							velocity[k * (1 + numFeatures) + j];
				}
			}
		}
		cout << "Total time of sending data to the fpga in msec: " << copy_to_time/(1000000) << "\n";
		cout << "Total time of kernel execution in msec: " << exec_time/(1000000) << "\n";
		cout << "Total time of getting data from the fpga in msec: " << copy_from_time/(1000000) << "\n";
	} catch (cl::sycl::exception const &e) {
		// Catches exceptions in the host code
		std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

		// Most likely the runtime couldn't find FPGA hardware!
		if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
			std::cerr << "If you are targeting an FPGA, please ensure that your "
			   "system has a correctly configured FPGA board.\n";
			std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
			std::cerr << "If you are targeting the FPGA emulator, compile with "
			   "-DFPGA_EMULATOR.\n";
		}
		std::terminate();
	}

    gettimeofday(&end, NULL);
	// Free any allocated memory
	for (int i = 0; i < NUM_KERNELS; i++) {
		delete(buf_labels[i]);
		delete(buf_features[i]);
		delete(buf_weights[i]);
		delete(buf_gradients[i]);
		free(grads[i]);
	}


  } else {
    // Invoke the software implementation of the algorithm
    gettimeofday(&start, NULL);
    for (int t = 0; t < iter; t++) {
      gradients_sw(labels, features, weights, gradients, numClasses,
                   numFeatures, numExamples);
      for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < (1 + numFeatures); j++) {
          velocity[k * (1 + numFeatures) + j] =
              gamma * velocity[k * (1 + numFeatures) + j] +
              (alpha / numExamples) * gradients[k * (16 + numFeatures) + j];
          weights[k * (16 + numFeatures) + j] -=
              velocity[k * (1 + numFeatures) + j];
        }
      }
    }
    gettimeofday(&end, NULL);
  }

  float time_us = ((end.tv_sec * 1000000) + end.tv_usec) -
                  ((start.tv_sec * 1000000) + start.tv_usec);
  float time_s = (end.tv_sec - start.tv_sec);

  cout << "! Time running Gradients Kernel: " << time_us / 1000 << " msec, "
       << time_s << " sec " << std::endl;

  // Compute the accuracy of the trained model on a given test dataset.
  predict(testFile, weights, numClasses, numFeatures);

  // Save the model to the specified user file
  write_output(modelFile, weights, numClasses, numFeatures);

  // Free any host allocated buffers
  free(labels);
  free(features);
  free(weights);
  free(gradients);
  free(velocity);
  return 0;
}

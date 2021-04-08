#define numClassesMax 32
#define numFeaturesMax 1023

#define numFeaturesPlusOneMax 64 //16 * 64 = 1024

#define chunk 8
#define parallelism 8

__attribute__((always_inline))
int roundTo(int value, int roundTo){
	return ((value + (roundTo - 1)) & (~(roundTo - 1)));
}

__attribute__((always_inline))
float accumulate(float16 vector){
	float8 tmp0 = vector.odd + vector.even;

	float4 tmp1 = tmp0.odd + tmp0.even;

	float2 tmp2 = tmp1.odd + tmp1.even;

	float scalar = tmp2.odd + tmp2.even;

	return scalar;
}

__kernel void Gradients_0(const __global int8 *restrict _label, const __global float16 *restrict _features, __global float16 *restrict _weights, __global float16 *restrict _gradients, const int numClasses, const int numFeatures, const int numExamples){

//	float16 __attribute__((numbanks(8), bankwidth(64), singlepump, numreadports(1), numwriteports(1))) features[numFeaturesPlusOneMax][chunk];
	float16 weights[numFeaturesPlusOneMax][numClassesMax];
	float16 gradients[numFeaturesPlusOneMax][numClassesMax];

//	float16 __attribute__((numbanks(1), bankwidth(64), singlepump, numreadports(2), numwriteports(1))) dotproduct[numClassesMax][chunk];
//	float __attribute__((numbanks(chunk), bankwidth(64), singlepump, numreadports(1), numwriteports(1))) prediction[numClassesMax][chunk];

	float16 features[numFeaturesPlusOneMax][chunk];
	float16 dotproduct[numClassesMax][chunk];
	float prediction[numClassesMax][chunk];

	int numFeaturesPlusOne = roundTo(numFeatures + 1, 16) >> 4;
	int numClassesMin = (13 > numClasses) ? 13 : numClasses;
	int numChunks = roundTo(numExamples, chunk) >> 3;

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
				prediction[k][c] = 1.0 / (1.0 + exp(-accumulate(dotproduct[k][c])));
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
}

__kernel void Gradients_1(const __global int8 *restrict _label, const __global float16 *restrict _features, __global float16 *restrict _weights, __global float16 *restrict _gradients, const int numClasses, const int numFeatures, const int numExamples){

	float16 weights[numFeaturesPlusOneMax][numClassesMax];
	float16 gradients[numFeaturesPlusOneMax][numClassesMax];

	float16 features[numFeaturesPlusOneMax][chunk];
	float16 dotproduct[numClassesMax][chunk];
	float prediction[numClassesMax][chunk];

	int numFeaturesPlusOne = roundTo(numFeatures + 1, 16) >> 4;
	int numClassesMin = (13 > numClasses) ? 13 : numClasses;
	int numChunks = roundTo(numExamples, chunk) >> 3;

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
				prediction[k][c] = 1.0 / (1.0 + exp(-accumulate(dotproduct[k][c])));
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
}

/*
__kernel void Gradients_2(const __global int8 *restrict _label, const __global float16 *restrict _features, __global float16 *restrict _weights, __global float16 *restrict _gradients, const int numClasses, const int numFeatures, const int numExamples){

	float16 weights[numFeaturesPlusOneMax][numClassesMax];
	float16 gradients[numFeaturesPlusOneMax][numClassesMax];

	float16 features[numFeaturesPlusOneMax][chunk];
	float16 dotproduct[numClassesMax][chunk];
	float prediction[numClassesMax][chunk];

	int numFeaturesPlusOne = roundTo(numFeatures + 1, 16) >> 4;
	int numClassesMin = (13 > numClasses) ? 13 : numClasses;
	int numChunks = roundTo(numExamples, chunk) >> 3;

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
				prediction[k][c] = 1.0 / (1.0 + exp(-accumulate(dotproduct[k][c])));
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
}

__kernel void Gradients_3(const __global int8 *restrict _label, const __global float16 *restrict _features, __global float16 *restrict _weights, __global float16 *restrict _gradients, const int numClasses, const int numFeatures, const int numExamples){


	float16 weights[numFeaturesPlusOneMax][numClassesMax];
	float16 gradients[numFeaturesPlusOneMax][numClassesMax];

	float16 features[numFeaturesPlusOneMax][chunk];
	float16 dotproduct[numClassesMax][chunk];
	float prediction[numClassesMax][chunk];

	int numFeaturesPlusOne = roundTo(numFeatures + 1, 16) >> 4;
	int numClassesMin = (13 > numClasses) ? 13 : numClasses;
	int numChunks = roundTo(numExamples, chunk) >> 3;

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
				prediction[k][c] = 1.0 / (1.0 + exp(-accumulate(dotproduct[k][c])));
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
}
*/

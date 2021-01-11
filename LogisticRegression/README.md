# Logistic Regression IP core

|   | One-api Stratix10 | One-api Arria10 | Opencl Arria10 |
| --- | --- | --- | --- |
| Num Engines/Kernels | 2 | 2 | 2 |
| Acceleration (compared to CPU only execution for 100 iterations) | 13.13x | 41.59x | 42.82x |
| ALUTs | 530547 | 240444 | 200474 |
| Registers | 1,673,644 | 560692 | 460165 |
| Logic Utilization | 70% | 61% | 51% |
| I/O pins | 72% | 38% | 38% |
| DSP blocks | 15% | 57% | 70% |
| Memory bits | 33% | 24% | 22% |
| RAM blocks | 41% | 45% | 35% |
| Actual clock freq | 261 | 192 | 171 |
| Kernel fmax | 261 | 192 | 171.7 |
| 1x clock fmax | 261 | 192 | 239.69 |
| 2x clock fmax | 522 | 384 | 343.4 |
| Highest non-global fanout | 307368 | 335372 | 252406 |

## Supported Platforms

|            Board            |
| --- |
| Intel Arria10 fpga |
| Intel Stratix10 fpga |

## Design Files

-   The application code is located in the src directory.
-   The Makefile will help you generate any host executable and accelerator one-api images .a files.

A listing of all the files in this repository is shown below:

    - Makefile
    - src/
		- host.cpp (contains exclusively code that executes on the host)
		- Gradients.cpp (contains almost exclusively code that executes on the device)
		- Gradients.hpp (contains only the forward declaration of the function containing the device code)
    - data/

## Preparation

- Before invoking any of the Makefile targets make sure you have sourced oneAPI setvars script.  

- Download train letters train dataset to data directory. Navigate to data directory and execute the following commands:

	``` bash
		wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_train.dat
		wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_test.dat
	```

## Compilation

The following build targets are provided:

- Compile for emulation (fast compile time, targets emulated FPGA device):

			make Gradients_emu

- Compile for Arria10 FPGA hardware (longer compile time, generates fpga image src/dev_image_a10.a):

			make Gradients_hw_a10

- Compile for Stratix10 FPGA hardware (longer compile time, generates fpga image src/dev_image_s10.a):

			make Gradients_hw_s10

- Compile for CPU only execution:

			make Gradients_sw


## Application Execution

The host application takes only one input argument, the number of iterations.
Example execution targeting Arria10 fpga card for 100 iterations: `./Gradients_hw_a10 100`

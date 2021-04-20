# Host compiler global settings
CC = g++ -Wall -O3 -Wno-deprecated-declarations

HOST_EXE = Gradients

HOST_DIR = host
KERNEL_DIR = device
AOCX_DIR = bin
AOCX_NAME = LogisticRegression.aocx
BOARD_NAME = pac_a10

# Host and Kernel sources
HOST_SRCS = $(wildcard $(HOST_DIR)/*/*.cpp) $(wildcard $(HOST_DIR)/*.cpp)
KERNEL_CL = $(wildcard $(KERNEL_DIR)/*.cl)

HOST_OBJECTS := $(HOST_SRCS:.cpp=.o)

# Include Libraries
HOST_CFLAGS = $(shell aocl compile-config)
HOST_LFLAGS = $(shell aocl link-config)

all:
	make _TEST_="-D _TEST_" host

host: ${HOST_EXE}

aocx:
	aoc -v -report -board=${BOARD_NAME} -high-effort ${KERNEL_CL} -o ${AOCX_DIR}/${AOCX_NAME}

# Building host
${HOST_EXE}: ${HOST_OBJECTS}
	${CC} ${HOST_OBJECTS} $(HOST_LFLAGS) -o $@
	${RM} -rf ${HOST_OBJECTS}

%.o: %.cpp
	${CC} ${_TEST_} ${HOST_CFLAGS} -c $< -o $@

help:
	@echo "Compile host executable only"
	@echo "make host"
	@echo ""
	@echo "Compile host executable only for SW version"
	@echo "make"
	@echo ""
	@echo "Generate bitstream .aocx file"
	@echo "make aocx"

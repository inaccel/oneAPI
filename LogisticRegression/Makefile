A10_PAC_BOARD_NAME = "intel_a10gx_pac:pac_a10"
S10_PAC_BOARD_NAME = "intel_s10sx_pac:pac_s10"

Gradients_sw: src/host_sw.o
	dpcpp --gcc-toolchain=/usr/local -fintelfpga -D_TEST_ src/host_sw.o  -o Gradients_sw

Gradients_hw_a10: src/host.o src/dev_image_a10.a
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/host.o src/dev_image_a10.a -o Gradients_hw_a10

Gradients_hw_s10: src/host.o src/dev_image_s10.a
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/host.o src/dev_image_s10.a -o Gradients_hw_s10

Gradients_emu: src/host_emu.o src/dev_emu.o
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/dev_emu.o src/host_emu.o -o Gradients_emu

report_a10: src/Gradients.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/Gradients.cpp -fsycl-link -Xshardware -Xsboard=$(A10_PAC_BOARD_NAME) -Xsopt-arg="-nocaching" -o report_a10

report_s10: src/Gradients.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/Gradients.cpp -fsycl-link -Xshardware -Xsboard=$(S10_PAC_BOARD_NAME) -Xsopt-arg="-nocaching" -Xsnum-reorder=6 -o report_s10

src/host_sw.o: src/host.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga -D_TEST_ src/host.cpp -c -o src/host_sw.o

src/host.o: src/host.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/host.cpp -c -o src/host.o

src/host_emu.o: src/host.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/host.cpp -DFPGA_EMULATOR -c -o src/host_emu.o

src/dev_emu.o: src/Gradients.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/Gradients.cpp -DFPGA_EMULATOR -c -o src/dev_emu.o

rc/dev_image_s10.a: src/dev.o
	dpcpp --gcc-toolchain=/usr/local -fintelfpga -fsycl-link=image src/dev.o  -o src/dev_image_s10.a -Xshardware -Xsboard=$(S10_PAC_BOARD_NAME) -Xsopt-arg="-nocaching" -Xsnum-reorder=6

src/dev_image_a10.a: src/dev.o
	dpcpp --gcc-toolchain=/usr/local -fintelfpga -fsycl-link=image src/dev.o  -o src/dev_image_a10.a -Xshardware -Xsboard=$(A10_PAC_BOARD_NAME)

src/dev.o: src/Gradients.cpp
	dpcpp --gcc-toolchain=/usr/local -fintelfpga src/Gradients.cpp -c -o src/dev.o

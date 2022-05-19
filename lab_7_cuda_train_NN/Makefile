SRC = main.cpp model.cpp Tensor.cu utils/initializations.cpp layers/Linear.cu layers/Sigmoid.cu utils/saver.cpp utils/reader.cpp criterion/MSELoss.cu optimizer/SGD.cu

compile:
	nvcc -arch sm_70 -lcublas -lm -use_fast_math $(SRC) -o network.out -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include

run:
	./network.out

profile:
	nsys profile -o prof -f true -t cuda ./network.out
	nsys stats prof.nsys-rep

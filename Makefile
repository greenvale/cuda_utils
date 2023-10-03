NVCC = nvcc
CFLAGS = -arch=sm_75
SRCS = main.cu
OUT = main

all: $(OUT)

$(OUT): $(SRCS)
	$(NVCC) $(CFLAGS) $(SRCS) -o $(OUT)

clean:
	rm -f $(OUT)
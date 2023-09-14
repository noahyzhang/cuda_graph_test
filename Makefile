
CC=g++
CFLAGS = -g -lcudart -lnvonnxparser -lnvinfer  -lonnx_proto -lprotobuf -lstdc++ -lm -lrt -ldl
INC_DIRS = -I/usr/include/x86_64-linux-gnu -I /usr/local/cuda-12.1/targets/x86_64-linux/include
LIB_DRS = -L/usr/local/cuda-12.1/targets/x86_64-linux/lib/

TARGET = main

SRCS = *.cpp 

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) $(INC_DIRS) $(LIB_DRS) -o $(TARGET) $(CFLAGS) 

clean:
	rm -f $(TARGET)
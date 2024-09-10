# Makefile

# 变量定义
CXX = g++
# CXXFLAGS = -Wall -O2 $(shell pkg-config --cflags eigen3) -fopenmp
CXXFLAGS = -Wall -O3 $(shell pkg-config --cflags eigen3) -fopenmp
# TARGET = eigen3omptest
# SRCS = eigen3omptest.cpp
# TARGET = gmres_example
# SRCS = gmres_example.cpp
TARGET = class_example
SRCS = class_example.cpp
#HEADERS = mat.h para_comput.h
OBJS = $(SRCS:.cpp=.o)

# 默认目标
all: $(TARGET)

# 链接目标
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 编译源文件
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJS) $(TARGET)

# 伪目标
.PHONY: all clean
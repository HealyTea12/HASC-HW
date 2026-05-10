CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -pedantic

BUILD_DIR := build
TARGET := $(BUILD_DIR)/transpose_study
RESULT_DIR := results
RESULT_FILE := $(RESULT_DIR)/transpose_results.csv
MINTIME ?= 0.10

.PHONY: all run clean

all: $(TARGET)

$(TARGET): transpose.cc time_experiment.hh
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) transpose.cc -o $(TARGET)

$(RESULT_FILE): $(TARGET)
	mkdir -p $(RESULT_DIR)
	./$(TARGET) $(MINTIME) > $(RESULT_FILE)
	@echo "wrote $(RESULT_FILE)"

run: $(RESULT_FILE)

clean:
	rm -rf $(BUILD_DIR)

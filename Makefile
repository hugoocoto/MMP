CXX = mpicxx
AR = ar

CXXFLAGS = -O2 -march=native -Wall -std=c++20 -fvisibility=hidden
CPPFLAGS = -Imalleable/include -MMD -MP

BUILD_DIR = build
LIB = $(BUILD_DIR)/libmalleable.a

SRCS = $(wildcard malleable/src/*.cpp malleable/builtin_policies/*.cpp)
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all examples clean re

all: $(LIB)

$(LIB): $(OBJS)
	$(RM) $@
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

examples: $(LIB)
	$(MAKE) -C examples

clean:
	$(RM) -r $(BUILD_DIR)

re: clean
	$(MAKE) all

-include $(OBJS:.o=.d)

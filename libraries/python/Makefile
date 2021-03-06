space :=
space +=
WEBOTS_HOME_PATH=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
include $(WEBOTS_HOME_PATH)/resources/Makefile.os.include

MODULE_NAME = managers
PYTHON_COMMAND ?= python
ifeq (, $(shell which $(PYTHON_COMMAND)))
$(error $(PYTHON_COMMAND) "is not installed on your system, please install it")
endif
PYTHON_VERSION := $(shell $(PYTHON_COMMAND) --version 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1.\2/')
PYTHON_SHORT_VERSION = $(subst .,,$(PYTHON_VERSION))

INTERFACE          = $(MODULE_NAME).i
SOURCE_HEADERS     = $(wildcard ../managers/include/*.hpp)
SWIG               = swig
SWIG_OPTS          = -c++ -python -outdir "." -I"$(WEBOTS_HOME)/resources/languages/python"
WRAPPER            = $(MODULE_NAME)$(PYTHON_SHORT_VERSION).cpp
WRAPPER_OBJECT     = $(WRAPPER:.cpp=.o)
PYOUT              = $(MODULE_NAME).py
INCLUDES           = -I"../managers/include" -I"$(WEBOTS_HOME)/include/controller/c" -I"$(WEBOTS_HOME)/include/controller/cpp"
LIB                = -L"../managers" -lmanagers

ifeq ($(OSTYPE),linux)
C_FLAGS         = -c -Wall -fPIC -Wno-unused-but-set-variable
LD_FLAGS        = -shared
LIBRARY         = _$(MODULE_NAME).so
PYTHON_INCLUDES = -I"/usr/include/python$(PYTHON_VERSION)"
endif

ifeq ($(OSTYPE),windows)
PYTHON_HOME    := $(dir $(shell which python))
C_FLAGS         = -c -O -Wall -DMS_WIN64 -D_hypot=hypot
LD_FLAGS        = -shared -Wl,--enable-auto-import
LIBRARY         = _$(MODULE_NAME).pyd
PYTHON_INCLUDES = -I$(PYTHON_HOME)include
LIB            += -L$(PYTHON_HOME)libs -lpython$(PYTHON_SHORT_VERSION)
endif

ifeq ($(OSTYPE),darwin)

PYTHON_PATH ?= /System/Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)

C_FLAGS         = -c -Wall -fPIC -mmacosx-version-min=$(MACOSX_MIN_SDK_VERSION) -stdlib=libc++
ifeq ($(findstring llvm-g++,$(shell ls -lF $(shell which c++))),)
C_FLAGS        += -Wno-self-assign
endif
LD_FLAGS        = -dynamiclib -install_name @rpath/lib/python/_$(INTERFACE:.i=.so) -Wl,-rpath,@loader_path/../../../../../.. -compatibility_version 1.0 -current_version 1.0.0 -mmacosx-version-min=$(MACOSX_MIN_SDK_VERSION) -stdlib=libc++
PYTHON_INCLUDES = -I$(PYTHON_PATH)/include/python$(PYTHON_VERSION)
LIB            += -L"$(PYTHON_PATH)/lib" -lpython$(PYTHON_VERSION)
LIBRARY         = _$(MODULE_NAME).so
endif

TARGET = $(PYOUT) $(LIBRARY)

release debug profile: $(TARGET)

$(PYOUT) $(WRAPPER):$(INTERFACE) $(SOURCE_HEADERS)
	$(SWIG) $(SWIG_OPTS) $(INCLUDES) -o $(WRAPPER) $<

$(LIBRARY):$(WRAPPER_OBJECT)
	$(CXX) $(LD_FLAGS) $< $(DEF) $(LIB) -o "$@"

$(WRAPPER_OBJECT):$(WRAPPER)
	$(CXX) $(C_FLAGS) $(INCLUDES) $(PYTHON_INCLUDES) $< -o $@

clean:
	@rm -fr *.o *.cpp $(PYOUT) $(LIBRARY) $(MODULE_NAME).pyc __pycache__

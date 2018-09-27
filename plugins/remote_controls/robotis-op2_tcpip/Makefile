# Webots Makefile system 
#
# You may add some variable definitions hereafter to customize the build process
# See documentation in $(WEBOTS_HOME_PATH)/resources/Makefile.include

space :=
space +=
WEBOTS_HOME_PATH=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))

include $(WEBOTS_HOME_PATH)/resources/Makefile.os.include

CXX_SOURCES = $(wildcard *.cpp)

USE_C_API = true

ifeq ($(OSTYPE),windows)
# on Windows, need to link with WinSock2
LIBRARIES = -lws2_32
endif

ifeq ($(OSTYPE),darwin)
INCLUDE = -I$(WEBOTS_HOME_PATH)/include/libjpeg/
endif

include $(WEBOTS_HOME_PATH)/resources/Makefile.include

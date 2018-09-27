MANAGERS_SOURCES_PATH = ./src
MANAGERS_INCLUDE_PATH = ./include
ROBOTISOP2_PATH = ../robotis-op2
ROBOTISOP2_FRAMEWORK_PATH = $(ROBOTISOP2_PATH)/robotis/Framework

CXX_SOURCES = \
  $(MANAGERS_SOURCES_PATH)/RobotisOp2DirectoryManager.cpp \
  $(MANAGERS_SOURCES_PATH)/RobotisOp2MotionManager.cpp \
  $(MANAGERS_SOURCES_PATH)/RobotisOp2GaitManager.cpp \
  $(MANAGERS_SOURCES_PATH)/RobotisOp2VisionManager.cpp

INCLUDE = -I"$(ROBOTISOP2_FRAMEWORK_PATH)/include" -I"$(MANAGERS_INCLUDE_PATH)"
CFLAGS = -DWEBOTS
LIBRARIES = -L"$(ROBOTISOP2_PATH)" -lrobotis-op2

space :=
space +=
WEBOTS_HOME_PATH=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
include $(WEBOTS_HOME_PATH)/resources/Makefile.include

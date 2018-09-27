space :=
space +=
WEBOTS_HOME_PATH=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
include $(WEBOTS_HOME_PATH)/resources/Makefile.os.include

TARGETS=robotis-op2.Makefile managers.Makefile python.Makefile

ifeq ($(OSTYPE),windows)
PYTHON_MAKE = PATH="$(PYTHON27_HOME):$(PATH)" make
else
PYTHON_MAKE = make
endif

.PHONY: release debug profile clean

release debug profile clean: $(TARGETS)

managers.Makefile: robotis-op2.Makefile

managers.Makefile robotis-op2.Makefile:
	+@echo "# make" $(MAKECMDGOALS) $(@:.Makefile=)
	+@make -s -C $(@:.Makefile=) $(MAKECMDGOALS)

python.Makefile: managers.Makefile
	+@echo "# make" $(MAKECMDGOALS) $(@:.Makefile=)
	+@$(PYTHON_MAKE) -s -C $(@:.Makefile=) $(MAKECMDGOALS)

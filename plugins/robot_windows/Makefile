TARGETS=robotis-op2_window.Makefile

.PHONY: release debug profile clean

release debug profile clean: $(TARGETS)

%.Makefile:
	+@echo "# make" $(MAKECMDGOALS) $(@:.Makefile=)
	+@make -s -C $(@:.Makefile=) $(MAKECMDGOALS)

VENV_PATH ?= venv
PYTHON ?= python
PYTEST ?= pytest
PIP ?= pip
YAPF ?= yapf

TORCH_VERSION ?= torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
UPDATE_PIP ?= python -m pip install --upgrade pip

ifeq ($(VENV_PATH),)
ACTIVATE = 
else
ACTIVATE = . $(VENV_PATH)/bin/activate &&
endif

.PHONY: clean doc doctest test test-gpu codecov check-formatting check-formatting-names clean-dacecaches
clean:
	! test -d $(VENV_PATH) || rm -r $(VENV_PATH)

venv: 
ifneq ($(VENV_PATH),)
	test -d $(VENV_PATH) || echo "Creating new venv" && $(PYTHON) -m venv ./$(VENV_PATH)
endif

install: venv
ifneq ($(VENV_PATH),)
	$(ACTIVATE) $(UPDATE_PIP)
endif
	$(ACTIVATE) $(PIP) install $(TORCH_VERSION) 
	$(ACTIVATE) $(PIP) install -e .[testing,debug,docs]

doc:
# suppress warnings in ONNXOps docstrings using grep -v
	$(ACTIVATE) cd doc && make clean html 2>&1 \
	| grep -v ".*daceml\/daceml\/onnx\/nodes\/onnx_op\.py:docstring of daceml\.onnx\.nodes\.onnx_op\.ONNX.*:[0-9]*: WARNING:"


doctest:
	$(ACTIVATE) cd doc && make doctest

test: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests

test-parallel: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests -n auto --dist loadfile

test-gpu: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests --gpu

codecov:
	curl -s https://codecov.io/bash | bash

clean-dacecaches:
	find . -name ".dacecache" -type d -not -path "*CMakeFiles*" -exec rm -r {} \;

check-formatting:
	$(ACTIVATE) $(YAPF) \
		--parallel \
		--diff \
		--recursive \
		daceml tests setup.py \
		--exclude daceml/onnx/shape_inference/symbolic_shape_infer.py

check-formatting-names:
	$(ACTIVATE) $(YAPF) \
		--parallel \
		--diff \
		--recursive \
		daceml tests setup.py \
		--exclude daceml/onnx/shape_inference/symbolic_shape_infer.py |  grep "+++" || echo "All good!"

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = educational_ai_analytics
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## 0. Make dataset (Descarga + Split) -> data/2_processed
.PHONY: data
data:
	$(PYTHON_INTERPRETER) educational_ai_analytics/dataset.py

## 1. Make features (Ingeniería de Características) -> data/2_processed/{split}/features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.features

## 2. Train Autoencoder
.PHONY: train_ae
train_ae:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.modeling.train_autoencoder

## 3. Generate Latent Space (Embeddings) -> data/3_embeddings
.PHONY: encode
encode:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.modeling.generate_embeddings

## 4. Train Clustering (K-Means, etc.)
.PHONY: train_clustering
train_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.modeling.train_clustering

## 5. Train Transformer (Future)
.PHONY: train_transformer
train_transformer:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.modeling.train_transformer

## 6. Plot Learning Curves
.PHONY: plot_loss
plot_loss:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.plots loss-curve

## 7. Plot Latent Space (Embeddings Visualization)
.PHONY: plot_latent
plot_latent:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.plots latent-space

## 8. Plot PCA Latent Space
.PHONY: plot_pca
plot_pca:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.plots latent-space --embeddings-file data/4_embeddings/training/latent_pca.csv --output-name latent_space_pca.png

## Prep latent space (Dataset -> Features -> Train AE -> Encode)
.PHONY: prep_latent
prep_latent: data features train_ae encode

## Run full pipeline
.PHONY: pipeline
pipeline: prep_latent train_clustering

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = educational_ai_analytics
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
METHOD = pca
SHELL := /bin/bash

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
	$(PYTHON_INTERPRETER) educational_ai_analytics/0_data/0_dataset.py

## 1. Make features (Ingeniería de Características) -> data/2_processed/{split}/features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) educational_ai_analytics/1_features/__main__.py

## 2. Train Autoencoder
.PHONY: train_ae
train_ae:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.ae.train_autoencoder

## 3. Generate Latent Space (Embeddings) -> data/3_embeddings
.PHONY: encode
encode:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.ae.generate_embeddings

## 4. Train Clustering (K-Means, etc.)
.PHONY: train_clustering
train_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.clustering.train_clustering

## 5. Predict Clustering (Inferencia GMM) -> data/4_embeddings/{split}/segmentation_gmm_ae.csv
.PHONY: predict_clustering
predict_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.clustering.predict_clustering

## 6. Generate Transformer Features -> data/6_transformer_features
.PHONY: transformer_features
transformer_features:
	$(PYTHON_INTERPRETER) educational_ai_analytics/1_features/__transformer_main__.py

## 7. Train Transformer
.PHONY: train_transformer
train_transformer:
	$(PYTHON_INTERPRETER) educational_ai_analytics/2_modeling/transformers/train_transformer.py

## 7. Plot Learning Curves (AE)
.PHONY: plot_ae_loss
plot_ae_loss:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae loss-curve

## 8. Plot Latent Space (AE)
.PHONY: plot_ae_latent
plot_ae_latent:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae latent-space --method $(METHOD)

## 9. Plot AE Sample Flow (Input -> Latent -> Recon)
.PHONY: plot_ae_sample
plot_ae_sample:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae ae-reconstruction

## 10. Plot comparison grid (PCA vs AE for multiple W)
.PHONY: plot_ae_compare_pca
plot_ae_compare_pca:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae compare-pca-vs-ae --method $(METHOD)

## 11. Plot comparison of latent spaces across W
.PHONY: plot_ae_compare_latent
plot_ae_compare_latent:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae compare-latent-spaces --method $(METHOD)

## 12. Plot all AE reports for default W
.PHONY: plot_ae
plot_ae:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae clean
	$(MAKE) plot_ae_latent plot_ae_sample plot_ae_compare_pca plot_ae_compare_latent

## Generar todos los plots de clustering (outcomes + profiles)
.PHONY: plot_clustering
plot_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering clean
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering weekly-reports

plot_profiles:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering profiles

## Prep latent space (Dataset -> Features -> Train AE -> Encode)
.PHONY: prep_latent
prep_latent: data features train_ae encode plot_ae

## Full AE Pipeline with UMAP (Train + Encode + Plot UMAP)
.PHONY: ae_full_pca
ae_full_pca: train_ae encode
	$(MAKE) plot_ae METHOD=pca

## Run full pipeline
.PHONY: pipeline
pipeline: prep_latent train_clustering predict_clustering

## End-to-end pipeline with clustering plots (data -> features -> AE -> clustering -> plots)
.PHONY: pipeline_clustering_plots
pipeline_clustering_plots: data features train_ae encode train_clustering predict_clustering plot_clustering

## Grouped: all plots (AE + Clustering + Profiles)
.PHONY: all_plots
all_plots: plot_ae plot_clustering plot_profiles

## Grouped: from raw data to clustering prediction
.PHONY: from_data_to_clustering_predict
from_data_to_clustering_predict: data features train_ae encode train_clustering predict_clustering

## Grouped: from AE training to clustering prediction
.PHONY: from_ae_train_to_predict
from_ae_train_to_predict: train_ae encode train_clustering predict_clustering

## Grouped: full pipeline (data -> features -> AE -> clustering -> prediction -> plots)
.PHONY: full_pipeline
full_pipeline: from_data_to_clustering_predict all_plots

## Menú interactivo con opciones numeradas
.PHONY: menu
menu:
	@echo ""
	@echo "================= 📋 MENU PIPELINES ================="
	@echo "  1) full_pipeline"
	@echo "  2) all_plots"
	@echo "  3) from_data_to_clustering_predict"
	@echo "  4) from_ae_train_to_predict"
	@echo "  5) plot_clustering"
	@echo "  6) help"
	@echo "====================================================="
	@read -p "Elige una opción [1-6]: " opt; \
	case $$opt in \
		1) $(MAKE) full_pipeline ;; \
		2) $(MAKE) all_plots ;; \
		3) $(MAKE) from_data_to_clustering_predict ;; \
		4) $(MAKE) from_ae_train_to_predict ;; \
		5) $(MAKE) plot_clustering ;; \
		6) $(MAKE) help ;; \
		*) echo "❌ Opción inválida. Ejecuta 'make menu' de nuevo." ;; \
	esac

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := menu

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

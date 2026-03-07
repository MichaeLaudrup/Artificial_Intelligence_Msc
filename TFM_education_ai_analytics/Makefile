#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = educational_ai_analytics
PYTHON_VERSION = 3.12
# Use the active Python from PATH by default so `make` follows the current
# environment (for example the Blackwell TensorFlow wheel installed in a venv).
# You can still override per command, e.g. `make train_ae PYTHON_INTERPRETER=/path/to/python`.
PYTHON_INTERPRETER ?= $(shell command -v python)
METHOD = pca
HPARAM_MAX_TRIALS ?= 5000
SHELL := /bin/bash

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .

## Install optional TensorFlow wheel dependencies in the current environment
.PHONY: requirements_tf
requirements_tf:
	pip install -e .[tensorflow]

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
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.ae.train_autoencoder 2> >(grep -vE "\+ptx85|could not open file to read NUMA node|Your kernel may have been built without NUMA support\.|XLA service|Compiled cluster using XLA|All log messages before absl::InitializeLog\(\)" >&2)

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
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.transformers.train_transformer

## 7.1. Train Transformer for every configured window
.PHONY: train_transformer_windows
train_transformer_windows:
	@set -e; \
	WINDOWS="$$($(PYTHON_INTERPRETER) -c 'from educational_ai_analytics.config import W_WINDOWS; print(" ".join(str(int(w)) for w in W_WINDOWS))')"; \
	echo "🧠 Transformer TRAIN | windows=$$WINDOWS"; \
	for W in $$WINDOWS; do \
		echo "➡️  train_transformer upto_week=$$W"; \
		TMP_CONFIG="$$(mktemp)"; \
		printf '{"upto_week": %s}\n' "$$W" > "$$TMP_CONFIG"; \
		$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.transformers.train_transformer --config-json "$$TMP_CONFIG"; \
		rm -f "$$TMP_CONFIG"; \
	done

## Build custom TensorFlow wheel for Blackwell GPUs (RTX 50xx / sm_120)
.PHONY: build_tf_blackwell_wheel
build_tf_blackwell_wheel:
	bash scripts/build_tf220_wheel.sh

## Install latest locally built TensorFlow wheel
.PHONY: install_tf_blackwell_wheel
install_tf_blackwell_wheel:
	bash scripts/install_tf220_wheel.sh

## Data-free Blackwell TensorFlow runtime check
.PHONY: check_tf_blackwell
check_tf_blackwell:
	bash scripts/check_tf_blackwell.sh

## Quick Blackwell TensorFlow benchmark / compatibility check
.PHONY: bench_tf_blackwell
bench_tf_blackwell:
	bash scripts/bench_tf220.sh

## 7.5. Run Full Transformer Pipeline (Features + Train)
.PHONY: full_transformer
full_transformer: transformer_features train_transformer_windows plot_transformer_summary

## 7.5.1. Generate global transformer validation summary (CSV + PNG)
.PHONY: plot_transformer_summary
plot_transformer_summary:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_transformers

## 7.6. Super finetuning automático (Random Search Transformer)
.PHONY: hparam_search
hparam_search:
	$(PYTHON_INTERPRETER) educational_ai_analytics/2_modeling/transformers/utils/tune_random_search.py --max-trials $(HPARAM_MAX_TRIALS)

## 7.7. Super finetuning en background (nocturno)
.PHONY: hparam_search_bg
hparam_search_bg:
	mkdir -p reports/hparam_search
	nohup $(PYTHON_INTERPRETER) educational_ai_analytics/2_modeling/transformers/utils/tune_random_search.py --max-trials $(HPARAM_MAX_TRIALS) > reports/hparam_search/night.log 2>&1 & echo $$! > reports/hparam_search/night.pid
	@echo "✅ Random search lanzado en background. PID guardado en reports/hparam_search/night.pid"

## 7.8. Estado del super finetuning
.PHONY: hparam_status
hparam_status:
	@if [ -f reports/hparam_search/night.pid ]; then \
		PID=$$(cat reports/hparam_search/night.pid); \
		echo "PID: $$PID"; \
		ps -p $$PID -o pid=,etime=,cmd= || true; \
	else \
		echo "⚠️ No hay PID registrado (reports/hparam_search/night.pid)"; \
	fi
	@echo "--- Últimas líneas del log ---"
	@tail -n 30 reports/hparam_search/night.log 2>/dev/null || echo "⚠️ Sin log todavía (reports/hparam_search/night.log)"

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

## Grouped: full pipeline end-to-end (data -> features -> AE -> clustering -> transformer, sin XAI)
.PHONY: full_pipeline
full_pipeline: from_data_to_transformer

## Grouped: from raw data to transformer training
.PHONY: from_data_to_transformer
from_data_to_transformer: data features train_ae encode train_clustering predict_clustering all_plots transformer_features train_transformer_windows plot_transformer_summary

## Grouped: from AE training to full transformer pipeline
.PHONY: from_ae_train_to_transformer
from_ae_train_to_transformer: train_ae encode train_clustering predict_clustering all_plots transformer_features train_transformer_windows plot_transformer_summary

## Menú interactivo con opciones numeradas
.PHONY: menu
menu:
	@echo ""
	@echo "================= 📋 MENU PIPELINES ================="
	@echo "  1) full_pipeline (hasta transformer, sin XAI)"
	@echo "  2) all_plots"
	@echo "  3) from_data_to_clustering_predict"
	@echo "  4) from_ae_train_to_predict"
	@echo "  5) plot_clustering"
	@echo "  6) from_data_to_transformer"
	@echo "  7) from_ae_train_to_transformer"
	@echo "  8) help"
	@echo "====================================================="
	@read -p "Elige una opción [1-8]: " opt; \
	case $$opt in \
		1) $(MAKE) full_pipeline ;; \
		2) $(MAKE) all_plots ;; \
		3) $(MAKE) from_data_to_clustering_predict ;; \
		4) $(MAKE) from_ae_train_to_predict ;; \
		5) $(MAKE) plot_clustering ;; \
		6) $(MAKE) from_data_to_transformer ;; \
		7) $(MAKE) from_ae_train_to_transformer ;; \
		8) $(MAKE) help ;; \
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

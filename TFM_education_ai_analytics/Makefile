#################################################################################

#################################################################################

PROJECT_NAME = educational_ai_analytics
PYTHON_VERSION = 3.12



PYTHON_INTERPRETER ?= $(shell command -v python)
METHOD = pca
HPARAM_MAX_TRIALS ?= 5000
SHELL := /bin/bash

#################################################################################

#################################################################################

#
.PHONY: requirements
requirements:
	pip install -e .

#
.PHONY: requirements_tf
requirements_tf:
	pip install -e .[tensorflow]

#
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#
.PHONY: lint
lint:
	ruff format --check
	ruff check

#
.PHONY: format
format:
	ruff check --fix
	ruff format

#################################################################################

#################################################################################

#
.PHONY: data
data:
	$(PYTHON_INTERPRETER) educational_ai_analytics/0_data/0_dataset.py

#
.PHONY: features
features:
	$(PYTHON_INTERPRETER) educational_ai_analytics/1_features/__main__.py

.PHONY: ae_feature_engienering
ae_feature_engienering:
	$(PYTHON_INTERPRETER) educational_ai_analytics/1_features/__main__.py

#
.PHONY: train_ae
train_ae:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.ae.train_autoencoder 2> >(grep -vE "\+ptx85|could not open file to read NUMA node|Your kernel may have been built without NUMA support\.|XLA service|Compiled cluster using XLA|All log messages before absl::InitializeLog\(\)" >&2)

#
.PHONY: encode
encode:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.ae.generate_embeddings

#
.PHONY: train_clustering
train_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.clustering.train_clustering

#
.PHONY: predict_clustering
predict_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.clustering.predict_clustering

#
.PHONY: transformer_features
transformer_features:
	$(PYTHON_INTERPRETER) educational_ai_analytics/1_features/__transformer_main__.py

#
.PHONY: train_transformer
train_transformer:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.transformers.train_transformer

#
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

#
.PHONY: test_transformer
test_transformer:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.transformers.evaluate_transformer

#
.PHONY: test_transformer_windows
test_transformer_windows:
	@set -e; \
	WINDOWS="$$($(PYTHON_INTERPRETER) -c 'from educational_ai_analytics.config import W_WINDOWS; print(" ".join(str(int(w)) for w in W_WINDOWS))')"; \
	echo "🧪 Transformer TEST | windows=$$WINDOWS"; \
	for W in $$WINDOWS; do \
		echo "➡️  test_transformer upto_week=$$W"; \
		TMP_CONFIG="$$(mktemp)"; \
		printf '{"upto_week": %s}\n' "$$W" > "$$TMP_CONFIG"; \
		$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.transformers.evaluate_transformer --config-json "$$TMP_CONFIG"; \
		rm -f "$$TMP_CONFIG"; \
	done

#
.PHONY: build_tf_blackwell_wheel
build_tf_blackwell_wheel:
	bash scripts/build_tf220_wheel.sh

#
.PHONY: install_tf_blackwell_wheel
install_tf_blackwell_wheel:
	bash scripts/install_tf220_wheel.sh

#
.PHONY: check_tf_blackwell
check_tf_blackwell:
	bash scripts/check_tf_blackwell.sh

#
.PHONY: bench_tf_blackwell
bench_tf_blackwell:
	bash scripts/bench_tf220.sh

#
.PHONY: full_transformer
full_transformer: transformer_features train_transformer_windows test_transformer_windows plot_transformer_summary xai_transformer

#
.PHONY: plot_transformer_summary
plot_transformer_summary:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_transformers

#
.PHONY: xai_transformer
xai_transformer:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.2_modeling.xai.xai run-all

#
.PHONY: hparam_search
hparam_search:
	$(PYTHON_INTERPRETER) educational_ai_analytics/2_modeling/transformers/utils/tune_random_search.py --max-trials $(HPARAM_MAX_TRIALS)

#
.PHONY: hparam_search_bg
hparam_search_bg:
	mkdir -p reports/hparam_search
	nohup $(PYTHON_INTERPRETER) educational_ai_analytics/2_modeling/transformers/utils/tune_random_search.py --max-trials $(HPARAM_MAX_TRIALS) > reports/hparam_search/night.log 2>&1 & echo $$! > reports/hparam_search/night.pid
	@echo "✅ Random search lanzado en background. PID guardado en reports/hparam_search/night.pid"

#
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

#
.PHONY: plot_ae_loss
plot_ae_loss:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae loss-curve

#
.PHONY: plot_ae_latent
plot_ae_latent:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae latent-space --method $(METHOD)

#
.PHONY: plot_ae_sample
plot_ae_sample:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae ae-reconstruction

#
.PHONY: plot_ae_compare_pca
plot_ae_compare_pca:
	@echo "⚠️ plot_ae compare-pca-vs-ae ya no existe; usando ae-reconstruction como salida equivalente disponible"
	$(MAKE) plot_ae_sample

#
.PHONY: plot_ae_compare_latent
plot_ae_compare_latent:
	@echo "⚠️ plot_ae compare-latent-spaces ya no existe; usando latent-space como salida equivalente disponible"
	$(MAKE) plot_ae_latent METHOD=$(METHOD)

#
.PHONY: plot_ae
plot_ae:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_ae clean
	$(MAKE) plot_ae_latent plot_ae_sample

#
.PHONY: plot_clustering
plot_clustering:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering clean
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering weekly-reports

plot_profiles:
	$(PYTHON_INTERPRETER) -m educational_ai_analytics.3_plots.plot_clustering profiles

#
.PHONY: prep_latent
prep_latent: data features train_ae encode plot_ae

#
.PHONY: ae_full_pca
ae_full_pca: train_ae encode
	$(MAKE) plot_ae METHOD=pca

#
.PHONY: pipeline
pipeline: prep_latent train_clustering predict_clustering

#
.PHONY: pipeline_clustering_plots
pipeline_clustering_plots: data features train_ae encode train_clustering predict_clustering plot_clustering

#
.PHONY: all_plots
all_plots: plot_ae plot_clustering plot_profiles

#
.PHONY: from_data_to_clustering_predict
from_data_to_clustering_predict: data features train_ae encode train_clustering predict_clustering

#
.PHONY: from_ae_train_to_predict
from_ae_train_to_predict: train_ae encode train_clustering predict_clustering

#
.PHONY: full_pipeline
full_pipeline: from_data_to_transformer

#
.PHONY: from_data_to_transformer
from_data_to_transformer: data features train_ae encode train_clustering predict_clustering all_plots transformer_features train_transformer_windows test_transformer_windows plot_transformer_summary xai_transformer

#
.PHONY: from_ae_train_to_transformer
from_ae_train_to_transformer: train_ae encode train_clustering predict_clustering all_plots transformer_features train_transformer_windows test_transformer_windows plot_transformer_summary xai_transformer

#
.PHONY: from_transformer_feat_to_xai from_transfor_feat_to_xai
from_transformer_feat_to_xai: transformer_features train_transformer_windows test_transformer_windows plot_transformer_summary xai_transformer

#
from_transfor_feat_to_xai: from_transformer_feat_to_xai

#
.PHONY: menu
menu:
	@echo ""
	@echo "================= 📋 MENU PIPELINES ================="
	@echo "  1) full_pipeline (incluye transformer, summary y XAI)"
	@echo "  2) all_plots"
	@echo "  3) from_data_to_clustering_predict"
	@echo "  4) from_ae_train_to_predict"
	@echo "  5) plot_clustering"
	@echo "  6) from_data_to_transformer"
	@echo "  7) from_ae_train_to_transformer"
	@echo "  8) from_transformer_feat_to_xai"
	@echo "  9) help"
	@echo "====================================================="
	@read -p "Elige una opción [1-9]: " opt; \
	case $$opt in \
		1) $(MAKE) full_pipeline ;; \
		2) $(MAKE) all_plots ;; \
		3) $(MAKE) from_data_to_clustering_predict ;; \
		4) $(MAKE) from_ae_train_to_predict ;; \
		5) $(MAKE) plot_clustering ;; \
		6) $(MAKE) from_data_to_transformer ;; \
		7) $(MAKE) from_ae_train_to_transformer ;; \
		8) $(MAKE) from_transformer_feat_to_xai ;; \
		9) $(MAKE) help ;; \
		*) echo "❌ Opción inválida. Ejecuta 'make menu' de nuevo." ;; \
	esac

#################################################################################

#################################################################################

.DEFAULT_GOAL := menu

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n#
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

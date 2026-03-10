import os
from importlib import import_module

from educational_ai_analytics.config import TRANSFORMER_PROFILE

from .schema import TransformerHyperparams

_PROFILE_MODULES = {
	"binary": ".binary",
	"binary-paper": ".binary_paper",
	"binary_paper": ".binary_paper",
	"paper": ".binary_paper",
	"trinary": ".trinary",
	"ternary": ".trinary",
	"quaternary": ".quaternary",
	"4class": ".quaternary",
}


def _resolve_profile_name() -> str:
	requested = os.getenv("TFM_TRANSFORMER_PROFILE", TRANSFORMER_PROFILE).strip().lower()
	if requested not in _PROFILE_MODULES:
		valid = ", ".join(sorted(_PROFILE_MODULES))
		raise ValueError(
			f"TFM_TRANSFORMER_PROFILE inválido: {requested}. Usa uno de: {valid}"
		)
	return requested


ACTIVE_PROFILE = _resolve_profile_name()
_profile_module = import_module(_PROFILE_MODULES[ACTIVE_PROFILE], package=__name__)

TRANSFORMER_PARAMS = _profile_module.TRANSFORMER_PARAMS

__all__ = ["ACTIVE_PROFILE", "TRANSFORMER_PARAMS", "TransformerHyperparams"]
import runpy

from educational_ai_analytics.config import PROJ_ROOT


def main():
    """
    Compatibilidad legacy: delega al pipeline activo en 1_features/__main__.py.
    """
    runpy.run_path(
        str(PROJ_ROOT / "educational_ai_analytics" / "1_features" / "__main__.py"),
        run_name="__main__",
    )

if __name__ == "__main__":
    main()

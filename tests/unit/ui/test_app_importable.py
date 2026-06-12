def test_app_module_imports_without_running_streamlit():
    # Importing must not execute any Streamlit calls (main() is guarded).
    # Run in a subprocess to avoid contaminating the test-process's sys.modules
    # with Streamlit's native-library side effects (e.g. weasyprint visibility).
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "mod = importlib.import_module('syneva.ui.app'); "
                "assert hasattr(mod, 'main') and callable(mod.main), "
                "'main() not found or not callable'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

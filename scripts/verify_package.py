"""Check an installed wheel against the existing local run without advancing replay."""
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import identitygraph
from identitygraph.api import create_app


def main():
    expected = Path(sys.argv[1]).resolve()
    package_file = Path(identitygraph.__file__).resolve()
    assert package_file.is_relative_to(expected), package_file
    root = Path(__file__).resolve().parents[1]
    assert Path.cwd() == root
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json()["status"] == "ready"
    for path in ("/", "/static/app.js", "/static/styles.css", "/openapi.json", "/api/summary"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.status_code)
    cli = subprocess.run([sys.executable, "-m", "identitygraph.cli", "--project-root", str(root),
                          "serve", "--help"], capture_output=True, text=True, check=True)
    assert "--port" in cli.stdout
    result = {"version": identitygraph.__version__, "imported_from": str(package_file),
              "current_folder_project_root_verified": True, "packaged_static_files_verified": True,
              "real_model_readiness_verified": True, "explicit_cli_root_verified": True}
    (root / "artifacts/verification/v1/package_checks.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import subprocess


def test_dynamic_versioning():
    obs = subprocess.run(["poetry", "version"], capture_output=True)
    assert obs.stdout.decode(encoding="utf-8").strip().split("pyam-iamc ")[1] != "0.0.0"

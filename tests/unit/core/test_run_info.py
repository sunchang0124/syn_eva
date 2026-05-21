from syneva.core.run_info import RunInfo


def test_capture_collects_versions_and_seed():
    info = RunInfo.capture(random_state=7)
    assert info.random_state == 7
    assert "pandas" in info.library_versions
    assert "syneva" in info.library_versions
    assert info.warnings == []


def test_to_dict_is_serializable():
    info = RunInfo.capture(random_state=1)
    d = info.to_dict()
    import json

    json.dumps(d)  # must not raise

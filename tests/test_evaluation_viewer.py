"""Read-only result browsing and HTTP ranges used by paired video playback."""

import json
import threading

import httpx
import pytest

from evaluation.viewer.server import Catalog, byte_range, create_server


@pytest.fixture
def viewer(tmp_path):
    directory = tmp_path / "drawer_1_0"
    directory.mkdir()
    gt = tmp_path / "reference.mp4"
    gen = tmp_path / "generated.mp4"
    gt.write_bytes(b"0123456789")
    gen.write_bytes(b"abcdefghij")
    (directory / "I_gt.jpg").write_bytes(b"gt-sheet")
    (directory / "I_gen.jpg").write_bytes(b"gen-sheet")
    result = {
        "case_id": directory.name,
        "status": "insufficient_evidence",
        "memory_score": None,
        "case": {
            "memory_target": "<cup>",
            "action_description": "open drawer",
            "gt_video": str(gt),
            "gen_video": str(gen),
        },
        "scores": {"dimensions": {"appearance": {"normalized": 0.6}}},
    }
    (directory / "result.json").write_text(json.dumps(result))
    (tmp_path / "run.json").write_text(json.dumps({"model": "test-model"}))
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
    ) as client:
        yield client, tmp_path, result
    server.shutdown()
    server.server_close()
    thread.join()


def test_catalog_preserves_partial_scores_and_null_overall(viewer):
    client, _, result = viewer
    catalog = client.get("/api/catalog").json()
    assert catalog["model"] == "test-model"
    case = catalog["cases"][0]
    assert case["category"] == "drawer"
    assert case["memory_score"] is None
    assert case["dimensions"]["appearance"]["normalized"] == 0.6
    assert client.get("/api/case/drawer_1_0").json() == result
    assert client.get("/").status_code == 200
    assert (
        client.get("/app.js")
        .headers["content-type"]
        .startswith(("text/javascript", "application/javascript"))
    )


def test_video_get_head_seek_and_invalid_range(viewer):
    client, _, _ = viewer
    path = "/media/drawer_1_0/gt_video"
    assert client.get(path).content == b"0123456789"
    response = client.get(path, headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["Content-Range"] == "bytes 2-5/10"
    assert client.get(path, headers={"Range": "bytes=-3"}).content == b"789"
    response = client.head(path, headers={"Range": "bytes=5-"})
    assert response.status_code == 206
    assert response.headers["Content-Length"] == "5"
    assert response.content == b""
    response = client.get(path, headers={"Range": "bytes=10-"})
    assert response.status_code == 416
    assert response.headers["Content-Range"] == "bytes */10"


def test_routes_expose_only_registered_artifacts(viewer):
    client, _, _ = viewer
    assert client.get("/media/drawer_1_0/gt_sheet").content == b"gt-sheet"
    for path in [
        "/api/case/unknown",
        "/media/unknown/gt_video",
        "/media/drawer_1_0/result.json",
        "/media/drawer_1_0/%2e%2e%2freference.mp4",
        "/%2e%2e/pyproject.toml",
    ]:
        assert client.get(path).status_code == 404


@pytest.mark.parametrize(
    "header", ["bytes=-0", "bytes=-", "bytes=9-2", "bytes=0-1,4-5", "bad"]
)
def test_invalid_single_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 10)


def test_catalog_rejects_case_identity_mismatch(viewer):
    _, root, result = viewer
    result["case_id"] = "../other"
    (root / "drawer_1_0" / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Invalid case identity"):
        Catalog(root)

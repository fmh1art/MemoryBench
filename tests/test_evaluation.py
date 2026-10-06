"""Protocol regressions: real timestamped videos, deterministic metrics and mocked VLM calls."""

import asyncio
import json
from fractions import Fraction
from pathlib import Path

import av
import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from evaluation.appearance.schema import Judgment as Appearance
from evaluation.environment.schema import Judgment as Environment
from evaluation.gate.schema import Judgment as Gate
from evaluation.inventory.schema import Judgment as Inventory
from evaluation.inventory.scoring import score_inventory
from evaluation.judge import Judge, validate_evidence
from evaluation.run import evaluate_case, load_cases, main
from evaluation.schema import Case, Metric
from evaluation.scoring import (
    average_memory_frames,
    score_memory,
    select_metrics,
    summarize,
)
from evaluation.spatial.schema import Judgment as Spatial
from evaluation.video import make_memory_pairs, make_sheets

EVIDENCE = [
    {
        "frame_indices": [9, 12],
        "observation": "GT and Gen show the same red cup in the drawer.",
    }
]


def metric(score=5, status="scored"):
    return {"status": status, "score": score, "evidence": EVIDENCE}


def replies():
    return {
        "gate": {
            "observability": {"label": 1, "evidence": EVIDENCE},
            "action_following": {"label": 1, "evidence": EVIDENCE},
        },
        "inventory": {
            "region_assessable": True,
            "objects": [
                {
                    "object_id": "cup",
                    "description": "red cup",
                    "gt_count": 1,
                    "gen_count": 1,
                    "evidence": EVIDENCE,
                }
            ],
            "evidence": EVIDENCE,
        },
        "appearance": {"visual_appearance": metric(), "geometry_shape": metric()},
        "spatial": {
            "object_localization": metric(),
            "relative_spatial_relation": metric(),
        },
        "environment": {
            "scene_structure": metric(),
            "background_appearance": metric(),
            "lighting_style": metric(None, "not_applicable"),
        },
    }


def make_video(path, times=None, size=(160, 80), color=(240, 20, 30)):
    times = times or [i / 10 for i in range(21)]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for timestamp in times:
            frame = av.VideoFrame.from_image(Image.new("RGB", size, color))
            frame.pts = round(timestamp * 1000)
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def case(tmp_path):
    gt, gen = tmp_path / "gt.mp4", tmp_path / "gen.mp4"
    make_video(gt)
    make_video(gen, size=(80, 160))
    return Case(
        case_id="demo",
        gt_video=str(gt),
        gen_video=str(gen),
        memory_target="red cup inside the drawer",
        action_description="open drawer",
        success_criterion="drawer open and interior visible",
        read_start_seconds=1.0,
    )


def test_uniform_physical_alignment_and_padding(case, tmp_path):
    gt_path, gen_path, data = make_sheets(case, tmp_path / "sheets")
    assert len(data["frames"]) == 12
    assert [f["time_seconds"] for f in data["frames"]] == pytest.approx(
        [2 * i / 11 for i in range(12)]
    )
    assert data["read_frame_indices"] == list(range(7, 13))
    assert all(
        abs(f["gt_actual_seconds"] - f["gen_actual_seconds"]) < 1e-9
        for f in data["frames"]
    )
    assert gt_path.name == "I_gt.jpg" and gen_path.name == "I_gen.jpg"
    with Image.open(gt_path) as gt, Image.open(gen_path) as gen:
        assert gt.size == gen.size == (2880, 748)
        # First tile content starts at y=76. Wide source gets vertical padding;
        # tall source gets horizontal padding. Neither is stretched.
        assert max(gt.getpixel((240, 90))) < 50
        assert gt.getpixel((240, 200))[0] > 200
        assert max(gen.getpixel((20, 200))) < 50
        assert gen.getpixel((240, 200))[0] > 200


def test_offsets_unequal_duration_and_vfr(case, tmp_path):
    # Different initial PTS are normalized independently; offsets refer to decoded local clocks.
    make_video(Path(case.gt_video), times=[5 + i / 10 for i in range(31)])
    make_video(Path(case.gen_video), times=[10 + i / 10 for i in range(41)])
    adjusted = case.model_copy(
        update={"gt_start_seconds": 0.5, "gen_start_seconds": 1.0}
    )
    _, _, data = make_sheets(adjusted, tmp_path / "offset")
    assert data["shared_end_seconds"] == pytest.approx(2.5)
    assert data["gen_uncovered_tail_seconds"] == pytest.approx(0.5)
    assert data["frames"][0]["gt_frame_index"] == 5
    assert data["frames"][0]["gen_frame_index"] == 10
    assert data["frames"][-1]["gen_actual_seconds"] == pytest.approx(2.5)
    # A large VFR gap must fail alignment rather than silently select distant frames.
    make_video(Path(case.gen_video), times=[0.0, 0.1, 1.8, 2.0])
    with pytest.raises(ValueError, match="cannot align"):
        make_sheets(case, tmp_path / "vfr")


def test_sparse_repeated_frames_and_read_coverage(case, tmp_path):
    make_video(Path(case.gt_video), times=[0.0, 1.0])
    make_video(Path(case.gen_video), times=[0.0, 1.0])
    sparse = case.model_copy(update={"read_start_seconds": 0.0})
    _, _, data = make_sheets(sparse, tmp_path / "sparse", max_skew_seconds=0.6)
    assert data["gt_unique_frames"] == data["gen_unique_frames"] == 2
    assert len(data["frames"]) == 12
    narrow = case.model_copy(
        update={"read_start_seconds": 0.41, "read_end_seconds": 0.42}
    )
    with pytest.raises(ValueError, match="misses the Read"):
        make_sheets(narrow, tmp_path / "narrow")
    outside = case.model_copy(update={"read_start_seconds": 3.0})
    with pytest.raises(ValueError, match="within shared"):
        make_sheets(outside, tmp_path / "outside")


def inventory(pairs, assessable=True):
    return Inventory(
        region_assessable=assessable,
        evidence=EVIDENCE,
        objects=[
            {
                "object_id": str(i),
                "description": "group",
                "gt_count": gt,
                "gen_count": gen,
                "evidence": EVIDENCE,
            }
            for i, (gt, gen) in enumerate(pairs)
        ],
    )


def test_inventory_counts_missing_spurious_and_normalization():
    scores = score_inventory(inventory([(3, 1), (1, 2), (0, 1)]))
    assert scores["object_presence"] == 1
    assert scores["object_count"] == pytest.approx(5 / 18)
    assert scores["missing_objects"] == scores["spurious_objects"] == 2
    assert scores["precision"] == scores["recall"] == scores["f1"] == 0.5
    assert scores["normalized"] == pytest.approx(16 / 27)
    assert scores["score"] == pytest.approx(80 / 27)


@pytest.mark.parametrize(
    "pairs, expected", [([], 1), ([(0, 2)], 0), ([(2, 0)], 0), ([(2, 2)], 1)]
)
def test_empty_and_presence_conventions(pairs, expected):
    assert score_inventory(inventory(pairs))["normalized"] == expected


@pytest.mark.parametrize("pairs, assessable", [([(2, None)], True), ([], False)])
def test_unresolved_inventory_is_not_silently_filtered(pairs, assessable):
    assert score_inventory(inventory(pairs, assessable))["normalized"] is None


@pytest.mark.parametrize(
    "score,status",
    [
        (6, "scored"),
        (-1, "scored"),
        (3.0, "scored"),
        (True, "scored"),
        (None, "scored"),
        (0, "not_applicable"),
        (2, "insufficient_evidence"),
    ],
)
def test_metric_contract_rejects_invalid_scores(score, status):
    with pytest.raises(ValidationError):
        Metric.model_validate(metric(score, status))


def test_metric_evidence_read_only_and_optional_style():
    data = replies()
    judged = Appearance.model_validate(data["appearance"])
    with pytest.raises(ValueError, match="Read"):
        validate_evidence("appearance", judged, [12], False)
    env = Environment.model_validate(data["environment"])
    env.lighting_style = Metric.model_validate(metric())
    with pytest.raises(ValueError, match="Disabled"):
        validate_evidence("environment", env, [9, 12], False)


def test_equal_dimension_weights_not_metric_count():
    data = replies()
    data["appearance"]["visual_appearance"] = metric(0)
    judgments = {
        "inventory": Inventory.model_validate(data["inventory"]),
        "appearance": Appearance.model_validate(data["appearance"]),
        "spatial": Spatial.model_validate(data["spatial"]),
        "environment": Environment.model_validate(data["environment"]),
    }
    result = score_memory(judgments)
    assert result["memory_score"] == 0.875  # (1 + .5 + 1 + 1) / 4
    judgments["spatial"].relative_spatial_relation = Metric.model_validate(
        metric(None, "not_applicable")
    )
    assert score_memory(judgments)["memory_score"] == 0.875
    judgments["spatial"].object_localization = Metric.model_validate(
        metric(None, "insufficient_evidence")
    )
    assert score_memory(judgments)["memory_score"] == pytest.approx(5 / 6)


class MockService:
    def __init__(self):
        self.outputs = replies()
        self.calls = []
        self.contexts = []
        self.active = self.peak = 0
        self.fail = None
        self.invalid_once = None

    async def __call__(self, request):
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        prompt = content[0]["text"]
        schema = json.loads(prompt.split("REQUIRED JSON SCHEMA:\n")[1])
        first = next(iter(schema["properties"]))
        dimension = {
            "observability": "gate",
            "region_assessable": "inventory",
            "visual_appearance": "appearance",
            "object_localization": "spatial",
            "scene_structure": "environment",
        }[first]
        self.calls.append(dimension)
        context = json.loads(
            prompt.split("CASE AND SAMPLING:\n")[1].split("\nREQUIRED JSON SCHEMA:")[0]
        )
        self.contexts.append(context)
        image_positions = [
            i for i, value in enumerate(content) if value["type"] == "image_url"
        ]
        assert len(image_positions) == 2
        assert content[image_positions[0] - 1]["text"] == "I_gt: GT REFERENCE"
        assert content[image_positions[1] - 1]["text"] == "I_gen: GENERATED VIDEO"
        assert "GT defines" in body["messages"][0]["content"]
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if self.fail == dimension:
            return httpx.Response(400, text="Do not persist this provider error body")
        if self.invalid_once == dimension:
            self.invalid_once = None
            response = "invalid JSON"
        else:
            data = json.loads(json.dumps(self.outputs[dimension]))
            if context.get("memory_frame"):
                index = context["memory_frame"]["frame_index"]

                def update_evidence(value):
                    if isinstance(value, dict):
                        if "frame_indices" in value:
                            value["frame_indices"] = [index]
                        for child in value.values():
                            update_evidence(child)
                    elif isinstance(value, list):
                        for child in value:
                            update_evidence(child)

                update_evidence(data)
            data = {
                key: value for key, value in data.items() if key in schema["properties"]
            }
            response = json.dumps(data)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": response}}
                ],
                "usage": {"total_tokens": 25},
            },
        )


def run_mock(case, output, service, concurrency=3, retries=0, selected_metrics=None):
    async def execute():
        judge = Judge(
            "https://mock.test/v1/chat/completions",
            "mock-vlm",
            "fake-secret",
            concurrency=concurrency,
            retries=retries,
            transport=httpx.MockTransport(service),
        )
        try:
            return await evaluate_case(
                case, judge, output, selected_metrics=selected_metrics
            )
        finally:
            await judge.close()

    return asyncio.run(execute())


def test_gate_then_parallel_memory_calls_and_audit(case, tmp_path):
    service = MockService()
    result = run_mock(case, tmp_path / "passing", service)
    assert result["status"] == "scored" and result["memory_score"] == 1
    assert service.calls[0] == "gate"
    assert len(service.calls) == 13
    assert all(
        service.calls.count(dim) == 4 for dim in ("inventory", "appearance", "spatial")
    )
    assert "environment" not in service.calls
    assert len(result["memory_frames"]) == 4
    assert service.peak == 3
    record = json.loads(
        (tmp_path / "passing/judges/frame_1/inventory.json").read_text()
    )
    assert record["attempts"][0]["parsed"]["objects"][0]["gt_count"] == 1
    assert record["context"]["memory_frame"]["frame_index"] == 1
    assert len(record["context"]["sampling"]["frames"]) == 1

    assert "fake-secret" not in json.dumps(record)
    assert "base64" not in json.dumps(record)


@pytest.mark.parametrize("label", ["observability", "action_following"])
def test_gate_failure_skips_all_memory_judges(case, tmp_path, label):
    service = MockService()
    service.outputs["gate"][label]["label"] = 0
    result = run_mock(case, tmp_path / label, service)
    assert result["status"] == "gate_failed" and result["memory_score"] is None
    assert service.calls == ["gate"]


def test_api_failure_finishes_other_judges_and_does_not_score_zero(case, tmp_path):
    service = MockService()
    service.fail = "appearance"
    result = run_mock(case, tmp_path / "error", service, retries=2)
    assert result["status"] == "error" and result["memory_score"] is None
    assert service.calls.count("appearance") == 4  # one terminal 400 per pair
    assert all(len(frame["judgments"]) == 2 for frame in result["memory_frames"])
    assert (
        "Do not persist"
        not in (tmp_path / "error/judges/frame_1/appearance.json").read_text()
    )


def test_invalid_json_retries_and_preserves_attempts(case, tmp_path):
    service = MockService()
    service.invalid_once = "inventory"
    result = run_mock(case, tmp_path / "retry", service, retries=1)
    assert result["status"] == "scored"
    assert service.calls.count("inventory") == 5
    record = json.loads((tmp_path / "retry/judges/frame_1/inventory.json").read_text())
    assert len(record["attempts"]) == 2 and "error" in record["attempts"][0]
    repair = record["attempts"][1]["validation_repair"]
    assert "COMPLETE JSON object" in repair
    assert "region_assessable, objects, evidence" in repair


def test_global_concurrency_across_cases(case, tmp_path):
    service = MockService()

    async def execute():
        judge = Judge(
            "https://mock.test",
            "mock",
            concurrency=2,
            transport=httpx.MockTransport(service),
        )
        try:
            return await asyncio.gather(
                evaluate_case(case, judge, tmp_path / "one"),
                evaluate_case(case, judge, tmp_path / "two"),
            )
        finally:
            await judge.close()

    results = asyncio.run(execute())
    assert all(result["status"] == "scored" for result in results)
    assert service.peak == 2 and len(service.calls) == 26
    summary = summarize(
        results
        + [
            {
                "status": "gate_failed",
                "gate": {
                    "observability": {"label": 0},
                    "action_following": {"label": 1},
                },
            },
            {"status": "error", "gate": None},
        ]
    )
    assert summary["scored_cases"] == 2 and summary["error_cases"] == 1
    assert summary["gate_evaluated"] == 3 and summary[
        "gate_pass_rate"
    ] == pytest.approx(2 / 3)
    assert (
        summary["memory_score"] == 1
        and "environment.lighting_style" not in summary["metrics"]
    )


def test_load_cases_paths_ids_and_duplicate_guard(case, tmp_path):
    data = case.model_dump()
    data.update(gt_video="gt.mp4", gen_video="gen.mp4")
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([data]))
    assert load_cases(path)[0].gt_video == str(tmp_path / "gt.mp4")
    path.write_text(json.dumps([data, data]))
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)
    with pytest.raises(ValidationError):
        Case.model_validate({**data, "case_id": "../unsafe"})


def test_output_overwrite_refused(case, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_mock(case, output, MockService())


def test_cli_input_error_writes_summary_and_exit_one(case, tmp_path, monkeypatch):
    case_file = tmp_path / "cases.json"
    data = case.model_dump()
    data["read_start_seconds"] = 20.0
    case_file.write_text(json.dumps([data]))
    output = tmp_path / "cli"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation.run",
            "--cases",
            str(case_file),
            "--endpoint",
            "https://unused.test",
            "--model",
            "mock",
            "--output",
            str(output),
        ],
    )
    assert main() == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["error_cases"] == 1 and summary["memory_score"] is None


@pytest.mark.parametrize("label", [True, False, 0.0, 1.0, "1", 2, -1])
def test_gate_requires_strict_binary_integer(label):
    data = replies()["gate"]
    data["observability"]["label"] = label
    with pytest.raises(ValidationError):
        Gate.model_validate(data)


def test_inventory_rejects_duplicate_and_empty_groups():
    data = replies()["inventory"]
    data["objects"].append(data["objects"][0].copy())
    with pytest.raises(ValidationError, match="unique"):
        Inventory.model_validate(data)
    with pytest.raises(ValidationError, match="absent in both"):
        inventory([(0, 0)])


def test_rate_pacer_uses_stricter_budget_and_does_not_burst(monkeypatch):
    from evaluation.rate_limit import RequestPacer

    clock = [0.0]
    monkeypatch.setattr("evaluation.rate_limit.time.monotonic", lambda: clock[0])

    async def advance(delay):
        clock[0] += delay

    monkeypatch.setattr("evaluation.rate_limit.asyncio.sleep", advance)

    async def execute():
        pacer = RequestPacer(qpm=300, tpm=60000, tokens_per_request=10000)
        starts = []
        for _ in range(3):
            await pacer.wait()
            starts.append(clock[0])
        assert starts == [0, 10, 20]
        clock[0] = 100
        await pacer.wait()
        await pacer.wait()
        assert clock[0] == 110  # no catching up after idle time
        with pytest.raises(ValueError, match="exceeding"):
            pacer.observe({"total_tokens": 11000})

    asyncio.run(execute())


@pytest.mark.parametrize(
    "kwargs",
    [{"qpm": 0}, {"tpm": float("nan")}, {"tpm": 100, "tokens_per_request": 101}],
)
def test_rate_pacer_rejects_invalid_limits(kwargs):
    from evaluation.rate_limit import RequestPacer

    with pytest.raises(ValueError):
        RequestPacer(**kwargs)


def test_temperature_can_be_omitted_for_terra(case, tmp_path):
    service = MockService()
    service.outputs["gate"]["observability"]["label"] = 0

    async def handle(request):
        assert "temperature" not in json.loads(request.content)
        return await service(request)

    async def execute():
        judge = Judge(
            "https://mock.test",
            "terra",
            temperature=None,
            transport=httpx.MockTransport(handle),
        )
        try:
            return await evaluate_case(case, judge, tmp_path / "default-temp")
        finally:
            await judge.close()

    assert asyncio.run(execute())["status"] == "gate_failed"


def test_connection_timeout_does_not_use_full_generation_timeout():
    async def execute():
        judge = Judge("https://mock.test", "mock", timeout=240)
        try:
            assert judge.client.timeout.connect == 15
            assert judge.client.timeout.read == 240
        finally:
            await judge.close()

    asyncio.run(execute())


def test_memory_pairs_sample_middle_sixty_percent_without_collages(case, tmp_path):
    _, _, gate_sampling = make_sheets(case, tmp_path)
    metadata = make_memory_pairs(case, tmp_path, gate_sampling)
    assert metadata["fractions"] == [0.2, 0.4, 0.6, 0.8]
    assert [frame["time_seconds"] for frame in metadata["frames"]] == pytest.approx(
        [0.4, 0.8, 1.2, 1.6]
    )
    assert [frame["is_read"] for frame in metadata["frames"]] == [
        False,
        False,
        True,
        True,
    ]
    with Image.open(tmp_path / "memory_frames/1_gt.jpg") as image:
        assert image.size == (160, 80)
    with Image.open(tmp_path / "memory_frames/1_gen.jpg") as image:
        assert image.size == (80, 160)


def test_only_selected_submetrics_are_judged_and_scored(case, tmp_path):
    service = MockService()
    selected = select_metrics(metrics=["appearance.visual_appearance"])
    result = run_mock(case, tmp_path / "selected", service, selected_metrics=selected)
    assert result["status"] == "scored"
    assert service.calls == ["gate"] + ["appearance"] * 4
    assert set(result["scores"]["dimensions"]) == {"appearance"}
    assert set(result["scores"]["metrics"]) == {"appearance.visual_appearance"}
    assert set(result["memory_frames"][0]["judgments"]["appearance"]) == {
        "visual_appearance"
    }
    summary = summarize([result])
    assert summary["dimensions"]["inventory"]["n"] == 0
    assert summary["inventory"]["missing_objects"]["n"] == 0


def test_environment_can_be_explicitly_enabled(case, tmp_path):
    service = MockService()
    result = run_mock(
        case,
        tmp_path / "environment",
        service,
        selected_metrics=select_metrics(["environment"]),
    )
    assert result["status"] == "scored"
    assert service.calls.count("environment") == 4
    assert "inventory" not in service.calls


def test_four_frame_scores_skip_unreliable_samples_and_retain_zero():
    frames = []
    for value in (0.0, 0.2, 0.6, 1.0):
        frames.append(
            {
                "status": "scored",
                "memory_score": value,
                "scores": {
                    "dimensions": {
                        "appearance": {"status": "scored", "normalized": value}
                    },
                    "metrics": {
                        "appearance.visual_appearance": {
                            "status": "scored",
                            "score": 5 * value,
                            "normalized": value,
                        }
                    },
                },
            }
        )
    aggregate = average_memory_frames(frames)
    assert aggregate["memory_score"] == pytest.approx(0.45)
    assert aggregate["metrics"]["appearance.visual_appearance"][
        "score"
    ] == pytest.approx(2.25)
    frames[0]["status"] = "insufficient_evidence"
    frames[0]["memory_score"] = None
    for group in ("dimensions", "metrics"):
        for item in frames[0]["scores"][group].values():
            item.update(status="insufficient_evidence", normalized=None)
    aggregate = average_memory_frames(frames)
    assert aggregate["memory_score"] == pytest.approx(0.6)
    assert aggregate["scored_frames"] == 3
    assert aggregate["excluded_frames"] == 1
    assert aggregate["metrics"]["appearance.visual_appearance"]["scored_frames"] == 3


def test_frame_averaging_retains_not_applicable_and_uses_available_denominator():
    frames = []
    for _ in range(4):
        item = {"status": "not_applicable", "normalized": None, "score": None}
        frames.append(
            {
                "status": "not_applicable",
                "memory_score": None,
                "scores": {
                    "dimensions": {"environment": dict(item)},
                    "metrics": {"environment.lighting_style": dict(item)},
                },
            }
        )
    aggregate = average_memory_frames(frames)
    assert aggregate["status"] == "not_applicable"
    assert aggregate["memory_score"] is None
    frames[0]["status"] = "scored"
    frames[0]["memory_score"] = 1.0
    for group in ("dimensions", "metrics"):
        for item in frames[0]["scores"][group].values():
            item.update(status="scored", normalized=1.0, score=5)
    assert average_memory_frames(frames)["status"] == "scored"
    assert average_memory_frames(frames)["memory_score"] == 1.0
    assert average_memory_frames(frames)["scored_frames"] == 1
    frames[0]["status"] = "insufficient_evidence"
    frames[0]["memory_score"] = None
    for group in ("dimensions", "metrics"):
        for item in frames[0]["scores"][group].values():
            item.update(status="insufficient_evidence", normalized=None, score=None)
    assert average_memory_frames(frames)["status"] == "insufficient_evidence"
    assert average_memory_frames(frames)["memory_score"] is None


def test_inventory_metric_selection_changes_only_the_aggregation():
    judgment = inventory([(1, 2)])
    scores = score_memory({"inventory": judgment}, ["inventory.object_presence"])
    assert scores["memory_score"] == 1.0
    assert set(scores["metrics"]) == {"inventory.object_presence"}
    assert scores["dimensions"]["inventory"]["spurious_objects"] == 1


def test_partial_metric_visibility_does_not_discard_reliable_scores():
    judgment = Appearance.model_validate(
        {
            "visual_appearance": metric(0),
            "geometry_shape": metric(None, "insufficient_evidence"),
        }
    )
    scores = score_memory({"appearance": judgment})
    assert scores["status"] == "scored"
    assert scores["memory_score"] == 0.0
    assert scores["dimensions"]["appearance"]["scored_samples"] == 1
    assert scores["metrics"]["appearance.geometry_shape"]["normalized"] is None

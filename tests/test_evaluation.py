"""v2 protocol checks use a scripted API; they do not claim to measure VLM accuracy."""

import json
from fractions import Fraction
from pathlib import Path
import av
import httpx
import numpy as np
import pytest
from evaluation.common.config import (
    Config,
    APIConfig,
    SamplingConfig,
    WeightConfig,
)
from evaluation.pipeline.pipeline import evaluate, pair_frames
from evaluation.metric.schemas import PairJudgment, GateJudgment
from evaluation.ground_truth.schema import MemoryAnnotation
from evaluation.metric.scoring import aggregate, score_pair
from evaluation.common.video import Frame, uniform_frames
from evaluation.common.vlm import VLM, EvaluationError
from evaluation.case_definition.assets import crop_video


def judgment(wrong=False):
    obj = {
        "id": "o1",
        "description": "有蓝色条纹和缺口的杯子",
        "identity_features": "独特蓝色条纹和杯沿缺口",
    }
    return {
        "gt_readable": True,
        "generated_readable": True,
        "gt_objects": [obj],
        "generated_objects": (
            [obj | {"identity_features": "不同的结构和花纹"}] if wrong else [obj]
        ),
        "verified_matches": (
            []
            if wrong
            else [
                {
                    "gt_id": "o1",
                    "gen_id": "o1",
                    "evidence": "两图独特条纹和杯沿缺口对应",
                }
            ]
        ),
        "appearance_items": [
            {"gt_id": "o1", "score": 0 if wrong else 0.75, "reason": "单物体外观判断"}
        ],
        "gt_spatial_applicable": False,
        "dimensions": {
            "inventory": {"score": 0 if wrong else 1, "reason": "实例数量"},
            "appearance": {"score": 0 if wrong else 0.75, "reason": "细节"},
            "spatial": {"score": None, "reason": "单物体，无可判定内部位置关系"},
        },
        "summary": "图像证据",
    }


def test_category_similarity_cannot_earn_memory_credit():
    j = PairJudgment.model_validate(judgment(wrong=True))
    row = {"scores": score_pair(j)[0]}
    assert aggregate([row], 1)["memory_score"] == 0
    raw = judgment(wrong=True)
    raw["dimensions"]["appearance"]["score"] = 0.5
    assert score_pair(PairJudgment.model_validate(raw))[0]["appearance"] == 0


def test_unmatched_objects_cannot_be_forced_or_receive_appearance_credit():
    raw = judgment()
    raw["verified_matches"][0]["gt_id"] = "unknown"
    with pytest.raises(ValueError, match="reference"):
        PairJudgment.model_validate(raw)
    raw = judgment()
    raw["gt_objects"].append(raw["gt_objects"][0] | {"id": "o2"})
    raw["dimensions"]["inventory"]["score"] = 2 / 3
    with pytest.raises(ValueError, match="appearance_items"):
        PairJudgment.model_validate(raw)


def test_weighted_average_and_na():
    rows = [
        {
            "scores": {
                "inventory": 1.0,
                "appearance": 0.5,
                "spatial": 0.0,
            }
        },
        {
            "scores": {
                "inventory": 0.0,
                "appearance": 0.0,
                "spatial": None,
            }
        },
    ]
    r = aggregate(rows, 1, {"inventory": 2, "appearance": 1, "spatial": 1})
    assert r["metrics"] == {
        "inventory": 0.5,
        "appearance": 0.25,
        "spatial": 0.0,
    }
    assert r["memory_score"] == 0.3125 and "state" not in r["effective_weights"]
    assert (
        aggregate([], 0)["memory_score"] == 0
        and aggregate([], 0)["metrics"]["inventory"] is None
    )
    with pytest.raises(ValueError):
        WeightConfig(inventory=0, appearance=0, spatial=0)


@pytest.mark.parametrize("a,b", [(1, 8), (3, 8), (8, 2), (3, 5), (1, 1), (8, 8)])
def test_short_clips_use_all_unique_frames_with_explicit_pairing(a, b):
    gt = [Frame("gt", i, i, float(i * i)) for i in range(a)]
    gen = [Frame("gen", i, i, float(i * 3)) for i in range(b)]
    pairs = pair_frames(gt, gen)
    assert len(pairs) == max(a, b)
    for role, frames in [("gt", gt), ("generated", gen)]:
        selected = [p[role].index for p in pairs]
        assert selected == sorted(selected) and set(selected) == {
            f.index for f in frames
        }
    assert len(uniform_frames(gt, 8)) == a


def write_video(path, count):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width = 96
        s.height = 64
        s.pix_fmt = "yuv420p"
        for i in range(count):
            f = av.VideoFrame.from_ndarray(
                np.full((64, 96, 3), i * 7, np.uint8), format="rgb24"
            )
            for packet in s.encode(f):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)


@pytest.fixture
def inputs(tmp_path):
    for name, count in [("gt", 12), ("gen", 20), ("action", 10)]:
        write_video(tmp_path / f"{name}.mp4", count)
    a = {
        "evaluation_case_id": "test",
        "memory_target": "抽屉里的物体",
        "read_action": "turn_view",
        "action_description": "转动视角",
        "success_criterion": "转向已打开的抽屉并看到内部",
    }
    p = tmp_path / "annotation.json"
    p.write_text(json.dumps(a))
    return [tmp_path / f"{n}.mp4" for n in ("gt", "gen", "action")] + [p]


class ScriptedAPI:
    def __init__(self, gate=True, wrong=False, repair=False, failure=False):
        self.gate = gate
        self.wrong = wrong
        self.repair = repair
        self.failure = failure
        self.calls = []

    def __call__(self, request):
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        text = content[0]["text"]
        data = json.loads(
            text.split("\nINPUT:\n")[1].split("\nREQUIRED JSON SCHEMA:\n")[0]
        )
        schema = json.loads(text.split("\nREQUIRED JSON SCHEMA:\n")[1])
        task = schema["title"]
        images = [x for x in content if x["type"] == "image_url"]
        self.calls.append((task, len(images), data))
        assert len(images) <= 8 and "MEMORY" in body["messages"][0]["content"]
        if self.failure:
            return httpx.Response(401, json={"error": "scripted"})
        if self.repair and len(self.calls) == 1:
            raw = {"wrong": "schema"}
        elif task == "GateJudgment":
            assert len(images) == 6 and data["annotation"]["read_action"] == "turn_view"
            raw = {
                "action_correct": self.gate,
                "target_visible": self.gate,
                "evidence_frame_ids": [data["requested_frames"][-1]["frame_id"]],
                "reason": "观察动作序列",
            }
        else:
            assert task == "PairJudgment" and len(images) == 2
            raw = judgment(self.wrong)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(raw)}, "finish_reason": "stop"}
                ]
            },
        )


def run(inputs, tmp_path, api):
    gt, gen, action, annotation = inputs
    return evaluate(
        gt,
        gen,
        Config(api=APIConfig(model="scripted", retries=1)),
        tmp_path / "out",
        action_path=action,
        annotation_path=annotation,
        transport=httpx.MockTransport(api),
        progress=lambda _: None,
    )


def test_exact_new_protocol(inputs, tmp_path):
    api = ScriptedAPI()
    r = run(inputs, tmp_path, api)
    assert r["framework_version"] == "2.1.0" and r["evaluated_frame_pairs"] == 8
    assert [x[0] for x in api.calls] == ["GateJudgment"] + ["PairJudgment"] * 8
    assert r["memory_score"] == 0.875 and len(r["selected_frames"]["gt_images"]) == 8
    assert (tmp_path / "out/pairs/pair_08.json").exists()


def test_gate_failure_skips_all_memory_queries(inputs, tmp_path):
    api = ScriptedAPI(gate=False)
    r = run(inputs, tmp_path, api)
    assert (
        r["memory_score"] == 0
        and r["evaluated_frame_pairs"] == 0
        and len(api.calls) == 1
    )


def test_reject_wrong_same_category_instances_in_full_pipeline(inputs, tmp_path):
    r = run(inputs, tmp_path, ScriptedAPI(wrong=True))
    assert r["memory_score"] == 0


def test_json_repair_and_api_failure(inputs, tmp_path):
    api = ScriptedAPI(repair=True)
    r = run(inputs, tmp_path, api)
    assert r["vlm_request_count"] == 10


def test_api_failure_is_error_not_zero(inputs, tmp_path):
    with pytest.raises(EvaluationError, match="401"):
        run(inputs, tmp_path, ScriptedAPI(failure=True))
    r = json.loads((tmp_path / "out/result.json").read_text())
    assert r["memory_score"] is None and r["status"] == "error"


def test_max_eight_images_enforced_before_http(tmp_path):
    client = VLM(
        APIConfig(model="test"),
        tmp_path,
        transport=httpx.MockTransport(lambda _: pytest.fail("HTTP must not run")),
    )
    try:
        with pytest.raises(ValueError, match="at most 8"):
            client.ask("action_gate", {}, [("x", "missing")] * 9, GateJudgment)
    finally:
        client.close()
    with pytest.raises(ValueError):
        APIConfig(max_images_per_query=9)


def test_action_crop_exact_half_open_interval(inputs, tmp_path):
    out = tmp_path / "crop.mp4"
    assert crop_video(inputs[1], out, 2, 8) == 6
    with av.open(str(out)) as c:
        assert len(list(c.decode(video=0))) == 6


def test_visual_judgments_survive_llm_arithmetic_error():
    raw = judgment()
    raw["gt_objects"].append(raw["gt_objects"][0] | {"id": "o2"})
    raw["appearance_items"].append(
        {
            "gt_id": "o2",
            "score": 1,
            "reason": "bad arithmetic: this unmatched object must count zero",
        }
    )
    raw["dimensions"]["inventory"]["score"] = 0.9
    raw["dimensions"]["appearance"]["score"] = 1
    scores, changes = score_pair(PairJudgment.model_validate(raw))
    assert scores["inventory"] == pytest.approx(2 / 3) and scores["appearance"] == 0.375
    assert len(changes) == 2

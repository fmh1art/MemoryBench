"""Exact 2026-10-11 archive CSV schema; preserve all unrelated raw records.

This is the server-side operational serializer, with no spreadsheet/runtime
dependency. Scores belong to the evaluator, never to the inference publisher.
"""
import csv
import io


FIELDS = tuple(('version,checkpoint,step,inference_status,evaluation_status,expected_cases,'
    'prediction_cases,vlm_cases,automated_cases,vlm_error_cases,automated_error_cases,'
    'memory_score_100,gate_pass_rate,inventory_score_100,appearance_score_100,spatial_score_100,'
    'benchmark_repo,benchmark_commit,evaluator_commit,scoring_version,prediction_run,'
    'evaluation_run,completed_at,summary_path,automated_summary_path,run_config_path,'
    'updated_at,notes').split(','))


def records(text):
    """Return parsed rows and their physical-line spans, including quoted newlines."""
    lines = text.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text, newline=''), strict=True)
    if tuple(next(reader, ())) != FIELDS:
        raise ValueError('Unsupported EVAL_CHECKLIST.csv columns; review current schema')
    newline = '\r\n' if lines[0].endswith('\r\n') else '\n'
    if not text.endswith(newline):
        raise ValueError('Checklist requires a terminated final record')
    previous, rows, seen = reader.line_num, [], set()
    for values in reader:
        if len(values) != len(FIELDS):
            raise ValueError('Malformed checkpoint CSV record')
        row = dict(zip(FIELDS, values))
        key = (row['version'], row['checkpoint'])
        if not all(key) or key in seen:
            raise ValueError('Missing or duplicate checkpoint identity')
        seen.add(key)
        rows.append((row, previous, reader.line_num))
        previous = reader.line_num
    return rows, lines, newline


def validate(text):
    return [row for row, _, _ in records(text)[0]]


def encoded_row(row, newline):
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator=newline).writerow([row[key] for key in FIELDS])
    return stream.getvalue()


def append_checkpoint(text, name, inference, model, timestamp):
    rows, _, newline = records(text)
    if any(row['version'] == 'v0.5' and row['checkpoint'] == name for row, _, _ in rows):
        raise ValueError('Checkpoint already in checklist; explicit reconciliation required')
    from gpu_policy import ELIGIBLE_GPUS, POLICY_ID
    if (inference['cases'] != 649 or inference['gpus'] != ELIGIBLE_GPUS
            or inference['gpu_policy'] != POLICY_ID):
        raise ValueError('Unexpected daily inference scope')
    from kda_contract import ALGORITHM, identity
    if (model['algorithm'] != ALGORITHM or model['source_dense_step'] != 80000
            or any(model.get(k) != v for k,v in identity(name).items())):
        raise ValueError('Unexpected KDA/VSA staged lineage')
    prefix = 'predictions/'
    output = inference['output']
    if not output.startswith(prefix) or '/' in output[len(prefix):] or not output[len(prefix):]:
        raise ValueError('Expected one named prediction run')
    row = dict.fromkeys(FIELDS, '')
    # Existing Direct64 rows leave ambiguous "step" blank. Record both distinct
    # counters in notes rather than mislabeling the data cursor as optimizer steps.
    row.update(version='v0.5', checkpoint=name, inference_status='待推理', evaluation_status='待评估',
        expected_cases=str(inference['cases']), prediction_cases='0', benchmark_repo=inference['dataset'],
        benchmark_commit=inference['commit'], prediction_run=output[len(prefix):], updated_at=timestamp,
        notes=f"Dense80k→KDA23/VSA7 LW20 模型归档完成（光轮固定保留20%样本、EgoQuest全保留、桶内循环，非20%采样权重）；stage={model['stage']}；stage_step={model['stage_step']}；pipeline_step={model['pipeline_step']}；"
              f"next_data_cursor={model['next_data_cursor']}；计划 649 case，仅 GPU 0–3；训练保存完整 checkpoint 后让出，评测后恢复；推理成功后自动接 38k v3.4 口径评测")
    return text + encoded_row(row, newline)


def complete_inference(text, name, run_name, files, cases, benchmark_repo, benchmark_commit, timestamp):
    rows, lines, newline = records(text)
    matches = [(row, start, end) for row, start, end in rows
               if row['version'] == 'v0.5' and row['checkpoint'] == name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one published checkpoint checklist row')
    row, start, end = matches[0]
    if (row['expected_cases'] != str(cases) or row['benchmark_repo'] != benchmark_repo
            or row['benchmark_commit'] != benchmark_commit or row['prediction_run'] != run_name):
        raise ValueError('Checklist benchmark/run identity changed; preserve generated predictions')
    receipt = f'v0.5/ckps/{name}/predictions/{run_name}/batch-completion.json'
    if receipt in row['notes']:
        if row['inference_status'] != '已完成' or row['prediction_cases'] != str(cases):
            raise ValueError('Conflicting existing inference completion')
        return text
    row.update(inference_status='已完成', prediction_cases=str(cases), updated_at=timestamp)
    row['notes'] += f'；本轮推理 {cases}/{cases}，产物 {files} 个；本脚本未运行评分；完成记录 {receipt}'
    # Evaluation status/counts/scores/completed_at/results paths are untouched.
    return ''.join(lines[:start]) + encoded_row(row, newline) + ''.join(lines[end:])


def complete_evaluation(text, name, prediction_run, evaluation_run, summary, automated, recipe, timestamp):
    from eval_contract import require_scores_complete
    require_scores_complete(summary, automated)
    rows, lines, newline = records(text)
    matches = [(row, start, end) for row, start, end in rows
               if row['version'] == 'v0.5' and row['checkpoint'] == name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one scoring checklist row')
    row, start, end = matches[0]
    if (row['inference_status'] != '已完成' or row['prediction_cases'] != '649'
            or row['expected_cases'] != '649' or row['prediction_run'] != prediction_run
            or row['benchmark_repo'] != recipe['benchmark_repo']
            or row['benchmark_commit'] != recipe['benchmark_commit']):
        raise ValueError('Checklist inference/dataset is not the evaluated run')
    if row['evaluation_status'] == '已完成' or row['memory_score_100'] or row['evaluation_run']:
        raise ValueError('Refuse to replace existing scores without explicit reconciliation')
    prefix = f'v0.5/ckps/{name}/results/{evaluation_run}'
    row.update(evaluation_status='已完成',vlm_cases='649',automated_cases='649',
        vlm_error_cases='0',automated_error_cases='0',memory_score_100=str(summary['memory_score_100']),
        gate_pass_rate=str(summary['gate_pass_rate']),inventory_score_100=str(summary['report']['inventory']),
        appearance_score_100=str(summary['report']['appearance']),spatial_score_100=str(summary['report']['spatial']),
        evaluator_commit=recipe['evaluator_commit'],scoring_version=recipe['scoring_version'],
        evaluation_run=evaluation_run,completed_at=timestamp,updated_at=timestamp,
        summary_path=prefix+'/summary.json',automated_summary_path=prefix+'/automated/summary.json',
        run_config_path=prefix+'/run.json')
    from gpu_policy import validate_active
    active = validate_active(recipe['active_gpus'])
    row['notes'] += f'；后续自动评测完成 649/649；复用 38k 原生 v3.4 口径；评分 TP1/DP{len(active)}，实际 GPU {active}'
    return ''.join(lines[:start]) + encoded_row(row, newline) + ''.join(lines[end:])

"""Bounded repair of the observed Inventory both-zero format error only.

Valid native replies, Gate failures, prompts, frames and scores are untouched.
There is no generic response editing, coercion, or score-selected retry.
"""
import json
from pathlib import Path
from pydantic import ValidationError
from inventory_decoder import constrained_inventory_schema, InventorySchemaTransport

POLICY = 'inventory_both_zero_native_schema_repair_v1'
ERROR = 'Do not list object groups absent in both videos'


def require_empty_group_failure(record):
    from evaluation.judge import SCHEMAS
    native = SCHEMAS['inventory']
    if record['dimension'] != 'inventory' or record['schema'] != native.model_json_schema():
        raise ValueError('Inventory schema or dimension differs; no guessed repair')
    last = record['attempts'][-1]
    if last.get('http_status') != 200 or last.get('finish_reason') != 'stop':
        raise ValueError('Only complete, successful-HTTP Inventory responses can be format-repaired')
    try:
        native.model_validate_json(last['response_text'])
    except ValidationError as error:
        errors = error.errors()
        if not errors or any(e['type'] != 'value_error' or len(e['loc']) != 2
                or e['loc'][0] != 'objects' or not isinstance(e['loc'][1],int)
                or str(e.get('ctx',{}).get('error')) != ERROR for e in errors):
            raise ValueError('Unsupported Inventory validation error; original failure retained') from error
    else:
        raise ValueError('Never repair a schema-valid Inventory response')


async def constrained_inventory(record, output, endpoint, model, options, *, limit=None, pacer=None):
    from evaluation.judge import Judge, SCHEMAS, write_json
    require_empty_group_failure(record)
    output = Path(output)
    schema = constrained_inventory_schema(SCHEMAS['inventory'].model_json_schema())
    write_json(output/'inventory.decoder-schema.json',schema)
    audit = []
    def record_request(event):
        audit.append(event)
        write_json(output/'inventory.decoder-requests.json',audit)
    if options.get('transport') is not None:
        raise ValueError('Nested transport is unsupported')
    judge = Judge(endpoint,model,**dict(options,transport=InventorySchemaTransport(schema,record_request)))
    try:
        if (judge.rubrics['inventory'] != record['rubric'] or judge.memory_common != record['common_rubric']
                or model != record['model'] or judge.temperature != record['temperature']
                or judge.max_tokens != record['max_tokens']):
            raise ValueError('Frozen Inventory prompt/model/sampling changed')
        if limit is not None: judge.limit = limit
        if pacer is not None: judge.pacer = pacer
        images = tuple(Path(item['path']) for item in record['images'])
        judgment = await judge.ask('inventory',record['context'],images,output)
    finally:
        await judge.close()
    write_json(output/'inventory.format-repair.json',dict(policy=POLICY,
        original_prompt_unchanged=True,original_images_unchanged=True,
        native_schema_unchanged=True,native_scoring_unchanged=True,
        manual_response_editing=False,first_schema_valid_response_final=True,
        transport_requests=len(audit),documentation='https://docs.vllm.ai/en/latest/features/structured_outputs/'))
    return judgment


def make_judge(endpoint, model, **options):
    """Keep the native call path; only a diagnosed failed Inventory gets repair."""
    from evaluation.judge import Judge
    class InventoryRepairJudge(Judge):
        async def ask(self,dimension,context,images,output,*,schema_override=None,rubric_override=None):
            try:
                return await super().ask(dimension,context,images,output,
                    schema_override=schema_override,rubric_override=rubric_override)
            except RuntimeError:
                if dimension != 'inventory' or schema_override is not None or rubric_override is not None:
                    raise
                output = Path(output)
                source = output/'inventory.json'
                record = json.loads(source.read_text())
                require_empty_group_failure(record)
                if (record['context'] != context
                        or [item['path'] for item in record['images']] != [str(p) for p in images]):
                    raise ValueError('Failed Inventory call identity differs')
                # Preserve the original native attempts before its log is replaced.
                with (output/'inventory.unconstrained.failed.json').open('xb') as backup:
                    backup.write(source.read_bytes())
                return await constrained_inventory(record,output,endpoint,model,options,
                    limit=self.limit,pacer=self.pacer)
    return InventoryRepairJudge(endpoint,model,**options)

"""Encode the existing native (gt_count, gen_count) != (0, 0) validator.

This is the FM90k-tested decoder, not a change to observations or scoring.
Native Pydantic validation remains authoritative for all other predicates.
"""
import copy
import hashlib
import json
import httpx


def constrained_inventory_schema(native):
    schema = copy.deepcopy(native)
    group = schema['$defs']['ObjectGroup']
    for field in ('gt_count', 'gen_count'):
        if group['properties'][field].get('anyOf') != [{'minimum': 0, 'type': 'integer'}, {'type': 'null'}]:
            raise ValueError('Unrecognized native count schema; no guessed conversion')
    nullable = {'anyOf': [{'type': 'integer', 'minimum': 0}, {'type': 'null'}]}
    integer = {'type': 'integer', 'minimum': 0}
    positive = {'type': 'integer', 'minimum': 1}
    null = {'type': 'null'}
    zero = {'type': 'integer', 'const': 0}
    variants = []
    # Disjoint native-valid count pairs, excluding exactly (0, 0).
    for gt, gen in ((null, nullable), (integer, null), (positive, integer), (zero, positive)):
        branch = copy.deepcopy(group)
        branch['properties']['gt_count'] = copy.deepcopy(gt)
        branch['properties']['gen_count'] = copy.deepcopy(gen)
        variants.append(branch)
    schema['$defs']['ObjectGroup'] = {'anyOf': variants, 'title': group['title']}
    return schema


class InventorySchemaTransport(httpx.AsyncBaseTransport):
    """For a dedicated Inventory judge only; change response_format, nothing else."""
    def __init__(self, schema, audit, backend=None):
        self.schema, self.audit = schema, audit
        self.backend = backend if backend is not None else httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        body = json.loads(request.content)
        if (str(request.url) != 'http://127.0.0.1:18047/v1/chat/completions'
                or body['response_format'] != {'type': 'json_object'}
                or body['model'] != 'qwen3.8-27b' or body['temperature'] != 0
                or body['max_tokens'] != 8192 or body['chat_template_kwargs'] != {'enable_thinking': False}):
            raise ValueError('Unsupported inventory request identity')
        original = copy.deepcopy(body)
        body['response_format'] = dict(type='json_schema', json_schema={
            'name': 'native_inventory_nonempty_object_group', 'schema': self.schema, 'strict': True})
        if {k:v for k,v in body.items() if k != 'response_format'} != {
                k:v for k,v in original.items() if k != 'response_format'}:
            raise AssertionError('Native request changed beyond response format')
        self.audit(dict(native_request_sha256=hashlib.sha256(request.content).hexdigest(),
            messages_sha256=hashlib.sha256(json.dumps(body['messages'],ensure_ascii=False).encode()).hexdigest(),
            unchanged_fields=[k for k in original if k != 'response_format'],
            original_response_format=original['response_format'],
            response_format_type='json_schema',
            decoder_schema_sha256=hashlib.sha256(json.dumps(self.schema,sort_keys=True).encode()).hexdigest()))
        headers = {k:v for k,v in request.headers.items() if k.lower() != 'content-length'}
        forwarded = httpx.Request(request.method,request.url,headers=headers,json=body,extensions=request.extensions)
        return await self.backend.handle_async_request(forwarded)

    async def aclose(self):
        await self.backend.aclose()

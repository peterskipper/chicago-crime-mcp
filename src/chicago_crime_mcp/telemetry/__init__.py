"""Observability: structured JSON logging of each tool call (trace id, args,
resolved query plan, row count, latency, compacted result) and the batch
outcome/failure telemetry rollup (empty-result rate, resolve_neighborhood
misses, malformed-arg rate) that treats failures as the product signal.
"""

import pyarrow as pa

TASKS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), False),
        pa.field("submission_time", pa.int64(), False),
        pa.field("duration", pa.int64(), False),
        pa.field("cpu_count", pa.int32(), False),
        pa.field("cpu_capacity", pa.float64(), False),
        pa.field("mem_capacity", pa.int64(), False),
        pa.field("gpu_count", pa.int32(), False),
        pa.field("gpu_capacity", pa.float64(), False),
        pa.field("gpu_mem_capacity", pa.int64(), False),
        pa.field("model_name", pa.string(), False),
        pa.field("gpu_name", pa.string(), False),
        pa.field("total_tokens", pa.int64(), False),
    ]
)

FRAGMENTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), False),
        pa.field("duration", pa.int64(), False),
        pa.field("cpu_count", pa.int32(), False),
        pa.field("cpu_usage", pa.float64(), False),
        pa.field("gpu_count", pa.int32(), False),
        pa.field("gpu_usage", pa.float64(), False),
    ]
)

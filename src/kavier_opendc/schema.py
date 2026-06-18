import pyarrow as pa

TASKS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32(), False),
        pa.field("submission_time", pa.timestamp("ms"), False),
        pa.field("duration", pa.int64(), False),
        pa.field("cpu_count", pa.int32(), False),
        pa.field("cpu_capacity", pa.float64(), False),
        pa.field("mem_capacity", pa.int64(), False),
        pa.field("gpu_count", pa.int32(), True),
        pa.field("gpu_capacity", pa.float64(), True),
        # carried through for the kavier-energy efficiency step (per-token normalisation).
        # OpenDC reads tasks by column name and ignores this extra field.
        pa.field("total_tokens", pa.int64(), True),
    ]
)

FRAGMENTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32(), False),
        pa.field("duration", pa.duration("ms"), False),
        pa.field("cpu_count", pa.int32(), False),
        pa.field("cpu_usage", pa.float64(), False),
        pa.field("gpu_count", pa.int32(), True),
        pa.field("gpu_usage", pa.float64(), True),
    ]
)

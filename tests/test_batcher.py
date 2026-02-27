from nerva.engine.batcher import BatchConfig


def test_batch_config_defaults() -> None:
    cfg = BatchConfig()
    assert cfg.max_batch_size == 32
    assert cfg.max_delay_ms == 10.0
    assert cfg.queue_capacity == 2048
    assert cfg.queue_timeout_ms == 100.0
    assert cfg.min_remaining_deadline_ms == 5.0


def test_batch_config_custom() -> None:
    cfg = BatchConfig(max_batch_size=8, max_delay_ms=5.0)
    assert cfg.max_batch_size == 8
    assert cfg.max_delay_ms == 5.0
    assert cfg.queue_capacity == 2048

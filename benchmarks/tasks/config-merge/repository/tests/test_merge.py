from config_merge import merge_config


def test_replaces_scalar_value() -> None:
    result = merge_config(
        {"timeout": 10},
        {"timeout": 30},
    )

    assert result == {"timeout": 30}


def test_adds_new_top_level_value() -> None:
    result = merge_config(
        {"timeout": 10},
        {"retries": 3},
    )

    assert result == {
        "timeout": 10,
        "retries": 3,
    }


def test_merges_nested_dictionary() -> None:
    result = merge_config(
        {
            "service": {
                "host": "localhost",
                "port": 8000,
            }
        },
        {
            "service": {
                "port": 9000,
            }
        },
    )

    assert result == {
        "service": {
            "host": "localhost",
            "port": 9000,
        }
    }


def test_merges_multiple_nested_levels() -> None:
    result = merge_config(
        {
            "database": {
                "pool": {
                    "minimum": 1,
                    "maximum": 10,
                },
                "host": "localhost",
            }
        },
        {
            "database": {
                "pool": {
                    "maximum": 20,
                }
            }
        },
    )

    assert result == {
        "database": {
            "pool": {
                "minimum": 1,
                "maximum": 20,
            },
            "host": "localhost",
        }
    }


def test_result_does_not_share_nested_values() -> None:
    base = {
        "service": {
            "timeout": 10,
        }
    }

    result = merge_config(base, {})
    service = result["service"]

    assert isinstance(service, dict)

    service["timeout"] = 99

    assert base == {
        "service": {
            "timeout": 10,
        }
    }


def test_override_replaces_list() -> None:
    result = merge_config(
        {
            "regions": [
                "us-east-1",
                "us-west-2",
            ]
        },
        {
            "regions": [
                "eu-west-1",
            ]
        },
    )

    assert result == {
        "regions": [
            "eu-west-1",
        ]
    }

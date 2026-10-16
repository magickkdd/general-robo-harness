from harness.brain.prompt import parse_action_json, repair_prompt


def test_valid_move():
    a, err = parse_action_json(
        '{"dx": 0.1, "dy": -0.2, "dz": 0.3, "skill": "none", "object": ""}')
    assert a == {"move": [0.1, -0.2, 0.3], "skill": "none", "object": ""} and err == ""


def test_valid_pick():
    a, err = parse_action_json(
        '{"dx": 0, "dy": 0, "dz": 0, "skill": "pick", "object": "cube_red"}')
    assert a == {"move": [0.0, 0.0, 0.0], "skill": "pick",
                 "object": "cube_red"} and err == ""


def test_embedded_json():
    a, err = parse_action_json(
        'Thinking... {"dx":0,"dy":0,"dz":0.1,"skill":"none","object":""} done')
    assert a["move"] == [0.0, 0.0, 0.1] and a["skill"] == "none" and err == ""


def test_defaults_fill():
    a, err = parse_action_json('{"skill": "none"}')
    assert a == {"move": [0.0, 0.0, 0.0], "skill": "none", "object": ""} and err == ""


def test_no_json():
    a, err = parse_action_json("no object here")
    assert a is None and "no JSON" in err


def test_unknown_skill():
    a, err = parse_action_json('{"skill": "fly", "object": "x"}')
    assert a is None and "unknown skill" in err


def test_missing_object_for_skill():
    a, err = parse_action_json('{"skill": "pick"}')
    assert a is None and "missing 'object'" in err


def test_out_of_range_rejected():
    a, err = parse_action_json('{"dx": 99.0, "skill": "none"}')
    assert a is None and "range" in err


def test_repair_prompt_mentions_error():
    p = repair_prompt("bad", "invalid JSON: x")
    assert "invalid JSON" in p and "bad" in p

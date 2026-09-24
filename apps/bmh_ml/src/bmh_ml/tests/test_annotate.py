from bmh_ml.tracking.annotate import BEGIN, END, format_number, get_kind, merge_description, run_link


def test_the_generated_description_replaces_the_earlier_one_and_keeps_text_written_by_hand():
    first = merge_description(None, "generated 1")
    assert first == f"{BEGIN}\ngenerated 1\n{END}"

    edited = f"My note above.\n\n{first}\n\nMy note below."
    second = merge_description(edited, "generated 2")

    assert second == f"My note above.\n\n{BEGIN}\ngenerated 2\n{END}\n\nMy note below."
    assert merge_description("Only my note.", "generated") == f"Only my note.\n\n{BEGIN}\ngenerated\n{END}"


def test_the_kind_of_a_run_follows_from_its_params_and_tags():
    assert get_kind({"sweep.model": "mlp"}, {"sweep": "s"}) == "sweep"
    assert get_kind({"refine.base": "B"}, {"refine": "r"}) == "refine"
    assert get_kind({"model": "mlp"}, {"sweep": "s", "trial": "3"}) == "sweep-trial"
    assert get_kind({"model": "mlp"}, {"refine": "r", "round": "2"}) == "refine-round"
    assert get_kind({"model": "mlp"}, {"refine": "r", "round": "-1"}) == "refine-control"
    assert get_kind({"model": "mlp"}, {}) == "training"
    assert get_kind({}, {}) == "other"


def test_links_and_numbers_are_formatted_for_the_ui():
    assert run_link("1", "abcdef123456") == "[abcdef12](#/experiments/1/runs/abcdef123456)"
    assert format_number(0.123456) == "0.123"
    assert format_number(250000.0) == "250,000"
    assert format_number(float("nan")) == ""

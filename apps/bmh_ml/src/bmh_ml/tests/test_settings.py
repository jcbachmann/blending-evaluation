import argparse

import pytest

from bmh_ml.settings import DEFAULT_MODEL_SET, add_model_set_argument, get_model_file, get_model_set, get_scaler_file


def test_the_default_model_set_is_the_one_trained_on_the_random_dataset():
    # A deliberate choice, the results of the experiments made so far are based on the models of this set
    assert DEFAULT_MODEL_SET == "random_training_data"


def test_files_of_a_model_set_keep_the_names_of_the_existing_models():
    assert get_model_file("lstm_model_f1", "random_training_data") == "data/lstm_model_f1_random_training_data.keras"
    assert get_model_file("lstm_model_f2", "random_training_data") == "data/lstm_model_f2_random_training_data.keras"
    assert get_scaler_file("random_training_data") == "data/scaler_random_training_data.pkl"


@pytest.mark.parametrize("name", ["random_training_data", "grouped-1", "A1"])
def test_valid_model_set_names(name):
    assert get_model_set(name) == name


@pytest.mark.parametrize("name", ["", "../other", "a/b", "a b", "a.b", "ä"])
def test_invalid_model_set_names_are_rejected(name):
    with pytest.raises(argparse.ArgumentTypeError, match="not a valid model set name"):
        get_model_set(name)


def test_model_set_argument_defaults_to_the_default_set():
    parser = argparse.ArgumentParser()
    add_model_set_argument(parser)

    assert parser.parse_args([]).model_set == DEFAULT_MODEL_SET
    assert parser.parse_args(["--model-set", "other"]).model_set == "other"


def test_a_required_model_set_argument_has_no_default():
    parser = argparse.ArgumentParser()
    add_model_set_argument(parser, required=True)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["--model-set", "other"]).model_set == "other"


def test_an_invalid_model_set_argument_is_rejected():
    parser = argparse.ArgumentParser()
    add_model_set_argument(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--model-set", "../other"])

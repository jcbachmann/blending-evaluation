import os


def get_identifier(path: str) -> str:
    """
    Split by slashes and get last non-empty entry
    :param path: path from which the identifier is derived
    :return: identifier - last entry in path
    """
    # Replace all \ with / to be platform agnostic
    normalized_path = path.replace("\\", "/")
    return [entry for entry in normalized_path.split("/") if entry][-1]


def get_identifiers(paths: list[str]) -> list[str]:
    """
    Call get_identifier() for each path in the list
    :param paths: list of paths containing identifiers
    :return: list of identifiers
    """
    return [get_identifier(path) for path in paths]

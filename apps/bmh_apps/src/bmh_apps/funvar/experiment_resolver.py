import logging
import os
import re

EXPERIMENT_PATTERN = re.compile(r"E(XPERIMENT)?-?(([0-9a-f]{1,8})(-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?)")


def shorten_experiment_label(label: str) -> str:
    match = EXPERIMENT_PATTERN.fullmatch(label)
    if match:
        return f"E{match.group(3)}"
    return label


class ExperimentResolver:
    def __init__(self, root_dir: str | None = None):
        self.root_dir = root_dir
        self.experiments = {}
        self.indexed = False

    def index_experiments(self, root_dir: str):
        self.indexed = True
        count = 0
        for parent_dir, dir_names, _ in os.walk(root_dir):
            for dir_name in dir_names:
                if EXPERIMENT_PATTERN.fullmatch(dir_name):
                    path = os.path.abspath(os.path.join(parent_dir, dir_name))
                    if dir_name not in self.experiments:
                        count += 1
                        self.experiments[dir_name] = path
                    else:
                        logging.warning(f"Experiment {dir_name} already indexed at {self.experiments[dir_name]} - skipping {path}")
        logging.info(f"Indexed {count} experiments")

    def ensure_indexed(self, file_path: str):
        # Walking the experiments directory is slow, so only do it once an experiment actually needs to be resolved
        if self.indexed:
            return
        if self.root_dir is None:
            raise ValueError(f"Cannot resolve experiment {file_path} without an experiments root directory (set BMH_EXPERIMENTS_PATH)")
        self.index_experiments(self.root_dir)

    def resolve(self, file_paths: list[str]) -> list[str]:
        resolved_paths = []
        for file_path in file_paths:
            match = EXPERIMENT_PATTERN.fullmatch(file_path)
            if match:
                self.ensure_indexed(file_path)
                uuid_prefix = match.group(2)
                matching_keys = [k for k in self.experiments if EXPERIMENT_PATTERN.fullmatch(k).group(2).startswith(uuid_prefix)]
                if len(matching_keys) == 0:
                    raise ValueError(f"Experiment {file_path} not indexed")
                if len(matching_keys) == 1:
                    resolved_paths.append(self.experiments[matching_keys[0]] + "/*/")
                else:
                    raise ValueError(f"Experiment {file_path} not unique")
            else:
                resolved_paths.append(file_path)
        return resolved_paths

class Evaluation:
    def __init__(self, identifier: str, path: str):
        self.identifier = identifier
        self.path = path
        self.references = None

    def __str__(self):
        return self.identifier

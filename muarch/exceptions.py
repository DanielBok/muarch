class NotFittedError(Exception):
    def __init__(self):
        super().__init__("ArchModel instance is not fitted yet")


class InvalidModelError(Exception):
    def __init__(self, index: int):
        super().__init__(f'The model in index {index} must be an instance of UArch')


class EndogInputError(Exception):
    def __init__(self):
        super().__init__("number of columns in data 'y' does not match expected dimension of MUArch object")

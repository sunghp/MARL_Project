class EnvironmentParametersChannel:
    def __init__(self):
        self.params = {}

    def set_float_parameter(self, key, value):
        self.params[key] = float(value)

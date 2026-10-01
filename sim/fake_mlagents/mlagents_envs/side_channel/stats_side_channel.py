class StatsSideChannel:
    def __init__(self):
        self.stats = {}

    def get_and_reset_stats(self):
        out = self.stats
        self.stats = {}
        return out

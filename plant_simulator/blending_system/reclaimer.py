class Reclaimer:
    def __init__(self, max_speed: float):
        self.stockpile = None
        self.max_speed = max_speed

    def reclaim(self):
        if self.reclaiming_finished():
            return 0, 0
        else:
            return self.stockpile.reclaim(self.max_speed)

    def reclaiming_finished(self):
        return self.stockpile is None or self.stockpile.reclaiming_finished()

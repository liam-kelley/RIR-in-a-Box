import time

class LKTimer:
    def __init__(self):
        self.times = {}

    def time(self, key):
        return TimerContext(self, key)

    def log_time(self, key, duration):
        if key not in self.times:
            self.times[key] = []
        self.times[key].append(duration)

    def get_logs(self, key=None):
        if key:
            return self.times.get(key, [])
        return self.times
    
    def reset(self):
        self.times = {}

class TimerContext:
    def __init__(self, timer_logger, key):
        self.timer_logger = timer_logger
        self.key = key

    def __enter__(self):
        print(self.key)
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        self.timer_logger.log_time(self.key, duration)
        # print(f"Block '{self.key}' executed in {duration:.4f} seconds")

def main():
    # Example usage:
    timer_logger = LKTimer()
    with timer_logger.time("block1"):
        time.sleep(2)
    with timer_logger.time("block2"):
        time.sleep(1)
    print(timer_logger.get_logs())
    # Output
    # Block 'block1' executed in 2.0012 seconds
    # Block 'block2' executed in 1.0006 seconds
    # {'block1': [2.0012459754943848], 'block2': [1.0006451606750488]}

if __name__ == "__main__":
    main()
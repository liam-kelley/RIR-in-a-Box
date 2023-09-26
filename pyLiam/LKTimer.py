import time

class LKTimer:
    def __init__(self,print_time=True):
        self.times = {}
        self.print_time = print_time

    def time(self, key, print_time=None):
        if print_time==None:
            return TimerContext(self, key, print_time=self.print_time)
        else:
            return TimerContext(self, key, print_time=print_time)

    def log_time(self, key, duration):
        self.times[key]=duration

    def get_logs(self, key=None):
        if key:
            return self.times.get(key, [])
        return self.times
    
    def reset(self):
        self.times = {}

class TimerContext:
    def __init__(self, timer_logger, key, print_time):
        self.timer_logger = timer_logger
        self.key = key
        self.print_time = print_time

    def __enter__(self):
        if self.print_time : print(self.key)
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        self.timer_logger.log_time(self.key, duration)
        if self.print_time : print(f"LKTimer : {duration:.4f} seconds")

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
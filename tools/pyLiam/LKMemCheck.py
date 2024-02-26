from collections.abc import Iterable
from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated
import pandas as pd

def format_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format, e.g.:
    1253656 => '1.20MiB', 1253656678 => '1.17GiB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
    return f"{bytes:.2f}Y{suffix}"

class LKMemCheck:
    def __init__(self, print_during_mem_check=False, print_at_last_mem_check=True, always_reset_lists=True):
        self.print_during_mem_check = print_during_mem_check
        self.print_at_last_mem_check = print_at_last_mem_check
        self.always_reset_lists = always_reset_lists

        self.enable_flag = True

        self.current_iteration = 0
        self.iteration_index = []
        self.step_name_index = []
        self.memory_allocated =  []
        self.memory_reserved = []

    def disable(self):
        self.enable_flag = False

    def enable(self):
        self.enable_flag = True
    
    def reset_lists(self):
        self.iteration_index = []
        self.step_name_index = []
        self.memory_allocated =  []
        self.memory_reserved = []
    
    def next_iteration(self):
        self.current_iteration += 1

    def memcheck(self, step_name="", last_step=False):
        if self.enable_flag:
            if self.print_during_mem_check:
                print(step_name, " memory_allocated: " , format_size(memory_allocated()), "\t\tmemory_reserved: ", format_size(memory_reserved()))
            self.iteration_index.append(self.current_iteration)
            self.step_name_index.append(step_name)
            self.memory_allocated.append(format_size(memory_allocated()))
            self.memory_reserved.append(format_size(memory_reserved()))

            if last_step:
                if self.print_at_last_mem_check:
                    self.print_mem_report()
                self.next_iteration()
                if self.always_reset_lists:
                    self.reset_lists()
        
    def get_df(self):
        return pd.DataFrame({"iteration": self.iteration_index, "step_name": self.step_name_index,
                             "memory_allocated": self.memory_allocated, "memory_reserved": self.memory_reserved})
    
    def print_mem_report(self):
        df = self.get_df()
        
        # only where current iteration is the current iteration
        df  = df[df["iteration"]==self.current_iteration]
        print(df)
    
    def print_complete_mem_report(self):
        print(self.get_df())

    def print_max(self):
        if self.enable_flag:
            print("max memory allocated: ", format_size(max_memory_allocated()))

    

# log_redirect.py
import sys


class Tee(object):
    def __init__(self, *files):
        self.files = files
        self._original_stdout = sys.__stdout__  # Original stdout
        self._original_stderr = sys.__stderr__  # Original stderr

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate output

    def flush(self):
        for f in self.files:
            f.flush()

    def __getattr__(self, attr):
        return getattr(self._original_stdout, attr)  # Access original stdout attributes


def setup_logging(log_file_path='outlog.log'):
    log_file = open(log_file_path, 'a')  # append mode
    original_stdout = sys.stdout  # Save original stdout
    original_stderr = sys.stderr  # Save original stderr

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    return log_file, original_stdout, original_stderr


def restore_logging(log_file, original_stdout, original_stderr):
    log_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
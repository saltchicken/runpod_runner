import os
import sys
from contextlib import contextmanager


@contextmanager
def quiet(suppress_stdout=True, suppress_stderr=True):
    """
    Redirects C-level and Python-level stdout/stderr to /dev/null
    to silence libraries like PyTorch and ComfyUI.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = sys.stdout.fileno()
        old_stderr_fd = sys.stderr.fileno()

        saved_stdout_fd = os.dup(old_stdout_fd)
        saved_stderr_fd = os.dup(old_stderr_fd)

        try:
            if suppress_stdout:
                sys.stdout.flush()
                os.dup2(devnull.fileno(), old_stdout_fd)
            if suppress_stderr:
                sys.stderr.flush()
                os.dup2(devnull.fileno(), old_stderr_fd)

            yield

        finally:
            if suppress_stdout:
                os.dup2(saved_stdout_fd, old_stdout_fd)
            if suppress_stderr:
                os.dup2(saved_stderr_fd, old_stderr_fd)

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

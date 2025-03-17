import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
import signal
import select
import psutil
from pprint import pprint

import numpy as np

from openrlhf.remote_rm.ds_prover.lean.ast_parser import lean4_parser
from openrlhf.remote_rm.ds_prover.scheduler import ProcessScheduler
from openrlhf.remote_rm.ds_prover.utils import AttrDict


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'

DEFAULT_LEAN_WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mathlib4')


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, 
                      last_env=None, verbose=False, timeout=300, allTactics=False, 
                      ast=False, premises=False, tactics=False, threads=4):
    
    unique_id = f"{os.getpid()}_{threading.get_ident()}_{int(time.time()*1000)}"
    tmp_file_path = None
    process = None
    children = []
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{unique_id}.lean', dir=lean_workspace, delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            tmp_file.write(code)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        
        process = subprocess.Popen(
            [lake_path, 'env', 'lean', 
             f'--threads={threads}',  # multi thread
             f'--memory={8*1024}',    
             os.path.basename(tmp_file_path)],
            cwd=lean_workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Use a new process group
        )
        
        # Record the PID of the main process for subsequent cleanup
        main_pid = process.pid
        
        # Set timeout
        start_time = time.time()
        stdout_chunks = []
        stderr_chunks = []
        
        
        while process.poll() is None:

            if time.time() - start_time > timeout:
                # Timeout handling - recursively terminate the process tree using psutil
                try:
                    parent = psutil.Process(main_pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        try:
                            child.terminate()
                        except:
                            pass
                    
        
                    gone, still_alive = psutil.wait_procs(children, timeout=1)
                    
                    # Forcefully terminate still alive processes
                    for p in still_alive:
                        try:
                            p.kill()
                        except:
                            pass
                    
                    # Terminate the main process
                    try:
                        parent.terminate()
                        parent.wait(1)
                    except:
                        try:
                            parent.kill()
                        except:
                            pass
                except:
                    # If psutil method fails, fall back to traditional method
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
                
                return {
                    'pass': False,
                    'complete': False,
                    'status': 'timeout',
                    'system_errors': f"Process timed out after {timeout}s",
                    'system_messages': f"Process timed out after {timeout}s",
                    'output': ''.join(stdout_chunks)
                }
            
            # Non-blocking read some output
            stdout_ready = select.select([process.stdout], [], [], 0.1)[0]
            if stdout_ready:
                output = process.stdout.readline()
                if output:
                    stdout_chunks.append(output)
            
            stderr_ready = select.select([process.stderr], [], [], 0.1)[0]
            if stderr_ready:
                error = process.stderr.readline()
                if error:
                    stderr_chunks.append(error)
                    
            time.sleep(0.1)  # Avoid CPU overuse
            
        # Read remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_chunks.append(remaining_stdout)
        if remaining_stderr:
            stderr_chunks.append(remaining_stderr)
        
        stdout = ''.join(stdout_chunks)
        stderr = ''.join(stderr_chunks)
        
        return {
            'pass': process.returncode == 0,
            'complete': process.returncode == 0,
            'system_errors': stderr if stderr else None,
            'system_messages': stderr,
            'output': stdout
        }
        
    except Exception as e:
        # Cleanup process
        if process:
            try:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except:
                        pass
                parent.kill()
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
                    
        return {
            'pass': False,
            'complete': False,
            'status': 'error',
            'system_errors': str(e),
            'system_messages': str(e)
        }
    finally:
        # 
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
        # Additional process cleanup, find missing processes
        try:
            # Find related processes by filename
            if tmp_file_path:
                file_base = os.path.basename(tmp_file_path)
                ps = subprocess.run(
                    f"ps aux | grep '{file_base}' | grep -v grep | awk '{{print $2}}'",
                    shell=True, text=True, capture_output=True
                )
                for pid in ps.stdout.strip().split("\n"):
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except:
                            pass
        except:
            pass


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
    
    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        
        total_tasks = 0
        completed_tasks = 0
        
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            
            batch_size = len(inputs)
            total_tasks += batch_size
            
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if 'timeout' not in task:
                    task['timeout'] = self.timeout
                    
                try:
                    result = verify_lean4_file(**task)
                    completed_tasks += 1
                    
                    # Concise progress display
                    if completed_tasks % 10 == 0:  # Update every 10 tasks
                        progress = completed_tasks / total_tasks * 100
                        print(f"\rProcess-{self.idx} Progress: {progress:.1f}%", end="", flush=True)
                        
                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1
                        
                except Exception as e:
                    with self.lock:
                        self.request_statuses[request_id] = {
                            'pass': False,
                            'complete': False,
                            'system_errors': str(e)
                        }
        
        if total_tasks > 0:
            print(f"\nProcess-{self.idx} completed {completed_tasks}/{total_tasks} tasks")


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier'):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Complete launching {len(self.processes)} LeanServerProcesses')

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(['killall', 'repl', f'--older-than={int(self.timeout) + 10}s'], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f'All {len(self.processes)} LeanServerProcesses stopped')


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.code.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    pprint(outputs_list)

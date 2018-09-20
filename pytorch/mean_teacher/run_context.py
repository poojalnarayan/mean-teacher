from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os

from pandas import DataFrame
from collections import defaultdict


class TrainLog:
    """Saves training logs in Pandas msgpacks"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.msgpack".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        col_val_dict_np = {}
        for (k,v) in col_val_dict.items():
            if str(type(v)) == '<class \'torch.Tensor\'>':
                col_val_dict_np[k] = col_val_dict[k].cpu().numpy()
            else:
                col_val_dict_np[k] = col_val_dict[k]
        self._record(step, col_val_dict_np)

    def save(self):
        df = self._as_dataframe()
        df.to_msgpack(self.log_file_path, compress='zlib')

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
                self._last_update_time = time.time()
                self.save()

    def _as_dataframe(self):
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx, run_name):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        runner_name = os.path.basename(runner_file).split(".")[0] + '_' + run_name
        self.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}/{run_idx}".format(
            root='results',
            runner_name=runner_name,
            date=datetime.now(),
            run_idx=run_idx
        )
        self.transient_dir = self.result_dir + "/transient"
        os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)

    def create_train_log(self, name):
        return TrainLog(self.result_dir, name)

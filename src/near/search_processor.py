import queue
import threading
import subprocess
import numpy as np
import struct
from typing import Optional, Tuple
from time import sleep


class AsyncNearResultsProcessor:
    def __init__(self, output_file,
                       query_data,
                       target_data,
                       target_labels):

        self.output_file = output_file
        self.queue = queue.Queue()
        self.done = False
        self.error: Optional[Exception] = None

        # Start processing thread
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()

        # Will be set by first batch
        self.total_hits = None
        self.process = None

    def add_to_queue(self, scores: np.ndarray, target_indices: np.ndarray, query_ids: np.ndarray) -> None:
        """
        Add a batch of results to the processing queue.
        batch: Tuple of (query_ids, target_ids, scores) arrays
        """
        if self.error:
            raise RuntimeError("Processor encountered an error") from self.error
        self.queue.put((scores, target_indices, query_ids))

    def _start_process(self, first_batch_size: int) -> None:
        """Initialize the C process with total hits count"""
        self.process = subprocess.Popen(
            ["near.process_near_results"],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Write initial hit count
        self.process.stdin.write(struct.pack('Q', first_batch_size))
        self.process.stdin.flush()

    def _process_queue(self) -> None:
        """Worker thread that processes the queue"""
        try:
            while not self.done or not self.queue.empty():
                try:
                    # Wait for data with timeout to check done flag periodically
                    batch = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                query_ids, target_ids, scores = batch

                # Initialize process if this is the first batch
                if self.process is None:
                    self._start_process(len(scores))

                # Write batch data to process
                query_ids.astype(np.uint64).tofile(self.process.stdin)
                target_ids.astype(np.uint64).tofile(self.process.stdin)
                scores.astype(np.float32).tofile(self.process.stdin)
                self.process.stdin.flush()

                self.queue.task_done()

        except Exception as e:
            self.error = e
        finally:
            if self.process:
                self.process.stdin.close()
                stderr = self.process.stderr.read()
                if self.process.wait() != 0:
                    self.error = RuntimeError(
                        f"Process failed: {stderr.decode()}"
                    )

    @property
    def not_done(self) -> bool:
        """Check if processing is still ongoing"""
        return self.thread.is_alive() or not self.queue.empty()

    def finalize(self) -> None:
        """Signal completion and wait for processing to finish"""
        self.done = True
        self.thread.join()
        if self.error:
            raise self.error
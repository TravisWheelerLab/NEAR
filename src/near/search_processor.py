import queue
import threading
import subprocess
import numpy as np
import struct
from .fasta_data import FASTAData
from typing import Optional, Tuple
from time import sleep
from importlib.resources import files
import os
import sys

class AsyncNearResultsProcessor:
    """Asynchronously processes NEAR search results using a C subprocess.

    This class handles the communication with a C process that formats and writes
    search results to an output file. It uses a queue to buffer results and processes
    them in a separate thread.

    Parameters
    ----------
    output_file : int
        Number of input (and output) channels.
    query_data : FASTAData
        The query sequence data
    target_data : FASTAData
        The target sequence data
    hits_per_emb : int
        The number of hits per query embedding
    embeddings_per_target : np.array
        The number of embeddings corresponding to each target sequence.
    """

    def __init__(self, output_file: str,
                        query_data: FASTAData,
                        target_data: FASTAData,
                        query_lengths,
                        target_lengths,
                        hits_per_emb: int,
                        filter_1,
                        filter_2,
                        sparsity,
                        angle_deviation_data,
                        stats):

        self.output_file = output_file

        self.query_data = query_data
        self.target_data = target_data

        self.query_lengths = query_lengths
        self.target_lengths = target_lengths

        self.hits_per_emb = hits_per_emb

        self.filter_1 = filter_1
        self.filter_2 = filter_2

        self.sparsity = sparsity

        self.angle_deviation_data = angle_deviation_data
        self.stats = stats

        self.queue = queue.Queue()
        self.done = False
        self.error: Optional[Exception] = None

        # Start processing thread
        self.thread = threading.Thread(target=self._start_process)
        self.thread.start()

        self.process = None

    def add_to_queue(self, query_ids: np.array, target_ids, scores: np.ndarray) -> None:
        """Add a batch of scores, query_ids, and target_ids to the queue.

        Parameters
        ----------
        scores : np.ndarray
            [N, k] array of nearest neighbor scores
        query_ids : np.array
            [N] query IDs
        target_ids : FASTAData
            [N, k] target IDs
        """

        if self.error:
            raise RuntimeError("Processor encountered an error") from self.error
        self.queue.put((query_ids, target_ids, scores))

    def _start_process(self) -> None:
        executable_path = str(files('near').joinpath('bin/process_near_results'))

        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Could not find executable at {executable_path}")
        self.log_file1 = open('near_log1.txt', 'w')
        self.log_file2 = open('near_log2.txt', 'w')

        self.process = subprocess.Popen(
            [executable_path,
             self.output_file,
             str(self.hits_per_emb),
             str(self.filter_1),
             str(self.filter_2),
             str(self.sparsity),
             str(len(self.stats[0])),
             str(128),
             ],
            stdin=subprocess.PIPE,
           # stdout=self.log_file1,
           # stderr=self.log_file2#,
            bufsize=1024 * 1024 * 512
        )

       # print(self.query_data.seqid_to_name[0], self.query_data.seqid_to_name[-1])
       # print(self.target_data.seqid_to_name[0], self.target_data.seqid_to_name[-1])
        #sys.stdout.flush()

        log_adds = self.stats[0]
        distributions = self.stats[1]

        self.process.stdin.write(log_adds.tobytes())
        self.process.stdin.flush()

        self.process.stdin.write(distributions[0].flatten().tobytes())  # Shape
        self.process.stdin.flush()

        self.process.stdin.write(distributions[1].flatten().tobytes())  # Loc
        self.process.stdin.flush()

        self.process.stdin.write(distributions[2].flatten().tobytes())  # Scale
        self.process.stdin.flush()

        self.process.stdin.write(self.angle_deviation_data.tobytes())  # cosine info
        self.process.stdin.flush()

        # Write the query data sequence names
        self.process.stdin.write(struct.pack('Q', len(self.query_data.seqid_to_name)))
        self.process.stdin.flush()

        query_seq_names = ('\0'.join(self.query_data.seqid_to_name) + '\0').encode('utf-8')

        self.process.stdin.write(struct.pack('Q', len(query_seq_names)))
        self.process.stdin.flush()

        self.process.stdin.write(query_seq_names)
        self.process.stdin.flush()

        self.process.stdin.write(self.query_lengths.tobytes())
        self.process.stdin.flush()

        # Write the target data sequence names
        self.process.stdin.write(struct.pack('Q', len(self.target_data.seqid_to_name)))
        self.process.stdin.flush()


        target_seq_names = ('\0'.join(self.target_data.seqid_to_name) + '\0').encode('utf-8')
        self.process.stdin.write(struct.pack('Q', len(target_seq_names)))
        self.process.stdin.flush()

        self.process.stdin.write(target_seq_names)
        self.process.stdin.flush()

        self.process.stdin.write(self.target_lengths.tobytes())
        self.process.stdin.flush()

        self._process_queue()

    def _process_queue(self) -> None:
        try:
            while not self.done or not self.queue.empty():
                try:
                    # Wait for data with timeout to check done flag periodically
                    batch = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                query_ids, target_ids, scores = batch
                if query_ids is None:
                    self.process.stdin.write(struct.pack('Q', 0))
                    self.process.stdin.flush()
                    self.queue.task_done()
                    return

                query_ids, target_ids, scores = query_ids.flatten(), target_ids.flatten(), scores.flatten()
                # Write batch data to process
                #print(query_ids.shape, target_ids.shape, scores.shape)
                #print(query_ids.dtype, target_ids.dtype, scores.dtype)
                scores = scores.astype(np.float64)
                self.process.stdin.write(struct.pack('Q', len(query_ids)))
                self.process.stdin.write(query_ids)
                self.process.stdin.write(target_ids)
                self.process.stdin.write(scores)
                self.process.stdin.flush()

                self.queue.task_done()

        except Exception as e:
            self.error = e
        finally:
            if self.process:
                self.process.stdin.close()

    def not_done(self) -> bool:
        return self.thread.is_alive() or not self.queue.empty()

    def finalize(self) -> None:
        """Signal completion and wait for processing to finish"""
        self.done = True
        self.thread.join()
        self.log_file1.close()
        self.log_file2.close()

        if self.error:
            raise self.error
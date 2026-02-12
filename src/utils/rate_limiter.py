import json
import time
import os
import fcntl
from pathlib import Path
from typing import Optional

class FileLock:
    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.fd = None

class RateLimiter:
    """
    Cross-process rate limiter using a Token Bucket algorithm backed by a file.
    
    Attributes:
        max_tokens (float): Maximum burst capacity.
        refill_rate (float): Tokens added per second.
        state_file (str): Path to the shared state file.
        trading_reserve (float): Number of tokens reserved for HIGH priority requests.
    """
    def __init__(self, 
                 max_tokens: float = 20.0, 
                 refill_rate: float = 10.0, 
                 state_file: str = ".cache/rate_limit.state",
                 trading_reserve: float = 5.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.state_file = state_file
        self.trading_reserve = trading_reserve
        
        # Ensure directory exists
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if not exists
        if not os.path.exists(self.state_file):
            with open(self.state_file, 'w') as f:
                json.dump({"tokens": max_tokens, "last_update": time.time()}, f)

    def _load_state(self, fd) -> dict:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            content = os.read(fd, 1024).decode('utf-8')
            if not content:
                return {"tokens": self.max_tokens, "last_update": time.time()}
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return {"tokens": self.max_tokens, "last_update": time.time()}

    def _save_state(self, fd, state: dict):
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(state).encode('utf-8'))

    def wait(self, priority: str = "LOW", timeout: float = 30.0) -> bool:
        """
        Blocks until a token is available.
        
        Args:
            priority (str): "HIGH" for trading, "LOW" for monitoring.
            timeout (float): Max wait time in seconds.
            
        Returns:
            bool: True if token acquired, False if timeout.
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                return False

            with FileLock(self.state_file) as lock:
                state = self._load_state(lock.fd)
                now = time.time()
                
                # 1. Refill
                elapsed = now - state["last_update"]
                new_tokens = elapsed * self.refill_rate
                state["tokens"] = min(self.max_tokens, state["tokens"] + new_tokens)
                state["last_update"] = now
                
                # 2. Check Availability
                # HIGH priority needs 1 token.
                # LOW priority needs 1 token + reserve.
                needed = 1.0
                threshold = 1.0 if priority == "HIGH" else (1.0 + self.trading_reserve)
                
                if state["tokens"] >= threshold:
                    state["tokens"] -= 1.0
                    self._save_state(lock.fd, state)
                    return True
                
                # 3. Wait time calculation (prevent busy spin)
                # How many tokens we need to reach threshold?
                missing = threshold - state["tokens"]
                wait_sec = missing / self.refill_rate
                
            # Sleep outside the lock
            time.sleep(min(wait_sec, 0.1)) # Sleep at least a bit, but cap at 0.1s check

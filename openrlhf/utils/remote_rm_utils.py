import time
import ray
import requests
import torch
import threading
import uuid

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

class RequestQueue:
    def __init__(self):
        self._queue = {}
        self._lock = threading.Lock()
        self._stats = {
            'processed': 0,
            'failed': 0,
            'last_stats_time': time.time()
        }
    
    def _log_stats(self, force=False):
        now = time.time()
        if force or now - self._stats['last_stats_time'] >= 60:  # 每60秒记录一次统计
            with self._lock:
                queue_size = len(self._queue)
                processed = self._stats['processed']
                failed = self._stats['failed']
                
                logger.info(
                    f"Queue Stats: waiting={queue_size}, "
                    f"processed={processed}/min, failed={failed}/min"
                )
                
                self._stats.update({
                    'processed': 0,
                    'failed': 0,
                    'last_stats_time': now
                })
    
    def add_request(self, request_id, data):
        with self._lock:
            self._queue[request_id] = {
                'data': data,
                'timestamp': time.time(),
                'retries': 0
            }
        self._log_stats()
    
    def get_request(self, request_id):
        with self._lock:
            return self._queue.get(request_id)
    
    def remove_request(self, request_id, success=True):
        with self._lock:
            if success:
                self._stats['processed'] += 1
            else:
                self._stats['failed'] += 1
            return self._queue.pop(request_id, None)

request_queue = RequestQueue()


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def request_api_wrapper_lean(url, data, score_key="rewards", try_max_times=3):
    headers = {"Content-Type": "application/json"}
    request_id = str(uuid.uuid4())
    request_queue.add_request(request_id, data)
    
    # check lean server status
    try:
        status_resp = requests.get(f"{url.split('/predict')[0]}/status", timeout=5)
        if status_resp.status_code == 200:
            server_status = status_resp.json()
            queue_size = server_status.get('queue_size', 0)
            active_requests = server_status.get('active_requests', 0)
            
            # if queue_size > 50, wait 15 seconds
            if queue_size > 50:
                logger.warning(f"Server queue size ({queue_size}) is very high, waiting more time before sending request")
                time.sleep(15)  
            elif queue_size > 20:
                logger.info(f"Server queue size ({queue_size}) is high, waiting before sending request")
                time.sleep(5) 
    except Exception as e:
        logger.warning(f"Failed to check server status: {e}")
    
    try:
        max_retries = 20  
        retries = 0
        
        while retries < max_retries:
            try:
                headers['X-Request-ID'] = request_id
                response = requests.post(
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=600  
                )
                
                if response.status_code == 500:  # Internal Server Error
                    logger.info(f"Server error for request {request_id}")
                    request_queue.remove_request(request_id, success=False)
                    return [-1.0] * len(data.get('query', []))
                    
                if response.status_code == 202:  # Processing
                    retries += 1
                    wait_time = min(10, 2 * retries)  
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                result = response.json()
                
                if score_key not in result:
                    request_queue.remove_request(request_id, success=False)
                    return [-1.0] * len(data.get('query', []))
                    
                request_queue.remove_request(request_id, success=True)
                return result[score_key]
                
            except (requests.Timeout, requests.HTTPError) as e:
                retries += 1
                if isinstance(e, requests.HTTPError) and e.response.status_code == 429:
                    time.sleep(5)
                else:
                    time.sleep(2)
                continue
        
        # Maximum retries exceeded
        logger.warning(f"Request {request_id} exceeded maximum retries")
        request_queue.remove_request(request_id, success=False)
        return [-0.8] * len(data.get('query', []))  
                
    except Exception as e:
        request_queue.remove_request(request_id, success=False)
        logger.info(f"Request {request_id} failed: {str(e)}")
        return [-1.0] * len(data.get('query', []))

def remote_rm_fn(api_url, queries, score_key="rewards"):
    """Remote reward model API with batch processing"""
    if api_url.endswith("/predict_lean"):
        scores = request_api_wrapper_lean(api_url, {"query": queries}, score_key)
    else:
        scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return torch.tensor(scores)

@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)

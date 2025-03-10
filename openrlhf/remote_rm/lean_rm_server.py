import traceback
import logging
import time
import re
import asyncio  
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict, Any
from asyncio import Queue, Lock
import random
import re

# 在已有导入后添加
from transformers import AutoTokenizer


try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
except:
    tokenizer = None

from openrlhf.remote_rm.ds_prover.lean.verifier import Lean4ServerScheduler
from openrlhf.remote_rm.ds_prover.lean.proof import ProofSummarizer
from openrlhf.remote_rm.ds_prover.utils import AttrDict

class Lean4ServerManager:
    def __init__(self, max_concurrent=8):
        self.queue = Queue()
        self.active_requests = 0
        self.lock = Lock()
        self.max_concurrent = max_concurrent
        self.stats = {
            'processed': 0,
            'failed': 0,
            'timeouts': 0,
            'total_processing_time': 0,
            'last_report_time': time.time()
        }
    
        self.monitor_task = asyncio.create_task(self._monitor_queue())
    
    async def _monitor_queue(self):
        while True:
            try:
                queue_size = self.queue.qsize()
                now = time.time()
                if now - self.stats['last_report_time'] >= 60:  
                    avg_time = 0
                    if self.stats['processed'] > 0:
                        avg_time = self.stats['total_processing_time'] / self.stats['processed']
                    
                    logger.info(
                        f"Service Stats: active={self.active_requests}, queue={queue_size}, "
                        f"processed={self.stats['processed']}, failed={self.stats['failed']}, "
                        f"timeouts={self.stats['timeouts']}, avg_time={avg_time:.2f}s"
                    )
                
                    self.stats.update({
                        'processed': 0,
                        'failed': 0,
                        'timeouts': 0,
                        'total_processing_time': 0,
                        'last_report_time': now
                    })
            
                if queue_size > 100:
                    logger.warning(f"High load detected! Queue size: {queue_size}")
                
                await asyncio.sleep(10) 
            except Exception as e:
                logger.error(f"Error in monitor task: {e}")
                await asyncio.sleep(30)  

    async def process_request(self, request):
        if self.active_requests >= self.max_concurrent:
            await self.queue.put(request)
            return await self.queue.get()
        
        async with self.lock:
            self.active_requests += 1
        try:
            return await self._process(request)
        finally:
            self.active_requests -= 1


app = FastAPI()


lean4_scheduler = Lean4ServerScheduler(
    max_concurrent_requests=256,  
    timeout=600,               
    memory_limit=80,           
    name='verifier'
)

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class InputText(BaseModel):
    query: List[Union[str, Dict[str, str]]]  

class OutputPrediction(BaseModel):
    rewards: List[float]

class DetailedOutputPrediction(BaseModel):
    rewards: List[float]
    details: List[dict]  # Detailed verification info

def clean_proof_backticks(proof_text):
    """Clean extra backticks from proof text"""
    cleaned_text = re.sub(r'\s*```+\s*$', '', proof_text)
    cleaned_text = re.sub(r'^```+\s*', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\n)```(?!\n|[a-zA-Z]+)', '', cleaned_text)
    return cleaned_text

def parse_error_positions(error_output, proof_content):
    """Parse error positions and convert to proof-relative positions"""
    error_positions = []
    if not error_output:
        return error_positions
    
    HEADER_LINES = 11  # Fixed header lines in test file
        
    for line in error_output.split('\n'):
        match = re.match(r'.*?:(\d+):(\d+):\s*error:\s*(.*)', line)
        if match:
            file_line = int(match.group(1))
            error_column = int(match.group(2))
            error_message = match.group(3)
            
            proof_line = file_line - HEADER_LINES
            
            if proof_line > 0:  
                proof_lines = proof_content.split('\n')
                if 0 <= proof_line - 1 < len(proof_lines):
                    relative_pos = {
                        'line': proof_line,
                        'file_line': file_line,
                        'column': error_column,
                        'position': sum(len(l) + 1 for l in proof_lines[:proof_line-1]) + error_column,
                        'message': error_message,
                        'content': proof_lines[proof_line - 1]
                    }
                    error_positions.append(relative_pos)
                    logger.debug(f"Error at proof line {proof_line} (file line {file_line}): {error_message}")
                
    return error_positions

def extract_proof_content(text):
    """Extract proof content, handle the following formats:
    - Extract formal_statement from the line starting with theorem
    - Extract proof_content from the part starting with := to the end
    Return tuple (formal_statement, proof_content)
    """
    # Find line containing theorem
    lines = text.split('\n')
    theorem_line = None
    for line in lines:
        if 'theorem' in line:
            theorem_line = line.strip()
            break
    
    if not theorem_line:
        return '', ''
    
    # Extract formal_statement (without :=)
    formal_statement = theorem_line
    if ':=' in theorem_line:
        formal_statement = theorem_line.split(':=')[0].strip()
    
    # Extract proof_content (from := to end)
    all_content = '\n'.join(lines)
    proof_content = ''
    if ':=' in all_content:
        proof_content = ':=' + all_content.split(':=')[1].strip()
        
    # Clean backticks in proof_content
    proof_content = clean_proof_backticks(proof_content)
    
    return formal_statement, proof_content


async def wait_for_proof_result(proof, timeout=300): 
    try:
        start_time = time.time()
        elapsed_time = 0
        while not proof.is_result_ready():
            elapsed_time = time.time() - start_time
            if elapsed_time > 180 and elapsed_time % 60 < 0.2: 
                logger.warning(f"Proof verification taking long time: {elapsed_time:.1f}s")
            
            if elapsed_time > timeout:
                logger.warning(f"Proof verification timeout after {timeout}s")
                return {
                    'pass': False,
                    'complete': False,
                    'status': 'timeout',
                    'system_errors': None,
                    'system_messages': f'Verification timeout after {timeout}s',
                    'output': '',
                    'elapsed_time': elapsed_time  
                }
            await asyncio.sleep(0.1)
            
        verification_time = time.time() - start_time
        if verification_time > 60:
            logger.info(f"Verification completed after {verification_time:.1f}s")
            
        result = proof.result
        result['elapsed_time'] = verification_time  
        return result
    except Exception as e:
        logger.error(f"Error waiting for proof: {e}")
        logger.error(traceback.format_exc())
        return {
            'pass': False,
            'complete': False,
            'status': 'error',
            'system_errors': str(e),
            'system_messages': 'Verification error',
            'output': ''
        }


@app.get("/status")
async def get_status():
    # 获取活跃请求数和队列长度
    active_requests = len(lean4_scheduler.request_statuses)
    queue_size = len(lean4_scheduler.task_queue)
    workers = len(lean4_scheduler.processes)
    
    try:
        load1, load5, load15 = os.getloadavg()
        mem_info = psutil.virtual_memory()
        mem_percent = mem_info.percent
    except:
        load1 = load5 = load15 = 0
        mem_percent = 0
    
    return {
        "active_requests": active_requests,
        "queue_size": queue_size,
        "workers": workers,
        "system_load": {
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "memory_percent": mem_percent
        },
        "status": "overloaded" if queue_size > 200 else "normal",
        "timestamp": time.time()
    }

@app.post("/predict")
async def predict(input_text: InputText) -> OutputPrediction:
    queue_size = len(lean4_scheduler.task_queue)
    if queue_size > 1000:
        logger.warning(f"Queue size: {queue_size}")
    
    queue_start_time = time.time()
    rewards = []
    
    for i, query in enumerate(input_text.query):
        try:
            start_time = time.time()
            
            request_id = f"{int(time.time())}-{i}"
            logger.info(f"Request {request_id}: Processing query {i+1}/{len(input_text.query)}")
            
            # If formal_statement and proof are directly provided
            if isinstance(query, dict) and 'formal_statement' in query:
                formal_statement = query['formal_statement']
                formal_statement = formal_statement.strip(':=')[0] if ':=' in formal_statement else formal_statement
                proof_content = query.get('proof', '')
                proof_content = proof_content.strip()
            else:
                # Handle full text input case
                text = query if isinstance(query, str) else str(query)
                formal_statement, proof_content = extract_proof_content(text)

            logger.info(f"Extracted formal statement: {formal_statement}")
            logger.info(f"Extracted proof: {proof_content}")

            summarizer = ProofSummarizer(
                data={
                    'formal_statement': formal_statement,
                },
                scheduler=lean4_scheduler
            )

            logger.info(f"Analyzing proof: {proof_content}")

            proof = summarizer.analyze(
                code=proof_content,
                require_verification=True
            )

            logger.info("Waiting for verification result...")
            try:
                result = await wait_for_proof_result(proof)
                logger.info(f"Verification result: {result}")
                
                error_positions = parse_error_positions(
                    result.get('output', ''), 
                    proof_content
                )
                
                reward = calculate_reward(result, error_positions, proof_content)
                rewards.append(reward)
                
                logger.info(f"Calculated reward: {reward}")
            except TimeoutError:
                logger.error("Verification timeout")
                rewards.append(-0.6)  # Lighter penalty for timeout
                continue

            # 添加超时预警
            if time.time() - start_time > 30:  # 如果已经运行了30秒，记录一个警告
                logger.warning(f"Request {request_id}: Processing taking longer than expected ({time.time() - start_time:.1f}s)")

            logger.info(f"Request {request_id}: Completed in {time.time() - start_time:.2f} seconds with reward {reward}")

        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)
    
    total_time = time.time() - queue_start_time
    logger.info(f"Batch completed: {len(rewards)}/{len(input_text.query)} in {total_time:.2f}s (avg {total_time/len(input_text.query):.2f}s per query)")
    
    return {"rewards": rewards}


@app.post("/predict_detail", response_model=DetailedOutputPrediction)
async def predict_detail(input_text: InputText):
    rewards = []
    details = []

    for query in input_text.query:
        try:
            start_time = time.time()

            # 处理不同的输入格式
            if isinstance(query, dict):
                formal_statement = query.get('formal_statement', '')
                proof_content = query.get('proof', '')
            else:
                # 处理字符串格式
                formal_statement, proof_content = extract_proof_content(query)

            detail = {
                'formal_statement': formal_statement,
                'verification_time': None,
                'errors': None,
                'status': None,
                'complete': False,
                'pass': False,
                'output': None,
                'system_messages': None
            }

            if proof_content:
                cleaned_proof = clean_proof_backticks(proof_content)
                if proof_content != cleaned_proof:
                    detail['cleaned_proof'] = True
                proof_content = cleaned_proof

            try:
                summarizer = ProofSummarizer(
                    data={'formal_statement': formal_statement},
                    scheduler=lean4_scheduler
                )

                proof = summarizer.analyze(
                    code=proof_content,
                    require_verification=True
                )

                result = await wait_for_proof_result(proof)
                verification_time = time.time() - start_time

                error_positions = parse_error_positions(
                    result.get('output', ''), 
                    proof_content
                )

                reward = calculate_reward(result, error_positions, proof_content)
                rewards.append(reward)

                detail.update({
                    'verification_time': verification_time,
                    'errors': result.get('errors', []),
                    'status': result.get('status', 'unknown'),
                    'complete': result.get('complete', False),
                    'pass': result.get('pass', False),
                    'output': result.get('output', ''),
                    'error_positions': error_positions,
                    'proof_segments': proof.segmentation(result) if hasattr(proof, 'segmentation') else None,
                    'reward': reward
                })

            except TimeoutError as e:
                logger.error("Verification timeout")
                rewards.append(-0.6)
                detail.update({
                    'errors': ['Verification timeout'],
                    'status': 'timeout',
                    'reward': -0.6
                })
            except Exception as e:
                logger.error(f"Error during verification: {str(e)}")
                logger.error(traceback.format_exc())
                rewards.append(-1.0)
                detail.update({
                    'errors': [str(e)],
                    'status': 'error',
                    'reward': -1.0
                })

            details.append(detail)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)
            details.append({
                'formal_statement': formal_statement if 'formal_statement' in locals() else 'Unknown',
                'errors': [str(e)],
                'status': 'error',
                'complete': False,
                'pass': False,
                'reward': -1.0
            })

    return {"rewards": rewards, "details": details}


@app.post("/predict_zero")
async def predict_zero(input_text: InputText) -> OutputPrediction:
    # for testing purposes, return random rewards
    logger.info(f"Received {len(input_text.query)} queries to predict_zero")
    start_time = time.time()
    rewards = []

    for i, query in enumerate(input_text.query):
        # 提取生成的证明
        if isinstance(query, dict):
            text = query.get("proof", "")
        else:
            text = query
            # 尝试提取证明部分
            if ":=" in text:
                text = text.split(":=", 1)[1]

        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = text.strip()


        if tokenizer:
            token_count = len(tokenizer.encode(text))
        else:
            token_count = len(text.split()) + (len(text) // 10)  

        if token_count > 1500:
            reward = -0.4
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")
        elif token_count > 1400:
            reward =-0.3
        elif token_count > 1300:
            reward =0.1
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")
        elif token_count > 1200:
            reward = 0.2
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")
        elif token_count < 100:
            reward = -0.5
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")
        elif token_count < 10:
            reward = -0.1
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")
        else:
            reward = round(random.uniform(0, 0.8), 1)
            if random.random() < 0.6: 
                reward = abs(reward)
            logger.info(f"Query {i}:  ({token_count} tokens), reward={reward}")

        rewards.append(reward)

    elapsed = time.time() - start_time
    logger.info(f"predict_zero completed {len(rewards)} queries in {elapsed:.2f}s")

    return {"rewards": rewards}


def calculate_reward_time(result, error_positions, proof_content):
    """Calculate reward based on verification result with adjusted timeout penalty"""
    if result.get('status') == 'timeout':
        elapsed_time = result.get('elapsed_time', 0)
        # 根据花费时间给予不同程度的惩罚，鼓励模型生成可以更快验证的代码
        if elapsed_time > 250:  # 接近超时限制
            return -0.8  # 严重惩罚
        elif elapsed_time > 150:  # 超过一半时间
            return -0.6  # 中等惩罚
        else:
            return -0.4  # 轻微惩罚

    if not result.get('pass', False):
        return -1.0
    if result.get('complete', False):
        # 对于成功验证的情况，可以根据验证时间给予额外奖励
        elapsed_time = result.get('elapsed_time', 0)
        if elapsed_time < 30:  # 快速验证
            return 1.2  # 额外奖励
        return 1.0
    return 0.0


def calculate_reward(result, error_positions, proof_content):
    """Calculate fine-grained reward"""
    if result.get("complete", False):
        return 1.0
    elif result.get("pass", False):
        return 0.5

    # Start handling failure cases with detailed rewards
    if not proof_content or not proof_content.strip():
        return -1.0  # Empty proof

    if not proof_content.startswith(":="):
        return -0.8  # Basic format error

    # Calculate penalty based on error positions
    if error_positions:
        first_error_line = min(pos["line"] for pos in error_positions)
        total_lines = len(proof_content.split("\n"))

        # Earlier errors result in heavier penalties
        progress_ratio = first_error_line / total_lines
        base_penalty = -0.7

        # Adjust penalty based on progress, lighter near the end
        adjusted_penalty = base_penalty + (0.4 * progress_ratio)

        # Further adjust based on error count
        error_count_penalty = min(0.1 * len(error_positions), 0.3)
        return adjusted_penalty - error_count_penalty

    # Other unknown errors
    return -0.5


@app.on_event("shutdown")
async def shutdown_event():
    lean4_scheduler.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

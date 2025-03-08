import traceback
import logging
import time
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from asyncio import Queue, Lock

from openrlhf.remote_rm.ds_prover.lean.verifier import Lean4ServerScheduler
from openrlhf.remote_rm.ds_prover.lean.proof import ProofSummarizer
from openrlhf.remote_rm.ds_prover.utils import AttrDict

class Lean4ServerManager:
    def __init__(self, max_concurrent=8):
        self.queue = Queue()
        self.active_requests = 0
        self.lock = Lock()
        self.max_concurrent = max_concurrent

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

# 初始化FastAPI应用
app = FastAPI()

# 配置验证器
lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=300, memory_limit=10, name='verifier')

# 配置logging为DEBUG级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InputText(BaseModel):
    query: List[str]  # List of strings containing theorem and proof

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
    """提取证明内容，处理以下格式:
    - 从theorem开始的行提取formal_statement
    - 从:=开始到结束的部分作为proof_content
    返回tuple (formal_statement, proof_content)
    """
    # 查找包含theorem的行
    lines = text.split('\n')
    theorem_line = None
    for line in lines:
        if 'theorem' in line:
            theorem_line = line.strip()
            break
    
    if not theorem_line:
        return '', ''
    
    # 提取formal_statement (不包含:=)
    formal_statement = theorem_line
    if ':=' in theorem_line:
        formal_statement = theorem_line.split(':=')[0].strip()
    
    # 提取proof_content (从:=开始到结束)
    all_content = '\n'.join(lines)
    proof_content = ''
    if ':=' in all_content:
        proof_content = ':=' + all_content.split(':=')[1].strip()
        
    # 清理proof_content中的反引号
    proof_content = clean_proof_backticks(proof_content)
    
    return formal_statement, proof_content

# 1. 修改等待验证结果的方式
async def wait_for_proof_result(proof, timeout=300):
    start_time = time.time()
    while not proof.is_result_ready():
        if time.time() - start_time > timeout:
            raise TimeoutError("Proof verification timeout")
        await asyncio.sleep(0.1)  # 使用asyncio.sleep避免CPU忙等待
    return proof.result

# 2. 在predict函数中使用
@app.post("/predict")
async def predict(input_text: InputText) -> OutputPrediction:
    logger.info(f"Received request: {input_text}")
    rewards = []

    for query in input_text.query:
        try:
            start_time = time.time()
            
            # 如果直接传入formal_statement和proof
            if isinstance(query, dict) and 'formal_statement' in query:
                formal_statement = query['formal_statement']
                formal_statement = formal_statement.strip(':=')[0] if ':=' in formal_statement else formal_statement
                proof_content = query.get('proof', '')
                proof_content = proof_content.strip()
            else:
                # 处理整个文本输入的情况
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
            except TimeoutError:
                logger.error("Verification timeout")
                rewards.append(-1.0)
                continue

            if result.get('complete', False):
                logger.info("Proof is complete and correct")
                rewards.append(1.0)  
            elif result.get('pass', False):
                logger.info("Proof passes but may use sorry")
                rewards.append(0.5)  
            else:
                logger.info(f"Proof has errors: {result.get('errors', [])}")
                rewards.append(-1.0)  

            logger.info(f"Verification completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Final rewards: {rewards}")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)

    return {"rewards": rewards}

@app.post("/predict_detail", response_model=DetailedOutputPrediction)
async def predict_detail(input_text: InputText):
    rewards = []
    details = []
    
    for query in input_text.query:
        try:
            start_time = time.time()
            detail = {
                'formal_statement': query['formal_statement'],
                'verification_time': None,
                'errors': None,
                'status': None,
                'complete': False,
                'pass': False,
                'output': None,  
                'system_messages': None  
            }
            
    
            if 'proof' in query and query['proof']:
                original_proof = query['proof']
                cleaned_proof = clean_proof_backticks(original_proof)
                if original_proof != cleaned_proof:
                    detail['cleaned_proof'] = True
                query['proof'] = cleaned_proof
            
            summarizer = ProofSummarizer(
                data={'formal_statement': query['formal_statement']},
                scheduler=lean4_scheduler
            )
            
            proof = summarizer.analyze(
                code=query['proof'],
                require_verification=True
            )
            
            while not proof.is_result_ready():
                pass
            
            result = proof.result
            verification_time = time.time() - start_time
            
            error_positions = parse_error_positions(
                result.get('output', ''), 
                query['proof']  
            )
            
            detail.update({
                'verification_time': verification_time,
                'errors': result.get('errors', []),
                'status': result.get('status', 'unknown'),
                'complete': result.get('complete', False),
                'pass': result.get('pass', False),
                'output': result.get('output', ''),
                'error_positions': error_positions,  
                'proof_segments': proof.segmentation(result)  
            })
            
            if result.get('complete', False):
                rewards.append(1.0)
            elif result.get('pass', False):
                rewards.append(0.5)
            else:
                rewards.append(-1.0)
                
            details.append(detail)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            rewards.append(-1.0)
            details.append({
                'formal_statement': query['formal_statement'],
                'verification_time': None,
                'errors': [str(e)],
                'status': 'error',
                'complete': False,
                'pass': False
            })
            
    return {"rewards": rewards, "details": details}

@app.on_event("shutdown")
async def shutdown_event():
    lean4_scheduler.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

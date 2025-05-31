import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import uvicorn
from verify import Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define the payload class
class Payload(BaseModel):
    response: str  # response from model
    prompt: str  # prompt
    answer: str  # answer (not formatted)
    solution: str  # solution (formatted)
    data_source: str  # data source
    reward_verifier: str  # reward verifier
    reward_verifier_parm: str  # reward verifier parm
    format_ratio: Optional[float] = None
    accuracy_ratio: Optional[float] = None


def check_payload(payload_dict):
    # Validate required fields
    required_fields = [
        "response", "answer", "prompt", "solution", "data_source", "reward_verifier", "reward_verifier_parm"
    ]
    for field in required_fields:
        if field not in payload_dict:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Reward Judge Server"}


@app.post("/judge")
async def judge_reward(payload: Payload):
    # Check payload
    check_payload(payload.dict())

    # Initialize default result
    result = {
        "rewards": {
            "format_reward": 0.0,
            "accuracy_reward": 0.0,
            "reflection_reward": 0.0,
            "final_reward": 0.0,
        }
    }

    try:
        if payload.reward_verifier in Verifier.list_verifiers():
            verifier_cls = Verifier.get(payload.reward_verifier)
        else:
            raise HTTPException(status_code=404, detail=f"Invalid reward verifier: {payload.reward_verifier}")

        # Create verifier instance
        verifier_parm = json.loads(payload.reward_verifier_parm)
        verifier = verifier_cls(**verifier_parm)
        # Judge the response against the answer/solution

        format_score = verifier.verify_format(payload.response)
        accuracy_score_gathered = verifier.verify_accuracy(payload.response, payload.solution)
        if isinstance(accuracy_score_gathered, dict):
            accuracy_score = accuracy_score_gathered['final_score']

            # to log the score of each metric
            for socre_key, socre_value in accuracy_score_gathered.items():
                if socre_key != 'final_score':
                    result['rewards'][f'{socre_key}_reward'] = socre_value
        else:
            accuracy_score = accuracy_score_gathered

        result['rewards']['format_reward'] = float(format_score)
        result['rewards']['accuracy_reward'] = float(accuracy_score)

        # Define the ratio of each reward
        # accuracy_ratio = 1

        if payload.accuracy_ratio is not None:
            accuracy_ratio = payload.accuracy_ratio
        else:
            accuracy_ratio = 1.0
            logger.warning(f"Accuracy ratio is not provided, using default {accuracy_ratio}")

        if payload.format_ratio is not None:
            # Use optimal format_ratio
            format_ratio = payload.format_ratio
        else:
            format_ratio = 0.1
            logger.warning(f"Format ratio is not provided, using default {format_ratio}")

        normalzied_score = accuracy_ratio + format_ratio

        # final reward
        result['rewards']['final_reward'] = accuracy_score * (accuracy_ratio / normalzied_score) + format_score * (
            format_ratio / normalzied_score)

        if random.random() < 0.3:
            print_dict = {
                'reward_verifier': payload.reward_verifier,
                'reward_verifier_parm': payload.reward_verifier_parm,
                'prediction': payload.response,
                'solution': payload.solution,
                'format_score': format_score,
                'accuracy_score': accuracy_score,
                'final_reward': result['rewards']['final_reward'],
            }
            if isinstance(accuracy_score_gathered, dict):
                for key, value in accuracy_score_gathered.items():
                    if key != 'final_score':
                        print_dict[key] = value
            logger.info(json.dumps(print_dict))

    except Exception as e:
        # Log the error but return a valid result structure
        import traceback
        print(f"Error during verification: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Return default zero rewards on error

    return result


if __name__ == "__main__":
    # Run the server with Uvicorn
    uvicorn.run(
        "reward_serving_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=32  # For load balancing
    )

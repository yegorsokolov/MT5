from __future__ import annotations

"""Simple feature transformation worker service.

This FastAPI service performs heavier feature calculations on behalf of
resource constrained clients.  It accepts basic job specifications and
returns transformed feature data as JSON serialisable records.
"""

from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class FeatureJob(BaseModel):
    symbol: str
    start: str
    end: str


@app.post("/compute")
async def compute(job: FeatureJob) -> List[Dict[str, Any]]:
    """Perform a toy "heavy" transformation.

    The worker simply generates a DataFrame with a range of integers between
    ``start`` and ``end`` (interpreted as integers) and returns their squares.
    In practice this module would house CPU/GPU intensive feature logic, but
    for testing purposes the computation is intentionally lightweight.
    """

    start = int(job.start)
    end = int(job.end)
    data = [{"symbol": job.symbol, "value": i * i} for i in range(start, end)]
    df = pd.DataFrame(data)
    return df.to_dict(orient="records")


if __name__ == "__main__":  # pragma: no cover - service entry point
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

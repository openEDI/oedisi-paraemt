"""
Runs a server that can be used when running in a Kubernetes or Docker Compose environment.

Not needed outside of this.
"""

import logging
import socket
import sys
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from oedisi.types.common import BrokerConfig, HeathCheck, ServerReply

from ParaEMT import run_simulator

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint that returns the health check information of the server.

    Returns:
        JSONResponse: The health check information in JSON format.
    """
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()
    return JSONResponse(response, 200)


@app.post("/run/")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    """
    Endpoint to run the simulator with the provided broker configuration.

    Args:
        broker_config (BrokerConfig): The broker configuration.
        background_tasks (BackgroundTasks): Background tasks to be executed.

    Returns:
        JSONResponse: The response indicating the task has been successfully added.
    """
    try:
        logging.info(broker_config)

        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail="Task successfully added.").dict()
        return JSONResponse(response, 200)
    except Exception as _:
        err = traceback.format_exc()
        HTTPException(500, str(err))


if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)

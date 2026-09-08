from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from conditional_RAG import app


app_api = FastAPI()


app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):

    message: str
    programme: str


@app_api.post("/chat")
def chat(request: ChatRequest):

    result = app.invoke({

        "programme": request.programme,

        "messages": [
            ("human", request.message)
        ]

    })

    answer = result["messages"][-1].content

    return {
        "response": answer
    }
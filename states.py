# So now we are creating a graph and the first thing you create a state

import os

# 1). typed Dict
from typing import TypedDict

class State(TypedDict):
    topic: str
    summary: str
    score: int


#2) pydantic approach
# It is good at data validation and type checking at runtime

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic: str
    score: int
    summary: str = ""

    @field_validator #Decorator
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("Score must be positive")


# 3)python dataclasses
#standard python dataclass but it is used very rarely

from dataclasses import dataclass, field

@dataclass
class State:
    topic: str = ""
    summary: str = ""
    score : int
    messages: list = field(default_factory=list)


# 4) Using base langgraph
from langgraph.graph import MessagesState

class State(MessagesState):
    #messages field is already included with add_messages reducer
    # just add your extra fields
    user_name: str
    language: str

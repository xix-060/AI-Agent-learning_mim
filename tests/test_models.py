"""models 的单元测试""" 
import pytest 
from pydantic import ValidationError 
from src.models import Message, RoleEnum, ChatRequest 

def test_message_valid(): 
    m = Message(role=RoleEnum.USER, content="hello") 
    assert m.role == RoleEnum.USER 
    assert m.content == "hello" 
    
def test_message_blank_content(): 
    with pytest.raises(ValidationError): 
        Message(role=RoleEnum.USER, content=" ") 
        
def test_chat_request_valid(): 
    req = ChatRequest( 
        messages=[Message(role=RoleEnum.USER, content="hi")], 
        temperature=0.5, 
    )
    assert req.model == "deepseek-chat" 
    assert req.temperature == 0.5 
    
def test_chat_request_invalid_temperature(): 
    with pytest.raises(ValidationError): 
        ChatRequest( 
            messages=[Message(role=RoleEnum.USER, content="hi")], 
            temperature=3.0, # 超过 2.0 
        ) 
        
def test_chat_request_empty_messages(): 
    with pytest.raises(ValidationError): 
        ChatRequest(messages=[])
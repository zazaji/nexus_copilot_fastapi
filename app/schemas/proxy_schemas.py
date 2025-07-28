# backend/app/schemas/proxy_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

class ModelInfo(BaseModel):
    name: str
    capabilities: List[str]
    max_tokens: Optional[int] = Field(None, alias="maxTokens")

    class Config:
        populate_by_name = True

class ApiProvider(BaseModel):
    id: str
    name: str
    baseUrl: str
    apiKey: str
    models: List[ModelInfo]
    proxy: Optional[str] = None

class ModelEndpoint(BaseModel):
    providerId: str
    modelName: str

class ModelAssignments(BaseModel):
    chat: Optional[ModelEndpoint] = None
    suggestion: Optional[ModelEndpoint] = None
    vision: Optional[ModelEndpoint] = None
    imageGen: Optional[ModelEndpoint] = None
    embedding: Optional[ModelEndpoint] = None
    tts: Optional[ModelEndpoint] = None

class OtherApiKeys(BaseModel):
    tavily: Optional[str] = ""
    bing: Optional[str] = ""

class OnlineKnowledgeBase(BaseModel):
    id: str
    name: str
    url: str
    token: str

class KnowledgeBaseSettings(BaseModel):
    indexedDirectories: List[str] = []
    scriptsDirectories: List[str] = []
    defaultSaveDirectory: Optional[str] = None
    topK: int = 5
    scoreThreshold: float = 0.6
    defaultInternetSearchEngine: str = Field("tavily", alias="default_internet_search_engine")

    class Config:
        populate_by_name = True

class ExecutionSettings(BaseModel):
    pythonPath: str
    nodePath: str
    workingDirectory: str
    autoStartBackend: bool
    backendUrl: str

class AppearanceSettings(BaseModel):
    theme: str
    language: str
    copilotAutoHideDelay: int
    editorFontSize: int

class ApiConfig(BaseModel):
    providers: List[ApiProvider]
    assignments: ModelAssignments
    keys: OtherApiKeys
    onlineKbs: Optional[List[OnlineKnowledgeBase]] = []
    knowledgeBase: Optional[KnowledgeBaseSettings] = None
    execution: Optional[ExecutionSettings] = None
    appearance: Optional[AppearanceSettings] = None

class ProxyMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ProxyChatPayload(BaseModel):
    model: str
    messages: List[ProxyMessage]
    stream: bool = False
    provider_config: ApiProvider # Use the full ApiProvider model
    knowledge_base_selection: Optional[str] = None
    api_config: Optional[ApiConfig] = None

class ProxyEmbeddingPayload(BaseModel):
    model: str
    input: List[str]
    provider_config: ApiProvider # Use the full ApiProvider model

class KnowledgeSource(BaseModel):
    id: str
    file_path: str
    source_name: str
    content_snippet: str
    score: float

class SearchRequest(BaseModel):
    query: str
    api_config: ApiConfig
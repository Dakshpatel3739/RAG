"""
LLM-based answer generation module
"""
from typing import List, Dict, Optional
from openai import OpenAI
from utils.logger import log
from config.settings import settings


class LLMGenerator:
    """Generate answers using LLM with retrieved context"""
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize LLM generator
        
        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = (settings.LLM_PROVIDER or "openai").lower()
        if self.provider == "gemini":
            self.model = model or settings.GEMINI_LLM_MODEL
        else:
            self.model = model or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI LLM usage")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.gemini_model = None
            log.info(f"LLM Generator initialized with OpenAI model: {self.model}")
        elif self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required for Gemini LLM usage")
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.genai = genai
            self.gemini_model = genai.GenerativeModel(self._gemini_model_name())
            self.client = None
            log.info(f"LLM Generator initialized with Gemini model: {self.model}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        Generate answer using LLM with context
        
        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            system_prompt: Optional custom system prompt
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            log.info(f"Generating answer for query: {query[:100]}...")
            
            # Build context from chunks
            context = self._build_context(context_chunks)
            
            # Build prompt
            user_prompt = self._build_user_prompt(query, context)
            
            # Default system prompt
            if system_prompt is None:
                system_prompt = self._get_default_system_prompt()
            
            # Call LLM
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
            elif self.provider == "gemini":
                prompt = self._build_gemini_prompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                answer = self._gemini_generate(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            # Prepare response
            result = {
                "answer": answer,
                "query": query,
                "model": self.model,
                "context_chunks_used": len(context_chunks)
            }
            
            if include_sources:
                result["sources"] = self._extract_sources(context_chunks)
            
            log.info("Answer generated successfully")
            return result
            
        except Exception as e:
            log.error(f"Error generating answer: {str(e)}")
            raise
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("filename", "Unknown")
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {score:.2f})]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with query and context"""
        return f"""Context information is below:
---
{context}
---

Using the context information above, please answer the following question. If the answer cannot be found in the context, say so clearly.

Question: {query}

Answer:"""
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """You are a helpful AI assistant that answers questions based on provided context. 

Guidelines:
- Answer questions accurately based on the provided context
- If the context doesn't contain enough information, clearly state that
- Cite specific sources when making claims
- Be concise but thorough
- If there are conflicting pieces of information, acknowledge them
- Don't make up information not present in the context"""
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        seen_sources = set()
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_id = metadata.get("filename", "Unknown")
            
            if source_id not in seen_sources:
                sources.append({
                    "filename": source_id,
                    "source": metadata.get("source", ""),
                    "relevance_score": chunk.get("score", 0.0)
                })
                seen_sources.add(source_id)
        
        return sources
    
    def generate_with_conversation_history(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate answer with conversation history
        
        Args:
            query: Current user question
            context_chunks: Retrieved chunks
            conversation_history: Previous messages [{"role": "user/assistant", "content": "..."}]
            system_prompt: Optional system prompt
            
        Returns:
            Generated response with metadata
        """
        try:
            log.info("Generating answer with conversation history")
            
            # Build context
            context = self._build_context(context_chunks)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt or self._get_default_system_prompt()}
            ]
            
            # Add conversation history
            messages.extend(conversation_history[-6:])  # Last 3 exchanges
            
            # Add current query with context
            current_message = f"""Context information:
---
{context}
---

Question: {query}"""
            
            messages.append({"role": "user", "content": current_message})
            
            # Generate response
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
            elif self.provider == "gemini":
                prompt = self._build_gemini_prompt(
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    user_prompt=current_message,
                    conversation_history=conversation_history[-6:]
                )
                answer = self._gemini_generate(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            return {
                "answer": answer,
                "query": query,
                "model": self.model,
                "context_chunks_used": len(context_chunks),
                "sources": self._extract_sources(context_chunks)
            }
            
        except Exception as e:
            log.error(f"Error generating answer with history: {str(e)}")
            raise
    
    def stream_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ):
        """
        Stream answer generation
        
        Args:
            query: User question
            context_chunks: Retrieved chunks
            system_prompt: Optional system prompt
            
        Yields:
            Chunks of generated text
        """
        try:
            context = self._build_context(context_chunks)
            user_prompt = self._build_user_prompt(query, context)
            
            if self.provider == "openai":
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt or self._get_default_system_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            elif self.provider == "gemini":
                prompt = self._build_gemini_prompt(
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    user_prompt=user_prompt
                )
                try:
                    stream = self.gemini_model.generate_content(
                        prompt,
                        generation_config=self._gemini_generation_config(),
                        stream=True
                    )
                    for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            yield text
                except Exception:
                    yield self._gemini_generate(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                    
        except Exception as e:
            log.error(f"Error streaming answer: {str(e)}")
            raise

    def _gemini_generation_config(self) -> Dict[str, any]:
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens
        }
    
    def _gemini_model_name(self) -> str:
        if self.model.startswith("models/"):
            return self.model
        return f"models/{self.model}"

    def _gemini_generate(self, prompt: str) -> str:
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=self._gemini_generation_config()
        )
        text = self._extract_gemini_text(response)
        if text:
            return text
        return "Gemini returned no text (response may have been blocked by safety settings)."

    def _build_gemini_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        parts = []
        if system_prompt:
            parts.append(f"System:\\n{system_prompt}")
        if conversation_history:
            for message in conversation_history:
                role = message.get("role", "user")
                label = "User" if role == "user" else "Assistant"
                parts.append(f"{label}:\\n{message.get('content', '')}")
        parts.append(f"User:\\n{user_prompt}")
        parts.append("Assistant:")
        return "\\n\\n".join(parts)

    def _extract_gemini_text(self, response) -> Optional[str]:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    return text
        return None

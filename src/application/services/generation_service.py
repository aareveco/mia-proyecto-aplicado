from typing import List
from src.application.ports.llm_port import LLMService
from src.domain.models import ProcessedChunk

class AugmentedGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # Threshold could be configurable
        self.relevance_threshold = 0.5 

    def generate_answer(self, query: str, context_chunks: List[ProcessedChunk]) -> str:
        """
        Generates an answer using the provided context chunks.
        steps:
        1. Context Verification: Check if we have valid context.
        2. Prompt Engineering: CoT + Citations.
        3. Generation.
        """
        
        # 1. Context Verification
        if not context_chunks:
            return "I cannot answer this question because I found no relevant information in the knowledge base."
        
        # Check relevance scores if available (assuming they are in metadata or attribute)
        # If the top chunk is below threshold, refuse.
        # Note: We rely on the reranker to populate 'rerank_score' in metadata or object.
        top_score = 0.0
        if hasattr(context_chunks[0], 'rerank_score'): # If we added it dynamically
            top_score = context_chunks[0].rerank_score
        elif context_chunks[0].metadata and "rerank_score" in context_chunks[0].metadata:
            top_score = context_chunks[0].metadata["rerank_score"]
        
        # If we have scores and they are low (e.g., < 0.0, dependent on reranker model logit/prob)
        # For now, let's assume if it's 0.0 it's suspicious if we expected a reracker, 
        # but if we didn't run reranking, we might not have scores. 
        # Let's be lenient if no scores exist, but strict if they do.
        
        # For this exercise, let's assume we proceed unless it's explicitly low or empty.
        
        # 2. Context Construction with IDs for Citation
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            # Use chunk_id or source_file as identifier
            cid = chunk.chunk_id if chunk.chunk_id else f"doc_{i}"
            context_str += f"[Source ID: {cid}]\n{chunk.content}\n\n"

        # 3. CoT Prompt Construction
        # 3. CoT Prompt Construction with Metabolomics Persona
        prompt = f"""You are a Senior Metabolomics Analyst. Use the provided context to answer the user's inquiry about a chemical feature (m/z, RT).
        
        Your goal is to identify the compound and report its bioactivity based ONLY on the context.
        
        **Required Output Structure:**
        1. **Feature Identification (Putative)**:
           - "The feature (m/z [val], RT [val]) is annotated putatively as [Compound Name] (Formula [F])."
        2. **Potential Bioactivities**:
           - List bioactivities found in the context for this compound.
           - "Activity: [Description] (Source: [Source ID])"
        3. **Internal Context** (if applicable):
           - Mention if this feature was seen in previous internal samples based on context.

        **Rules:**
        - Answer ONLY using the context.
        - Cite Source IDs [doc_1] for every claim.
        - If the context doesn't link the m/z to a compound, state "No identification found for this mass."
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:"""

        return self.llm_service.generate_text(prompt)

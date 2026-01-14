from langchain_openai import ChatOpenAI


class MultimodalRAG:
    """Retrieval-Augmented Generation layer."""

    def __init__(self, retriever):
        """Initialize the RAG layer with a retriever and an LLM.

        Args:
            retriever: A LangChain-compatible retriever (e.g., Chroma.as_retriever()).

        The RAG uses the retriever to fetch relevant documents and an LLM
        to synthesize answers from the retrieved context.
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
        )

    def query(self, question: str) -> str:
        """Answer a question by retrieving documents and calling the LLM.

        Args:
            question: The user query string.

        Returns:
            A string containing the LLM's answer based on retrieved context.
        """
        
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
                    You are a helpful assistant.
                    Use the context below to answer the question.

                    Context:
                    {context}

                    Question:
                    {question}
                    """

        full_response = []
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                full_response.append(chunk.content)

        return "".join(full_response)
       

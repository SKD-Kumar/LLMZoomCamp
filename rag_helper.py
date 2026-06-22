INSTRUCTIONS = '''Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know.'''


USER_PROMPT_TEMPLATE = '''
Question:
{question}

Context:
{context}'''


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="MBZUAI-IFM/K2-Think-v2"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
    
    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
    
#to use different LLM
class K2ThinkRAG(RAGBase): #extend the prev
    #Override the llm function
    def llm(self, prompt):
        message_history = [
            {'role': 'system', 'content': self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages = message_history
        )
        ans = response.choices[0].message.content
        parts = ans.split("</think>", 1) 
        thinks = parts[0]
        act_resp = parts[1]
        return act_resp, response.usage
    
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["content"])
            lines.append("FileName " + doc["filename"])
            lines.append("")

        return "\n".join(lines).strip()
    
    def search(self, query, num_results=1):
        return self.index.search(
            query,
            num_results=num_results,
        )
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, usage = self.llm(prompt)
        return answer, usage
    

class K2ThinkRAG_nonhw(RAGBase): #extend the prev
    #Override the llm function
    def llm(self, prompt):
        search_tool = {
            "type": "function",
            "name": "search",
            "description": "Search the FAQ database for entries matching the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look up in the course FAQ."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
        message_history = [
            {'role': 'system', 'content': self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages = message_history,
            tools=[search_tool]
        )
        ans = response.choices[0].message.content
        parts = ans.split("</think>", 1) 
        thinks = parts[0]
        act_resp = parts[1]
        return act_resp
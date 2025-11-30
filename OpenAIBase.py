from openai import OpenAI

class OpenAIBase:

    def __init__(self, api_key: str, model: str="gpt-4.1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_text: str, text_format: type):
        response = self.client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            max_output_tokens=12000,
            text_format=text_format
        )
        return response.output_parsed
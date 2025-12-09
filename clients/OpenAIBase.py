from openai import OpenAI
class OpenAIBase:

    def __init__(self, api_key: str, model: str="gpt-4.1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_response(self, inputs: list, max_output_tokens = 12000):
        response = self.client.responses.create(
            model=self.model,
            input=inputs,
            max_output_tokens=max_output_tokens
        )
        return response

    def parse_response(self, inputs: list, text_format, max_output_tokens = 12000):
        response = self.client.responses.parse(
            model=self.model,
            input=inputs,
            max_output_tokens=max_output_tokens,
            text_format=text_format
        )
        return response
    
class OpenAIStructuredOutputClient(OpenAIBase):

    def call(self, conversation, text_format: type, max_output_tokens = 12000):
        response = super().parse_response(conversation, text_format, max_output_tokens)
        return response.output_parsed

class OpenAITextOutputClient(OpenAIBase):

    def call(self, conversation, max_output_tokens = 12000):
        response = super().create_response(conversation, max_output_tokens)
        return response.output_text.strip()
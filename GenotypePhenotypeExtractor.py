from clients.OpenAIBase import OpenAIStructuredOutputClient
from models.models import MutationList, OptimizedSystemPrompt
import json

class GenotypePhenotypeExtractor:
    SYSTEM_PROMPT_EXTRACT_MUTATIONS_PATH = "./prompts/extract_mutations_system_prompt.txt"
    SYSTEM_PROMPT_EXTRACT_MUTATIONS_EVALUATION_PATH = "./prompts/extract_mutations_evaluator_system_prompt.txt"

    conversation = []
    full_text = None
    annotations = []

    def __init__(self, api_key: str, full_text: str, model: str="gpt-4.1"):
        self.openai_structured_output_client = OpenAIStructuredOutputClient(api_key, model)
        self.full_text = full_text

        with open(self.SYSTEM_PROMPT_EXTRACT_MUTATIONS_PATH) as f:
            self.system_prompt_extract_mutations = f.read()
            self.current_system_prompt = self.system_prompt_extract_mutations
            f.close()

        with open(self.SYSTEM_PROMPT_EXTRACT_MUTATIONS_EVALUATION_PATH) as f:
            self.system_prompt_extract_mutations_evaluation = f.read()
            f.close()

    def extract(self):
        if len(self.conversation) == 0:
            self.conversation = [
                {"role": "system", "content": self.system_prompt_extract_mutations},
                {"role": "user", "content": self.full_text}
            ]
        response = self.openai_structured_output_client.call(
            self.conversation,
            MutationList
        )
        self.annotations = response.mutations

    def evaluate(self):
        user_prompt = ("Current system prompt:\n" +
                       self.current_system_prompt +
                       "\n\nText used an input:\n" +
                       self.full_text +
                       "\n\nExtracted annotations:\n" +
                       json.dumps([m.model_dump() for m in self.annotations], indent=2))
        response = self.openai_structured_output_client.call(
            [
                {"role": "system", "content": self.system_prompt_extract_mutations_evaluation},
                {"role": "user", "content": user_prompt}
            ],
            OptimizedSystemPrompt
        )
        if not response.converged:
            # Update conversation
            self.conversation.extend([
                {
                    "role": "user",
                    "content": "I evaluated the extracted annotations and have additional instructions to add to the original prompt. The rationale for the updated instructions is below," +
                    response.rationale +
                    "\n\n" +
                    "Based on this rationale I have new instructions to append to the original system prompt. Redo the extractions along with the updated instructions below," +
                    "\n\n" +
                    response.system_prompt
                }
            ])
        return response

    def iteratively_extract(self, max_steps: int = 10):
        for step in range(max_steps):
            self.extract()
            evaluation_response = self.evaluate()
            if evaluation_response.converged:
                break

    def write_conversation_to_file(self, out_path):
        with open(out_path, "w") as f:
            json.dump(self.conversation, f, indent=2)
            f.close()

    def write_annotations_to_file(self, out_path):
        with open(out_path, "w") as f:
            annotations_to_dump = [m.model_dump() for m in self.annotations]
            json.dump(annotations_to_dump, f, indent=4, ensure_ascii=False)
            f.close()
import os
from PyPDF2 import PdfReader, PdfWriter
import tempfile
from clients.OpenAIBase import OpenAITextOutputClient, OpenAIStructuredOutputClient
from base64 import b64encode
from models.models import ContainsTable

def get_base64_pdf_user_prompt(file: str):
    with open(file, "rb") as fin:
        base64_pdf = b64encode(fin.read()).decode("utf-8")
        return {
            "type": "input_file",
            "file_data": f"data:application/pdf;base64,{base64_pdf}",
            "filename": "input_paper.pdf",
        }


class PdfToTextConverter:

    SYSTEM_PROMPT_EXTRACT_TEXT = "Extract and return ONLY the plain text from the provided PDF. Do not summarize, analyze, or interpret."
    SYSTEM_PROMPT_CHECK_TABLE = "Does the provided PDF page contain a table?"
    SYSTEM_PROMPT_EXTRACT_TABLE = "Extract and return ONLY tables from the PDF page in markdown format. Do not analyze or interpret."

    full_paper_text = ""
    full_paper_text_path = None

    def __init__(self, file_path, api_key, tmp_dir="./tmp"):
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.openai_text_client = OpenAITextOutputClient(api_key)
        self.openai_structured_output_client = OpenAIStructuredOutputClient(api_key)
        self.file_path = file_path
        self.split_pdf_folder = tempfile.mkdtemp(dir=self.tmp_dir)

        self.pages = []

    def split_pdf(self):
        reader = PdfReader(self.file_path)
        path_to_pages = []

        for i, page in enumerate(reader.pages):
            path_to_page = os.path.join(self.split_pdf_folder, f"page_{i:02d}.pdf")
            with open(path_to_page, "wb") as out_file:
                writer = PdfWriter()
                writer.add_page(page)
                writer.write(out_file)
                writer.close()
                path_to_pages.append(path_to_page)
        return path_to_pages

    def convert(self):
        path_to_pdf_pages = self.split_pdf()

        full_text = []
        for page_idx, path_to_pdf_page in enumerate(path_to_pdf_pages):
            print(path_to_pdf_page)
            # Extract raw text from PDF page via GPT prompt
            base64_pdf_page_user_prompt = get_base64_pdf_user_prompt(path_to_pdf_page)

            page_text = self.openai_text_client.call([
                    {"role": "system", "content": self.SYSTEM_PROMPT_EXTRACT_TEXT},
                    {"role": "user", "content": [base64_pdf_page_user_prompt]}
                ])

            full_text.append(page_text)

            # Check if page has table
            contains_table_response = self.openai_structured_output_client.call([
                    {"role": "system", "content": self.SYSTEM_PROMPT_CHECK_TABLE},
                    {"role": "user", "content": [base64_pdf_page_user_prompt]}
                ],
                ContainsTable
            )

            table_text = None
            if contains_table_response.has_table:
                # Extract tables from PDF via GPT prompt
                table_text = self.openai_text_client.call([
                    {"role": "system", "content": self.SYSTEM_PROMPT_EXTRACT_TABLE},
                    {"role": "user", "content": [base64_pdf_page_user_prompt]}
                ])
                full_text.append(table_text)

            # Store page content
            self.pages.append({
                "page": page_idx,
                "text": page_text.strip(),
                "tables": table_text.strip() if table_text else ""
            })

        # get full table
        self.full_paper_text = "\n".join(
            p["text"] + ("\n" + p["tables"] if p["tables"] else "")
            for p in self.pages
        )

    def write_full_paper_text(self):
        self.full_paper_text_path = os.path.join(self.split_pdf_folder, "full_paper_text.txt")
        with open(self.full_paper_text_path, "w", encoding="utf-8") as f:
            f.write(self.full_paper_text)

# -*- coding: utf-8 -*-
"""
title: Multimodal RAG Pipeline using Colpali
author: Stephane Ancelot
date: 2025-03-07
version: 4.0
license: MIT
description: A pipeline for retrieving relevant information using Vision Language models.
requirements: pdf2image, qdrant-client, colpali-engine, Pillow
"""
import logging
import shutil
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor, ColQwen2, ColQwen2Processor
from qdrant_client.http import models
from qdrant_client.models import PointStruct
import base64
from io import BytesIO
from typing import List, Dict, Any, Sequence, Tuple
import requests
from qdrant_client import QdrantClient
import asyncio
import os
from typing import List, Union, Generator, Iterator
from pathlib import Path
import time
import sys
import json
import threading
import io
sys.path.append("e:\\workspace\\multimrag")

from pdf_optimizer import apply_patch  # noqa
apply_patch()
from pdf2image import convert_from_path  # noqa
from pdf_optimizer import OptimizedPDFConverter  # noqa

converter = OptimizedPDFConverter()

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
# Create file handler
file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.DEBUG)
# Add handler to logger
logger.addHandler(file_handler)


def check_poppler_in_path():
    """
    Checks if Poppler's 'pdftotext.exe' is in the system's PATH.
    """
    # We check for a specific Poppler executable, like pdftotext.exe.
    # You could also use pdfinfo.exe, pdftocairo.exe, etc.
    if sys.platform == "win32":
        poppler_executable = "pdftotext.exe"
    else:
        pass  # FIXME: add support for other platforms
    # shutil.which() returns the path to the executable if found, otherwise None.
    location = shutil.which(poppler_executable)

    if location:
        print(f"  Success: Poppler is in the PATH.")
        print(f"   Found at: {location}")
        return True
    else:
        print(
            f" Failure: Poppler's '{poppler_executable}' was not found in the PATH.")
        print("\n--- How to Fix ---")
        print("1. Make sure you have downloaded and unzipped Poppler for Windows.")
        print("2. Add the 'bin' directory from your Poppler folder to the system's PATH environment variable.")
        print("   (e.g., C:\\path\\to\\poppler-23.11.0\\Library\\bin)")
        print("3. Restart your terminal or IDE for the new PATH to take effect.")
        return False


# Set your OpenWebUI API base URL and (optionally) your API key
BASE_URL = "http://xxxx:8080/api/v1"
API_KEY = "sk-xxxx"
headers = {
    "Content-Type": "application/json"
}

if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

downloads_dir = "downloads"


class Pipeline:
    def __init__(self):
        print("=========== pipeline init ===========")
        if not check_poppler_in_path():
            print("Poppler not found in PATH.")
            return

        # Initialize state tracking
        self.initialization_state_file = "pipeline_state.json"
        self.is_fully_initialized = False
        self.initialization_in_progress = False
        self.initialization_lock = threading.Lock()

        # Load previous state if exists
        self.load_initialization_state()

        print("===========pipeline ready=========")

    def load_initialization_state(self):
        """Load the initialization state from file"""
        try:
            if os.path.exists(self.initialization_state_file):
                with open(self.initialization_state_file, 'r') as f:
                    state = json.load(f)
                    self.is_fully_initialized = state.get(
                        'is_fully_initialized', False)
                    print(
                        f"Loaded state: fully_initialized={self.is_fully_initialized}")
            else:
                print("No previous state found - first run")
        except Exception as e:
            print(f"Error loading state: {e}")
            self.is_fully_initialized = False

    def save_initialization_state(self):
        """Save the current initialization state to file"""
        try:
            state = {
                'is_fully_initialized': self.is_fully_initialized,
                'timestamp': time.time()
            }
            with open(self.initialization_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(
                f"Saved state: fully_initialized={self.is_fully_initialized}")
        except Exception as e:
            print(f"Error saving state: {e}")

    def reset_initialization(self):
        """Reset initialization state - useful for debugging or forced reinitialization"""
        self.is_fully_initialized = False
        self.initialization_in_progress = False
        if os.path.exists(self.initialization_state_file):
            os.remove(self.initialization_state_file)
        print("Initialization state reset")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def load_image(self, filename):
        img = Image.open(filename)
        return img

    def convert_pdf_to_images(self, pdf_file: Path,  image_format='PNG', dpi=200):
        """
        Convert  PDF file in a folder to images

        Args:
            docs_folder (str): Path to folder containing PDF files
            output_folder (str): Path to output folder (default: docs_folder/images)
            image_format (str): Output format - PNG, JPEG, etc.
            dpi (int): Resolution for output images
        """

        # Set up paths
        output_path = Path("images")
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Converting {pdf_file.name}...")
            start = time.time()
            total_threads = os.cpu_count()
            half_threads = total_threads // 2
            half_threads = 1
            # Convert PDF to images
            # images = convert_from_path(pdf_file, dpi = dpi, thread_count = half_threads, use_pdftocairo = True)
            images = converter.convert_from_path_optimized(
                pdf_file, optimization_strategy="async", dpi=dpi, fmt="png")

            end = time.time()
            print(f"spent {end-start}s")
            # Save each page as separate image
            for i, image in enumerate(images):
                if len(images) == 1:
                    # Single page PDF
                    image_name = f"{pdf_file.stem}.{image_format.lower()}"
                else:
                    # Multi-page PDF
                    image_name = f"{pdf_file.stem}_page_{i+1:03d}.{image_format.lower()}"

                image_path = output_path / image_name
                image.save(image_path, image_format)

            print(f"✓ Converted {pdf_file.name} ({len(images)} pages)")

        except Exception as e:
            print(f" Error converting {pdf_file.name}: {str(e)}")

        print(f"\nConversion complete! Images saved to: {output_path}")

    def ingest_document(self, document_path, collection_name):

        # Define the images directory
        images_dir = Path("images")

        # Empty the images folder
        for file in images_dir.glob("*.*"):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

        # Convert PDFs to images if needed
        self.convert_pdf_to_images(
            Path(os.path.join(downloads_dir, document_path)))
        images = []
        available_files = list(Path("images").glob("*.*"))
        for doc_path in available_files:
            images.append(self.load_image(doc_path))

        for i, img in enumerate(images):
            print(i)
            batch_image = self.processor.process_images(
                [img]).to(self.model.device)
            with torch.no_grad():
                img_embedding = self.model(**batch_image)
            multivector = torch.unbind(img_embedding.to("cpu"))[
                0].float().numpy()
            print(multivector.shape)
            points = [
                PointStruct(
                    id=i,
                    vector=multivector,
                    payload={
                        "source": doc_path,
                        "document": f"{doc_path}",
                        "image_index": i+1,
                        # Store original shape for reference
                        "original_shape": list(img_embedding.shape),
                        "base64": self.pil_image_to_base64(img)
                    }
                )
            ]

            self.dbclient.upsert(
                collection_name=collection_name,
                points=points
            )

    def pil_image_to_base64(self, image, format='PNG'):
        """Convert a PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format=format)
        image_data = buffer.getvalue()
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return base64_string

    def download_knowledge_file(self, doc_id: str, filename: str):
        # Ensure the downloads directory exists
        os.makedirs(downloads_dir, exist_ok=True)
        url = f"{BASE_URL}/files/{doc_id}/content#"
        try:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            file_path = os.path.join(downloads_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"File downloaded successfully: {filename}")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")

    def get_knowledge_docs(self):
        try:
            print(f"request call")
            response = requests.get(
                f"{BASE_URL}/knowledge", headers=headers, timeout=30)
            print(f"response received")
            # Check if response is successful
            if response.status_code != 200:
                print(f"API returned status code {response.status_code}")
                print(f"Response content: {response.text}")
                return

            # Check if response is empty
            if not response.text.strip():
                print(f"Response is empty")
                return
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                for doc in data:
                    print(f"- ID: {doc.get('id')}, Name: {doc.get('name')}")
                    print(doc.get("files"))
            else:
                print("Unexpected response format:", data)
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def perform_full_initialization(self):
        """Perform the heavy initialization work (models, documents, etc.)"""
        print(f"========== Starting full initialization =============")

        try:
            # Wait a bit to ensure API is fully available
            time.sleep(2)

            self.dbclient = QdrantClient(path="mydb")

            model_name = "vidore/colqwen2.5-v0.1"
            model_name = "vidore/colqwen2-v1.0"
            try:
                print("Loading ColPali models...")
                self.model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0",  # or "mps" if on Apple Silicon
                    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                ).eval()

                self.processor = ColQwen2Processor.from_pretrained(model_name)
                print("✓ ColPali models loaded successfully")
            except Exception as e:
                print(f"Cannot load colpali models: {e}")
                return False
            if not self.is_fully_initialized:
                data = self.get_knowledge_docs()
                if not data:
                    print("No knowledge documents found or API error")
                    return False

                ids = []
                metadatas = []
                self.id_map = {}         # Maps int -> UUID
                self.reverse_id_map = {}  # Maps UUID -> int
                index = 0
                for doc in data:

                    print(f"Processing knowledge: {doc.get('name')}")
                    collection_name = doc.get('name')
                    if collection_name == "SAV_Soft":

                        try:
                            # Try to delete existing collection
                            self.dbclient.delete_collection(collection_name)
                            print(
                                f"Deleted existing collection: {collection_name}")
                        except Exception as e:
                            print(
                                f"Collection {collection_name} didn't exist or couldn't be deleted: {e}")

                        collection = self.dbclient.create_collection(
                            collection_name,
                            on_disk_payload=True,
                            optimizers_config=models.OptimizersConfigDiff(
                                indexing_threshold=100
                            ),
                            vectors_config=models.VectorParams(
                                size=128,
                                distance=models.Distance.COSINE,
                                multivector_config=models.MultiVectorConfig(
                                    comparator=models.MultiVectorComparator.MAX_SIM
                                ),
                                quantization_config=models.ScalarQuantization(
                                    scalar=models.ScalarQuantizationConfig(
                                        type=models.ScalarType.INT8,
                                        quantile=0.99,
                                        always_ram=True,
                                    ),
                                ),
                            ),
                        )

                        for file in doc.get('files'):
                            file_id = file.get('id')
                            filename = file.get('meta').get('name')
                            print(
                                f"Processing file: {filename} (ID: {file_id})")

                            self.download_knowledge_file(file_id, filename)
                            ids.append(index)
                            metadatas.append(file.get('meta'))
                            self.id_map[index] = file_id
                            self.reverse_id_map[file_id] = index
                            self.ingest_document(filename, collection_name)
                            index += 1

                print(f"✓ Full initialization completed successfully")
                self.is_fully_initialized = True
                self.save_initialization_state()
            return True

        except Exception as e:
            print(f"Error during full initialization: {e}")
            return False
        finally:
            self.initialization_in_progress = False

    def start_background_initialization(self):
        """Start the full initialization in a background thread"""
        def background_init():
            print("Starting background initialization...")
            self.perform_full_initialization()

        thread = threading.Thread(target=background_init, daemon=True)
        thread.start()
        print("Background initialization thread started")

    async def on_startup(self):
        """Lightweight startup - defer heavy work to background"""
        print(f"========== RAG pipeline startup (lightweight) =============")

        # If already fully initialized, nothing to do
        # if self.is_fully_initialized:
        #     print("Pipeline already fully initialized")
        #     return

        # If this is the first time or initialization was reset
        with self.initialization_lock:
            if not self.initialization_in_progress:
                self.initialization_in_progress = True
                # Start background initialization after startup completes
                # Use a small delay to ensure backend is fully up

                def delayed_init():
                    time.sleep(1)  # Give backend time to be fully ready
                    self.perform_full_initialization()

                thread = threading.Thread(target=delayed_init, daemon=True)
                thread.start()
                print("Background initialization scheduled")

        print(f"========== RAG pipeline startup completed quickly =============")

    async def on_shutdown(self):
        print(f"=========== RAG Pipeline shutdown =================")
        pass

    def ensure_initialized(self):
        """Ensure the pipeline is fully initialized before processing queries"""
        if self.is_fully_initialized:
            return True

        # If initialization is in progress, wait for it
        if self.initialization_in_progress:
            print("Initialization in progress, please wait...")
            max_wait = 300  # 5 minutes max wait
            waited = 0
            while self.initialization_in_progress and waited < max_wait:
                asyncio.sleep(1)
                waited += 1

            if self.is_fully_initialized:
                return True
            else:
                print("Initialization timed out or failed")
                return False

        # Try to initialize now if not in progress
        print("Pipeline not initialized - attempting initialization...")
        with self.initialization_lock:
            if not self.initialization_in_progress:
                self.initialization_in_progress = True
                success = self.perform_full_initialization()
                return success

        return False

    def query_db(self, question, top_k=5):
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
            raise Exception(
                "Models not loaded - initialization incomplete - retry in few minutes")
        results = None
        batch_queries = self.processor.process_queries(
            [question]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**batch_queries)
            multivector = torch.unbind(query_embedding.to("cpu"))[
                0].float().numpy()
            results = self.dbclient.query_points(
                collection_name="SAV_Soft",
                query=multivector,
                limit=top_k
            )
        return results

#     def query_vlm(self, query: str, image_documents: List[ImageDocument]) -> str:
#         """
#         Query the VLM with retrieved images

#         Args:
#             query: User question
#             image_documents: Retrieved image documents

#         Returns:
#             VLM response
#         """
#         if not image_documents:
#             return "No relevant documents found to answer your question."

#         # Prepare the prompt
#         context_info = []
#         for i, img_doc in enumerate(image_documents):
#             doc_info = f"Document {i+1}"
#             if 'doc_id' in img_doc.metadata:
#                 doc_info += f" (ID: {img_doc.metadata['doc_id']})"
#             if 'page_num' in img_doc.metadata:
#                 doc_info += f" - Page {img_doc.metadata['page_num']}"
#             if 'score' in img_doc.metadata:
#                 doc_info += f" - Relevance: {img_doc.metadata['score']:.3f}"
#             context_info.append(doc_info)

#         prompt = f"""Based on the following retrieved documents, please answer this question: {query}

# Retrieved documents:
# {chr(10).join(context_info)}

# Please analyze the images and provide a comprehensive answer. If the answer cannot be found in the provided documents, please say so clearly."""

#         try:
#             print("Querying VLM...")
#             response = self.vlm.complete(
#                 prompt=prompt,
#                 image_documents=image_documents
#             )
#             return response.text

#         except Exception as e:
#             return f"Error querying VLM: {str(e)}"
    def process_retrieved_images(self, results: List[Dict]) -> List:
        """
        Convert retrieved results to ImageDocument objects for VLM processing

        Args:
            results: Results from ColPali search

        Returns:
            List of image objects
        """
        image_documents = []
        logger.debug("process doc")
        for doc in results:
            logger.debug(
                f"{doc.payload['source']} page  {doc.payload['image_index']}")

        for i, result in enumerate(results):
            # Extract base64 image data
            # result = result.dict()
            # print(result.keys())
            # print("page ", result.get('page_num'),
            #       " ", result.get('doc_id'))
            logger.debug(f"type data {type(result)}")
            # logger.debug(f"{result.keys()}")
            if 'base64' in result.payload:
                logger.debug("cas1")
                base64_data = result.payload['base64']
                # Remove data URL prefix if present
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                image_documents.append(base64_data)

            elif 'image' in result.payload:
                logger.debug("cas2")

                # Handle PIL Image objects
                pil_image = result.payload['image']
                if isinstance(pil_image, Image.Image):
                    # Convert PIL image to base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(
                        buffered.getvalue()).decode()
                    image_documents.append(img_base64)
        logger.debug("process doc done")
        return image_documents

    def add_images_to_messages(self, json_data, images_list=None, message_index=None):
        """
        Add an 'images' field to specific message(s) in the Ollama JSON request.

        Args:
            json_data: Dictionary or JSON string containing the request
            images_list: List of base64 encoded images (default: empty list)
            message_index: Index of message to add images to (default: all user messages)

        Returns:
            Modified dictionary with images field added to messages
        """
        if images_list is None:
            images_list = []

        # Handle string input
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data.copy()  # Don't modify original

        # Add images to specific message or all user messages
        if "messages" in data:
            for i, message in enumerate(data["messages"]):
                # Add to specific message index or all user messages
                if message_index is None:
                    if message.get("role") == "user":
                        message["images"] = images_list
                elif i == message_index:
                    message["images"] = images_list

        return data

    def add_images_to_last_user_message(self, json_data, images_list=None):
        """
        Add images to the last user message in the conversation.

        Args:
            json_data: Dictionary or JSON string containing the request
            images_list: List of base64 encoded images

        Returns:
            Modified dictionary with images added to last user message
        """
        if images_list is None:
            images_list = []

        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data.copy()

        if "messages" in data:
            # Find last user message and add images
            for message in reversed(data["messages"]):
                if message.get("role") == "user":
                    message["images"] = images_list
                    break

        return data

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.debug("==============================================")
        # logger.debug(f"messages {messages}") fait peter le log
        logger.debug(f"user_message {user_message}")

        logger.debug(f"stream {body['stream']}")
        logger.debug(f"usermsg {body['user']}")
        # Check if title generation is requested
        # as of 12/28/24, these were standard greetings
        if ("broad tags categorizing" in user_message.lower()) or ("create a concise" in user_message.lower()):
            # ## Create a concise, 3-5 word title with
            # ## Task:\nGenerate 1-3 broad tags categorizing the main themes
            logger.debug(f"Title Generation (aborted): {user_message}")
            return "(title generation disabled)"

        # Ensure initialization before processing
        if not self.ensure_initialized():
            logger.debug(
                "Pipeline initialization failed or incomplete. Please wait a moment and try again.")
            return "Pipeline initialization failed or incomplete. Please wait a moment and try again."

        try:

            try:
                if body["stream"]:
                    retrieved_docs = self.query_db(user_message)
                    images = self.process_retrieved_images(
                        retrieved_docs.points)
                    try:
                        logger.debug("add images")
                        body = self.add_images_to_last_user_message(
                            body, images)
                        logger.debug("add images done")
                    except Exception as e:
                        return f"Error adding images: {e}"
                    try:
                        logger.debug("=========== BODY CONTENT======")
                        logger.debug(json.dumps(
                            body, indent=2, ensure_ascii=False))
                        logger.debug("=========== BODY CONTENT END======")
                    except:
                        pass

                model_id = "qwen2.5vl:latest"
                r = requests.post(
                    url="http://192.168.0.115:11434/api/chat",
                    json={**body, "model": model_id},
                    headers={"Content-Type": "application/json"},  # Add this
                    stream=True,
                )

                r.raise_for_status()

                if body.get("stream", False):
                    # Handle streaming response
                    def stream_generator():
                        for line in r.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line.decode('utf-8'))
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        yield chunk['message']['content']
                                except json.JSONDecodeError:
                                    continue
                    return stream_generator()
                else:
                    # Handle non-streaming response
                    response_json = r.json()
                    logger.debug(
                        f"Response JSON: {json.dumps(response_json, indent=2)}")

                    # Extract the actual message content
                    if 'message' in response_json and 'content' in response_json['message']:
                        return response_json['message']['content']
                    else:
                        logger.debug("Unexpected response format")
                        return f"Unexpected response format: {response_json}"
            except Exception as e:
                return f"Error: {e}"
        except Exception as e:
            logger.debug(f"Error during query processing: {e}")
            print(f"Error during query processing: {e}")
            return f"Error processing query: {str(e)}"


if __name__ == "__main__":
    pipeline = Pipeline()

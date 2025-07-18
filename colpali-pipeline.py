"""
title: Multimodal RAG Pipeline using Colpali
author: Stephane Ancelot
date: 2025-03-07
version: 4.0
license: MIT
description: A pipeline for retrieving relevant information using Vision Language models.
requirements: pdf2image, qdrant-client, colpali-engine, Pillow
"""
import shutil
from pdf2image import convert_from_path
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
BASE_URL = "http://10.1.42.88:8080/api/v1"
API_KEY = "sk-5d9ab3bd43c846f2a6da49e68dacbbf5"
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
                    self.is_fully_initialized = state.get('is_fully_initialized', False)
                    print(f"Loaded state: fully_initialized={self.is_fully_initialized}")
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
            print(f"Saved state: fully_initialized={self.is_fully_initialized}")
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

            total_threads = os.cpu_count()
            half_threads = total_threads // 2
            # Convert PDF to images
            images = convert_from_path(
                pdf_file, dpi=dpi, thread_count=half_threads)

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
            
            data = self.get_knowledge_docs()
            if not data:
                print("No knowledge documents found or API error")
                return False
                
            ids = []
            metadatas = []
            self.id_map = {}         # Maps int -> UUID
            self.reverse_id_map = {}  # Maps UUID -> int
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
                
            index = 0
            for doc in data:
                print(f"Processing knowledge: {doc.get('name')}")
                collection_name = doc.get('name')
                try:
                    # Try to delete existing collection
                    self.dbclient.delete_collection(collection_name)
                    print(f"Deleted existing collection: {collection_name}")
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
                    print(f"Processing file: {filename} (ID: {file_id})")
                    
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
        if self.is_fully_initialized:
            print("Pipeline already fully initialized")
            return
            
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

    async def ensure_initialized(self):
        """Ensure the pipeline is fully initialized before processing queries"""
        if self.is_fully_initialized:
            return True
            
        # If initialization is in progress, wait for it
        if self.initialization_in_progress:
            print("Initialization in progress, please wait...")
            max_wait = 300  # 5 minutes max wait
            waited = 0
            while self.initialization_in_progress and waited < max_wait:
                await asyncio.sleep(1)
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

    def query(self, question, top_k=5):
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
            raise Exception("Models not loaded - initialization incomplete")
            
        batch_queries = self.processor.process_queries(
            [question]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**batch_queries)
            multivector = torch.unbind(query_embedding.to("cpu"))[
                0].float().numpy()
            results = self.dbclient.query_points(
                collection_name="documents",
                query=multivector,
                limit=top_k
            )
            for point in results.points:
                print(point)
        return results

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        print(f"messages {messages}")
        print(f"user_message {user_message}")

        # Ensure initialization before processing
        if not await self.ensure_initialized():
            return "Pipeline initialization failed or incomplete. Please wait a moment and try again."

        try:
            response = self.query(user_message)
            return response
        except Exception as e:
            print(f"Error during query processing: {e}")
            return f"Error processing query: {str(e)}"


if __name__ == "__main__":
    pipeline = Pipeline()

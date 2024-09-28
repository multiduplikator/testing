import lancedb
import configparser
from datetime import timedelta
from pathlib import Path
from typing import List
from alive_progress import alive_bar
from langchain_community.document_loaders import (
    JSONLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dtos.Metadata import Metadata


class UnsupportedFileTypeError(Exception):
    """Exception for unsupported file types."""

    pass


class ConfigManager:
    """Manages configuration file access."""

    def __init__(self, config_path: Path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get(self, section: str, key: str, default: str = "") -> str:
        """Retrieve a string value from the config."""
        return self.config.get(section, key, fallback=default)

    def get_list(self, section: str, key: str) -> List[str]:
        """Retrieve a list of strings from the config."""
        return [item.strip() for item in self.get(section, key).split(",")]

    def get_path(self, section: str, key: str) -> Path:
        """Retrieve and expand a path from the config."""
        return Path(self.get(section, key)).expanduser()

    def get_int(self, section: str, key: str, default: int = 14) -> int:
        """Retrieve a string value from the config and convert it to an integer."""
        value = self.get(section, key, default=str(default))

        try:
            return int(value)
        except ValueError:
            return default


class Vectorizer:
    """Handles file processing and indexing."""

    def __init__(
        self,
        base_directory: Path,
        lance_directory: Path,
        embeddings: str,
        config_manager: ConfigManager,
    ):
        self.config_manager = config_manager
        self.base_directory = base_directory.resolve()
        self.strip_path = Path(__file__).parent.resolve()
        # Setup LangChain
        self.lance_directory = lance_directory
        self.embeddings = OpenAIEmbeddings(
            base_url=config_manager.get("config", "ollama_uri_emb"), model=embeddings, api_key="placeholder", check_embedding_ctx_length=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200
        )
        self.vector_index = LanceDB(
            uri=str(self.lance_directory),
            embedding=self.embeddings,
            mode="append",
            table_name=config_manager.get("vectorizer", "table_name"),
        )
        # File type handlers
        self.loaders = {
            ".pdf": self._load_pdf,
            ".json": self._load_json,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
        }

    def _metadata_func(self, record: dict, metadata: dict) -> dict:
        """Update metadata with additional information."""
        metadata.update({"start": record.get("start"), "end": record.get("end")})
        return metadata

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load and process PDF files."""
        loader = UnstructuredPDFLoader(
            str(file_path),
            mode="elements",
            chunking_strategy="basic",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            languages=self.config_manager.get_list("vectorizer", "languages"),
            combine_text_under_n_chars=800,
        )
        return self._process_documents(loader.load_and_split(self.text_splitter), "pdf")

    def _load_json(self, file_path: Path) -> List[Document]:
        """Load and process JSON files."""
        loader = JSONLoader(
            str(file_path),
            jq_schema=".[]",
            content_key="text",
            metadata_func=self._metadata_func,
        )
        return self._process_documents(loader.load(), "video")

    def _load_excel(self, file_path: Path) -> List[Document]:
        """Load and process Excel files."""
        loader = UnstructuredExcelLoader(str(file_path), mode="elements")
        return self._process_documents(loader.load(), "excel")

    def _process_documents(
        self, documents: List[Document], file_type: str
    ) -> List[Document]:
        """Filter and add context to documents."""
        return [
            self._add_context(doc, file_type)
            for doc in filter_complex_metadata(documents)
        ]

    def _add_context(self, doc: Document, file_type: str) -> Document:
        """Add context metadata to a document."""
        context = self._process_context(doc, file_type)
        doc.metadata = context.model_dump()
        return doc

    def _load_file(self, file_path: Path) -> List[Document]:
        """Load file based on its extension."""
        loader = self.loaders.get(file_path.suffix)
        if loader:
            return loader(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")

    def _get_all_files(self) -> List[Path]:
        """Get all files with supported extensions from the base directory."""
        return [
            file
            for file in self.base_directory.rglob("*")
            if file.suffix in self.loaders
        ]

    def process_files(self) -> None:
        """Process and index all files."""
        all_files = self._get_all_files()
        print("Indexing files...")
        with alive_bar(len(all_files)) as bar:
            for file_path in all_files:
                pages = self._load_file(file_path)
                if pages:
                    self.vector_index.add_documents(pages)
                bar.text(f"Processed: {file_path}")
                bar()

    def _strip_path_to_root(self, full_path: Path) -> str:
        """Get relative path from base directory."""
        try:
            return str(full_path.relative_to(self.base_directory))
        except ValueError:
            return str(full_path)

    def _process_context(self, doc: Document, file_type: str) -> Metadata:
        """Generate Metadata object from document."""
        metadata = doc.dict()["metadata"]
        source = self._strip_path_to_root(Path(metadata["source"]))
        context_kwargs = {
            "file": source,
            "doctype": file_type,
            "page": "",
            "excerpt": doc.dict()["page_content"],
            "context": "",
        }

        if file_type == "pdf":
            context_kwargs["page"] = str(
                metadata.get("page_number", metadata.get("page", 0) + 1)
            )
        elif file_type == "excel":
            context_kwargs["page"] = str(metadata.get("page_name", ""))
        elif file_type == "video":
            context_kwargs.update(
                {
                    "file": source.replace(".json", ".mp4"),
                    "start": str(metadata.get("start", "")),
                    "end": str(metadata.get("end", "")),
                }
            )

        return Metadata(**context_kwargs)


if __name__ == "__main__":
    # Initialize configuration manager and vectorizer
    config_manager = ConfigManager(Path("./config.ini"))
    vectorizer = Vectorizer(
        base_directory=config_manager.get_path("paths", "base_directory"),
        lance_directory=config_manager.get_path("paths", "lance_db"),
        embeddings=str(config_manager.get_path("model", "embedding")),
        config_manager=config_manager,
    )

    # Open vectorstore for fast rebuild and version cleanup
    db_path = Path(config_manager.get_path("paths", "lance_db"))
    table_name = config_manager.get("vectorizer", "table_name")
    days_to_keep = config_manager.get_int("vectorizer", "days_to_keep")

    # Connect to the database
    db = lancedb.connect(str(db_path))

    # Check if table exists and process accordingly
    try:
        table = db.open_table(table_name)
        table.delete("true")
    except Exception:
        table = None

    if table is None:
        vectorizer.process_files()
        table = db.open_table(table_name)
    else:
        vectorizer.process_files()

    # Cleanup old versions and compact files
    table.cleanup_old_versions(older_than=timedelta(days=days_to_keep))
    table.compact_files()

import os
import tempfile
import json
import re
import ast
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging

# MCP Server imports
from mcp.server.fastmcp import FastMCP, Context

# Git repository handling
import git
from git import Repo

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    print("Installing langchain...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain"])
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

# Vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("mongodb-adoption-assistant")

# Global variables for MongoDB docs
REPO_URL = "https://github.com/mongodb/docs"
MONGO_REPO_DIR = None
VECTOR_INDEX = None
DOC_CONTENTS = []
EMBEDDER = None
FILE_PATHS = []

# Codebase analysis results
CODEBASE_ANALYSIS = {}
CURRENT_WORKING_DIR = os.getcwd()
INITIALIZED = False


@dataclass
class DatabasePattern:
    """Represents a detected database usage pattern in the codebase."""
    pattern_type: str  # 'sql', 'orm', 'nosql', 'file_storage'
    technology: str  # 'mysql', 'postgresql', 'sqlite', 'redis', etc.
    files: List[str]  # Files where pattern is found
    code_snippets: List[str]  # Example code snippets
    frequency: int  # How often this pattern appears
    complexity: str  # 'simple', 'medium', 'complex'


@dataclass
class CodebaseAnalysis:
    """Results of codebase analysis."""
    project_type: str  # 'web_app', 'api', 'desktop_app', etc.
    languages: List[str]
    frameworks: List[str]
    database_patterns: List[DatabasePattern]
    data_models: List[Dict[str, Any]]
    migration_complexity: str
    recommendations: List[str]
    project_path: str  # Store the analyzed project path


def initialize_resources():
    """Initialize global resources lazily."""
    global MONGO_REPO_DIR, EMBEDDER, CURRENT_WORKING_DIR, INITIALIZED
    
    if INITIALIZED:
        return "Already initialized"
    
    try:
        MONGO_REPO_DIR = tempfile.mkdtemp()
        logger.info(f"MongoDB docs will be cloned to: {MONGO_REPO_DIR}")

        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        CURRENT_WORKING_DIR = os.getcwd()
        INITIALIZED = True
        
        logger.info(f"Current working directory: {CURRENT_WORKING_DIR}")
        logger.info("Resources initialized successfully")
        return "Resources initialized successfully"
        
    except Exception as e:
        logger.error(f"Error initializing resources: {e}")
        return f"Error initializing resources: {e}"


# Database pattern detection
DATABASE_PATTERNS = {
    "sql": {
        "keywords": [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE", 
            "ALTER TABLE", "DROP TABLE", "JOIN", "WHERE", "ORDER BY"
        ],
        "imports": ["sqlite3", "psycopg2", "mysql.connector", "pymysql", "sqlalchemy"],
        "file_extensions": [".sql"],
    },
    "orm": {
        "keywords": [
            "models.Model", "db.Model", "Entity", "@Entity", "class.*Model",
            "Session", "Query", "relationship", "ForeignKey"
        ],
        "imports": [
            "django.db", "sqlalchemy", "flask_sqlalchemy", "peewee", 
            "tortoise", "sequelize", "typeorm", "prisma"
        ],
        "patterns": ["models.py", "entities/", "schemas/"],
    },
    "nosql": {
        "keywords": ["find(", "aggregate(", "insertOne", "updateMany", "collection"],
        "imports": ["pymongo", "motor", "redis", "cassandra", "elasticsearch"],
        "file_patterns": ["*.json", "data/*.json"],
    },
    "file_storage": {
        "keywords": [
            "json.load", "json.dump", "pickle.load", "csv.reader", "open(.*\.json"
        ],
        "imports": ["json", "pickle", "csv", "pandas"],
        "patterns": ["data/", "storage/", "*.json", "*.csv", "*.pickle"],
    },
}


def get_project_info(project_path: str) -> Dict[str, Any]:
    """Get basic information about the current project."""
    path = Path(project_path)
    info = {
        "path": str(path.absolute()),
        "name": path.name,
        "is_git_repo": (path / ".git").exists(),
        "files_count": 0,
        "directories_count": 0,
        "size_mb": 0,
    }

    try:
        # Count files and directories
        for item in path.rglob("*"):
            if item.is_file():
                info["files_count"] += 1
                try:
                    info["size_mb"] += item.stat().st_size / (1024 * 1024)
                except:
                    pass
            elif item.is_dir():
                info["directories_count"] += 1

        info["size_mb"] = round(info["size_mb"], 2)

        # Get git info if available
        if info["is_git_repo"]:
            try:
                repo = Repo(project_path)
                info["git_branch"] = repo.active_branch.name
                info["git_remote"] = repo.remotes.origin.url if repo.remotes else None
            except:
                pass

    except Exception as e:
        logger.error(f"Error getting project info: {e}")

    return info


def detect_project_type(codebase_path: str) -> str:
    """Detect the type of project based on files and structure."""
    path = Path(codebase_path)

    # Check for common framework files
    if (path / "package.json").exists():
        try:
            package_json = json.loads((path / "package.json").read_text())
            dependencies = package_json.get("dependencies", {})
            if any(dep in dependencies for dep in ["express", "koa", "fastify"]):
                return "node_api"
            elif any(dep in dependencies for dep in ["react", "vue", "angular"]):
                return "web_frontend"
            else:
                return "node_app"
        except:
            return "node_app"

    elif (path / "requirements.txt").exists() or (path / "pyproject.toml").exists():
        if (path / "manage.py").exists():
            return "django_app"
        elif any((path / name).exists() for name in ["app.py", "main.py"]):
            return "flask_app"
        else:
            return "python_app"

    elif (path / "pom.xml").exists() or (path / "build.gradle").exists():
        return "java_app"

    elif (path / "Cargo.toml").exists():
        return "rust_app"

    elif (path / "go.mod").exists():
        return "go_app"

    else:
        return "unknown"


def analyze_python_file(file_path: str) -> List[DatabasePattern]:
    """Analyze a Python file for database patterns."""
    patterns = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Parse AST for more accurate analysis
        try:
            tree = ast.parse(content)

            # Check imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Detect patterns based on imports and code
            for pattern_type, config in DATABASE_PATTERNS.items():
                matches = []

                # Check imports
                for imp in imports:
                    if any(db_import in imp for db_import in config.get("imports", [])):
                        matches.append(f"Import: {imp}")

                # Check keywords in content
                for keyword in config.get("keywords", []):
                    if re.search(keyword, content, re.IGNORECASE):
                        matches.append(f"Keyword: {keyword}")

                if matches:
                    # Extract code snippets
                    snippets = []
                    for keyword in config.get("keywords", [])[:3]:  # Limit to first 3
                        matches_iter = re.finditer(f".*{keyword}.*", content, re.IGNORECASE)
                        for match in list(matches_iter)[:2]:  # Max 2 per keyword
                            snippets.append(match.group().strip())

                    patterns.append(
                        DatabasePattern(
                            pattern_type=pattern_type,
                            technology=imports[0] if imports else "unknown",
                            files=[file_path],
                            code_snippets=snippets,
                            frequency=len(matches),
                            complexity=(
                                "simple" if len(matches) < 5
                                else "medium" if len(matches) < 20 
                                else "complex"
                            ),
                        )
                    )

        except SyntaxError:
            # Fallback to regex analysis
            pass

    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")

    return patterns


def analyze_javascript_file(file_path: str) -> List[DatabasePattern]:
    """Analyze a JavaScript/TypeScript file for database patterns."""
    patterns = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Common JS/TS database patterns
        js_patterns = {
            "sql": ["mysql", "pg", "sqlite3", "mssql", "oracle"],
            "orm": ["sequelize", "typeorm", "prisma", "mongoose", "knex"],
            "nosql": ["mongodb", "redis", "elasticsearch", "dynamodb"],
        }

        for pattern_type, technologies in js_patterns.items():
            for tech in technologies:
                if (re.search(f"require\\(['\"].*{tech}.*['\"]\\)", content) or 
                    re.search(f"import.*from.*['\"].*{tech}.*['\"]", content)):

                    # Extract some code snippets
                    snippets = []
                    for match in re.finditer(f".*{tech}.*", content)[:3]:
                        snippets.append(match.group().strip())

                    patterns.append(
                        DatabasePattern(
                            pattern_type=pattern_type,
                            technology=tech,
                            files=[file_path],
                            code_snippets=snippets,
                            frequency=len(re.findall(tech, content, re.IGNORECASE)),
                            complexity="simple",
                        )
                    )

    except Exception as e:
        logger.error(f"Error analyzing JS file {file_path}: {e}")

    return patterns


@mcp.tool()
def auto_analyze_codebase_for_mongodb() -> str:
    """
    One-click MongoDB adoption analysis - automatically analyzes current codebase, 
    loads MongoDB docs, and provides comprehensive adoption recommendations.
    """
    global CODEBASE_ANALYSIS, CURRENT_WORKING_DIR
    
    try:
        # Initialize if needed
        init_result = initialize_resources()
        logger.info(f"Initialization: {init_result}")
        
        # Step 1: Analyze current codebase
        logger.info("Step 1: Analyzing current codebase...")
        codebase_result = analyze_current_codebase()
        
        # Step 2: Clone and process MongoDB docs if needed
        if not VECTOR_INDEX:
            logger.info("Step 2: Setting up MongoDB documentation...")
            clone_result = clone_repository()
            logger.info(f"Clone result: {clone_result}")
            
            process_result = process_docs()
            logger.info(f"Process result: {process_result}")
        
        # Step 3: Generate adoption plan with documentation context
        logger.info("Step 3: Generating MongoDB adoption plan...")
        adoption_plan = get_mongodb_adoption_plan()
        
        # Combine results
        result = f"# MongoDB Adoption Analysis for Your Codebase\n\n"
        result += f"## Current Project Analysis\n"
        result += f"**Location:** {CURRENT_WORKING_DIR}\n\n"
        
        if CODEBASE_ANALYSIS:
            result += f"**Project Type:** {CODEBASE_ANALYSIS.project_type}\n"
            result += f"**Languages:** {', '.join(CODEBASE_ANALYSIS.languages)}\n"
            result += f"**Database Patterns Found:** {len(CODEBASE_ANALYSIS.database_patterns)}\n\n"
            
            if CODEBASE_ANALYSIS.database_patterns:
                result += f"### Current Database Technologies\n"
                for pattern in CODEBASE_ANALYSIS.database_patterns:
                    result += f"- **{pattern.technology}** ({pattern.pattern_type}): {pattern.frequency} occurrences\n"
                result += "\n"
        
        result += adoption_plan
        
        return result
        
    except Exception as e:
        logger.error(f"Error in auto analysis: {e}")
        return f"Error during auto-analysis: {str(e)}"


@mcp.tool()
def get_current_project_info() -> str:
    """Get information about the current working directory project."""
    global CURRENT_WORKING_DIR

    try:
        info = get_project_info(CURRENT_WORKING_DIR)

        result = f"# Current Project Information\n\n"
        result += f"**Project Name:** {info['name']}\n"
        result += f"**Path:** {info['path']}\n"
        result += f"**Files:** {info['files_count']}\n"
        result += f"**Directories:** {info['directories_count']}\n"
        result += f"**Size:** {info['size_mb']} MB\n"

        if info["is_git_repo"]:
            result += f"**Git Repository:** Yes\n"
            if "git_branch" in info:
                result += f"**Current Branch:** {info['git_branch']}\n"
            if "git_remote" in info and info["git_remote"]:
                result += f"**Remote:** {info['git_remote']}\n"
        else:
            result += f"**Git Repository:** No\n"

        return result

    except Exception as e:
        return f"Error getting project info: {str(e)}"


@mcp.tool()
def analyze_current_codebase(include_patterns: str = "all") -> str:
    """
    Analyze the current working directory codebase to understand data storage patterns and architecture.

    Args:
        include_patterns: Comma-separated patterns to include ('sql,orm,nosql,file_storage' or 'all')
    """
    global CODEBASE_ANALYSIS, CURRENT_WORKING_DIR

    try:
        codebase_path = CURRENT_WORKING_DIR
        path = Path(codebase_path)

        # Basic project analysis
        project_type = detect_project_type(codebase_path)
        project_info = get_project_info(codebase_path)

        # Find relevant files
        code_files = []
        for pattern in ["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.go", "**/*.rs"]:
            code_files.extend(path.glob(pattern))

        # Limit to reasonable number of files and exclude common ignore patterns
        code_files = [
            f for f in code_files
            if not any(skip in str(f) for skip in [
                ".git", "node_modules", "__pycache__", ".venv", 
                "dist", "build", ".next", "target", "venv"
            ])
        ][:200]  # Increased limit

        # Analyze each file
        all_patterns = []
        languages = set()

        for file_path in code_files:
            file_str = str(file_path)

            if file_str.endswith(".py"):
                languages.add("Python")
                all_patterns.extend(analyze_python_file(file_str))
            elif file_str.endswith((".js", ".ts")):
                languages.add("JavaScript/TypeScript")
                all_patterns.extend(analyze_javascript_file(file_str))

        # Consolidate patterns
        consolidated_patterns = {}
        for pattern in all_patterns:
            key = f"{pattern.pattern_type}_{pattern.technology}"
            if key in consolidated_patterns:
                consolidated_patterns[key].files.extend(pattern.files)
                consolidated_patterns[key].code_snippets.extend(pattern.code_snippets)
                consolidated_patterns[key].frequency += pattern.frequency
            else:
                consolidated_patterns[key] = pattern

        # Create analysis results
        analysis = CodebaseAnalysis(
            project_type=project_type,
            languages=list(languages),
            frameworks=[],  # Could be enhanced to detect frameworks
            database_patterns=list(consolidated_patterns.values()),
            data_models=[],  # Could be enhanced to extract data models
            migration_complexity="medium",  # Could be calculated based on patterns
            recommendations=[],  # Will be generated by another tool
            project_path=codebase_path,
        )

        CODEBASE_ANALYSIS = analysis

        # Format results
        result = f"# Codebase Analysis Results\n\n"
        result += f"**Project:** {project_info['name']}\n"
        result += f"**Type:** {project_type}\n"
        result += f"**Languages:** {', '.join(languages) if languages else 'None detected'}\n"
        result += f"**Files Analyzed:** {len(code_files)}\n"
        result += f"**Total Size:** {project_info['size_mb']} MB\n\n"

        result += f"## Database Patterns Detected\n\n"

        if not consolidated_patterns:
            result += "No database patterns detected in the analyzed files.\n"
            result += "This could mean:\n"
            result += "- The project doesn't use databases yet (perfect for MongoDB!)\n"
            result += "- Database code is in files not analyzed\n"
            result += "- The project uses database patterns not in our detection rules\n\n"
        else:
            for i, (key, pattern) in enumerate(consolidated_patterns.items(), 1):
                result += f"### {i}. {pattern.pattern_type.title()} Pattern - {pattern.technology}\n"
                result += f"- **Files:** {len(pattern.files)} files\n"
                result += f"- **Frequency:** {pattern.frequency} occurrences\n"
                result += f"- **Complexity:** {pattern.complexity}\n"

                if pattern.code_snippets:
                    result += f"- **Example Code:**\n"
                    for snippet in pattern.code_snippets[:2]:  # Show max 2 examples
                        result += f"  ```\n  {snippet}\n  ```\n"
                result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error analyzing codebase: {e}")
        return f"Error analyzing current codebase: {str(e)}"


@mcp.tool()
def clone_repository() -> str:
    """Clone the MongoDB documentation repository."""
    global MONGO_REPO_DIR

    if not MONGO_REPO_DIR:
        initialize_resources()

    try:
        logger.info(f"Cloning repository from {REPO_URL} to {MONGO_REPO_DIR}...")
        Repo.clone_from(REPO_URL, MONGO_REPO_DIR, depth=1)  # Shallow clone for speed
        return f"Repository cloned successfully to {MONGO_REPO_DIR}"
    except git.GitCommandError as e:
        if "already exists" in str(e):
            return f"Repository already exists at {MONGO_REPO_DIR}"
        return f"Error cloning repository: {str(e)}"
    except Exception as e:
        return f"Error cloning repository: {str(e)}"


def clean_rst_content(content: str) -> str:
    """Clean reStructuredText content for better processing."""
    rst_patterns = [
        r":orphan:",
        r"\.\. default-domain:: \w+",
        r"\.\. contents::.*?(?=\n\n|\n[^\s]|\Z)",
        r"\.\. \w+::.*?(?=\n\n|\n[^\s]|\Z)",
        r":\w+:`[^`]*`",
        r"\.\. _[\w-]+:",
        r"`([^`]*) <[^>]*>`__?",
        r"\|(\w+)\|",
        r"{\+[^}]+\+}",
        r"=+\s*$",
        r"-+\s*$",
        r"~+\s*$",
        r"\^+\s*$",
    ]

    cleaned = content
    for pattern in rst_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"^\s+", "", cleaned, flags=re.MULTILINE)

    return cleaned.strip()


def extract_rst_title(content: str) -> str:
    """Extract the main title from RST content."""
    lines = content.split("\n")

    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if len(next_line) > 0 and all(c in "=-~^" for c in next_line):
                title = line.strip()
                if title and not title.startswith(":") and not title.startswith(".."):
                    return title

    for line in lines:
        line = line.strip()
        if (line and not line.startswith(":") and 
            not line.startswith("..") and not line.startswith("=")):
            return line

    return "Untitled"


def extract_documentation_files(repo_dir: str, subfolder: Optional[str] = None) -> List[str]:
    """Extract all documentation files (.txt and .md) from the repository."""
    base_dir = Path(repo_dir)
    if subfolder:
        base_dir = base_dir / subfolder

    doc_files = []
    for pattern in ["**/*.txt", "**/*.md", "**/*.rst"]:
        for path in base_dir.glob(pattern):
            if any(skip_dir in str(path) for skip_dir in [
                ".git", "__pycache__", ".pytest_cache", ".vscode"
            ]):
                continue
            doc_files.append(str(path))

    return doc_files


@mcp.tool()
def process_docs(subfolder: str = "source") -> str:
    """Process MongoDB documentation files from a specific subfolder."""
    global DOC_CONTENTS, FILE_PATHS, MONGO_REPO_DIR, EMBEDDER, VECTOR_INDEX

    if not MONGO_REPO_DIR:
        return "Repository not cloned yet. Please run clone_repository first."

    try:
        doc_files = extract_documentation_files(MONGO_REPO_DIR, subfolder)

        if not doc_files:
            return f"No documentation files found in subfolder: {subfolder}"

        chunks = []
        FILE_PATHS = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        processed_count = 0
        for file_path in doc_files[:500]:  # Limit to prevent timeout
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if len(content.strip()) < 100:
                    continue

                rel_path = os.path.relpath(file_path, MONGO_REPO_DIR)

                if file_path.endswith((".txt", ".rst")):
                    cleaned_content = clean_rst_content(content)
                    title = extract_rst_title(content)
                    file_type = "rst"
                else:
                    cleaned_content = content
                    title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                    title = (title_match.group(1) if title_match 
                            else os.path.basename(file_path))
                    file_type = "markdown"

                if len(cleaned_content.strip()) < 50:
                    continue

                file_chunks = text_splitter.create_documents(
                    texts=[cleaned_content],
                    metadatas=[{"source": rel_path, "title": title, "file_type": file_type}],
                )

                chunks.extend(file_chunks)
                FILE_PATHS.append(rel_path)
                processed_count += 1

                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count} files...")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")

        DOC_CONTENTS = chunks

        if not chunks:
            return "No valid content found after processing files."

        logger.info("Generating embeddings...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = EMBEDDER.encode(chunk.page_content)
            embeddings.append(embedding)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{len(chunks)} embeddings...")

        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]

        VECTOR_INDEX = faiss.IndexFlatL2(dimension)
        VECTOR_INDEX.add(embeddings_array)

        return f"Successfully processed {len(chunks)} chunks from {processed_count} documentation files."

    except Exception as e:
        logger.error(f"Error processing documentation: {e}")
        return f"Error processing documentation: {str(e)}"


@mcp.tool()
def get_mongodb_adoption_plan(focus_areas: str = "all") -> str:
    """
    Generate MongoDB adoption recommendations based on the current codebase analysis.

    Args:
        focus_areas: Comma-separated focus areas ('migration,performance,schema,integration' or 'all')
    """
    global CODEBASE_ANALYSIS, VECTOR_INDEX, DOC_CONTENTS, EMBEDDER, CURRENT_WORKING_DIR

    if not CODEBASE_ANALYSIS:
        return "No codebase analysis available. Please run analyze_current_codebase first."

    try:
        analysis = CODEBASE_ANALYSIS
        project_info = get_project_info(CURRENT_WORKING_DIR)

        # Generate search queries based on detected patterns
        search_queries = []

        # Base query
        base_query = f"MongoDB adoption {analysis.project_type} {' '.join(analysis.languages)}"

        # Pattern-specific queries
        for pattern in analysis.database_patterns:
            if pattern.pattern_type == "sql":
                search_queries.append(f"MongoDB migration from SQL {pattern.technology} relational database")
            elif pattern.pattern_type == "orm":
                search_queries.append(f"MongoDB ODM object document mapper {pattern.technology}")
            elif pattern.pattern_type == "nosql":
                search_queries.append(f"MongoDB compared to {pattern.technology}")
            elif pattern.pattern_type == "file_storage":
                search_queries.append("MongoDB document storage JSON files")

        # If no patterns detected, use generic queries
        if not search_queries:
            search_queries = [
                f"MongoDB getting started {analysis.project_type}",
                f"MongoDB best practices {' '.join(analysis.languages)}",
                "MongoDB schema design document database",
            ]

        # Search MongoDB documentation for relevant information if available
        all_recommendations = []
        if VECTOR_INDEX and DOC_CONTENTS:
            for query in search_queries[:5]:  # Limit to 5 searches
                try:
                    # Generate embedding for search
                    query_embedding = EMBEDDER.encode(query)
                    query_embedding = np.array([query_embedding]).astype("float32")

                    # Search the index
                    distances, indices = VECTOR_INDEX.search(query_embedding, 3)  # Top 3 per query

                    for idx in indices[0]:
                        if idx < len(DOC_CONTENTS):
                            doc = DOC_CONTENTS[idx]
                            all_recommendations.append({
                                "source": doc.metadata.get("source", "Unknown"),
                                "title": doc.metadata.get("title", "Untitled"),
                                "content": doc.page_content,
                                "query": query,
                            })
                except Exception as e:
                    logger.error(f"Error searching for query '{query}': {e}")

        # Format comprehensive adoption plan
        result = f"# MongoDB Adoption Plan for {project_info['name']}\n\n"

        # Executive Summary
        result += f"## Executive Summary\n\n"
        result += f"**Project:** {project_info['name']} ({analysis.project_type})\n"
        result += f"**Location:** {CURRENT_WORKING_DIR}\n"
        result += f"**Languages:** {', '.join(analysis.languages) if analysis.languages else 'Mixed/Unknown'}\n"
        result += f"**Current Database Patterns:** {len(analysis.database_patterns)} different patterns detected\n"

        # Determine migration complexity
        if not analysis.database_patterns:
            migration_complexity = "Low - Greenfield Project"
            complexity_note = "No existing database patterns detected. This is ideal for MongoDB adoption."
        else:
            total_complexity = (
                sum(1 for p in analysis.database_patterns if p.complexity == "simple") +
                sum(2 for p in analysis.database_patterns if p.complexity == "medium") +
                sum(3 for p in analysis.database_patterns if p.complexity == "complex")
            )

            if total_complexity <= 3:
                migration_complexity = "Low"
                complexity_note = "Simple patterns detected. Migration should be straightforward."
            elif total_complexity <= 8:
                migration_complexity = "Medium"
                complexity_note = "Multiple patterns detected. Requires careful planning."
            else:
                migration_complexity = "High"
                complexity_note = "Complex patterns detected. Consider gradual migration."

        result += f"**Migration Complexity:** {migration_complexity}\n"
        result += f"{complexity_note}\n\n"

        # Current State Analysis
        result += f"## Current State Analysis\n\n"

        if analysis.database_patterns:
            result += f"### Detected Database Technologies\n"
            for pattern in analysis.database_patterns:
                result += f"- **{pattern.technology}** ({pattern.pattern_type}): {pattern.frequency} occurrences in {len(pattern.files)} files\n"
            result += "\n"
        else:
            result += f"### No Database Technologies Detected\n"
            result += f"This appears to be either:\n"
            result += f"- A new project without database integration\n"
            result += f"- A project using database technologies not in our detection patterns\n"
            result += f"- A frontend-only or static project\n\n"

        # Migration Strategy
        result += f"## Recommended Migration Strategy\n\n"

        if not analysis.database_patterns:
            result += f"### Greenfield MongoDB Implementation\n"
            result += f"Since no existing database patterns were detected, you can implement MongoDB from scratch:\n\n"
            result += f"1. **Design Phase**\n"
            result += f"   - Define your data models based on application requirements\n"
            result += f"   - Plan document structure around your query patterns\n"
            result += f"   - Consider embedding vs referencing for relationships\n\n"
            result += f"2. **Implementation Phase**\n"
            result += f"   - Set up MongoDB (local or cloud)\n"
            result += f"   - Install MongoDB driver for {', '.join(analysis.languages) if analysis.languages else 'your language'}\n"
            result += f"   - Implement data models and connections\n\n"
        else:
            if any(p.pattern_type == "sql" for p in analysis.database_patterns):
                result += f"### SQL to MongoDB Migration\n"
                result += f"- **Approach:** Gradual migration with dual-write pattern\n"
                result += f"- **Schema Design:** Document-oriented schema based on access patterns\n"
                result += f"- **Data Migration:** Use MongoDB Compass or custom scripts\n\n"

            if any(p.pattern_type == "orm" for p in analysis.database_patterns):
                result += f"### ORM to ODM Transition\n"
                result += f"- **Replace ORM:** Consider Mongoose (Node.js) or MongoEngine (Python)\n"
                result += f"- **Schema Validation:** Use MongoDB schema validation\n"
                result += f"- **Relationship Modeling:** Embed vs Reference decisions\n\n"

        # Technology-specific recommendations
        if "Python" in analysis.languages:
            result += f"### Python-Specific Recommendations\n"
            result += f"- **Driver:** Use PyMongo for direct access or MongoEngine for ODM\n"
            result += f"- **Async Support:** Consider Motor for async applications\n"
            result += f"- **Integration:** Flask-PyMongo or Django-MongoDB-Engine\n\n"

        if "JavaScript/TypeScript" in analysis.languages:
            result += f"### JavaScript/TypeScript Recommendations\n"
            result += f"- **Driver:** Use official MongoDB Node.js driver or Mongoose ODM\n"
            result += f"- **Type Safety:** Mongoose with TypeScript definitions\n"
            result += f"- **Integration:** Express.js with MongoDB middleware\n\n"

        # Specific Recommendations from Documentation
        if all_recommendations:
            result += f"## MongoDB Documentation Insights\n\n"

            seen_sources = set()
            for i, rec in enumerate(all_recommendations[:6], 1):  # Show top 6 unique recommendations
                source = rec["source"]
                if source not in seen_sources:
                    seen_sources.add(source)
                    result += f"### {i}. {rec['title']}\n"
                    result += f"**Source:** {source}\n"
                    result += f"**Context:** {rec['query']}\n\n"
                    result += f"{rec['content'][:400]}...\n\n"
                    result += "---\n\n"

        # Implementation Steps
        result += f"## Implementation Steps\n\n"
        result += f"### Phase 1: Planning (Week 1)\n"
        result += f"- [ ] Review current data models and access patterns\n"
        result += f"- [ ] Design MongoDB schema based on queries\n"
        result += f"- [ ] Choose deployment option (local, Atlas, self-hosted)\n"
        result += f"- [ ] Plan migration strategy\n\n"

        result += f"### Phase 2: Setup (Week 2)\n"
        result += f"- [ ] Install and configure MongoDB\n"
        result += f"- [ ] Set up development environment\n"
        result += f"- [ ] Install MongoDB drivers/ODM\n"
        result += f"- [ ] Create initial connection and basic models\n\n"

        result += f"### Phase 3: Implementation (Week 3-4)\n"
        if analysis.database_patterns:
            result += f"- [ ] Implement dual-write pattern for existing data\n"
            result += f"- [ ] Migrate schema and data incrementally\n"
        else:
            result += f"- [ ] Implement MongoDB models and operations\n"
            result += f"- [ ] Create CRUD operations\n"
        result += f"- [ ] Update application code\n"
        result += f"- [ ] Implement comprehensive testing\n\n"

        result += f"### Phase 4: Optimization (Week 5-6)\n"
        result += f"- [ ] Create appropriate indexes\n"
        result += f"- [ ] Monitor performance and optimize queries\n"
        result += f"- [ ] Implement backup and monitoring\n"
        result += f"- [ ] Document the new architecture\n\n"

        # Next Steps
        result += f"## Immediate Next Steps\n\n"
        result += f"1. **Install MongoDB** locally or sign up for MongoDB Atlas\n"
        result += f"2. **Install drivers** for your technology stack\n"
        result += f"3. **Experiment** with basic operations using MongoDB shell\n"
        result += f"4. **Design** your first document schemas\n\n"

        result += f"💡 **Tip:** Start with a single feature or data model to gain experience before full migration.\n"

        return result

    except Exception as e:
        logger.error(f"Error generating adoption plan: {e}")
        return f"Error generating adoption plan: {str(e)}"


@mcp.tool()
def search_docs_for_current_project(query: str, top_k: int = 5) -> str:
    """
    Search MongoDB documentation with context from the current codebase analysis.

    Args:
        query: Search query for MongoDB documentation
        top_k: Number of results to return
    """
    global VECTOR_INDEX, DOC_CONTENTS, EMBEDDER, CODEBASE_ANALYSIS, CURRENT_WORKING_DIR

    if not VECTOR_INDEX or not DOC_CONTENTS:
        return "Documentation not processed yet. Please run clone_repository and process_docs first."

    try:
        # Enhance search with current codebase context
        search_text = query

        if CODEBASE_ANALYSIS:
            # Add context from current codebase
            context_parts = []
            if CODEBASE_ANALYSIS.languages:
                context_parts.append(f"Languages: {', '.join(CODEBASE_ANALYSIS.languages)}")
            if CODEBASE_ANALYSIS.project_type:
                context_parts.append(f"Project type: {CODEBASE_ANALYSIS.project_type}")
            if CODEBASE_ANALYSIS.database_patterns:
                patterns = [f"{p.pattern_type} ({p.technology})" for p in CODEBASE_ANALYSIS.database_patterns]
                context_parts.append(f"Current database patterns: {', '.join(patterns)}")

            if context_parts:
                search_text = f"{query} {' '.join(context_parts)}"

        # Generate embedding for the search text
        query_embedding = EMBEDDER.encode(search_text)
        query_embedding = np.array([query_embedding]).astype("float32")

        # Search the index
        distances, indices = VECTOR_INDEX.search(query_embedding, top_k)

        # Format results
        results = []
        project_info = get_project_info(CURRENT_WORKING_DIR)

        for i, idx in enumerate(indices[0]):
            if idx >= len(DOC_CONTENTS):
                continue

            doc = DOC_CONTENTS[idx]
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "Untitled")
            file_type = doc.metadata.get("file_type", "unknown")
            distance = distances[0][i]
            relevance_score = 1 / (1 + distance)

            result = f"## {title}\n"
            result += f"**Source:** {source} ({file_type})\n"
            result += f"**Relevance:** {relevance_score:.3f}\n\n"
            result += doc.page_content
            result += "\n\n---\n\n"

            results.append(result)

        if not results:
            return f"No relevant documentation found for: {query}"

        # Create comprehensive header with project context
        header = f"# MongoDB Documentation Search Results\n\n"
        header += f"**Project:** {project_info['name']}\n"
        header += f"**Query:** {query}\n"

        if CODEBASE_ANALYSIS:
            header += f"**Project Context:** {CODEBASE_ANALYSIS.project_type} using {', '.join(CODEBASE_ANALYSIS.languages)}\n"
            if CODEBASE_ANALYSIS.database_patterns:
                header += f"**Current Database Tech:** {', '.join([p.technology for p in CODEBASE_ANALYSIS.database_patterns])}\n"

        header += f"\nFound {len(results)} relevant documentation sections:\n\n"

        return header + "".join(results)

    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
        return f"Error searching documentation: {str(e)}"


@mcp.tool()
def change_working_directory(new_path: str) -> str:
    """
    Change the current working directory for codebase analysis.

    Args:
        new_path: New directory path to analyze
    """
    global CURRENT_WORKING_DIR, CODEBASE_ANALYSIS

    try:
        # Resolve the path
        new_path = os.path.abspath(os.path.expanduser(new_path))

        if not os.path.exists(new_path):
            return f"Error: Directory '{new_path}' does not exist."

        if not os.path.isdir(new_path):
            return f"Error: '{new_path}' is not a directory."

        # Update the working directory
        old_path = CURRENT_WORKING_DIR
        CURRENT_WORKING_DIR = new_path

        # Clear previous analysis since we're analyzing a new directory
        CODEBASE_ANALYSIS = {}

        # Get basic info about the new directory
        project_info = get_project_info(new_path)

        result = f"# Working Directory Changed\n\n"
        result += f"**Previous Directory:** {old_path}\n"
        result += f"**New Directory:** {new_path}\n\n"
        result += f"**New Project Info:**\n"
        result += f"- **Name:** {project_info['name']}\n"
        result += f"- **Files:** {project_info['files_count']}\n"
        result += f"- **Size:** {project_info['size_mb']} MB\n"
        result += f"- **Git Repository:** {'Yes' if project_info['is_git_repo'] else 'No'}\n\n"
        result += f"**Note:** Previous codebase analysis cleared. Run `analyze_current_codebase()` to analyze the new directory.\n"

        return result

    except Exception as e:
        return f"Error changing working directory: {str(e)}"


# Run the server
if __name__ == "__main__":
    logger.info("MongoDB Adoption Assistant MCP Server starting...")
    logger.info(f"Working directory: {CURRENT_WORKING_DIR}")
    logger.info("Available tools:")
    logger.info("  - auto_analyze_codebase_for_mongodb: One-click analysis and recommendations")
    logger.info("  - get_current_project_info: Get info about current project")
    logger.info("  - analyze_current_codebase: Analyze current directory for database patterns")
    logger.info("  - get_mongodb_adoption_plan: Get MongoDB adoption recommendations")
    logger.info("  - clone_repository: Clone MongoDB documentation")
    logger.info("  - process_docs: Process MongoDB documentation")
    logger.info("  - search_docs_for_current_project: Search docs with project context")
    logger.info("  - change_working_directory: Change analysis directory")
    
    mcp.run()
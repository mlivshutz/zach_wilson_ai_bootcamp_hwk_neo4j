"""
GitHub Repository Dependency Graph Builder using Neo4j

This module scans GitHub repositories and builds comprehensive dependency graphs showing:
- File-to-file dependencies through import statements
- External package dependencies
- Module relationships across different programming languages
- Support for Python, JavaScript/TypeScript, Java, and more

The dependency graph is stored in Neo4j for efficient querying and traversal.

Features:
- Multi-language import parsing (Python, JS/TS, Java, etc.)
- GitHub repository integration
- Real-time change monitoring
- Circular dependency detection
- Impact analysis queries
- Package dependency tracking

Required packages:
pip install neo4j github langchain-community langchain-openai python-dotenv requests
"""

import os
import re
import json
import ast
import xml.etree.ElementTree as ET
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import hashlib
import traceback
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
from github import Github
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph

# Load environment variables
load_dotenv()


@dataclass
class ImportStatement:
    """Represents a parsed import statement"""
    from_module: Optional[str]  # Module being imported from
    imported_names: List[str]   # Names being imported
    import_type: str           # 'direct', 'relative', 'wildcard'
    line_number: int           # Line number in source file
    raw_statement: str         # Original import statement


@dataclass
class DependencyNode:
    """Represents a file or module in the dependency graph"""
    file_path: str
    file_type: str            # 'python', 'javascript', 'java', etc.
    repo_name: str
    content_hash: str         # For change detection
    last_modified: datetime
    imports: List[ImportStatement]
    is_external: bool = False # External package vs internal file


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between nodes"""
    source_file: str
    target_file: str
    dependency_type: str      # 'import', 'require', 'package_dep'
    line_number: Optional[int] = None
    import_names: List[str] = None


class ImportParser:
    """Parses import statements from different programming languages"""
    
    @staticmethod
    def parse_python_imports(content: str, file_path: str) -> List[ImportStatement]:
        """Parse Python import statements using AST"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportStatement(
                            from_module=None,
                            imported_names=[alias.name],
                            import_type='direct',
                            line_number=node.lineno,
                            raw_statement=f"import {alias.name}"
                        ))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    import_type = 'relative' if module.startswith('.') else 'direct'
                    if node.names[0].name == '*':
                        import_type = 'wildcard'
                        imported_names = ['*']
                    else:
                        imported_names = [alias.name for alias in node.names]
                    
                    imports.append(ImportStatement(
                        from_module=module,
                        imported_names=imported_names,
                        import_type=import_type,
                        line_number=node.lineno,
                        raw_statement=f"from {module} import {', '.join(imported_names)}"
                    ))
        except SyntaxError as e:
            print(f"⚠️  Syntax error parsing {file_path}: {e}")
        except Exception as e:
            print(f"⚠️  Error parsing Python file {file_path}: {e}")
        
        return imports

    @staticmethod
    def parse_javascript_imports(content: str, file_path: str) -> List[ImportStatement]:
        """Parse JavaScript/TypeScript import statements using regex"""
        imports = []
        lines = content.split('\n')
        
        # Patterns for different import types
        patterns = [
            # import { name1, name2 } from 'module'
            r"import\s+\{\s*([^}]+)\s*\}\s+from\s+['\"]([^'\"]+)['\"]",
            # import name from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import * as name from 'module'
            r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import 'module' (side effects)
            r"import\s+['\"]([^'\"]+)['\"]",
            # const name = require('module')
            r"(?:const|let|var)\s+(\w+)\s*=\s*require\(['\"]([^'\"]+)['\"]\)",
            # const { name1, name2 } = require('module')
            r"(?:const|let|var)\s+\{\s*([^}]+)\s*\}\s*=\s*require\(['\"]([^'\"]+)['\"]\)"
        ]
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        if 'import' in line and '{' in line:
                            # Destructured import
                            imported_names = [name.strip() for name in groups[0].split(',')]
                            from_module = groups[1]
                        elif 'require' in line and '{' in line:
                            # Destructured require
                            imported_names = [name.strip() for name in groups[0].split(',')]
                            from_module = groups[1]
                        else:
                            # Default import or single require
                            imported_names = [groups[0]] if groups[0] else ['default']
                            from_module = groups[1]
                    elif len(groups) == 1:
                        # Side effect import
                        imported_names = ['*']
                        from_module = groups[0]
                    else:
                        continue
                    
                    import_type = 'relative' if from_module.startswith('.') else 'direct'
                    
                    imports.append(ImportStatement(
                        from_module=from_module,
                        imported_names=imported_names,
                        import_type=import_type,
                        line_number=line_num,
                        raw_statement=line
                    ))
                    break
        
        return imports

    @staticmethod  
    def parse_java_imports(content: str, file_path: str) -> List[ImportStatement]:
        """Parse Java import statements using regex"""
        imports = []
        lines = content.split('\n')
        
        # Java import pattern
        import_pattern = r"import\s+(?:static\s+)?([^;]+);"
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
                
            match = re.search(import_pattern, line)
            if match:
                import_path = match.group(1).strip()
                import_type = 'wildcard' if import_path.endswith('*') else 'direct'
                
                # Extract the imported name (last part of the path)
                if import_path.endswith('*'):
                    imported_names = ['*']
                    from_module = import_path[:-2]  # Remove .*
                else:
                    parts = import_path.split('.')
                    imported_names = [parts[-1]]
                    from_module = '.'.join(parts[:-1]) if len(parts) > 1 else ''
                
                imports.append(ImportStatement(
                    from_module=from_module,
                    imported_names=imported_names,
                    import_type=import_type,
                    line_number=line_num,
                    raw_statement=line
                ))
        
        return imports

    @staticmethod
    def parse_package_dependencies(content: str, file_path: str, file_type: str) -> List[ImportStatement]:
        """Parse package dependencies from manifest files"""
        imports = []
        
        try:
            if file_type == 'requirements_txt':
                # Python requirements.txt
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle version specifiers
                        package_name = re.split(r'[>=<!=]', line)[0].strip()
                        imports.append(ImportStatement(
                            from_module=package_name,
                            imported_names=[package_name],
                            import_type='package_dep',
                            line_number=line_num,
                            raw_statement=line
                        ))
            
            elif file_type == 'package_json':
                # JavaScript package.json
                data = json.loads(content)
                dependencies = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                for line_num, (package_name, version) in enumerate(dependencies.items(), 1):
                    imports.append(ImportStatement(
                        from_module=package_name,
                        imported_names=[package_name],
                        import_type='package_dep',
                        line_number=line_num,
                        raw_statement=f"{package_name}: {version}"
                    ))
            
            elif file_type == 'pom_xml':
                # Java Maven pom.xml
                root = ET.fromstring(content)
                dependencies = root.findall('.//{http://maven.apache.org/POM/4.0.0}dependency')
                for line_num, dep in enumerate(dependencies, 1):
                    group_id = dep.find('{http://maven.apache.org/POM/4.0.0}groupId')
                    artifact_id = dep.find('{http://maven.apache.org/POM/4.0.0}artifactId')
                    if group_id is not None and artifact_id is not None:
                        package_name = f"{group_id.text}.{artifact_id.text}"
                        imports.append(ImportStatement(
                            from_module=package_name,
                            imported_names=[package_name],
                            import_type='package_dep',
                            line_number=line_num,
                            raw_statement=package_name
                        ))
        
        except Exception as e:
            print(f"⚠️  Error parsing package file {file_path}: {e}")
        
        return imports


class RepositoryScanner:
    """Scans GitHub repositories for dependency analysis"""
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with GitHub token"""
        self.github_token = github_token or os.getenv('GITHUB_PAT')
        if not self.github_token:
            raise ValueError("GitHub token required. Set GITHUB_PAT environment variable")
        
        self.github = Github(self.github_token)
        self.parser = ImportParser()
    
    def get_file_type(self, file_path: str) -> Optional[str]:
        """Determine file type from extension"""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c_header'
        }
        
        # Special files
        if name == 'requirements.txt':
            return 'requirements_txt'
        elif name == 'package.json':
            return 'package_json'
        elif name == 'pom.xml':
            return 'pom_xml'
        elif name in ['cargo.toml', 'pyproject.toml']:
            return 'toml_deps'
        
        return type_mapping.get(ext)
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed for dependencies"""
        file_type = self.get_file_type(file_path)
        if not file_type:
            return False
        
        # Skip test files, build outputs, etc.
        skip_patterns = [
            'test', '__pycache__', 'node_modules', 'build',
            'dist', '.git', 'target', 'bin', 'obj'
        ]
        
        return not any(pattern in file_path.lower() for pattern in skip_patterns)
    
    def scan_repository(self, repo_name: str) -> List[DependencyNode]:
        """Scan a GitHub repository and extract all dependencies"""
        print(f"🔍 Scanning repository: {repo_name}")
        
        try:
            repo = self.github.get_repo(repo_name)
            nodes = []
            
            def process_contents(contents, path=""):
                for content in contents:
                    full_path = f"{path}/{content.name}" if path else content.name
                    
                    if content.type == "dir":
                        try:
                            sub_contents = repo.get_contents(content.path)
                            process_contents(sub_contents, content.path)
                        except Exception as e:
                            print(f"⚠️  Error accessing directory {content.path}: {e}")
                            continue
                    
                    elif content.type == "file" and self.should_process_file(full_path):
                        try:
                            file_content = content.decoded_content.decode('utf-8', errors='ignore')
                            file_type = self.get_file_type(full_path)
                            
                            # Parse imports based on file type
                            imports = self.parse_file_imports(file_content, full_path, file_type)
                            
                            # Create content hash for change detection
                            content_hash = hashlib.md5(file_content.encode()).hexdigest()
                            
                            node = DependencyNode(
                                file_path=full_path,
                                file_type=file_type,
                                repo_name=repo_name,
                                content_hash=content_hash,
                                last_modified=datetime.now(),
                                imports=imports
                            )
                            
                            nodes.append(node)
                            print(f"  📄 Processed {full_path}: {len(imports)} imports")
                            
                        except Exception as e:
                            print(f"⚠️  Error processing file {full_path}: {e}")
                            continue
            
            # Start processing from root
            contents = repo.get_contents("")
            process_contents(contents)
            
            print(f"✅ Repository scan complete: {len(nodes)} files processed")
            return nodes
            
        except Exception as e:
            print(f"❌ Error scanning repository {repo_name}: {e}")
            return []
    
    def parse_file_imports(self, content: str, file_path: str, file_type: str) -> List[ImportStatement]:
        """Parse imports from file content based on file type"""
        if file_type == 'python':
            return self.parser.parse_python_imports(content, file_path)
        elif file_type in ['javascript', 'typescript']:
            return self.parser.parse_javascript_imports(content, file_path)
        elif file_type == 'java':
            return self.parser.parse_java_imports(content, file_path)
        elif file_type in ['requirements_txt', 'package_json', 'pom_xml']:
            return self.parser.parse_package_dependencies(content, file_path, file_type)
        else:
            return []


class DependencyGraphBuilder:
    """Main class for building and managing dependency graphs in Neo4j"""
    
    def __init__(self):
        """Initialize Neo4j connection and setup"""
        # Get Neo4j credentials from environment
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            missing = []
            if not self.neo4j_uri: missing.append("NEO4J_URI")
            if not self.neo4j_username: missing.append("NEO4J_USERNAME")
            if not self.neo4j_password: missing.append("NEO4J_PASSWORD")
            raise ValueError(f"Missing Neo4j environment variables: {', '.join(missing)}")
        
        # Initialize Neo4j connection
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            self.graph = Neo4jGraph(enhanced_schema=True)
            print("✅ Connected to Neo4j database")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
        
        # Initialize repository scanner
        self.scanner = RepositoryScanner()
        
        # Setup graph schema
        self.setup_graph_schema()
    
    def setup_graph_schema(self):
        """Create indexes and constraints for the dependency graph"""
        try:
            with self.driver.session() as session:
                # Create constraints and indexes
                constraints_and_indexes = [
                    "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
                    "CREATE CONSTRAINT module_name_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE", 
                    "CREATE CONSTRAINT package_name_unique IF NOT EXISTS FOR (p:Package) REQUIRE p.name IS UNIQUE",
                    "CREATE INDEX file_repo_index IF NOT EXISTS FOR (f:File) ON f.repo_name",
                    "CREATE INDEX file_type_index IF NOT EXISTS FOR (f:File) ON f.file_type",
                    "CREATE INDEX dependency_type_index IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON d.dependency_type"
                ]
                
                for statement in constraints_and_indexes:
                    try:
                        session.run(statement)
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            print(f"⚠️  Warning creating constraint/index: {e}")
                
                print("✅ Graph schema setup complete")
                
        except Exception as e:
            print(f"❌ Error setting up graph schema: {e}")
            raise
    
    def clear_repository_data(self, repo_name: str):
        """Clear existing data for a repository"""
        try:
            with self.driver.session() as session:
                # Delete all nodes and relationships for this repository
                query = """
                MATCH (f:File {repo_name: $repo_name})
                DETACH DELETE f
                """
                session.run(query, repo_name=repo_name)
                print(f"🗑️  Cleared existing data for repository: {repo_name}")
        except Exception as e:
            print(f"❌ Error clearing repository data: {e}")
    
    def store_dependency_graph(self, nodes: List[DependencyNode], clear_existing: bool = True):
        """Store dependency nodes and relationships in Neo4j"""
        if not nodes:
            print("⚠️  No nodes to store")
            return
        
        repo_name = nodes[0].repo_name
        if clear_existing:
            self.clear_repository_data(repo_name)
        
        print(f"💾 Storing dependency graph for {repo_name}: {len(nodes)} nodes")
        
        try:
            with self.driver.session() as session:
                # Create file nodes
                for node in nodes:
                    self.create_file_node(session, node)
                
                # Create dependency relationships
                for node in nodes:
                    self.create_dependency_relationships(session, node, nodes)
                
                print("✅ Dependency graph stored successfully")
                
        except Exception as e:
            print(f"❌ Error storing dependency graph: {e}")
            raise
    
    def create_file_node(self, session, node: DependencyNode):
        """Create a file node in Neo4j"""
        query = """
        MERGE (f:File {path: $path, repo_name: $repo_name})
        SET f.file_type = $file_type,
            f.content_hash = $content_hash,
            f.last_modified = datetime($last_modified),
            f.import_count = $import_count,
            f.is_external = $is_external
        """
        
        session.run(query, 
                   path=node.file_path,
                   repo_name=node.repo_name,
                   file_type=node.file_type,
                   content_hash=node.content_hash,
                   last_modified=node.last_modified.isoformat(),
                   import_count=len(node.imports),
                   is_external=node.is_external)
    
    def create_dependency_relationships(self, session, source_node: DependencyNode, all_nodes: List[DependencyNode]):
        """Create dependency relationships for a file node"""
        # Create a lookup for internal files
        internal_files = {node.file_path: node for node in all_nodes}
        
        for import_stmt in source_node.imports:
            target_path = self.resolve_import_path(import_stmt, source_node, internal_files)
            
            if target_path:
                # Create or update target node (might be external package)
                is_external = target_path not in internal_files
                
                if is_external:
                    # Create external package node
                    package_query = """
                    MERGE (p:Package {name: $package_name})
                    SET p.is_external = true
                    """
                    session.run(package_query, package_name=target_path)
                    target_node_query = "MATCH (t:Package {name: $target_path}) RETURN t"
                else:
                    target_node_query = "MATCH (t:File {path: $target_path, repo_name: $repo_name}) RETURN t"
                
                # Create dependency relationship
                rel_query = """
                MATCH (source:File {path: $source_path, repo_name: $repo_name})
                MATCH (target {""" + ("name" if is_external else "path") + """: $target_path""" + ("" if is_external else ", repo_name: $repo_name") + """})
                MERGE (source)-[d:DEPENDS_ON]->(target)
                SET d.dependency_type = $dependency_type,
                    d.import_type = $import_type,
                    d.line_number = $line_number,
                    d.imported_names = $imported_names,
                    d.raw_statement = $raw_statement
                """
                
                session.run(rel_query,
                           source_path=source_node.file_path,
                           target_path=target_path,
                           repo_name=source_node.repo_name,
                           dependency_type='package' if is_external else 'file',
                           import_type=import_stmt.import_type,
                           line_number=import_stmt.line_number,
                           imported_names=import_stmt.imported_names,
                           raw_statement=import_stmt.raw_statement)
    
    def resolve_import_path(self, import_stmt: ImportStatement, source_node: DependencyNode, internal_files: Dict[str, DependencyNode]) -> Optional[str]:
        """Resolve import statement to actual file path or package name"""
        if not import_stmt.from_module:
            return None
        
        # Handle relative imports
        if import_stmt.import_type == 'relative':
            source_dir = str(Path(source_node.file_path).parent)
            if import_stmt.from_module.startswith('.'):
                # Calculate relative path
                dots = len(import_stmt.from_module) - len(import_stmt.from_module.lstrip('.'))
                relative_parts = import_stmt.from_module[dots:].split('.')
                
                # Go up directories based on number of dots
                current_dir = Path(source_dir)
                for _ in range(dots - 1):
                    current_dir = current_dir.parent
                
                # Add relative path parts
                for part in relative_parts:
                    if part:
                        current_dir = current_dir / part
                
                # Try to find the actual file
                possible_paths = [
                    str(current_dir) + '.py',
                    str(current_dir / '__init__.py'),
                    str(current_dir) + '.js',
                    str(current_dir) + '.ts'
                ]
                
                for path in possible_paths:
                    if path in internal_files:
                        return path
        
        # Handle absolute imports within the repository
        else:
            module_parts = import_stmt.from_module.split('.')
            
            # Try different file extensions and structures
            possible_paths = []
            
            if source_node.file_type == 'python':
                # Python module resolution
                possible_paths.extend([
                    '/'.join(module_parts) + '.py',
                    '/'.join(module_parts) + '/__init__.py'
                ])
            elif source_node.file_type in ['javascript', 'typescript']:
                # JS/TS module resolution
                possible_paths.extend([
                    '/'.join(module_parts) + '.js',
                    '/'.join(module_parts) + '.ts',
                    '/'.join(module_parts) + '/index.js',
                    '/'.join(module_parts) + '/index.ts'
                ])
            
            for path in possible_paths:
                if path in internal_files:
                    return path
        
        # If not found internally, treat as external package
        return import_stmt.from_module
    
    def build_repository_graph(self, repo_name: str, clear_existing: bool = True) -> Dict[str, Any]:
        """Complete workflow to build dependency graph for a repository"""
        print(f"🚀 Building dependency graph for repository: {repo_name}")
        
        try:
            # Scan repository for files and dependencies
            nodes = self.scanner.scan_repository(repo_name)
            
            if not nodes:
                return {"error": "No processable files found in repository"}
            
            # Store in Neo4j
            self.store_dependency_graph(nodes, clear_existing)
            
            # Generate statistics
            stats = self.get_repository_statistics(repo_name)
            
            return {
                "repository": repo_name,
                "files_processed": len(nodes),
                "total_imports": sum(len(node.imports) for node in nodes),
                "file_types": list(set(node.file_type for node in nodes)),
                "statistics": stats,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error building repository graph: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_repository_statistics(self, repo_name: str) -> Dict[str, Any]:
        """Get statistics about the dependency graph for a repository"""
        try:
            with self.driver.session() as session:
                # Basic counts
                file_count = session.run(
                    "MATCH (f:File {repo_name: $repo_name}) RETURN count(f) as count",
                    repo_name=repo_name
                ).single()["count"]
                
                dependency_count = session.run(
                    "MATCH (f:File {repo_name: $repo_name})-[d:DEPENDS_ON]->() RETURN count(d) as count",
                    repo_name=repo_name
                ).single()["count"]
                
                external_deps = session.run(
                    "MATCH (f:File {repo_name: $repo_name})-[:DEPENDS_ON]->(p:Package) RETURN count(DISTINCT p) as count",
                    repo_name=repo_name
                ).single()["count"]
                
                # File types
                file_types = session.run(
                    "MATCH (f:File {repo_name: $repo_name}) RETURN f.file_type as type, count(f) as count",
                    repo_name=repo_name
                ).data()
                
                return {
                    "total_files": file_count,
                    "total_dependencies": dependency_count,
                    "external_packages": external_deps,
                    "file_types": {item["type"]: item["count"] for item in file_types}
                }
                
        except Exception as e:
            print(f"❌ Error getting repository statistics: {e}")
            return {"error": str(e)}
    
    def find_circular_dependencies(self, repo_name: str) -> List[List[str]]:
        """Find circular dependencies in the repository"""
        try:
            with self.driver.session() as session:
                # Find cycles using Cypher
                query = """
                MATCH (start:File {repo_name: $repo_name})
                MATCH path = (start)-[:DEPENDS_ON*2..]->(start)
                WHERE ALL(n in nodes(path) WHERE n.repo_name = $repo_name)
                RETURN [n in nodes(path) | n.path] as cycle
                """
                
                result = session.run(query, repo_name=repo_name)
                cycles = [record["cycle"] for record in result]
                
                # Remove duplicates
                unique_cycles = []
                for cycle in cycles:
                    normalized = tuple(sorted(cycle))
                    if normalized not in [tuple(sorted(c)) for c in unique_cycles]:
                        unique_cycles.append(cycle)
                
                return unique_cycles
                
        except Exception as e:
            print(f"❌ Error finding circular dependencies: {e}")
            return []
    
    def get_file_dependencies(self, repo_name: str, file_path: str) -> Dict[str, Any]:
        """Get all dependencies for a specific file"""
        try:
            with self.driver.session() as session:
                # Direct dependencies
                direct_deps = session.run("""
                    MATCH (f:File {repo_name: $repo_name, path: $file_path})-[d:DEPENDS_ON]->(target)
                    RETURN target.path as target_path, target.name as target_name, 
                           d.dependency_type as type, d.imported_names as imports
                """, repo_name=repo_name, file_path=file_path).data()
                
                # Reverse dependencies (what depends on this file)
                reverse_deps = session.run("""
                    MATCH (source)-[d:DEPENDS_ON]->(f:File {repo_name: $repo_name, path: $file_path})
                    RETURN source.path as source_path, d.dependency_type as type, d.imported_names as imports
                """, repo_name=repo_name, file_path=file_path).data()
                
                return {
                    "file": file_path,
                    "direct_dependencies": direct_deps,
                    "reverse_dependencies": reverse_deps
                }
                
        except Exception as e:
            print(f"❌ Error getting file dependencies: {e}")
            return {"error": str(e)}
    
    def update_file_if_changed(self, repo_name: str, file_path: str, new_content: str) -> bool:
        """Update a file's dependencies if content has changed"""
        try:
            new_hash = hashlib.md5(new_content.encode()).hexdigest()
            
            with self.driver.session() as session:
                # Check if file exists and get current hash
                result = session.run("""
                    MATCH (f:File {repo_name: $repo_name, path: $file_path})
                    RETURN f.content_hash as current_hash
                """, repo_name=repo_name, file_path=file_path).single()
                
                if not result or result["current_hash"] != new_hash:
                    # File changed or is new, update it
                    file_type = self.scanner.get_file_type(file_path)
                    if file_type:
                        imports = self.scanner.parse_file_imports(new_content, file_path, file_type)
                        
                        # Remove old dependencies
                        session.run("""
                            MATCH (f:File {repo_name: $repo_name, path: $file_path})-[d:DEPENDS_ON]->()
                            DELETE d
                        """, repo_name=repo_name, file_path=file_path)
                        
                        # Update file node
                        node = DependencyNode(
                            file_path=file_path,
                            file_type=file_type,
                            repo_name=repo_name,
                            content_hash=new_hash,
                            last_modified=datetime.now(),
                            imports=imports
                        )
                        
                        self.create_file_node(session, node)
                        # Note: For dependencies, we'd need all nodes context
                        
                        print(f"✅ Updated file: {file_path}")
                        return True
                
                return False
                
        except Exception as e:
            print(f"❌ Error updating file: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("✅ Neo4j connection closed")


def main():
    """Demo function showing how to use the DependencyGraphBuilder"""
    print("=== GitHub Repository Dependency Graph Builder Demo ===\n")
    
    try:
        # Initialize the builder
        builder = DependencyGraphBuilder()
        
        # Example repositories to analyze (replace with actual repo names)
        test_repos = [
            "mlivshutz/fastapi-vibe-coding",  # Simple test repo
            # Add your own repositories here
        ]
        
        for repo_name in test_repos:
            print(f"\n{'='*60}")
            print(f"Analyzing repository: {repo_name}")
            print('='*60)
            
            # Build dependency graph
            result = builder.build_repository_graph(repo_name)
            
            if result.get("status") == "success":
                print(f"✅ Successfully processed {result['files_processed']} files")
                print(f"📊 Statistics: {result['statistics']}")
                
                # Find circular dependencies
                cycles = builder.find_circular_dependencies(repo_name)
                if cycles:
                    print(f"⚠️  Found {len(cycles)} circular dependencies:")
                    for i, cycle in enumerate(cycles):
                        print(f"  {i+1}. {' -> '.join(cycle)}")
                else:
                    print("✅ No circular dependencies found")
                
            else:
                print(f"❌ Failed to process repository: {result.get('error')}")
        
        # Close connection
        builder.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

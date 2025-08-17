# GitHub Repository Dependency Graph Builder

A comprehensive system for analyzing GitHub repositories and building detailed dependency graphs using Neo4j. This tool scans repositories, parses import statements across multiple programming languages, and creates a graph database that enables powerful dependency analysis and querying.

## 🌟 Features

### Core Capabilities
- **Multi-Language Support**: Parse imports from Python, JavaScript/TypeScript, Java, and more
- **GitHub Integration**: Scan entire repositories automatically
- **Graph Database Storage**: Store dependency relationships in Neo4j for efficient querying
- **Dependency Analysis**: Detect circular dependencies, impact analysis, and connectivity metrics
- **Change Monitoring**: Track repository changes and update graphs accordingly
- **REST API**: FastAPI web interface for easy integration
- **Real-time Queries**: Advanced Cypher queries for dependency exploration

### Supported File Types
- **Python** (`.py`): `import`, `from...import` statements
- **JavaScript/TypeScript** (`.js`, `.ts`, `.jsx`, `.tsx`): `import`, `require()` statements  
- **Java** (`.java`): `import` statements
- **Package Files**: `requirements.txt`, `package.json`, `pom.xml`
- **Configuration**: `pyproject.toml`, `Cargo.toml` (planned)

### Dependency Types Tracked
- **Direct imports**: `from module import function`
- **Relative imports**: `from .local_module import something`
- **External packages**: Dependencies from manifest files
- **Wildcard imports**: `from module import *`
- **Module aliases**: `import module as alias`

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (local or cloud)
- GitHub Personal Access Token

### Installation

1. **Clone or download the files:**
   ```bash
   # Files needed:
   # - dependency_graph_builder.py
   # - dependency_graph_api.py  
   # - example_dependency_analysis.py
   # - requirements_dependency_graph.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_dependency_graph.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create a .env file or set environment variables:
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USERNAME="neo4j"
   export NEO4J_PASSWORD="your_password"
   export GITHUB_PAT="your_github_token"
   ```

4. **Start Neo4j:**
   ```bash
   # Make sure Neo4j is running and accessible at the URI above
   ```

### Basic Usage

#### 1. Command Line Analysis
```python
from dependency_graph_builder import DependencyGraphBuilder

# Initialize the builder
builder = DependencyGraphBuilder()

# Analyze a repository
result = builder.build_repository_graph("octocat/Hello-World")

# Get statistics
stats = builder.get_repository_statistics("octocat/Hello-World")
print(f"Files processed: {stats['total_files']}")

# Find circular dependencies
cycles = builder.find_circular_dependencies("octocat/Hello-World")
print(f"Circular dependencies found: {len(cycles)}")

# Query file dependencies
deps = builder.get_file_dependencies("octocat/Hello-World", "main.py")
print(f"Direct dependencies: {len(deps['direct_dependencies'])}")
```

#### 2. Web API
```bash
# Start the FastAPI server
python dependency_graph_api.py

# Or with uvicorn
uvicorn dependency_graph_api:app --host 0.0.0.0 --port 8000
```

Then use the REST API:
```bash
# Analyze a repository
curl -X POST "http://localhost:8000/analyze-repository" \
     -H "Content-Type: application/json" \
     -d '{"repo_name": "octocat/Hello-World"}'

# Get repository statistics
curl "http://localhost:8000/repository/octocat/Hello-World/stats"

# Find circular dependencies
curl "http://localhost:8000/repository/octocat/Hello-World/cycles"

# Get file dependencies
curl "http://localhost:8000/repository/octocat/Hello-World/dependencies?file_path=main.py"
```

#### 3. Example Script
```bash
# Run the comprehensive example
python example_dependency_analysis.py
```

## 📊 Neo4j Graph Schema

### Node Types
- **`:File`**: Represents source code files
  - `path`: File path within repository
  - `repo_name`: Repository name
  - `file_type`: Programming language
  - `content_hash`: For change detection
  - `last_modified`: Timestamp
  - `import_count`: Number of imports

- **`:Package`**: Represents external packages
  - `name`: Package name
  - `is_external`: Always true for packages

### Relationship Types
- **`:DEPENDS_ON`**: Dependency relationship
  - `dependency_type`: 'file' or 'package'
  - `import_type`: 'direct', 'relative', 'wildcard'
  - `line_number`: Line number of import
  - `imported_names`: List of imported symbols
  - `raw_statement`: Original import statement

### Example Queries

```cypher
// Find files with most dependencies
MATCH (f:File)-[d:DEPENDS_ON]->()
RETURN f.path, f.repo_name, count(d) as dep_count
ORDER BY dep_count DESC LIMIT 10

// Find circular dependencies
MATCH (start:File)-[:DEPENDS_ON*2..]->(start)
RETURN nodes(path) as cycle

// Most used external packages
MATCH ()-[d:DEPENDS_ON]->(p:Package)
RETURN p.name, count(d) as usage_count
ORDER BY usage_count DESC LIMIT 10

// Impact analysis - what would break if file X changes?
MATCH (f:File {path: "main.py"})<-[:DEPENDS_ON*]-(affected)
RETURN DISTINCT affected.path as affected_files
```

## 🔧 API Reference

### Core Classes

#### `DependencyGraphBuilder`
Main orchestrator for dependency graph operations.

**Key Methods:**
- `build_repository_graph(repo_name, clear_existing=True)`: Analyze complete repository
- `get_repository_statistics(repo_name)`: Get analysis statistics
- `find_circular_dependencies(repo_name)`: Detect dependency cycles
- `get_file_dependencies(repo_name, file_path)`: Get file-specific dependencies
- `update_file_if_changed(repo_name, file_path, content)`: Incremental updates

#### `RepositoryScanner`
Handles GitHub repository scanning and file processing.

**Key Methods:**
- `scan_repository(repo_name)`: Scan entire repository
- `get_file_type(file_path)`: Determine file type from extension
- `should_process_file(file_path)`: Filter files for processing

#### `ImportParser`  
Parses import statements from different programming languages.

**Key Methods:**
- `parse_python_imports(content, file_path)`: Parse Python imports using AST
- `parse_javascript_imports(content, file_path)`: Parse JS/TS imports with regex
- `parse_java_imports(content, file_path)`: Parse Java imports
- `parse_package_dependencies(content, file_path, file_type)`: Parse manifest files

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze-repository` | Analyze a GitHub repository |
| `GET` | `/repository/{owner}/{repo}/stats` | Get repository statistics |
| `GET` | `/repository/{owner}/{repo}/dependencies` | Get file dependencies |
| `GET` | `/repository/{owner}/{repo}/cycles` | Find circular dependencies |
| `POST` | `/repository/{owner}/{repo}/update` | Update repository analysis |
| `GET` | `/repositories` | List analyzed repositories |
| `GET` | `/search/dependencies` | Search dependencies |
| `GET` | `/health` | API health check |

## 🔍 Advanced Use Cases

### 1. Impact Analysis
Before making changes to a file, see what else might be affected:
```python
# Find all files that depend on a specific file
deps = builder.get_file_dependencies("myorg/myrepo", "core/database.py")
affected_files = [dep["source_path"] for dep in deps["reverse_dependencies"]]
```

### 2. Circular Dependency Detection
Identify problematic dependency cycles:
```python
cycles = builder.find_circular_dependencies("myorg/myrepo")
for cycle in cycles:
    print(f"Circular dependency: {' -> '.join(cycle)}")
```

### 3. Package Usage Analysis
Find your most critical external dependencies:
```cypher
MATCH ()-[d:DEPENDS_ON]->(p:Package)
WHERE p.is_external = true
RETURN p.name, count(d) as usage_count
ORDER BY usage_count DESC
```

### 4. Code Architecture Analysis
Identify architectural patterns and violations:
```cypher
// Find files that import from too many different modules
MATCH (f:File)-[:DEPENDS_ON]->(target)
WITH f, count(DISTINCT target) as import_diversity
WHERE import_diversity > 10
RETURN f.path, import_diversity
ORDER BY import_diversity DESC
```

## 🛠️ Development and Extension

### Adding New Language Support
1. Extend `ImportParser` with a new `parse_<language>_imports()` method
2. Update `get_file_type()` in `RepositoryScanner` 
3. Add file extension mappings
4. Test with sample files

### Custom Queries
The Neo4j graph can be queried directly for custom analysis:
```python
with builder.driver.session() as session:
    result = session.run("""
        MATCH (f:File {repo_name: $repo})
        WHERE f.file_type = 'python'
        RETURN count(f) as python_files
    """, repo=repo_name)
```

### Monitoring and Automation
- Set up webhooks to trigger analysis on repository changes
- Schedule periodic scans for dependency updates
- Integrate with CI/CD pipelines for impact analysis

## 🔧 Configuration

### Environment Variables
```bash
# Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GITHUB_PAT=your_github_token

# Optional  
MILVUS_COLLECTION_NAME=github_dense_index  # For future integration
```

### File Processing Configuration
The system automatically filters out common build/cache directories:
- `__pycache__`, `node_modules`, `build`, `dist`
- `.git`, `target`, `bin`, `obj`
- Test directories (configurable)

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check URI, username, and password
   - Ensure Neo4j is running and accessible
   - Verify network connectivity

2. **GitHub API Rate Limiting**
   - Use authenticated requests with GITHUB_PAT
   - Implement request throttling for large repositories
   - Consider GitHub Enterprise for higher limits

3. **Memory Issues with Large Repositories**
   - Process files in batches
   - Use streaming for large file contents
   - Increase available memory

4. **Import Parsing Errors**
   - Check file encoding (should be UTF-8)
   - Handle syntax errors gracefully
   - Add logging for debugging

### Performance Optimization
- Use Neo4j indexes for large datasets
- Batch Neo4j operations
- Cache GitHub API responses
- Process files in parallel

## 🤝 Contributing

To extend or improve the system:
1. Add test cases for new language parsers
2. Implement incremental update mechanisms  
3. Add visualization capabilities
4. Extend package manager support

## 📄 License

This project is provided as an educational example. Adapt the license as needed for your use case.

## 🔒 AI-Powered Vulnerability Analysis

### New Feature: Comprehensive Security Analysis
The system now includes advanced vulnerability detection and analysis capabilities:

#### **Vulnerability Detection Sources:**
- **CVE Database**: National Vulnerability Database integration
- **GitHub Security Advisories**: Real-time security advisories
- **Snyk Database**: Commercial vulnerability intelligence
- **Custom Sources**: Extensible for additional databases

#### **AI-Powered Analysis Features:**
- **Smart Graph Traversal**: AI agents intelligently navigate dependency graphs
- **Context-Aware Impact Assessment**: Real-world risk vs theoretical vulnerabilities
- **Dynamic Remediation Strategies**: Multiple options ranked by safety and effectiveness
- **Compatibility Analysis**: Version compatibility and breaking change assessment

#### **Usage Example:**
```python
from vulnerability_analysis_agent import VulnerabilityAnalysisAgent
from dependency_graph_builder import DependencyGraphBuilder

# Initialize
builder = DependencyGraphBuilder()
vuln_agent = VulnerabilityAnalysisAgent(builder)

# Configure rate limiting (optional)
vuln_agent.vuln_db.configure_rate_limits('snyk', requests_per_minute=30, min_delay=2.0)

# Check rate limit status
status = vuln_agent.vuln_db.get_rate_limit_status('snyk')
print(f"Snyk: {status['requests_last_minute']}/{status['limit_per_minute']} requests this minute")

# Run comprehensive analysis
reports = await vuln_agent.analyze_all_repositories()

# Generate reports
markdown_report = vuln_agent.generate_comprehensive_report("markdown")
json_report = vuln_agent.generate_comprehensive_report("json")
```

#### **Report Features:**
- **Severity Assessment**: AI-powered risk scoring with business context
- **Remediation Roadmaps**: Step-by-step fix instructions with rollback procedures
- **Impact Analysis**: Dependency chain analysis and blast radius assessment
- **Test Recommendations**: Verification strategies for safe remediation
- **Alternative Solutions**: Package replacement suggestions when updates aren't viable
- **Intelligent Rate Limiting**: Built-in rate limiting with exponential backoff for all vulnerability APIs

#### **Environment Setup for Vulnerability Analysis:**
```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional (for enhanced analysis)
export GITHUB_PAT="your-github-token"
export SNYK_TOKEN="your-snyk-token"
```

#### **Running Vulnerability Analysis:**
```bash
# Install additional dependencies
pip install aiohttp semver packaging

# Run comprehensive analysis
python example_vulnerability_analysis.py

# Or use the main vulnerability script
python vulnerability_analysis_agent.py
```

#### **Rate Limiting Features:**
The vulnerability analysis system includes comprehensive rate limiting to handle API restrictions:

**Built-in Rate Limits:**
- **Snyk API**: 50 requests/minute, 1000 requests/hour (configurable)
- **GitHub API**: 60 requests/minute, 5000 requests/hour  
- **CVE Database**: 30 requests/minute, 1000 requests/hour (conservative)

**Smart Rate Limiting Features:**
- **Automatic Throttling**: Respects API rate limits with configurable delays
- **Exponential Backoff**: Intelligent retry logic for 429 (rate limited) responses
- **Request History Tracking**: Monitors API usage across time windows
- **Jitter Support**: Adds randomization to prevent thundering herd problems

**Configuring Rate Limits:**
```python
# Adjust Snyk rate limits for your API tier
vuln_agent.vuln_db.configure_rate_limits('snyk', 
                                        requests_per_minute=100,  # Premium tier
                                        min_delay=0.6)

# Check current rate limit status
status = vuln_agent.vuln_db.get_rate_limit_status('snyk')
print(f"Usage: {status['requests_last_minute']}/{status['limit_per_minute']} requests/min")
```

**Handling Rate Limit Errors:**
- Automatic detection of 429 (Too Many Requests) responses
- Progressive backoff with retry attempts (default: 3 retries)
  - 1st retry: ~5.5 seconds
  - 2nd retry: ~11 seconds  
  - 3rd retry: ~22 seconds
- Graceful degradation when APIs are unavailable
- Detailed logging of rate limiting events and wait times

## 🧪 System Testing and Validation

### Comprehensive Test Suite
The vulnerability analysis system includes a complete test suite to validate functionality:

**Test Coverage:**
- **Rate Limiting**: Validates API rate limit enforcement and 429 error handling
- **Vulnerability Detection**: Tests multi-source vulnerability database integration
- **Data Structure Validation**: Ensures all data structures are properly formed
- **AI Analysis**: Validates AI-powered impact analysis and remediation generation
- **End-to-End Integration**: Tests complete workflow from dependency graph to reports
- **Error Handling**: Validates graceful handling of edge cases and failures
- **Performance**: Tests rate limiting effectiveness and response times

**Running Tests:**
```bash
# Run essential tests (recommended)
python run_system_tests.py

# Quick validation of core functionality
python run_system_tests.py --quick

# Comprehensive tests with detailed output
python run_system_tests.py --verbose

# Using pytest (requires pytest installation)
pytest test_vulnerability_system.py -v
```

**Test Guarantees:**
The test suite validates that the system:
- ✅ Returns valid vulnerability analysis results
- ✅ Generates structured remediation recommendations
- ✅ Handles errors gracefully without crashes
- ✅ Respects API rate limits and prevents 429 errors  
- ✅ Produces consistent data structures
- ✅ Integrates properly with dependency graph data

**Example Test Output:**
```
🔍 Running Essential Vulnerability Analysis System Tests
============================================================

1️⃣  Testing Rate Limiting Configuration...
   ✅ Rate limiting configuration works correctly

2️⃣  Testing Vulnerability Data Structures...
   ✅ Vulnerability data structures are valid

3️⃣  Testing Mock Vulnerability Detection...
   ✅ Vulnerability detection works with mock data

📊 TEST RESULTS SUMMARY
============================================================
✅ PASS Rate Limiting
✅ PASS Data Validation  
✅ PASS Mock Detection
✅ PASS AI Analysis
✅ PASS Integration

📈 Results: 5/5 tests passed
🎉 All tests passed! Vulnerability analysis system is working correctly.
```

## 🔗 Related Projects

- [LangChain](https://langchain.readthedocs.io/) - For LLM integration and AI agents
- [Neo4j](https://neo4j.com/) - Graph database platform
- [PyGithub](https://pygithub.readthedocs.io/) - GitHub API client
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [OpenAI](https://openai.com/) - AI models for analysis and recommendations

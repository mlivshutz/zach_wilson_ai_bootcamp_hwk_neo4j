#!/usr/bin/env python3
"""
Example Usage of GitHub Repository Dependency Graph Builder

This script demonstrates how to use the dependency graph builder to:
1. Analyze GitHub repositories 
2. Build dependency graphs in Neo4j
3. Query and analyze dependencies
4. Detect circular dependencies
5. Monitor changes and updates

Make sure to set up your environment variables before running:
- NEO4J_URI: Your Neo4j database URI 
- NEO4J_USERNAME: Your Neo4j username
- NEO4J_PASSWORD: Your Neo4j password  
- GITHUB_PAT: Your GitHub Personal Access Token

Example:
    python example_dependency_analysis.py
"""

import os
import sys
import asyncio
from typing import List, Dict
from dependency_graph_builder import DependencyGraphBuilder

def check_environment_setup():
    """Check if all required environment variables are set"""
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GITHUB_PAT"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables before running the script.")
        print("You can create a .env file with:")
        print("NEO4J_URI=bolt://localhost:7687")
        print("NEO4J_USERNAME=neo4j") 
        print("NEO4J_PASSWORD=your_password")
        print("GITHUB_PAT=your_github_token")
        return False
    
    print("✅ All environment variables are set")
    return True

def analyze_single_repository(builder: DependencyGraphBuilder, repo_name: str):
    """Analyze a single repository and display results"""
    print(f"\n{'='*60}")
    print(f"📊 Analyzing Repository: {repo_name}")
    print('='*60)
    
    # Build dependency graph
    result = builder.build_repository_graph(repo_name)
    
    if result.get("status") == "success":
        print(f"✅ Successfully analyzed {result['files_processed']} files")
        print(f"📁 File types found: {', '.join(result['file_types'])}")
        print(f"🔗 Total imports: {result['total_imports']}")
        
        # Show detailed statistics
        stats = result.get("statistics", {})
        print(f"\n📈 Detailed Statistics:")
        print(f"   • Total files: {stats.get('total_files', 0)}")
        print(f"   • Total dependencies: {stats.get('total_dependencies', 0)}")
        print(f"   • External packages: {stats.get('external_packages', 0)}")
        
        file_types = stats.get('file_types', {})
        if file_types:
            print(f"   • File type breakdown:")
            for file_type, count in file_types.items():
                print(f"     - {file_type}: {count} files")
        
        # Find circular dependencies
        print(f"\n🔄 Checking for circular dependencies...")
        cycles = builder.find_circular_dependencies(repo_name)
        
        if cycles:
            print(f"⚠️  Found {len(cycles)} circular dependencies:")
            for i, cycle in enumerate(cycles[:5]):  # Show first 5
                print(f"   {i+1}. {' → '.join(cycle[:4])}{'...' if len(cycle) > 4 else ''}")
            if len(cycles) > 5:
                print(f"   ... and {len(cycles) - 5} more cycles")
        else:
            print("✅ No circular dependencies found")
        
        # Show some example file dependencies
        print(f"\n📄 Sample file dependencies:")
        sample_files = [node.file_path for node in builder.scanner.scan_repository(repo_name)[:3]]
        for file_path in sample_files:
            deps = builder.get_file_dependencies(repo_name, file_path)
            if not deps.get("error"):
                direct_count = len(deps.get("direct_dependencies", []))
                reverse_count = len(deps.get("reverse_dependencies", []))
                print(f"   • {file_path}: {direct_count} dependencies, {reverse_count} dependents")
    
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

def print_dependency_graph_schema(builder: DependencyGraphBuilder):
    """
    Print the current dependency graph schema to understand the data structure
    This shows exactly what node types and relationships exist in the database
    """
    try:
        print("\n" + "="*60)
        print("📋 DEPENDENCY GRAPH SCHEMA")
        print("="*60)
        
        with builder.driver.session() as session:
            # Get raw schema information
            try:
                schema_query = "CALL db.schema.visualization()"
                schema_result = session.run(schema_query)
                print("Raw Schema Available ✅")
            except Exception:
                print("Raw Schema: Using manual queries")
            
            # Get node types (labels)
            node_types_query = """
            CALL db.labels() YIELD label
            RETURN collect(label) as labels
            """
            result = session.run(node_types_query)
            node_types = result.single()["labels"]
            
            # Get relationship types
            rel_types_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) as types
            """
            result = session.run(rel_types_query)
            rel_types = result.single()["types"]
            
            # Get detailed counts for each node type
            node_counts = {}
            for node_type in node_types:
                count_query = f"MATCH (n:`{node_type}`) RETURN count(n) as count"
                result = session.run(count_query)
                node_counts[node_type] = result.single()["count"]
            
            # Get relationship counts
            rel_counts = {}
            for rel_type in rel_types:
                count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                result = session.run(count_query)
                rel_counts[rel_type] = result.single()["count"]
            
            # Display results
            print(f"📊 Node Types ({len(node_types)}):")
            for node_type in sorted(node_types):
                count = node_counts.get(node_type, 0)
                print(f"   • {node_type}: {count:,} nodes")
            
            print(f"\n🔗 Relationship Types ({len(rel_types)}):")
            for rel_type in sorted(rel_types):
                count = rel_counts.get(rel_type, 0)
                print(f"   • {rel_type}: {count:,} relationships")
            
            # Get total counts
            total_nodes = sum(node_counts.values())
            total_rels = sum(rel_counts.values())
            
            # Get repository count
            repo_query = """
            MATCH (f:File)
            RETURN count(DISTINCT f.repo_name) as repo_count,
                   collect(DISTINCT f.repo_name) as repo_names
            """
            result = session.run(repo_query)
            repo_data = result.single()
            repo_count = repo_data["repo_count"] if repo_data else 0
            repo_names = repo_data["repo_names"] if repo_data else []
            
            print(f"\n📈 Summary Statistics:")
            print(f"   • Total Nodes: {total_nodes:,}")
            print(f"   • Total Relationships: {total_rels:,}")
            print(f"   • Repositories Analyzed: {repo_count}")
            
            if repo_names:
                print(f"\n📚 Repositories in Database:")
                for repo in sorted(repo_names):
                    print(f"   • {repo}")
            
            # Show sample node properties for each type
            print(f"\n🔍 Sample Node Properties:")
            for node_type in sorted(node_types):
                if node_counts.get(node_type, 0) > 0:
                    prop_query = f"""
                    MATCH (n:`{node_type}`)
                    WITH n, keys(n) as props
                    RETURN props[0..5] as sample_props
                    LIMIT 1
                    """
                    result = session.run(prop_query)
                    record = result.single()
                    if record and record["sample_props"]:
                        props = record["sample_props"]
                        print(f"   • {node_type}: {', '.join(props)}")
        
        print("="*60)
        print("NOTE: This schema represents the dependency structure of")
        print("analyzed repositories. Use these node and relationship")
        print("types for Cypher queries and graph analysis.")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error printing dependency graph schema: {e}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Full error message: {str(e)}")
        import traceback
        print("Full stack trace:")
        traceback.print_exc()

def demonstrate_queries(builder: DependencyGraphBuilder, repo_names: List[str]):
    """Demonstrate various dependency graph queries"""
    print(f"\n{'='*60}")
    print("🔍 Demonstrating Advanced Queries")
    print('='*60)
    
    try:
        with builder.driver.session() as session:
            # Query 1: Most connected files
            print("\n1. Most connected files (by total dependencies):")
            result = session.run("""
                MATCH (f:File)
                OPTIONAL MATCH (f)-[d:DEPENDS_ON]->()
                OPTIONAL MATCH ()-[rd:DEPENDS_ON]->(f)
                WITH f, count(d) as outgoing, count(rd) as incoming
                RETURN f.repo_name as repo, f.path as file, 
                       outgoing + incoming as total_connections
                ORDER BY total_connections DESC
                LIMIT 5
            """)
            
            for record in result:
                print(f"   • {record['file']} ({record['repo']}): {record['total_connections']} connections")
            
            # Query 2: External packages by popularity
            print("\n2. Most used external packages:")
            result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->(p:Package)
                RETURN p.name as package, count(d) as usage_count
                ORDER BY usage_count DESC
                LIMIT 5
            """)
            
            for record in result:
                print(f"   • {record['package']}: used {record['usage_count']} times")
            
            # Query 3: Files with no dependencies (leaf nodes)
            print("\n3. Files with no outgoing dependencies:")
            result = session.run("""
                MATCH (f:File)
                WHERE NOT (f)-[:DEPENDS_ON]->()
                RETURN f.repo_name as repo, f.path as file
                LIMIT 5
            """)
            
            files = [f"{record['file']} ({record['repo']})" for record in result]
            if files:
                print(f"   • " + "\n   • ".join(files))
            else:
                print("   • No files found without dependencies")
            
            # Query 4: Dependency depth analysis
            print("\n4. Dependency depth analysis:")
            result = session.run("""
                MATCH (f:File)
                OPTIONAL MATCH path = (f)-[:DEPENDS_ON*1..3]->()
                WITH f, max(length(path)) as max_depth
                WHERE max_depth IS NOT NULL
                RETURN max_depth, count(f) as file_count
                ORDER BY max_depth
            """)
            
            for record in result:
                depth = record['max_depth'] or 0
                count = record['file_count']
                print(f"   • Depth {depth}: {count} files")
    
    except Exception as e:
        print(f"❌ Error running queries: {e}")

def main():
    """Main demonstration function"""
    print("🔗 GitHub Repository Dependency Graph Builder - Example Usage")
    print("=" * 70)
    
    # Check environment setup
    if not check_environment_setup():
        sys.exit(1)
    
    try:
        # Initialize the builder
        print("\n🚀 Initializing Dependency Graph Builder...")
        builder = DependencyGraphBuilder()
        
        # Example repositories to analyze
        # Replace these with repositories you have access to
        example_repos = [
            "mlivshutz/fastapi-vibe-coding",  # Simple test repository
            # Add more repositories here, for example:
            # "microsoft/vscode",  # Large JavaScript/TypeScript project  
            # "django/django",     # Large Python project
            # "spring-projects/spring-boot",  # Java project
        ]
        
        print(f"\n📋 Will analyze {len(example_repos)} repositories:")
        for repo in example_repos:
            print(f"   • {repo}")
        
        # Analyze each repository
        for repo_name in example_repos:
            try:
                analyze_single_repository(builder, repo_name)
            except Exception as e:
                print(f"❌ Failed to analyze {repo_name}: {e}")
                continue
        
        # Print the dependency graph schema
        if example_repos:
            print_dependency_graph_schema(builder)
        
        # Demonstrate advanced queries
        if example_repos:
            demonstrate_queries(builder, example_repos)
        
        # Show overall statistics
        print(f"\n{'='*60}")
        print("📊 Overall Analysis Summary")
        print('='*60)
        
        with builder.driver.session() as session:
            # Total counts across all repositories
            result = session.run("""
                MATCH (f:File)
                RETURN count(DISTINCT f.repo_name) as total_repos,
                       count(f) as total_files,
                       count(DISTINCT f.file_type) as file_types
            """).single()
            
            deps_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN count(d) as total_dependencies
            """).single()
            
            packages_result = session.run("""
                MATCH (p:Package)
                RETURN count(p) as total_packages
            """).single()
            
            print(f"✅ Analysis Complete!")
            print(f"   • Repositories analyzed: {result['total_repos']}")
            print(f"   • Total files processed: {result['total_files']}")
            print(f"   • File types supported: {result['file_types']}")
            print(f"   • Total dependencies: {deps_result['total_dependencies']}")
            print(f"   • External packages identified: {packages_result['total_packages']}")
        
        print(f"\n🎯 Next Steps:")
        print("   • Use the FastAPI web interface (dependency_graph_api.py)")
        print("   • Query the Neo4j database directly with Cypher")
        print("   • Set up monitoring for repository changes")
        print("   • Integrate with CI/CD pipelines for impact analysis")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'builder' in locals():
            builder.close()

if __name__ == "__main__":
    main()

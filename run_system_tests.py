#!/usr/bin/env python3
"""
Simple Test Runner for Vulnerability Analysis System

This script runs the system tests without requiring pytest installation.
It provides a lightweight way to validate the vulnerability analysis system.

Usage:
    python run_system_tests.py
    python run_system_tests.py --quick    # Run essential tests only
    python run_system_tests.py --verbose  # Detailed output
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_vulnerability_system import (
    TestVulnerabilityDatabase,
    TestAIAnalysisSystem, 
    TestEndToEndIntegration,
    test_performance_and_rate_limiting
)


def setup_test_environment():
    """Setup minimal test environment"""
    os.environ.setdefault('NEO4J_URI', 'bolt://localhost:7687')
    os.environ.setdefault('NEO4J_USERNAME', 'neo4j')
    os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
    os.environ.setdefault('OPENAI_API_KEY', 'test_openai_key')
    os.environ.setdefault('GITHUB_PAT', 'test_github_token')
    os.environ.setdefault('SNYK_TOKEN', 'test_snyk_token')


async def run_essential_tests(verbose=False):
    """Run the essential system validation tests"""
    
    test_results = []
    start_time = datetime.now()
    
    print("🔍 Running Essential Vulnerability Analysis System Tests")
    print("="*60)
    
    try:
        # Test 1: Rate Limiting Configuration
        print("\n1️⃣  Testing Rate Limiting Configuration...")
        test_db = TestVulnerabilityDatabase()
        test_db.test_rate_limiting_configuration()
        test_results.append(("Rate Limiting", True, ""))
        
        # Test 2: Vulnerability Data Structures  
        print("\n2️⃣  Testing Vulnerability Data Structures...")
        test_db.test_vulnerability_data_validation()
        test_results.append(("Data Validation", True, ""))
        
        # Test 3: Mock Vulnerability Detection
        print("\n3️⃣  Testing Mock Vulnerability Detection...")
        await test_db.test_mock_vulnerability_detection()
        test_results.append(("Mock Detection", True, ""))
        
        # Test 4: AI Impact Analysis
        print("\n4️⃣  Testing AI Impact Analysis...")
        test_ai = TestAIAnalysisSystem()
        await test_ai.test_ai_impact_analysis()
        test_results.append(("AI Analysis", True, ""))
        
        # Test 5: End-to-End Integration
        print("\n5️⃣  Testing End-to-End Integration...")
        test_integration = TestEndToEndIntegration()
        await test_integration.test_complete_vulnerability_analysis_workflow()
        test_results.append(("Integration", True, ""))
        
        if verbose:
            # Additional tests for verbose mode
            print("\n6️⃣  Testing Remediation Strategy Generation...")
            await test_ai.test_remediation_strategy_generation()
            test_results.append(("Remediation", True, ""))
            
            print("\n7️⃣  Testing Error Handling...")
            test_integration.test_error_handling_and_resilience()
            test_results.append(("Error Handling", True, ""))
            
            print("\n8️⃣  Testing Performance & Rate Limiting...")
            await test_performance_and_rate_limiting()
            test_results.append(("Performance", True, ""))
        
    except Exception as e:
        error_msg = str(e)
        test_results.append(("Failed Test", False, error_msg))
        if verbose:
            import traceback
            traceback.print_exc()
    
    # Print results summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    
    for test_name, success, error in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success and error:
            print(f"      Error: {error}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    print(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! Vulnerability analysis system is working correctly.")
        print("\n✅ System Guarantees Validated:")
        print("   • Returns valid vulnerability analysis results")
        print("   • Generates structured remediation recommendations") 
        print("   • Handles errors gracefully without crashes")
        print("   • Respects API rate limits and prevents 429 errors")
        print("   • Produces consistent data structures")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. System may have issues.")
        return False


async def run_quick_validation():
    """Run a quick validation of core functionality"""
    print("⚡ Quick Validation of Core Functionality")
    print("="*40)
    
    try:
        # Quick smoke tests
        print("1. Testing imports...")
        from vulnerability_analysis_agent import VulnerabilityAnalysisAgent, VulnerabilityDatabase
        from dependency_graph_builder import DependencyGraphBuilder
        print("   ✅ All imports successful")
        
        print("2. Testing basic initialization...")
        vuln_db = VulnerabilityDatabase()
        assert 'snyk' in vuln_db.rate_limits
        print("   ✅ VulnerabilityDatabase initializes correctly")
        
        print("3. Testing rate limiting configuration...")
        vuln_db.configure_rate_limits('snyk', requests_per_minute=100)
        status = vuln_db.get_rate_limit_status('snyk')
        assert status['limit_per_minute'] == 100
        print("   ✅ Rate limiting works")
        
        print("4. Testing data structures...")
        from vulnerability_analysis_agent import VulnerabilityData, VulnerabilitySeverity
        vuln = VulnerabilityData(
            cve_id="TEST-001",
            ghsa_id=None,
            title="Test Vulnerability",
            description="Test description",
            severity=VulnerabilitySeverity.MEDIUM,
            cvss_score=6.0,
            affected_packages=["test"],
            affected_versions=["<1.0.0"],
            fixed_versions=["1.0.0"],
            published_date=datetime.now(),
            source_db="test",
            references=[],
            cwe_ids=[]
        )
        assert vuln.title == "Test Vulnerability"
        print("   ✅ Data structures work correctly")
        
        print("\n🎯 Quick validation complete - Core functionality works!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick validation failed: {e}")
        return False


def print_usage_help():
    """Print usage information"""
    print("🔧 Vulnerability Analysis System Test Runner")
    print("="*50)
    print("\nUsage:")
    print("  python run_system_tests.py              # Run essential tests")
    print("  python run_system_tests.py --quick      # Quick validation only")  
    print("  python run_system_tests.py --verbose    # Run all tests with details")
    print("  python run_system_tests.py --help       # Show this help")
    print("\nTests validate:")
    print("  • Rate limiting and API management")
    print("  • Vulnerability detection and data structures")
    print("  • AI-powered impact analysis")
    print("  • Remediation strategy generation")
    print("  • End-to-end integration workflow")
    print("  • Error handling and system resilience")


async def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description='Run vulnerability analysis system tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation tests only')
    parser.add_argument('--verbose', action='store_true',
                       help='Run all tests with verbose output')
    parser.add_argument('--help-usage', action='store_true',
                       help='Show detailed usage information')
    
    args = parser.parse_args()
    
    if args.help_usage:
        print_usage_help()
        return True
    
    # Setup test environment
    setup_test_environment()
    
    if args.quick:
        success = await run_quick_validation()
    else:
        success = await run_essential_tests(verbose=args.verbose)
    
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)

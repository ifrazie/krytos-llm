#!/usr/bin/env python
"""
Test runner script for Streamlit Ollama Chat application.
Provides convenient commands for running tests and generating coverage reports.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run tests for Streamlit Ollama Chat application")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--ui", action="store_true", help="Run UI tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Skip UI tests by default unless explicitly requested
    if not args.ui and not args.all:
        os.environ["SKIP_UI_TESTS"] = "true"
    
    # Determine which tests to run
    if args.unit:
        test_path = "tests/unit"
    elif args.integration:
        test_path = "tests/integration"
    elif args.ui:
        test_path = "tests/ui"
    else:
        # Default to running all non-UI tests
        test_path = "tests/unit tests/integration"
        if args.all:
            test_path = "tests"
    
    # Base command
    cmd = ["pytest", "-v"]
    
    # Add parallel execution if requested
    if args.parallel and not args.ui:  # UI tests shouldn't run in parallel
        cmd.extend(["-n", "auto"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=."])
        if args.html:
            cmd.append("--cov-report=html")
        else:
            cmd.append("--cov-report=term")
    
    # Add markers to skip slow tests if requested
    if args.skip_slow:
        cmd.append("-m 'not slow'")
    
    # Add test path
    cmd.extend(test_path.split())
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Open HTML report in browser if generated
    if args.coverage and args.html and result.returncode == 0:
        html_report = Path.cwd() / "htmlcov" / "index.html"
        if html_report.exists():
            print(f"Opening coverage report: {html_report}")
            webbrowser.open(f"file://{html_report}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
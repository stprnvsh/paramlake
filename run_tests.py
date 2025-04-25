#!/usr/bin/env python3
"""
Test runner script for ParamLake.
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ParamLake tests')
    
    parser.add_argument(
        '-m', '--module',
        default=None,
        help='Run tests for a specific module (e.g., storage, collectors)'
    )
    
    parser.add_argument(
        '-t', '--test',
        default=None,
        help='Run a specific test (e.g., test_store_tensor)'
    )
    
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '-x', '--xml',
        action='store_true',
        help='Generate XML report'
    )
    
    parser.add_argument(
        '-f', '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    return parser.parse_args()


def run_tests(args):
    """Run the tests with the specified options."""
    cmd = ['pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=paramlake', '--cov-report=term'])
        
        # Also generate HTML report
        cmd.append('--cov-report=html')
    
    # Add XML report
    if args.xml:
        cmd.append('--junitxml=test-results.xml')
    
    # Add fail fast
    if args.failfast:
        cmd.append('-x')
    
    # Add specific module
    if args.module:
        if os.path.exists(f'tests/test_{args.module}.py'):
            cmd.append(f'tests/test_{args.module}.py')
        else:
            print(f"Error: Test module 'tests/test_{args.module}.py' not found.")
            return 1
    
    # Add specific test
    if args.test:
        if args.module:
            cmd[-1] += f'::test_{args.test}'
        else:
            cmd.append(f'tests/::test_{args.test}')
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(run_tests(args)) 
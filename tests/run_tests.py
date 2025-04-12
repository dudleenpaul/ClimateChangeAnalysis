#!/usr/bin/env python3
"""
Run all tests for the Climate Change Impact Analyzer.
"""

import unittest
import os
import sys
import coverage

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    """Run all tests and calculate code coverage."""
    # Start coverage measurement
    cov = coverage.Coverage(
        source=['src'],
        omit=['src/__pycache__/*', 'src/templates/*']
    )
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    suite = loader.discover(tests_dir)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage measurement and save report
    cov.stop()
    cov.save()
    
    # Print coverage report
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML coverage report
    cov_dir = os.path.join(os.path.dirname(tests_dir), 'coverage_html')
    os.makedirs(cov_dir, exist_ok=True)
    cov.html_report(directory=cov_dir)
    print(f"HTML coverage report generated in: {cov_dir}")
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive Agg backend for tests
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Run the tests
    exit_code = run_tests()
    sys.exit(exit_code)
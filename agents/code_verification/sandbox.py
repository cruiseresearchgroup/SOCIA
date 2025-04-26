"""
Sandbox execution environment for code verification.
This module provides a secure isolation mechanism using Docker containers
for safely executing and validating generated simulation code.
"""

import os
import sys
import tempfile
import subprocess
import shutil
import json
import textwrap
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Add handler if none exists
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DockerSandbox:
    """
    A secure sandbox environment using Docker containers for code execution.
    Provides isolation for executing untrusted code with resource limitations.
    """

    def __init__(self, base_image: str = "python:3.9-slim", 
                 timeout: int = 30, 
                 max_memory: str = "512m",
                 network_enabled: bool = False):
        """
        Initialize the Docker sandbox environment.

        Args:
            base_image: Docker image to use as the execution environment
            timeout: Maximum execution time in seconds
            max_memory: Memory limit for the container
            network_enabled: Whether to allow network access
        """
        self.base_image = base_image
        self.timeout = timeout
        self.max_memory = max_memory
        self.network_enabled = network_enabled
        self.container_id = None
        self.temp_dir = None
        
        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not available on this system")
        except FileNotFoundError:
            raise RuntimeError("Docker is not installed or not in PATH")

    def __enter__(self):
        """Set up the Docker container and temporary directory."""
        # Create temporary directory for file exchange
        self.temp_dir = tempfile.mkdtemp(prefix="socia_sandbox_")
        
        # Create container
        network_param = "" if self.network_enabled else "--network=none"
        
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", network_param,
             "--memory", self.max_memory,
             "-v", f"{self.temp_dir}:/workspace",
             self.base_image, 
             "tail", "-f", "/dev/null"],  # Keep container running
            capture_output=True, text=True, check=True
        )
        
        self.container_id = result.stdout.strip()
        logger.debug(f"Started Docker container: {self.container_id}")
        
        # Install basic dependencies in container
        self._run_in_container("pip install --no-cache-dir numpy matplotlib pytest")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the Docker container and temporary directory."""
        # Stop and remove container
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True, check=False
                )
                logger.debug(f"Stopped Docker container: {self.container_id}")
            except Exception as e:
                logger.warning(f"Error stopping container: {str(e)}")
        
        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Removed temporary directory: {self.temp_dir}")
    
    def _run_in_container(self, command: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """
        Run a command in the Docker container.
        
        Args:
            command: The command to execute
            timeout: Override the default timeout if provided
            
        Returns:
            CompletedProcess instance with stdout, stderr, and return code
        """
        cmd_timeout = timeout or self.timeout
        
        result = subprocess.run(
            ["docker", "exec", self.container_id, "/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=cmd_timeout,
            check=False
        )
        
        return result
    
    def install_package(self, package_name: str) -> bool:
        """
        Install a Python package in the container.
        
        Args:
            package_name: Name of the package to install
            
        Returns:
            True if installation succeeded, False otherwise
        """
        result = self._run_in_container(f"pip install --no-cache-dir {package_name}")
        return result.returncode == 0
    
    def execute_code(self, code: str, entry_point: str = None) -> Dict[str, Any]:
        """
        Execute Python code in the container.
        
        Args:
            code: Python code to execute
            entry_point: Optional entry point function or class to call
            
        Returns:
            Dictionary with execution results
        """
        # Create wrapper to capture output and exceptions
        wrapper_code = self._create_wrapper_code(code, entry_point)
        
        # Write code to file in the shared directory
        code_path = os.path.join(self.temp_dir, "code_to_test.py")
        wrapper_path = os.path.join(self.temp_dir, "wrapper.py")
        results_path = os.path.join(self.temp_dir, "results.json")
        
        with open(code_path, "w") as f:
            f.write(code)
        
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)
        
        # Execute code in container with timeout
        result = self._run_in_container(f"python /workspace/wrapper.py")
        
        # Read results
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    execution_results = json.load(f)
            except json.JSONDecodeError:
                execution_results = {
                    "success": False,
                    "error": "Failed to parse results JSON file",
                    "stdout": "",
                    "stderr": "",
                }
        else:
            execution_results = {
                "success": False,
                "error": f"Execution failed with code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        
        return execution_results
    
    def _create_wrapper_code(self, code: str, entry_point: str = None) -> str:
        """
        Create a wrapper script that executes the code and captures results.
        
        Args:
            code: The code to execute
            entry_point: Optional entry point to call
            
        Returns:
            String containing the wrapper script
        """
        # Default smoke test execution if no entry point specified
        if not entry_point:
            entry_point_code = """
# Try to identify and call the main function or simulation class
entry_point_found = False

# Option 1: Call main() function if it exists
if 'main' in locals() and callable(locals()['main']):
    locals()['main']()
    entry_point_found = True

# Option 2: Create and run a Simulation if it exists
elif 'Simulation' in locals() and callable(locals()['Simulation']):
    # Use small test parameters for quick execution
    sim_params = {
        "population_size": 10,
        "initial_infected_count": 1,
        "transmission_probability": 0.1,
        "recovery_probability_per_step": 0.05,
        "simulation_steps": 3,
        "random_seed": 42
    }
    
    try:
        sim = locals()['Simulation'](sim_params)
        sim.run()
        entry_point_found = True
    except Exception as e:
        raise RuntimeError(f"Failed to run Simulation: {str(e)}")

# Report if no entry point was found
if not entry_point_found:
    raise RuntimeError("No entry point (main function or Simulation class) found")
"""
        else:
            # Custom entry point execution
            entry_point_code = f"{entry_point}"

        # Create the wrapper code
        wrapper_code = f"""
import sys
import io
import json
import time
import traceback

# Redirect stdout and stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = stdout_capture
sys.stderr = stderr_capture

# Track execution metrics
start_time = time.time()
peak_memory = 0

# Results dictionary
results = {{
    "success": False,
    "error": "",
    "stdout": "",
    "stderr": "",
    "execution_time": 0,
    "peak_memory_mb": 0
}}

try:
    # Execute the original code
    with open("/workspace/code_to_test.py", "r") as f:
        code = f.read()
    
    # Compile and execute the code
    code_namespace = {{}}
    exec(code, code_namespace)
    
    # Execute the entry point
    try:
{textwrap.indent(entry_point_code, "        ")}
        
        results["success"] = True
    except Exception as entry_point_error:
        results["success"] = False
        results["error"] = f"Entry point execution failed: {{str(entry_point_error)}}"
        results["stderr"] += traceback.format_exc()

except Exception as e:
    results["success"] = False
    results["error"] = f"Code execution failed: {{str(e)}}"
    results["stderr"] += traceback.format_exc()

finally:
    # Capture execution metrics
    results["execution_time"] = time.time() - start_time
    
    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    # Store captured output
    results["stdout"] = stdout_capture.getvalue()
    results["stderr"] = stderr_capture.getvalue()
    
    # Write results to file
    with open("/workspace/results.json", "w") as f:
        json.dump(results, f)
"""
        return wrapper_code


class DependencyAnalyzer:
    """
    Analyzes code dependencies to identify required packages.
    """
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        # Set of standard library modules
        self.stdlib_modules = set(sys.builtin_module_names)
        
        # Add modules from standard library
        stdlib_path = os.path.dirname(os.__file__)
        if os.path.exists(stdlib_path):
            for module in os.listdir(stdlib_path):
                if module.endswith('.py'):
                    self.stdlib_modules.add(module[:-3])
    
    def extract_imports(self, code: str) -> List[str]:
        """
        Extract all import statements from the code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List of import statements
        """
        import_statements = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                import_statements.append(line)
        
        return import_statements
    
    def identify_external_packages(self, import_statements: List[str]) -> List[str]:
        """
        Identify non-standard library packages from import statements.
        
        Args:
            import_statements: List of import statements
            
        Returns:
            List of external package names
        """
        external_packages = set()
        
        for statement in import_statements:
            parts = statement.split()
            if statement.startswith('import '):
                # Handle "import x" or "import x, y, z"
                for module in parts[1].split(','):
                    module_name = module.strip().split('.')[0]
                    if module_name not in self.stdlib_modules:
                        external_packages.add(module_name)
            elif statement.startswith('from '):
                # Handle "from x import y"
                module_name = parts[1].split('.')[0]
                if module_name not in self.stdlib_modules and module_name != '':
                    external_packages.add(module_name)
        
        return list(external_packages)
    
    def analyze_dependencies(self, code: str) -> List[str]:
        """
        Analyze code and identify external dependencies.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List of required external packages
        """
        imports = self.extract_imports(code)
        return self.identify_external_packages(imports)


class CodeVerificationSandbox:
    """
    Main class that manages the sandbox verification process.
    Combines dependency analysis with secure code execution.
    """
    
    def __init__(self, 
                 output_dir: str,
                 base_image: str = "python:3.9-slim", 
                 timeout: int = 30,
                 network_enabled: bool = False):
        """
        Initialize the code verification sandbox.
        
        Args:
            output_dir: Directory to store verification artifacts
            base_image: Docker image to use
            timeout: Maximum execution time in seconds
            network_enabled: Whether to allow network access
        """
        self.output_dir = output_dir
        self.base_image = base_image
        self.timeout = timeout
        self.network_enabled = network_enabled
        self.dependency_analyzer = DependencyAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def verify_syntax(self, code: str) -> Dict[str, Any]:
        """
        Verify the syntax of the code.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with verification results
        """
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', temp_file_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    "passed": True,
                    "errors": []
                }
            else:
                return {
                    "passed": False,
                    "errors": result.stderr.splitlines()
                }
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def verify_dependencies(self, code: str) -> Dict[str, Any]:
        """
        Verify that all dependencies can be installed.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with dependency verification results
        """
        # Extract dependencies
        required_packages = self.dependency_analyzer.analyze_dependencies(code)
        
        if not required_packages:
            return {
                "dependency_check_passed": True,
                "required_packages": [],
                "missing_packages": [],
                "error_messages": []
            }
        
        # Try to install dependencies in sandbox
        missing_packages = []
        error_messages = []
        
        with DockerSandbox(
            base_image=self.base_image,
            timeout=self.timeout,
            network_enabled=True  # Temporarily enable network for package installation
        ) as sandbox:
            for package in required_packages:
                if not sandbox.install_package(package):
                    missing_packages.append(package)
                    error_messages.append(f"Failed to install package: {package}")
        
        return {
            "dependency_check_passed": len(missing_packages) == 0,
            "required_packages": required_packages,
            "missing_packages": missing_packages,
            "error_messages": error_messages
        }
    
    def execute_smoke_test(self, code: str) -> Dict[str, Any]:
        """
        Execute a smoke test to check if the code runs without crashing.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with smoke test results
        """
        with DockerSandbox(
            base_image=self.base_image,
            timeout=self.timeout,
            network_enabled=self.network_enabled
        ) as sandbox:
            # Install dependencies first
            required_packages = self.dependency_analyzer.analyze_dependencies(code)
            for package in required_packages:
                sandbox.install_package(package)
            
            # Execute the code
            execution_results = sandbox.execute_code(code)
            
            # Save execution output for inspection
            output_file = os.path.join(self.output_dir, "smoke_test_output.json")
            with open(output_file, "w") as f:
                json.dump(execution_results, f, indent=2)
            
            return {
                "execution_success": execution_results["success"],
                "error_message": execution_results["error"],
                "stdout": execution_results["stdout"],
                "stderr": execution_results["stderr"],
                "execution_time": execution_results.get("execution_time", 0),
                "output_file": output_file
            }
    
    def verify_code(self, code: str) -> Dict[str, Any]:
        """
        Run a comprehensive verification of the code.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with complete verification results
        """
        logger.info("Starting code verification")
        
        # Step 1: Check syntax
        syntax_results = self.verify_syntax(code)
        logger.info(f"Syntax check: {'passed' if syntax_results['passed'] else 'failed'}")
        if not syntax_results["passed"]:
            logger.warning(f"Syntax errors: {syntax_results['errors']}")
        
        if not syntax_results["passed"]:
            verification_summary = {
                "passed": False,
                "stage": "syntax",
                "details": {
                    "syntax_check": syntax_results,
                    "dependency_check": None,
                    "execution_check": None
                },
                "critical_issues": [f"Syntax error: {err}" for err in syntax_results["errors"]]
            }
            
            # Save verification results
            results_file = os.path.join(self.output_dir, "verification_results.json")
            with open(results_file, "w") as f:
                json.dump(verification_summary, f, indent=2)
                
            logger.info(f"Verification failed at syntax check stage")
            logger.debug(f"Detailed verification result: {json.dumps(verification_summary, indent=2)}")
            
            return verification_summary
        
        # Step 2: Check dependencies
        dependency_results = self.verify_dependencies(code)
        logger.info(f"Dependency check: {'passed' if dependency_results['dependency_check_passed'] else 'failed'}")
        
        if dependency_results["required_packages"]:
            logger.info(f"Required packages: {', '.join(dependency_results['required_packages'])}")
        
        if not dependency_results["dependency_check_passed"]:
            logger.warning(f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}")
            logger.warning(f"Dependency errors: {dependency_results['error_messages']}")
            
            verification_summary = {
                "passed": False,
                "stage": "dependencies",
                "details": {
                    "syntax_check": syntax_results,
                    "dependency_check": dependency_results,
                    "execution_check": None
                },
                "critical_issues": [
                    f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}"
                ]
            }
            
            # Save verification results
            results_file = os.path.join(self.output_dir, "verification_results.json")
            with open(results_file, "w") as f:
                json.dump(verification_summary, f, indent=2)
                
            logger.info(f"Verification failed at dependency check stage")
            logger.debug(f"Detailed verification result: {json.dumps(verification_summary, indent=2)}")
            
            return verification_summary
        
        # Step 3: Execute smoke test
        logger.info("Starting smoke test execution")
        execution_results = self.execute_smoke_test(code)
        logger.info(f"Execution check: {'passed' if execution_results['execution_success'] else 'failed'}")
        
        if not execution_results["execution_success"]:
            logger.warning(f"Execution error: {execution_results['error_message']}")
            if execution_results["stderr"]:
                logger.debug(f"Execution stderr: {execution_results['stderr']}")
        else:
            logger.info(f"Smoke test executed successfully in {execution_results['execution_time']:.2f} seconds")
        
        # Combine all results
        verification_summary = {
            "passed": syntax_results["passed"] and 
                     dependency_results["dependency_check_passed"] and 
                     execution_results["execution_success"],
            "stage": "complete",
            "details": {
                "syntax_check": syntax_results,
                "dependency_check": dependency_results,
                "execution_check": execution_results
            },
            "critical_issues": []
        }
        
        # Collect critical issues
        if not syntax_results["passed"]:
            verification_summary["critical_issues"].extend(
                [f"Syntax error: {err}" for err in syntax_results["errors"]]
            )
        
        if not dependency_results["dependency_check_passed"]:
            verification_summary["critical_issues"].append(
                f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}"
            )
        
        if not execution_results["execution_success"]:
            verification_summary["critical_issues"].append(
                f"Execution failed: {execution_results['error_message']}"
            )
        
        # Save verification results
        results_file = os.path.join(self.output_dir, "verification_results.json")
        with open(results_file, "w") as f:
            json.dump(verification_summary, f, indent=2)
        
        # Log final verification result
        if verification_summary["passed"]:
            logger.info("Verification passed: Code is syntactically correct, all dependencies can be installed, and smoke test executed successfully")
        else:
            logger.warning(f"Verification failed with {len(verification_summary['critical_issues'])} critical issues")
            for issue in verification_summary["critical_issues"]:
                logger.warning(f"Critical issue: {issue}")
        
        logger.debug(f"Detailed verification result: {json.dumps(verification_summary, indent=2)}")
        
        return verification_summary 
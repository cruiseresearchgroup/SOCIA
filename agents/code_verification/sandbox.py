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

# Python standard library modules list
PYTHON_STDLIB_MODULES = [
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "base64", "bdb", "binascii", 
    "bz2", "calendar", "cgi", "cgitb", "chunk", "cmd", "code", "codecs", "codeop", 
    "collections", "colorsys", "compileall", "concurrent", "configparser", "contextlib", 
    "copy", "copyreg", "csv", "ctypes", "curses", "datetime", "dbm", "decimal", "difflib", 
    "dis", "distutils", "doctest", "dummy_threading", "email", "encodings", "ensurepip", 
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", 
    "formatter", "fractions", "ftplib", "functools", "gc", "getopt", "getpass", "gettext", 
    "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib", 
    "imghdr", "imp", "importlib", "inspect", "io", "ipaddress", "itertools", 
    "json", "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "macpath", 
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", 
    "msilib", "msvcrt", "multiprocessing", "netrc", "nis", "nntplib", "ntpath", 
    "numbers", "operator", "optparse", "os", "ossaudiodev", "parser", "pathlib", 
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib", 
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd", 
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline", 
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets", "select", 
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib", 
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics", 
    "string", "stringprep", "struct", "subprocess", "sunau", "symbol", "symtable", 
    "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", 
    "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter", "token", 
    "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo", 
    "types", "typing", "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", 
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound", "wsgiref", 
    "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib"
]

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
                 network_enabled: bool = False,
                 data_path: Optional[str] = None):
        """
        Initialize the Docker sandbox environment.

        Args:
            base_image: Docker image to use as the execution environment
            timeout: Maximum execution time in seconds
            max_memory: Memory limit for the container
            network_enabled: Whether to allow network access
            data_path: Data directory path (optional)
        """
        self.base_image = base_image
        self.timeout = timeout
        self.max_memory = max_memory
        self.network_enabled = network_enabled
        self.data_path = data_path
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
        # Initialize docker command
        docker_cmd = ["docker", "run", "-d", "--rm"]
        
        # Add network parameter only if needed
        if not self.network_enabled:
            docker_cmd.append("--network=none")
        else:
            # If network is enabled, we'll install packages during container creation
            docker_cmd.append("--network=host")
        
        # Add memory limit
        docker_cmd.extend(["--memory", self.max_memory])
        
        # Environment variables for data paths
        docker_cmd.extend(["-e", f"PROJECT_ROOT=/workspace"])
        
        # Add DATA_PATH environment variable if specified
        if self.data_path:
            docker_cmd.extend(["-e", f"DATA_PATH={self.data_path}"])
        else:
            docker_cmd.extend(["-e", "DATA_PATH=data"])
        
        # Mount code exchange directory to a dedicated /sandbox path to avoid overlap with /workspace
        docker_cmd.extend(["-v", f"{self.temp_dir}:/sandbox"])
        
        # Mount host data directory into container so files can be opened
        if self.data_path:
            host_data_dir = os.path.abspath(self.data_path)
            container_data_dir = os.path.join("/workspace", self.data_path)
            docker_cmd.extend(["-v", f"{host_data_dir}:{container_data_dir}"])
        
        # Add image and command to keep container running
        docker_cmd.extend([
            self.base_image, 
            "tail", "-f", "/dev/null"
        ])
        
        # Start the container
        result = subprocess.run(
            docker_cmd,
            capture_output=True, text=True, check=True
        )
        
        self.container_id = result.stdout.strip()
        logger.debug(f"Started Docker container: {self.container_id}")
        
        # Install essential packages in container
        logger.info("Installing essential packages in Docker container...")
        essential_packages = ["numpy", "pandas", "matplotlib", "networkx", "pytest"]
        if self.network_enabled:
            packages_str = " ".join(essential_packages)
            install_cmd = f"pip install --no-cache-dir {packages_str} && echo 'PACKAGES_INSTALLED_SUCCESSFULLY'"
            install_result = self._run_in_container(install_cmd)
            if "PACKAGES_INSTALLED_SUCCESSFULLY" not in install_result.stdout:
                logger.warning(f"Failed to install essential packages: {install_result.stderr}")
            else:
                logger.debug("Successfully installed essential packages in Docker container")
        else:
            logger.warning("Network is disabled, essential packages cannot be installed")
        
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
        
        logger.debug(f"Running command in container: {command}")
        
        docker_cmd = ["docker", "exec", self.container_id, "/bin/bash", "-c", command]
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=cmd_timeout,
                check=False
            )
            
            if result.returncode != 0:
                logger.debug(f"Command failed with exit code {result.returncode}")
                if result.stderr:
                    logger.debug(f"Command stderr: {result.stderr[:200]}...")
            else:
                logger.debug(f"Command succeeded, first 50 chars of output: {result.stdout[:50]}...")
            
            return result
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Command timed out after {cmd_timeout} seconds: {command}")
            return subprocess.CompletedProcess(
                args=docker_cmd,
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {cmd_timeout} seconds"
            )
    
    def install_package(self, package_name: str) -> bool:
        """
        Install a Python package in the container.
        
        Args:
            package_name: Name of the package to install
            
        Returns:
            True if installation succeeded, False otherwise
        """
        # Package name mapping for special cases
        package_mapping = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'pil': 'pillow',
            'tf': 'tensorflow',
            'plt': 'matplotlib',
            'px': 'plotly-express',
        }
        
        # Use the mapped package name if available
        actual_package = package_mapping.get(package_name.lower(), package_name)
        
        result = self._run_in_container(f"pip install --no-cache-dir {actual_package}")
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
        # Check for package imports and try to install them if network is enabled
        if self.network_enabled:
            try:
                # Extract required packages from the code
                analyzer = DependencyAnalyzer()
                packages = analyzer.analyze_dependencies(code)
                
                # Ensure essential packages are installed
                essential_packages = ["numpy", "pandas", "matplotlib", "networkx", "scikit-learn"]
                for package in essential_packages + packages:
                    self.install_package(package)
                    
                logger.debug(f"Pre-installed packages: {', '.join(essential_packages + packages)}")
            except Exception as e:
                logger.warning(f"Error pre-installing packages: {str(e)}")
        
        # Create wrapper to capture output and exceptions
        wrapper_code = self._create_wrapper_code(code, entry_point)
        
        # Write code to file in the shared sandbox directory
        code_path = os.path.join(self.temp_dir, "code_to_test.py")
        wrapper_path = os.path.join(self.temp_dir, "wrapper.py")
        results_path = os.path.join(self.temp_dir, "results.json")
        
        with open(code_path, "w") as f:
            f.write(code)
        
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)
        
        # Execute code in container with timeout (using /sandbox path)
        result = self._run_in_container(f"python /sandbox/wrapper.py")
        
        # Read results
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    execution_results = json.load(f)
                    
                # Check for "No module named" errors which indicate missing packages
                if not execution_results.get("success", False) and "No module named" in execution_results.get("error", ""):
                    # Extract the missing module name
                    import re
                    match = re.search(r"No module named '([^']+)'", execution_results.get("error", ""))
                    if match and not self.network_enabled:
                        missing_package = match.group(1)
                        execution_results["error"] = f"Missing package '{missing_package}'. Network is disabled, so packages cannot be installed. Enable network to install required packages."
                        logger.warning(f"Execution failed due to missing package and network is disabled: {missing_package}")
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
# Note: main() should be called directly at global scope in the code, not wrapped in if __name__ == "__main__"
if 'main' in code_namespace and callable(code_namespace['main']):
    code_namespace['main']()
    entry_point_found = True

# Option 2: Create and run a Simulation if it exists
elif 'Simulation' in code_namespace and callable(code_namespace['Simulation']):
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
        sim = code_namespace['Simulation'](sim_params)
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
    with open("/sandbox/code_to_test.py", "r") as f:
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
    with open("/sandbox/results.json", "w") as f:
        json.dump(results, f)
"""
        return wrapper_code


class DependencyAnalyzer:
    """
    Analyzes code dependencies to identify required packages.
    """
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        # Use the globally defined standard library modules list
        self.stdlib_modules = set(PYTHON_STDLIB_MODULES)
        
        # Add built-in modules
        self.stdlib_modules.update(sys.builtin_module_names)
        
        # Ensure we don't miss any key standard libraries
        additional_stdlib = ['json', 'collections', 'datetime', 'math', 'random', 
                            'time', 'argparse', 'csv', 'html', 'http', 'urllib', 
                            'xml', 'zlib', 'zipfile', 'tarfile', 'pickle', 'io',
                            'logging']  # Specifically ensure logging is recognized as standard library
        self.stdlib_modules.update(additional_stdlib)
    
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
                 network_enabled: bool = True,
                 data_path: Optional[str] = None):
        """
        Initialize the code verification sandbox.
        
        Args:
            output_dir: Directory to store verification artifacts
            base_image: Docker image to use
            timeout: Maximum execution time in seconds
            network_enabled: Whether to allow network access (default: True)
            data_path: Data directory path (optional)
        """
        self.output_dir = output_dir
        self.base_image = base_image
        self.timeout = timeout
        self.network_enabled = network_enabled
        self.data_path = data_path
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
        Verify that all required dependencies can be installed.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with dependency check results
        """
        # Extract required packages
        required_packages = self.dependency_analyzer.analyze_dependencies(code)
        
        if not required_packages:
            return {
                "dependency_check_passed": True,
                "required_packages": [],
                "missing_packages": [],
                "error_messages": []
            }
        
        # Package name mapping for special cases (same as in install_package)
        package_mapping = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'pil': 'pillow',
            'tf': 'tensorflow',
            'plt': 'matplotlib',
            'px': 'plotly-express',
        }
        
        # Try to install dependencies in sandbox
        missing_packages = []
        error_messages = []
        
        # Always enable network for dependency installation
        with DockerSandbox(
            base_image=self.base_image,
            timeout=self.timeout,
            network_enabled=True,  # Force network enabled for dependency installation
            data_path=self.data_path
        ) as sandbox:
            for package in required_packages:
                if not sandbox.install_package(package):
                    # Use original package name for reporting
                    missing_packages.append(package)
                    error_messages.append(f"Failed to install package: {package}")
        
        # Check if all dependencies were successfully installed
        dependency_check_passed = len(missing_packages) == 0
        
        return {
            "dependency_check_passed": dependency_check_passed,
            "required_packages": required_packages,
            "missing_packages": missing_packages,
            "error_messages": error_messages
        }
    
    def execute_smoke_test(self, code: str) -> Dict[str, Any]:
        """
        Execute a basic smoke test on the code to verify it runs.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with smoke test results
        """
        # Always enable network for smoke tests
        network_enabled = True
        
        try:
            # Create Docker sandbox for execution
            with DockerSandbox(
                base_image=self.base_image,
                timeout=self.timeout,
                network_enabled=network_enabled,
                data_path=self.data_path
            ) as sandbox:
                # Execute a basic smoke test
                result = sandbox.execute_code(code)
                
                # Check if execution was successful
                execution_success = result.get("success", False)
                error_message = result.get("error", "Unknown error")
                
                return {
                    "execution_check": execution_success,
                    "error": error_message,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", "")
                }
                
        except Exception as e:
            logger.error(f"Smoke test failed: {str(e)}")
            return {
                "execution_check": False,
                "error": f"Smoke test failed: {str(e)}",
                "stdout": "",
                "stderr": ""
            }
    
    def verify_code(self, code: str) -> Dict[str, Any]:
        """
        Verify the code for syntax, dependencies, and execution.
        
        Args:
            code: Python code to verify
            
        Returns:
            Dictionary with verification results
        """
        # Track critical issues
        critical_issues = []
        
        # Step 1: Verify Syntax
        logger.info("Starting code verification")
        syntax_results = self.verify_syntax(code)
        logger.info(f"Syntax check: {'passed' if syntax_results['passed'] else 'failed'}")
        
        if not syntax_results["passed"]:
            return {
                "passed": False,
                "stage": "syntax",
                "critical_issues": syntax_results["errors"],
                "summary": f"Code verification failed at syntax stage: {'; '.join(syntax_results['errors'])}"
            }
        
        # Step 2: Check Dependencies
        dependency_results = self.verify_dependencies(code)
        logger.info(f"Dependency check: {'passed' if dependency_results['dependency_check_passed'] else 'failed'}")
        
        if dependency_results["required_packages"]:
            logger.info(f"Required packages: {', '.join(dependency_results['required_packages'])}")
        
        if not dependency_results["dependency_check_passed"]:
            logger.warning(f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}")
            if dependency_results["error_messages"]:
                logger.warning(f"Dependency errors: {dependency_results['error_messages']}")
            
            logger.info("Verification failed at dependency check stage")
            return {
                "passed": False,
                "stage": "dependencies",
                "critical_issues": [f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}"],
                "summary": f"Code verification failed at dependencies stage: Missing dependencies: {', '.join(dependency_results['missing_packages'])}"
            }
        
        # Step 3: Execute Smoke Test
        logger.info("Starting smoke test execution")
        execution_results = self.execute_smoke_test(code)
        logger.info(f"Execution check: {'passed' if execution_results['execution_check'] else 'failed'}")
        
        if not execution_results["execution_check"]:
            logger.warning(f"Execution error: {execution_results['error']}")
            if execution_results["stderr"]:
                logger.debug(f"Execution stderr: {execution_results['stderr']}")
        else:
            logger.info(f"Smoke test executed successfully")
        
        # Combine all results
        verification_summary = {
            "passed": syntax_results["passed"] and 
                     dependency_results["dependency_check_passed"] and 
                     execution_results["execution_check"],
            "stage": "complete",
            "details": {
                "syntax_check": syntax_results["passed"],
                "dependency_check": dependency_results["dependency_check_passed"],
                "execution_check": execution_results["execution_check"],
                "required_packages": dependency_results["required_packages"],
                "missing_packages": dependency_results["missing_packages"],
                "error_messages": []
            },
            "critical_issues": critical_issues,
            "summary": "Code verification completed successfully."
        }
        
        # Collect critical issues
        if not syntax_results["passed"]:
            verification_summary["critical_issues"].extend(syntax_results["errors"])
        
        if not dependency_results["dependency_check_passed"]:
            verification_summary["critical_issues"].append(
                f"Missing dependencies: {', '.join(dependency_results['missing_packages'])}"
            )
        
        if not execution_results["execution_check"]:
            verification_summary["critical_issues"].append(
                f"Execution failed: {execution_results['error']}"
            )
        
        # Update summary based on issues
        if verification_summary["critical_issues"]:
            logger.warning(f"Verification failed with {len(verification_summary['critical_issues'])} critical issues")
            for issue in verification_summary["critical_issues"]:
                logger.warning(f"Critical issue: {issue}")
            verification_summary["summary"] = f"Code verification failed at complete stage: {verification_summary['critical_issues'][0]}"
            verification_summary["passed"] = False
        
        return verification_summary 
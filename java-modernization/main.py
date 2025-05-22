import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with proper configuration
mcp = FastMCP("java-modernization")

# Modernization indicators to check for
MODERNIZATION_INDICATORS = {
    "java.util.Date": "java.time.LocalDateTime",
    "new SimpleDateFormat": "DateTimeFormatter",
    "Vector": "ArrayList",
    "Hashtable": "HashMap or ConcurrentHashMap",
    "Enumeration": "Iterator or Stream",
    "StringBuffer": "StringBuilder",
    "synchronized": "concurrent collections or locks",
    "System.out.println": "proper logging framework",
    "throws Exception": "specific exception types",
    "catch (Exception": "specific exception handling",
    "Class.forName": "modern JDBC approaches",
    "ResultSet.*executeQuery": "try-with-resources",
    "FileInputStream": "Files.newInputStream or try-with-resources",
    "FileOutputStream": "Files.newOutputStream or try-with-resources",
    "new String\\(": "String literals or factory methods",
    '\\+ ".*" \\+': "StringBuilder or String.format",
}

# Additional patterns for more sophisticated detection
PATTERN_INDICATORS = {
    r"for\s*\(\s*int\s+\w+\s*=\s*0": "enhanced for-loop or streams",
    r"\.get\(\s*\w+\s*\)": "streams or modern collection operations",
    r"new\s+ArrayList\s*\(\s*\)": "List.of() or modern initialization",
    r"new\s+HashMap\s*\(\s*\)": "Map.of() or modern initialization",
    r"Statement\s+\w+\s*=.*createStatement": "PreparedStatement",
}


@mcp.tool()
def check_modernization_status() -> Dict:
    """
    Analyze Java files in the codebase for modernization opportunities.
    Returns a JSON object with modernization status for each file.
    """
    # For this example, we'll scan the current working directory
    # In a real implementation, you might want to make this configurable
    root_path = os.getcwd()

    # Scan for Java files
    java_files = scan_for_java_files(root_path)

    if not java_files:
        return {
            "status": "No Java files found",
            "needs_modernization": 0,
            "modernized": 0,
            "files_to_modernize": [],
            "modernized_files": [],
        }

    # Analyze each file
    results = {}
    for file_path in java_files:
        try:
            relative_path = os.path.relpath(file_path, root_path)
            needs_modernization, details = analyze_file(file_path)
            results[relative_path] = {
                "needs_modernization": needs_modernization,
                "details": details,
                "file_size": os.path.getsize(file_path),
                "last_modified": os.path.getmtime(file_path),
            }
        except Exception as e:
            results[relative_path] = {
                "needs_modernization": True,
                "details": {"Error analyzing file": str(e)},
                "file_size": 0,
                "last_modified": 0,
            }

    # Create dashboard data
    dashboard_data = create_dashboard_data(results)
    return dashboard_data


def scan_for_java_files(root_path: str) -> List[str]:
    """Find all Java files in the given root path."""
    java_files = []
    try:
        root_path_obj = Path(root_path)
        if not root_path_obj.exists():
            return java_files

        # Look for Java files, excluding common build directories
        exclude_dirs = {".git", "target", "build", ".gradle", "node_modules", ".idea"}

        for path in root_path_obj.rglob("*.java"):
            # Skip files in excluded directories
            if not any(excluded in path.parts for excluded in exclude_dirs):
                java_files.append(str(path))

    except Exception as e:
        print(f"Error scanning for Java files: {e}")

    return java_files


def analyze_file(file_path: str) -> Tuple[bool, Dict[str, int]]:
    """Analyze a Java file for modernization indicators."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        indicators_found = {}
        needs_modernization = False

        # Check for direct string matches
        for indicator, replacement in MODERNIZATION_INDICATORS.items():
            if indicator.startswith("\\") or "(" in indicator:
                # This is likely a regex pattern, handle it separately
                continue

            occurrences = content.count(indicator)
            if occurrences > 0:
                needs_modernization = True
                indicators_found[f"Replace '{indicator}' with {replacement}"] = (
                    occurrences
                )

        # Check for regex patterns
        for pattern, replacement in PATTERN_INDICATORS.items():
            try:
                matches = re.findall(pattern, content)
                if matches:
                    needs_modernization = True
                    indicators_found[
                        f"Modernize pattern (detected {len(matches)} occurrences): {replacement}"
                    ] = len(matches)
            except re.error as e:
                print(f"Regex error for pattern {pattern}: {e}")

        # Additional checks for common legacy patterns
        legacy_patterns = [
            (
                "Raw types usage",
                r"\b(ArrayList|HashMap|List|Map)\s+\w+\s*=\s*new\s+(ArrayList|HashMap)",
            ),
            ("Old-style loops", r"for\s*\(\s*int\s+\w+\s*=\s*0.*\.size\(\)"),
            ("Unsafe casting", r"\(\s*(String|Integer|Object|\w+)\s*\)"),
            (
                "Manual resource management",
                r"(FileInputStream|FileOutputStream|BufferedReader).*(?!try-with-resources)",
            ),
        ]

        for description, pattern in legacy_patterns:
            try:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                if matches:
                    needs_modernization = True
                    indicators_found[description] = len(matches)
            except re.error:
                continue

        return needs_modernization, indicators_found

    except Exception as e:
        return True, {"Error analyzing file": str(e)}


def create_dashboard_data(results: Dict[str, Dict]) -> Dict:
    """
    Create a dashboard representation of the modernization status.
    Returns a dictionary that can be easily processed.
    """
    files_to_modernize = []
    modernized_files = []
    total_issues = 0

    for file_path, status in results.items():
        if status["needs_modernization"]:
            issue_count = (
                sum(status["details"].values())
                if isinstance(status["details"], dict)
                else 1
            )
            total_issues += issue_count

            files_to_modernize.append(
                {
                    "filename": file_path,
                    "issues": status["details"],
                    "issue_count": issue_count,
                    "file_size": status.get("file_size", 0),
                }
            )
        else:
            modernized_files.append(
                {"filename": file_path, "file_size": status.get("file_size", 0)}
            )

    # Sort files by issue count (most problematic first)
    files_to_modernize.sort(key=lambda x: x["issue_count"], reverse=True)

    dashboard = {
        "summary": {
            "total_files": len(results),
            "needs_modernization": len(files_to_modernize),
            "modernized": len(modernized_files),
            "total_issues": total_issues,
            "modernization_percentage": (
                round((len(modernized_files) / len(results)) * 100, 1) if results else 0
            ),
        },
        "files_to_modernize": files_to_modernize,
        "modernized_files": modernized_files,
        "top_issues": get_top_issues(files_to_modernize),
    }

    return dashboard


def get_top_issues(files_to_modernize: List[Dict]) -> Dict[str, int]:
    """Get the most common modernization issues across all files."""
    issue_counts = {}

    for file_info in files_to_modernize:
        for issue, count in file_info["issues"].items():
            if isinstance(count, int):
                issue_counts[issue] = issue_counts.get(issue, 0) + count

    # Return top 10 issues
    return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])


@mcp.tool()
def get_modernization_recommendations(filename: str) -> Dict:
    """
    Get specific modernization recommendations for a given file.
    """
    if not filename.endswith(".java"):
        return {"error": "Please provide a Java file (.java extension)"}

    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}

    try:
        needs_modernization, details = analyze_file(file_path)

        recommendations = []
        for issue, count in details.items():
            recommendation = generate_recommendation(issue, count)
            if recommendation:
                recommendations.append(recommendation)

        return {
            "filename": filename,
            "needs_modernization": needs_modernization,
            "recommendations": recommendations,
            "priority": (
                "high"
                if len(recommendations) > 10
                else "medium" if len(recommendations) > 5 else "low"
            ),
        }

    except Exception as e:
        return {"error": f"Error analyzing file: {str(e)}"}


def generate_recommendation(issue: str, count: int) -> Optional[Dict]:
    """Generate a specific recommendation based on the issue found."""
    recommendations_map = {
        "Vector": {
            "description": "Replace Vector with ArrayList or ConcurrentLinkedQueue for thread-safety",
            "example": "List<String> list = new ArrayList<>(); // instead of Vector<String>",
            "priority": "high",
        },
        "Hashtable": {
            "description": "Replace Hashtable with HashMap or ConcurrentHashMap",
            "example": "Map<String, String> map = new HashMap<>(); // or ConcurrentHashMap for thread-safety",
            "priority": "high",
        },
        "synchronized": {
            "description": "Consider using concurrent collections or explicit locks",
            "example": "Use ConcurrentHashMap, ReentrantLock, or java.util.concurrent utilities",
            "priority": "medium",
        },
    }

    for key, recommendation in recommendations_map.items():
        if key in issue:
            return {
                "issue": issue,
                "occurrences": count,
                "description": recommendation["description"],
                "example": recommendation["example"],
                "priority": recommendation["priority"],
            }

    return {
        "issue": issue,
        "occurrences": count,
        "description": f"Found {count} occurrences that may need modernization",
        "priority": "medium",
    }


if __name__ == "__main__":
    # Run the server
    mcp.run()

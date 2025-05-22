# MCP Servers - Java Application Modernization and MongoDB Adoption Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
  - [Java Modernization Server](#java-modernization-server)
  - [MongoDB Adoption Server](#mongodb-adoption-server)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Claude Desktop](#claude-desktop)
  - [Claude Code CLI](#claude-code-cli)
  - [Java Modernization Server](#java-modernization-server-1)
  - [MongoDB Adoption Server](#mongodb-adoption-server-1)
- [Best Practices](#best-practices)
- [GenAI Integration](#genai-integration)
- [Demo](#demo)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Roadmap](#roadmap)

## Overview

This project consists of two MCP (Model Context Protocol) servers designed to assist in modernizing and analyzing legacy Java applications, with a specific focus on the 'kitchensink' application. The tool provides automated analysis and recommendations for both Java code modernization and MongoDB adoption strategies.

## Architecture

The project uses a dual-server architecture:
- **Java Modernization Server**: Handles code analysis and modernization recommendations
- **MongoDB Adoption Server**: Manages database pattern analysis and migration strategies

Both servers communicate through Claude's MCP protocol, enabling seamless integration with both Claude Desktop and Claude Code CLI.

## Project Structure

The project contains two main components:

1. **Java Modernization Server** (`java-modernization/`)
   - Analyzes Java code for modernization opportunities
   - Identifies legacy patterns and suggests modern alternatives
   - Provides detailed recommendations for code improvements

2. **MongoDB Adoption Server** (`mongodb-adoption/`)
   - Analyzes codebase for database usage patterns
   - Provides MongoDB adoption strategies
   - Offers documentation and best practices for migration

## Features

### Java Modernization Server
- Detects legacy Java patterns and suggests modern alternatives
- Identifies opportunities for:
  - Modern date/time API usage
  - Collection framework improvements
  - Exception handling best practices
  - Resource management modernization
  - String handling optimizations
- Generates detailed modernization reports
- Provides specific recommendations for each file

### MongoDB Adoption Server
- Analyzes existing database patterns
- Identifies potential MongoDB use cases
- Provides migration strategies
- Offers documentation search and recommendations
- Generates adoption plans based on codebase analysis

## Prerequisites

### Required
- Python 3.8 or higher
- Git
- MongoDB (for testing migrations)

### Recommended Tools
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/getting-started)
- [Claude Desktop](https://support.anthropic.com/en/articles/10065433-installing-claude-for-desktop)
- [uv](https://docs.astral.sh/uv/) for managing Python projects

## Quick Start

1. Install Claude Desktop and Claude Code CLI
2. Clone this repository
3. Set up both MCP servers using the installation instructions below
4. Start modernizing your Java codebase!

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up the Java Modernization server:
```bash
cd java-modernization
uv -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv run mcp install main.py  # sends server to Claude Desktop
```

3. Set up the MongoDB Adoption server:
```bash
cd mongodb-adoption
uv -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv install -r requirements.txt
uv run mcp install main.py  # sends server to Claude Desktop
```

## Usage

### Claude Desktop

Verify the MCP servers have been added. You can check by going to Setting > Developer > Edit Config. The file `claude_desktop_config.json` should look something like this:

```json
{
  "mcpServers": {
    "java-modernization": {
      "command": "YOUR_PATH_TO_UV_HERE",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "YOUR_PATH_TO_JAVA-MODERNIZATION_MAIN.PY_HERE"
      ]
    },
    "mongodb-adoption-assistant": {
      "command": "YOUR_PATH_TO_UV_HERE",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "YOUR_PATH_TO_MONGODB-ADOPTION-ASSISTANT_MAIN.PY_HERE"
      ]
    }
  }
}
```

### Claude Code CLI

Go to your terminal and open the 'kitchensink' directory or any other java codebase you want to modernize. Then, run the following:

```bash
claude mcp add-from-claude-desktop  # adds MCP servers from Claude Desktop to CLI
claude mcp list     # verify you have the two MCP servers
claude      # get started with Claude Code CLI
```

### Java Modernization Server

You can start with phrases like:
- Help me modernize my java files
- How can I modernize the java files in my codebase?

### MongoDB Adoption Server

You can start with phrases like:
- Analyze this codebase for MongoDB adoption
- How can I adopt MongoDB in this project?

## Best Practices

1. **Code Modernization**
   - Start with smaller, isolated files before tackling larger components
   - Review recommendations before applying changes
   - Keep a backup of original code
   - Test thoroughly after each modernization step

2. **MongoDB Adoption**
   - Begin with non-critical data models
   - Validate migration strategies in a staging environment
   - Document all schema changes
   - Plan for data migration downtime

## GenAI Integration

This project leverages Generative AI in several ways:

1. **Code Analysis**
   - Uses AI to identify patterns and anti-patterns in Java code
   - Generates context-aware modernization recommendations

2. **Documentation Processing**
   - Processes MongoDB documentation using AI-powered text analysis
   - Provides relevant documentation snippets based on codebase context

3. **Migration Planning**
   - Uses AI to generate customized migration strategies
   - Provides intelligent recommendations based on codebase analysis

4. **Claude**
    - Manages a lot of the built-in tooling and functionality

5. **Cursor**
    - Fast iteration and development of this project, everything from error handling to documentation

## Demo

### Java Modernization Server Demo - Using Claude Code CLI
[![Java Modernization with Claude Code CLI](demos/java-tool-claude-cli.mp4)](demos/java-tool-claude-cli.mp4)
- Demonstration of code analysis and modernization workflow
- Showcase of real-time pattern detection and recommendations
- Example of modernizing legacy Java code patterns
- Integration with Claude Code CLI for seamless development

### MongoDB Adoption Server Demo - Using Claude Desktop
[![MongoDB Adoption with Claude Desktop](demos/mongo-tool-calude-desktop.mp4)](demos/mongo-tool-calude-desktop.mp4)
- Database pattern analysis and migration strategy generation
- Documentation search and recommendations
- Adoption plan creation and implementation
- Integration with Claude Desktop for enhanced visualization

## Limitations

- Currently supports Java codebases only
- Requires Python 3.8 or higher
- MongoDB adoption analysis limited to common database patterns
- Large codebases may require additional processing time

## Troubleshooting

### Common Issues

1. **MCP Server Connection Issues**
   - Verify Claude Desktop is running
   - Check server paths in config file
   - Ensure Python environment is activated

2. **Modernization Analysis Problems**
   - Check file permissions
   - Verify Java file encoding
   - Ensure sufficient system resources

3. **MongoDB Adoption Issues**
   - Verify MongoDB connection
   - Check database credentials
   - Ensure proper network access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- Support for additional programming languages
- Visualize Java modernization with an autogenerated dashboard
    - priority matrix for better triaging
    - progress bar for easy reporting
- Enhanced MongoDB MCP server
    - data collection: collect more updated documentation through an API
    - data processing: enhance RAG pipeline for better document awareness and code context
    - data storage/retrieval: better caching mechanisms for faster processing and data retrieval 
- Integration with popular IDEs
- Automated testing framework